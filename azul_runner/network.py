"""Higher level interactions with Azul Dispatcher vs dispatcher.py."""

import logging
import time
import traceback
import typing

from azul_bedrock import dispatcher, exceptions
from azul_bedrock import models_network as azm

from . import network_transform, plugin
from .models import JobResult, State

logger = logging.getLogger(__name__)

# by default will retry getting events if it received corrupted events
CONTINUE_ON_RECV_CORRUPT_EVENT = True


class Network:
    """Handle plugin communications with dispatcher."""

    def __init__(self, p: plugin.Plugin):
        self.plugin = p

        # Cache of metadata for content that has been posted to the server already.
        # Entries are maintained in order by recent usage, so LRU entries can be discarded when the cache limit is
        # reached. Old entries are only dropped on completion of the current job, never during a run.
        self._cached_file_data: dict[tuple[str, str, str], azm.Datastream] = {}

        self.api = dispatcher.DispatcherAPI(
            events_url=self.plugin.cfg.events_url,
            data_url=self.plugin.cfg.data_url,
            retry_count=self.plugin.cfg.request_retry_count,
            timeout=self.plugin.cfg.request_timeout,
            author_name=self.plugin.NAME,
            author_version=self.plugin.VERSION,
            deployment_key=self.plugin.cfg.deployment_key or self.plugin.NAME,
        )

    def _clear_file_metadata(self):
        """Clear metadata cache between runs."""
        self._cached_file_data = {}

    def post_registrations(self):
        """Post all registration events to dispatcher."""
        for row in network_transform.get_registrations(self.plugin):
            logger.info(f"Registering plugin {row.author.name}-{row.author.version if row.author.version else ''}")
            self.api.submit_events([row], model=azm.ModelType.Plugin)

    def fetch_job(self) -> azm.BinaryEvent:
        """Fetches next job from the queue."""

        def fmt_dict_filters(filt: dict[str, list[str]]) -> list[str]:
            """Merge these filters to network format.

            Potentially should be cached as otherwise same work performed every query.
            """
            ret = []
            for k, v in filt.items():
                ret.append(f"{k},{','.join(v)}")
            return ret

        if not self.plugin.cfg.events_url:
            raise ValueError("Cannot fetch jobs when events_url is None")
        while True:
            try:
                info, events = self.api.get_binary_events(
                    count=1,  # only retrieve 1 event, as if plugin crashes these will not reprocess
                    is_task=True,
                    deadline=10,  # dispatcher has up to 10 seconds to retrieve/filter events
                    # filters
                    require_expedite=self.plugin.cfg.require_expedite,
                    require_live=self.plugin.cfg.require_live,
                    require_historic=self.plugin.cfg.require_historic,
                    require_under_content_size=self.plugin.cfg.filter_max_content_size,
                    require_over_content_size=self.plugin.cfg.filter_min_content_size,
                    require_actions=self.plugin.cfg.filter_allow_event_types,
                    deny_self=self.plugin.cfg.filter_self,
                    require_streams=fmt_dict_filters(self.plugin.cfg.filter_data_types),
                )
            except dispatcher.BadResponseException as e:
                logger.warning(f"Failed to decode event from server ({e.__class__.__name__}):\n{str(e)}")
                logger.warning(f"Failed content:\n{e.content}")
                logger.warning(f"Error:\n{traceback.format_exc()}")
                # immediately repoll for new events
                if CONTINUE_ON_RECV_CORRUPT_EVENT:
                    continue
                raise e

            if info.filtered:
                logger.info(f"{info.filtered} uninteresting events filtered by dispatcher")
                if not info.fetched:
                    # immediately repoll for new events
                    continue
            if not events:
                logger.info("No events fetched or filtered by dispatcher; backing off for 10 seconds")
                time.sleep(10)
                continue
            if len(events) > 1:
                raise ValueError(f"{len(events)} events fetched by dispatcher, only 1 allowed")

            event = events[0]
            return event

    def _gen_status(self, src: azm.BinaryEvent, result: JobResult, multiplugin: str | None = None) -> azm.StatusEvent:
        # transform hash-data dictionary to contain labels as well
        labelled_data = {}
        for hash, data in result.data.items():
            labelled_data.setdefault(hash, ([], data))
        for ev in result.events:
            for data in ev.data:
                labelled_data[data.hash][0].append(data.label)
        # post data to dispatcher and receive metadata
        hash_to_meta = self._post_data(src.source.name, labelled_data)
        author_tmpl = network_transform.gen_author(self.plugin, self.plugin.get_multiplugin(multiplugin))
        status_event = network_transform.gen_processing_events(self.plugin, hash_to_meta, author_tmpl, src, result)

        return status_event

    def ack_job(self, src: azm.BinaryEvent, result: JobResult, multiplugin: str | None = None) -> None:
        """Send the result of the job to the dispatcher."""
        try:
            status_event = self._gen_status(src, result, multiplugin)
        except Exception as e:
            # try to capture error
            status_event = self._gen_status(
                src,
                JobResult(
                    state=State(
                        State.Label.ERROR_OUTPUT,
                        failure_name=e.__class__.__name__,
                        message=traceback.format_exc(),
                    ),
                    date_start=result.date_start,
                    date_end=result.date_end,
                    runtime=result.runtime,
                ),
            )

        # use logging lib to format string for performance
        logger.debug("Posting status event: %s", status_event)
        # Post result
        try:
            self.api.submit_events([status_event], model=azm.ModelType.Status)
        except exceptions.NetworkDataException as e:
            # try to capture bad output from plugin
            # rerunning the plugin should produce the same result
            # so no need to capture detailed info
            status_event = self._gen_status(
                src,
                JobResult(
                    state=State(State.Label.ERROR_OUTPUT, message=e.args[0]),
                    date_start=result.date_start,
                    date_end=result.date_end,
                    runtime=result.runtime,
                ),
            )
            self.api.submit_events([status_event], model=azm.ModelType.Status)

        # clear file metadata after events have been submitted
        self._clear_file_metadata()

    def _post_data(
        self, source: str, data: dict[str, tuple[list[azm.DataLabel], typing.BinaryIO]]
    ) -> dict[str, azm.Datastream]:
        """Posts data and receives any calculated metadata.

        The server wants us to set metadata on the data streams in downstream jobs/events.
        After completion, self.posted_data should contain ContentEntry entries for each data stream in `data`.
        """
        ret: dict[str, azm.Datastream] = {}
        if not self.plugin.cfg.data_url or not data:
            return ret
        # post all data
        for data_hash, (labels, binary) in data.items():
            for label in labels:
                # attempt to load from cache
                if (source, label, data_hash) in self._cached_file_data:
                    continue

                logger.info("Posting data to server for %s %s %s" % (source, label, data_hash))
                file_info = self.api.submit_binary(source=source, label=label, data=binary)
                file_info.label = label
                # We'll take the returned APIContentEntry as-is, but later replace the 'label' value in _ack_job
                ret[data_hash] = self._cached_file_data[(source, label, data_hash)] = file_info
        return ret

    def get_cached_file(self, source: str, label: azm.DataLabel, data_hash: str) -> azm.Datastream | None:
        """Get cached file info or none if it doesn't exist."""
        return self._cached_file_data.get((source, label, data_hash))
