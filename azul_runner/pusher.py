"""Handle full Plugin execution when the plugin sources the original event itself rather than dispatcher.

* Utility to Generate and send sourced events to an external source.
* Run job based on an input event.
* Send job results to external source
* Run required multiplugins
* Handle timeouts and critical errors
"""

import hashlib
import io
import logging
from typing import Any, Type

import pendulum
from azul_bedrock import models_network as azm

from azul_runner import settings, storage
from azul_runner.coordinator import Coordinator
from azul_runner.network_transform import gen_author
from azul_runner.plugin import Plugin

logger = logging.getLogger(__name__)


def md5(text: str):
    """Return string md5 representing incoming text."""
    return hashlib.md5(text.encode()).hexdigest()  # noqa: S303 # nosec B303, B324


class Pusher(Coordinator):
    """Allows plugins to publish binary sourced events with some basic structure.

    Note: not thread safe because of the way plugin features are tracked.
    """

    def __init__(self, plugin_class: Type[Plugin], config: dict[str, Any], is_register_plugin: bool = True):
        """Start the pusher with the provided configuration."""
        super().__init__(plugin_class, settings.parse_config(plugin_class, config))
        if not self.plugin._IS_USING_PUSHER:
            raise ValueError("Pusher cannot be used unless the _IS_USING_PUSHER flag is set to true.")

        if is_register_plugin:
            # Register the plugin.
            self._network.post_registrations()

    def generate_base_mapped_source(self, source_label: str, references: dict[str, str], security: str) -> azm.Source:
        """Generate a base Source event that can be used for pushing depth 0 (root level) mapped events."""
        timestamp = pendulum.now(pendulum.UTC)
        return azm.Source(
            name=source_label,
            timestamp=timestamp,
            references=references,
            path=[],
            security=security,
        )

    def _gen_mapped_event(
        self,
        *,
        source_info: azm.Source,
        source_file_info: azm.FileInfo,
        security: str,
        relationship: dict[str, str] | None = None,
        filename: str = "",
    ) -> azm.BinaryEvent:
        """Generate mapped events based on the existing source and file information."""
        sha256 = source_file_info.sha256
        if not sha256:
            raise Exception(f"Source file info must have its sha256 set and it is currently {sha256}")

        input_entity = source_file_info.to_input_entity()
        if filename:
            input_entity.features.append(
                azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value=filename)
            )
        # If this is a root level event use the same timestamp as the source event.
        if len(source_info.path) == 0:
            timestamp = pendulum.from_timestamp(source_info.timestamp.timestamp())
        else:
            timestamp = pendulum.now(pendulum.UTC)

        if not relationship:
            relationship = {}

        author = gen_author(self.plugin, None)
        author.security = security

        source_copy = source_info.model_copy(deep=True)
        source_copy.path.append(
            azm.PathNode(
                sha256=sha256,
                action=azm.BinaryAction.Mapped,
                timestamp=timestamp,
                author=author,
                relationship=relationship,
                file_format=input_entity.file_format,
                file_format_legacy=input_entity.file_format_legacy,
                size=input_entity.size,
                filename=filename if filename else None,
            )
        )

        return azm.BinaryEvent(
            kafka_key=f"{author.name}-placeholder",  # temporary id so we can create the object
            dequeued=".".join(
                [
                    sha256,
                    author.name,
                    author.version if author.version else "",
                    timestamp.to_iso8601_string(),
                ]
            ),
            action=azm.BinaryAction.Mapped,
            model_version=azm.CURRENT_MODEL_VERSION,
            timestamp=timestamp,
            author=author,
            entity=input_entity,
            source=source_copy,
        )

    def _gen_sourced_event(
        self,
        *,
        input_entity: azm.BinaryEvent.Entity,
        source_label: str,
        references: dict[str, str],
        filename: str = "",
        security: str | None = None,
    ) -> azm.BinaryEvent:
        """Generate a sourced event that can have features added to it."""
        timestamp = pendulum.now(pendulum.UTC)
        if not input_entity.sha256:
            raise ValueError(f"Input entity must have it's sha256 set and it is currently {input_entity.sha256}")

        author = gen_author(self.plugin, None)
        author.security = security

        return azm.BinaryEvent(
            kafka_key=f"{author.name}-placeholder",  # temporary id so we can create the object
            dequeued=".".join(
                [
                    input_entity.sha256,
                    author.name,
                    author.version if author.version else "",
                    timestamp.to_iso8601_string(),
                ]
            ),
            action=azm.BinaryAction.Sourced,
            model_version=azm.CURRENT_MODEL_VERSION,
            timestamp=timestamp,
            author=author,
            entity=input_entity,
            source=azm.Source(
                name=source_label,
                timestamp=timestamp,
                references=references,
                path=[
                    azm.PathNode(
                        author=author,
                        action=azm.BinaryAction.Sourced,
                        timestamp=timestamp,
                        sha256=input_entity.sha256,
                        filename=filename if filename else None,
                        size=input_entity.size,
                        file_format=input_entity.file_format,
                        file_format_legacy=input_entity.file_format_legacy,
                    )
                ],
                security=security,
            ),
        )

    def push_once_sourced(
        self,
        *,
        content: bytes,
        source_label: str,
        references: dict[str, str],
        security: str,
        filename: str = "",
        local: list[storage.StorageProxyFile] | None = None,
    ) -> str:
        """Run the plugin on the provided input content and generate a sourced event for the provided source_label.

        Returns the sha256 of the provided content.
        """
        sha256 = storage.calc_stream_hash(io.BytesIO(content), hashlib.sha256)

        logger.info(f"Uploading - {sha256} - {filename} - {max(len(content) // 1024, 1)}kb - {security}")

        file_info_dict = self._network._post_data(
            source_label,
            {sha256: ([azm.DataLabel.CONTENT], io.BytesIO(content))},
        )
        file_info = file_info_dict[sha256]

        if file_info is None:
            raise Exception(f"The file with sha256 {sha256} was uploaded but no metadata was returned.")
        if file_info.sha256 is None:
            file_info.sha256 = sha256

        input_entity = file_info.to_input_entity()

        if filename:
            input_entity.features.append(
                azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value=filename)
            )
        # Sourced event because content was provied.
        root_event = self._gen_sourced_event(
            input_entity=input_entity,
            source_label=source_label,
            references=references,
            filename=filename,
            security=security,
        )
        self._run_plugin(root_event, local=local)
        return sha256

    def push_once_mapped(
        self,
        *,
        source_file_info: azm.FileInfo,
        source_info: azm.Source,
        security: str,
        relationship: dict[str, str] | None = None,
        filename: str = "",
        local: list[storage.StorageProxyFile] | None = None,
    ):
        """Map the provided metadata onto the provided source_info as a mapped event.

        NOTE - if this is a newly generated event use the generate_base_mapped_source function on pusher.
        """
        root_event = self._gen_mapped_event(
            source_file_info=source_file_info,
            security=security,
            source_info=source_info,
            relationship=relationship,
            filename=filename,
        )
        return self._run_plugin(root_event, local=local)

    def _run_plugin(
        self,
        root_event: azm.BinaryEvent,
        local: list[storage.StorageProxyFile] | None = None,
    ):
        if not self._cfg.events_url:
            raise ValueError("Cannot run and post results when events_url is None")

        if self._watched and self._watched.check_updated():
            self._recreate_plugin()
            logger.info("Plugin has been restarted due to a file change")

        try:
            # NOTE - root_event can be a mapped or sourced event depending on if a content stream was provided.
            for _result, _multi in self._run_job(root_event, local=local):
                self._network.ack_job(root_event, _result, _multi)
                # close any open handles in JobResult
                _result.close()

        except Exception as e:  # Unknown error so extra logging required.
            logger.error(
                f"Error {e} during job with kafka_key: '{root_event.kafka_key}', "
                + f"action: '{root_event.action}', author info: {root_event.author.model_dump()}"
                + f"\n source info: {root_event.source.model_dump()}."
            )
            raise
