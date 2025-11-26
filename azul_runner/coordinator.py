"""Handle full Plugin execution loop.

* Receive jobs from external source or local
* Send job results to external source or local
* Run required multiplugins
* Handle timeouts and critical errors
"""

import contextlib
import datetime
import logging
import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.util
import queue as queueLib
import signal
import subprocess  # noqa: S404 # nosec B404
import sys
import time
import traceback
from typing import Any, Generator, Optional, Type

from azul_bedrock import models_network as azm
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import models, network
from . import plugin as mplugin
from . import plugin_executor, settings
from .models import JobResult, State, TaskModel
from .storage import StorageError, StorageProxyFile

logger = logging.getLogger(__name__)


QUEUE_PUT_TIMEOUT = 0.5


class CriticalError(Exception):
    """Something has gone wrong that indicates the plugin needs to exit."""

    pass


class SigTermExitError(Exception):
    """Sig term has been provided to the process and it is not exiting."""

    pass


class RecreateException(Exception):
    """Plugin needs to be re-created by it's parent process.

    This is raised when the source plugin has detected a file change.
    """

    pass


def get_git_version_suffix(config: settings.Settings) -> str | None:
    """Get the git version suffix for the watched repo if there is one."""
    if config.watch_path and config.watch_type == settings.WatchTypeEnum.GIT:
        # wait for valid git repo on disk
        for _ in range(3):
            retcode = subprocess.call(  # noqa: S603, S607 # nosec B603 B607
                ["git", "status"],
                cwd=config.watch_path,
                shell=False,
            )
            if retcode == 0:
                break
            logger.warning(f"'git status' failed with {retcode} on {config.watch_path}, sleeping")
            time.sleep(config.watch_wait)

        # recalculate commit hash as version_suffix
        # requires git to be installed
        try:
            output = subprocess.check_output(  # noqa: S603, S607 # nosec B603 B607
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=config.watch_path,
                shell=False,
            )
        except subprocess.CalledProcessError as e:
            raise CriticalError(f"is git not installed or {config.watch_path} not a valid git checkout?") from e
        return output.decode().strip()
    return None


class WatchPath(FileSystemEventHandler):
    """Event handler for watchdog."""

    def __init__(self, watch_wait: int) -> None:
        self._restart_after = None
        self._watch_wait = watch_wait
        super().__init__()

    def on_created(self, event):
        """Event trigger."""
        self._update(event)

    def on_deleted(self, event):
        """Event trigger."""
        self._update(event)

    def on_moved(self, event):
        """Event trigger."""
        self._update(event)

    def on_modified(self, event):
        """Event trigger."""
        self._update(event)

    def _update(self, event: FileSystemEvent):
        """Trigger restart if necessary."""
        logger.info(f"Watched file {event.event_type}: {event.src_path}")
        if not self._restart_after:
            logger.info(f"Plugin will be restarted after at least {self._watch_wait} seconds.")
            # delay in case multiple files are being rewritten
            self._restart_after = time.time() + self._watch_wait

    def check_updated(self) -> bool:
        """Return true if path was updated."""
        if self._restart_after and self._restart_after < time.time():
            self._restart_after = None
            return True
        else:
            return False


class Coordinator:
    """Manage continuous plugin execution and prevent bad jobs from stalling plugin."""

    _plugin: mplugin.Plugin

    def __init__(
        self,
        plugin_class: Type[mplugin.Plugin],
        config: settings.Settings,
    ):
        self._plugin_class = plugin_class
        if isinstance(config, dict):
            raise ValueError("Config should be a Settings object not a dictionary!")
        self._cfg = config

        self.is_signalled_to_exit = False
        signal.signal(signal.SIGINT, self.set_signal_exit)
        signal.signal(signal.SIGTERM, self.set_signal_exit)

        self._watchdog: Any | None = None
        self._watched: WatchPath | None = None

        # start watchdog
        if self._cfg.watch_path:
            self._watchdog = Observer()
            self._watched = WatchPath(self._cfg.watch_wait)
            # sleep before starting plugin
            # In Kubernetes, the sidecar is unlikely to have completed sync yet
            # so this sleep avoids kubernetes pod restarts.
            time.sleep(self._cfg.watch_wait)
            self._watchdog.schedule(self._watched, self._cfg.watch_path, recursive=True)
            self._watchdog.start()
        self._recreate_plugin()

    def set_signal_exit(self, *args):
        """Set the option to exit the plugin based on a signal (SIGINT, SIGTERM...)."""
        self.is_signalled_to_exit = True

    @property
    def plugin(self):
        """Return plugin instance."""
        return self._plugin

    @property
    def cfg(self):
        """Return config instance."""
        return self._cfg

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_watchdog") and self._watchdog:
            self._watchdog.stop()

    def _recreate_plugin(self):
        """Recreate plugin including git version suffix."""
        version_suffix = get_git_version_suffix(self._cfg)
        if version_suffix:
            self._cfg.version_suffix = version_suffix
        self._plugin = self._plugin_class(config=self._cfg)
        self._network = network.Network(self._plugin)

    def run_once(
        self,
        event: azm.BinaryEvent,
        local_streams: list[StorageProxyFile] = None,
    ) -> dict[str, JobResult]:
        """Perform a local run of plugin, with timeout and error capture."""
        if self._watched and self._watched.check_updated():
            self._recreate_plugin()
            logger.info("Plugin has been restarted due to a file change")
        # keep newest result for each multiplugin
        results = {}
        for result, multiplugin in self._run_job(event, local=local_streams):
            results[multiplugin] = result
        return results

    def run_loop(
        self,
        *,
        queue: multiprocessing.Queue,
        job_limit: int = None,
    ):
        """Run continually and fetch jobs from the queue for processing while posting results to dispatcher.

        If `job_limit` is given, the loop will terminate after completing that many jobs (whether successful or not).
        Relies on network interactions to handle their own errors. If they fail then the runner will die.
        """
        if not self._cfg.events_url:
            raise ValueError("Cannot run fetch/execute loop when events_url is None")

        self._network.post_registrations()
        job_count = 0
        while job_limit is None or (job_limit and job_limit > job_count):
            if self._watched and self._watched.check_updated():
                logger.info("Plugin being restarted due to a file change")
                raise RecreateException()
            # ensure plugin is (still) ready to receive jobs
            if not self._plugin.is_ready():
                backoff = self._cfg.not_ready_backoff
                logger.info("Plugin is not ready to receive jobs; retrying in %ds" % backoff)
                time.sleep(backoff)
                continue

            # Set current job to None to indicate fetching a job.
            with contextlib.suppress(queueLib.Full):
                queue.put(None, block=False)

            # Exit gracefully when a SIGTERM Or SIGINT is provided. (placed after queue.put for testing reasons)
            if self.is_signalled_to_exit:
                raise SigTermExitError()

            event = self._network.fetch_job()
            # Put task immediately to account for time spend fetching a file.
            try:
                queue.put(TaskModel(in_event=event, start_time_epoch=time.time()), timeout=QUEUE_PUT_TIMEOUT)
            except queueLib.Full:
                logger.warning("monitoring queue is full bypassing, which may result in odd heartbeats.")

            job_count += 1
            try:
                for result, multiplugin in self._run_job(event, queue=queue):
                    self._network.ack_job(event, result, multiplugin)
                    # close any open handles in JobResult
                    result.close()
            except CriticalError:  # No extra logging required.
                logger.info("Exited on a critical error.")
                raise
            except Exception:  # Unknown error so extra logging required.
                traceback.print_exc()
                logger.error(
                    f"Error during job with kafka_key: '{event.kafka_key}', action: '{event.action}',"
                    + f" author info: {event.author.model_dump()}"
                    + f"\n source info: {event.source.model_dump()}."
                )
                raise
        sys.exit(0)

    def _run_job(
        self,
        event: azm.BinaryEvent,
        local: list[StorageProxyFile] | None = None,
        queue: multiprocessing.Queue = None,
    ) -> Generator[tuple[JobResult, str], None, None]:
        """Run plugin with supplied event and yield response.

        This is a generator so that yielded results may be published while plugin continues to run.
        Useful if a certain multiplugin exceeds max memory and crashes the process.
        """
        logger.info(
            "received plugin=%s file_format_legacy=%s size=%s sha256=%s"
            % (
                self._plugin.NAME,
                event.entity.file_format_legacy,
                event.entity.size,
                event.entity.sha256,
            )
        )
        run_start = datetime.datetime.now(datetime.timezone.utc)

        # check whether the maximum depth limit for the event has been reached, and opt-out if it has been.
        max_depth = self._cfg.plugin_depth_limit
        depth_reached = False
        if max_depth >= 0:
            if len(event.source.path) >= max_depth:
                depth_reached = True

        if depth_reached:
            result = JobResult(
                state=State(
                    State.Label.OPT_OUT,
                    message=f"{self._plugin.NAME} reached configured plugin_depth_limit",
                ),
                date_start=run_start,
                date_end=run_start,
                runtime=0,
            )
            logger.warning(f"{self._plugin.NAME}: hit max depth OPT_OUT for '{event.entity.sha256}'")
            yield (result, None)
            return  # Fetch next job

        job = models.Job(event=event)
        try:
            job.load_streams(dp=self._network.api, local=local)
        except (StorageError, FileNotFoundError):
            result = JobResult(
                state=State(
                    State.Label.ERROR_INPUT,
                    failure_name="Error opening data stream",
                    message=traceback.format_exc(),
                ),
                date_start=run_start,
                date_end=run_start,
                runtime=0,
            )
            logger.warning("Error opening data stream for %s" % event.entity.sha256)
            yield (result, None)
            return  # Fetch next job

        # run each multiplugin
        run_start = datetime.datetime.now(datetime.timezone.utc)

        # run first entrypoint to see if the whole plugin is opting out
        # first entrypoint == execute() function
        result = self._run_job_with_multiplugin(job, multiplugin=None, queue=queue)
        # start time should include multiprocessing code
        result.date_start = run_start
        if result.state.label not in azm.StatusEnumSuccess:
            # ending time should include multiprocessing code
            now = datetime.datetime.now(datetime.timezone.utc)
            result.date_end = now
            result.runtime = int((now - run_start).total_seconds())
            # yield heartbeats and other non-ok events
            yield result, None
            return  # fetch next job

        # Execute any multiplugins and always send the main result
        # NOTE - multiprocessing is not included in the time cost of these events
        try:
            for multiplugin in self._plugin.get_registered_multiplugins():
                if multiplugin is None:
                    continue
                yield self._run_job_with_multiplugin(job, multiplugin, queue), multiplugin
        finally:
            # the overall plugin finished without error, now need to raise event for it
            # send main plugin message with runtime covering all multiplugins and multiprocessing code
            now = datetime.datetime.now(datetime.timezone.utc)
            result.date_end = now
            result.runtime = int((now - run_start).total_seconds())
            yield result, None

    def _run_job_with_multiplugin(
        self, job: models.Job, multiplugin: Optional[str], queue: multiprocessing.Queue = None
    ) -> JobResult:
        """Run a multiplugin with the given event and streams."""
        try:
            # Don't enqueue job if multiplugin is None as it will have already been enqueued.
            if queue and multiplugin is not None:
                queue.put(
                    TaskModel(in_event=job.event, start_time_epoch=time.time(), multi_plugin_name=multiplugin),
                    timeout=QUEUE_PUT_TIMEOUT,
                )
            result = plugin_executor.run_plugin_with_job(self._plugin, job, multiplugin)
        except BaseException:
            logger.fatal("run_once terminated with uncaught exception")
            raise

        logger.info(
            "finish plugin=%s mp=%s state=%s file_format_legacy=%s size=%s sha256=%s"
            % (
                self._plugin.NAME,
                multiplugin,
                result.state.label.name,
                job.event.entity.file_format_legacy,
                job.event.entity.size,
                job.event.entity.sha256,
            )
        )
        return result
