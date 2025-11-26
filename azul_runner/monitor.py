"""Monitor the plugin co-ordinator and check if it is Out of Memory, perform heart beats and capture timeouts."""

import datetime
import json
import logging
import multiprocessing
import os
import pathlib
import shutil
import signal
import tempfile
import time
import traceback
import uuid
from typing import Callable

import psutil
import pydantic
from azul_bedrock import models_network as azm

from azul_runner.coordinator import (
    Coordinator,
    CriticalError,
    RecreateException,
    SigTermExitError,
    get_git_version_suffix,
)
from azul_runner.log_setup import AddLoggingQueueListener, LogLevel, setup_logger
from azul_runner.models import JobResult, State, TaskExitCodeEnum, TaskModel

from . import network
from . import plugin as mplugin
from . import settings
from .storage import StorageProxyFile

MAX_ATTEMPTS_TO_KILL_CHILD_PROCESS = 10
logger = logging.getLogger(__name__)

result_type_adapter = pydantic.TypeAdapter(dict[str | None, JobResult])


class PluginTimeoutError(RuntimeError):
    """Raised if a child process executing the plugin can't be killed, or there were too many plugin timeouts."""

    pass


class TerminateError(RuntimeError):
    """Raised if a child process exited and requested to be terminated."""

    pass


class NoNetworkResultError(Exception):
    """Error to raise when a heartbeat or OOM error can't be sent to dispatcher because there is no network."""

    def __init__(self, message: str, result: JobResult):
        super().__init__(message, result)
        self.result = result


def kill_child_proc_tree(pid: int, sig=signal.SIGKILL):
    """Kill all child processes via process tree (including grandchildren) with signal KILL."""
    try:
        # If the parent no longer exists you can't kill children.
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for p in children:
        try:
            p.send_signal(sig)
        except psutil.NoSuchProcess:
            pass


class RunOnceHandler:
    """Handle multiprocessing when only running the subprocess for one iteration."""

    def __init__(
        self,
        event: azm.BinaryEvent,
        local_streams: list[StorageProxyFile] = None,
        cache_dir_path: str = tempfile.gettempdir(),
    ):
        """Handle the communication between a child process and the parent process."""
        self.event = event
        self.local_streams = local_streams
        self.cache_dir_path = cache_dir_path
        self.run_guid = str(uuid.uuid4())

    def _get_result_file_path(self) -> str:
        """Get the results file path."""
        return os.path.join(self.cache_dir_path, self.run_guid + "-result")

    def _generate_stream_path(self, file_label) -> str:
        """Generate a path for a stream."""
        return os.path.join(self.cache_dir_path, file_label + "-" + self.run_guid)

    def load_job_results_from_temp(self) -> dict[str, JobResult]:
        """Loads results or raises an exception if there are no results.

        NOTE - leaves open file handles for all streams to allow for tests to inspect streams.
        """
        try:
            with open(self._get_result_file_path(), "rb") as read_result:
                results = dict()
                for multiplugin_name, result_value in json.loads(read_result.read()).items():
                    # Account for serialization/deserialization of none
                    if multiplugin_name.lower() == "none":
                        multiplugin_name = None
                    results[multiplugin_name] = JobResult.model_validate(result_value)
        except Exception as e:
            with open(self._get_result_file_path(), "rb") as read_result:
                file_data = read_result.read()
            logger.info(f"Input data was {file_data} with error {e}")
            raise
        # Leave all the files as open file handles.
        for r in results.values():
            for file_label in r.data:
                r.data[file_label] = open(self._generate_stream_path(file_label), "rb")
        return results

    def _save_job_results_to_temp(self, result_dict: dict[str, JobResult]):
        """Save the job result to the provided temp directory."""
        for result in result_dict.values():
            for label, file_handle in result.data.items():
                file_handle.seek(0)
                with open(self._generate_stream_path(label), "wb") as stream_file:
                    stream_file.write(file_handle.read())
                # Close the file because this is the final read.
                file_handle.close()
                result.data[label] = b""
        json_results = result_type_adapter.dump_json(result_dict, round_trip=True)
        with open(self._get_result_file_path(), "wb") as out_file:
            out_file.write(json_results)

    def start_run_once_coordinator(
        self,
        *,
        plugin: type[mplugin.Plugin],
        config: settings.Settings,
        job_limit: int,  # Ignored but kept for compatibility.
        log_level: LogLevel,
        queue: multiprocessing.Queue,
        logging_queue: multiprocessing.Queue,
    ):
        """Run a plugin once and then write the result to the cache_dir_path."""
        setup_logger(log_level, logging_queue)
        loop = Coordinator(plugin, config)
        # Note - ignoring multi-plugin for timeouts.
        if queue:
            queue.put(TaskModel(in_event=self.event, start_time_epoch=time.time()), timeout=0.5)
        results = loop.run_once(self.event, self.local_streams)
        self._save_job_results_to_temp(results)


def _start_loop_coordinator(
    *,
    plugin: type[mplugin.Plugin],
    config: settings.Settings,
    job_limit: int,
    log_level: LogLevel,
    queue: multiprocessing.Queue,
    logging_queue: multiprocessing.Queue,
):
    """Start the coordinator and plugin and return exit code if it exits."""
    setup_logger(log_level, logging_queue)
    loop = Coordinator(plugin, config)
    pid = os.getpid()
    try:
        loop.run_loop(queue=queue, job_limit=job_limit)
        # clean exit
        logger.info("Plugin has stopped cleanly.")
        exit(TaskExitCodeEnum.COMPLETED)
    except CriticalError:
        logger.info("closing program after critical error.")
        exit(TaskExitCodeEnum.TERMINATE)
    except SigTermExitError:
        logger.info("closing program after SigTerm recieved error.")
        exit(TaskExitCodeEnum.TERMINATE)
    except RecreateException:
        # Kill process tree to remove any bad child processes launched with subprocess module.
        logger.info("Re-creating the plugin.")
        kill_child_proc_tree(pid)
        exit(TaskExitCodeEnum.RECREATE_PLUGIN)
    except Exception:
        logger.error(f"Plugin crashed with generic error {traceback.format_exc()}")
        exit(TaskExitCodeEnum.TERMINATE)


class MonitorTask:
    """Object that has the current states of a running plugin that is being tracked by Monitor.

    Coordinator is the child_process that is being tracked.
    Queue is the multiprocessing queue that is held by the main process and this process.
        This enables the child_process to communicate how many and what jobs it has worked on.

    The other class attributes are used to track heart beating and queuing information.
    """

    def __init__(self, child_process: multiprocessing.Process, queue: multiprocessing.Queue):
        self.child_process = child_process
        self.current_job: TaskModel | None = None
        self.job_sent_heartbeats = 0
        self.job_low_memory_warning_raised = False
        self.timeout_count = 0
        self.queue = queue

    def set_current_job_and_count_completed_jobs(self) -> int:
        """Count the number of completed jobs and set the current_job to None or a task.

        Sets the current_job to a non-None value if the plugin is actively working on a job.
        """
        counted_jobs = 0
        while not self.queue.empty():
            self.current_job = self.queue.get(block=False)
            # Only increase count for main plugin runs, not multiplugins.
            if self.current_job and self.current_job.multi_plugin_name is None:
                counted_jobs += 1
            self.job_sent_heartbeats = 0
            self.job_low_memory_warning_raised = False
        return counted_jobs


class Monitor:
    """Monitor the plugin coordinator and check if it is Out of Memory, perform heart beats and capture timeouts."""

    def __init__(self, plugin_class: type[mplugin.Plugin], config: dict):
        """Init."""
        self._plugin_class = plugin_class
        self._cfg = settings.parse_config(self._plugin_class, config)
        # Using multiprocess so dill can be used for serialization rather than pickle.
        if self._cfg.use_multiprocessing_fork:
            self.mp_ctx = multiprocessing.get_context("fork")
        else:
            self.mp_ctx = multiprocessing.get_context("forkserver")

        # Setup plugin
        self._recreate_plugin()

        # Calculate time to wait once as it won't change.
        # Default to 1 second because monitor needs to continually clear the Queue between monitor and coordinator
        # or there is a risk the queue becomes full and then the plugin will be stuck until the queue clears.
        self.time_to_wait_between_checks = 1
        if self._cfg.enable_mem_limits:
            self.time_to_wait_between_checks = min(
                self.time_to_wait_between_checks, self._cfg.mem_poll_frequency_milliseconds / 1000
            )

        # Should only be set to true for run once local runs to prevent sending messages to dispatcher.
        self.no_network = False

        # Memory limit setup
        self.max_memusage_bytes = -1
        self.warn_memusage_bytes = -1
        self.error_memusage_bytes = -1
        if self._cfg.enable_mem_limits:
            try:
                max_mem_bytes = -1
                with open(self._cfg.max_mem_file_path, "r") as f:
                    max_mem_bytes = f.read()
                    self.max_memusage_bytes = int(max_mem_bytes)
                    self.warn_memusage_bytes = int(self.max_memusage_bytes * self._cfg.used_mem_warning_frac)
                    self.error_memusage_bytes = int(self.max_memusage_bytes * self._cfg.used_mem_force_exit_frac)

            except ValueError:
                logger.warning(f"Invalid mem usage from file ('{max_mem_bytes}'), disabling mem checks!")
                self._cfg.enable_mem_limits = False
            except Exception:
                if max_mem_bytes == -1:
                    logger.error(f"Could not find file '{self._cfg.max_mem_file_path}' it doesn't exist.")
                else:
                    logger.error(
                        f"Could not cast the value '{max_mem_bytes}' into an integer to set max_memusage_bytes."
                    )
                raise

    def _recreate_plugin(self):
        """Recreate the plugin and networking for monitor."""
        version_suffix = get_git_version_suffix(self._cfg)
        if version_suffix:
            self._cfg.version_suffix = version_suffix
        self._plugin = self._plugin_class(config=self._cfg)
        self._network = network.Network(self._plugin)

    # Tempfile.template should be "tmp" which is the default tempfile prefix when creating files.
    @staticmethod
    def delete_tempfiles(file_prefix: str = tempfile.template):
        """Delete temporary files if any exists, this prevents lost tempfiles building up."""
        try:
            temp_dir = pathlib.Path(tempfile.gettempdir())
            for file in temp_dir.iterdir():
                # All temp files created by a plugin should start with tmp
                if file.is_file() and file.name.lower().startswith(file_prefix.lower()):
                    file.unlink(missing_ok=True)
                    logger.info(f"azul-runner is deleting temporary file {file} for cleanup purposes.")
        except Exception:
            logger.warning(
                f"Unable to clear all the files from temp dir '{tempfile.gettempdir()}'"
                + f" while restarting the plugin with error: {traceback.format_exc()}"
            )

    @staticmethod
    def purge_temp_directory():
        """Delete everything in the temporary directory (needed during plugin recreation requests)."""
        folder = tempfile.gettempdir()
        try:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                # Delete file
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception:
            logger.warning(f"Failed to fully cleanup temp: {traceback.format_exc()}")

    def _create_and_start_child_process(
        self,
        start_child_process_func: Callable[
            [
                type[mplugin.Plugin],
                settings.Settings,
                int,
                LogLevel,
                multiprocessing.Queue,
                multiprocessing.Queue,
            ],
            None,
        ],
        job_limit: int,
        queue: multiprocessing.Queue,
        logging_queue: multiprocessing.Queue,
    ) -> multiprocessing.Process:
        """Create the child co-ordinator process."""
        # Ensure that if the process was re-created the parent process waits until fetching is done.
        queue.empty()
        queue.put(None, timeout=0.5)
        log_level = logging.getLogger().level
        child_process = self.mp_ctx.Process(
            target=start_child_process_func,
            kwargs=dict(
                plugin=self._plugin_class,
                config=self._cfg,
                job_limit=job_limit,
                log_level=log_level,
                queue=queue,
                logging_queue=logging_queue,
            ),
        )
        child_process.start()

        return child_process

    def _kill_child_processes(self, concurrent_task_list: list[MonitorTask]):
        """Continually kill the child process until it's no longer alive."""
        killing_attempt = 0
        for cur_task in concurrent_task_list:
            while cur_task.child_process.is_alive():
                # Keep trying to kill the child process and all of it's children.
                kill_child_proc_tree(cur_task.child_process.pid)
                cur_task.child_process.kill()
                time.sleep(0.1)
                killing_attempt += 1
                if killing_attempt > MAX_ATTEMPTS_TO_KILL_CHILD_PROCESS:
                    # Failed to kill child process.
                    raise TimeoutError()

    def run_once(
        self,
        event: azm.BinaryEvent,
        local_streams: list[StorageProxyFile] = None,
        cache_dir_path: str = tempfile.tempdir,
        no_network=True,
    ) -> dict[str, JobResult]:
        """Run the plugin once in a subprocess and get the result.

        no_network prevents heartbeats from being sent and errors if there is memory or timeout issues.
        """
        roh = RunOnceHandler(event, local_streams, cache_dir_path)
        # Shouldn't wait anytime between checks in testing.
        self.time_to_wait_between_checks = 0
        self.no_network = no_network
        # Run and get result
        self._run(roh.start_run_once_coordinator)
        return roh.load_job_results_from_temp()

    def run_loop(self, job_limit: int | None = None):
        """Run a plugin in an infinite loop or until the job limit is reached."""
        self._run(_start_loop_coordinator, job_limit)

    def propagate_termination_signal(self, tasks: list[MonitorTask], send_sig: signal.Signals):
        """Setup a function to propagate the termination of this process to the child process."""

        def func_terminate_all_child_processes(*args):
            for t in tasks:
                t.child_process.terminate()

        signal.signal(send_sig, func_terminate_all_child_processes)

    def _run(
        self,
        start_child_process_func: Callable[
            [
                type[mplugin.Plugin],
                settings.Settings,
                int,
                LogLevel,
                multiprocessing.Queue,
                multiprocessing.Queue,
            ],
            None,
        ],
        job_limit=None,
    ):
        """Run a child plugin in a subprocess infinite loop or until the job limit is reached."""
        job_count = 0
        logging_queue = self.mp_ctx.Queue()
        concurrent_task_list: list[MonitorTask] = []
        recreate_plugin_requested = False
        plugin_clean_exit_requested = False
        is_any_job_active = False
        try:
            with AddLoggingQueueListener(logging_queue, logging.getLogger()):
                # Delete temp files out of temp if any exist. This prevents tmp files building up.
                self.delete_tempfiles()
                for _ in range(self._cfg.concurrent_plugin_instances):
                    queue = self.mp_ctx.Queue()
                    child_process = self._create_and_start_child_process(
                        start_child_process_func, job_limit, queue, logging_queue
                    )
                    concurrent_task_list.append(MonitorTask(child_process, queue))
                # Ensure SIGINT and SIGTERM cause the child process to terminate.
                self.propagate_termination_signal(concurrent_task_list, signal.SIGINT)
                self.propagate_termination_signal(concurrent_task_list, signal.SIGTERM)

                while True:
                    # Sleep only if the job_limit isn't set and if the limit is set start waiting once you have
                    # processed that many jobs. (makes testing much faster)
                    if job_limit and job_count > job_limit:
                        logger.info("Exiting after reaching job limit.")
                        return
                    elif job_limit is None or job_count > job_limit:
                        time.sleep(self.time_to_wait_between_checks)
                        pass

                    # Confirm at least one task wants to be recreated and none have any active jobs.
                    if recreate_plugin_requested and not is_any_job_active:
                        self.purge_temp_directory()
                        self._recreate_plugin()
                        # Ensure all child processes were terminated before re-creating them.
                        self._kill_child_processes(concurrent_task_list)
                        for monitor_task in concurrent_task_list:
                            # Ensure the jobs in the child tasks queue are cleared before starting plugin to prevent
                            # Deadlocks.
                            monitor_task.set_current_job_and_count_completed_jobs()
                            # Start child process again
                            monitor_task.child_process = self._create_and_start_child_process(
                                start_child_process_func, job_limit, queue, logging_queue
                            )

                    # Confirm at least one task wants to exit and there are no active tasks.
                    if plugin_clean_exit_requested and not is_any_job_active:
                        logger.info("Closing down successfully completed plugin.")
                        return

                    recreate_plugin_requested = False
                    plugin_clean_exit_requested = False
                    is_any_job_active = False

                    # If the child process has stopped handle that case.
                    for monitor_task in concurrent_task_list:
                        if not child_process.is_alive():
                            if monitor_task.child_process.exitcode == TaskExitCodeEnum.COMPLETED.value:
                                plugin_clean_exit_requested = True
                                continue
                            elif monitor_task.child_process.exitcode == TaskExitCodeEnum.RECREATE_PLUGIN.value:
                                recreate_plugin_requested = True
                                continue
                            elif monitor_task.child_process.exitcode == TaskExitCodeEnum.TERMINATE.value:
                                raise TerminateError("Critical error occurred crashing process.")
                            else:
                                error_message = (
                                    "Unknown exit code from child process code: "
                                    + f"'{monitor_task.child_process.exitcode}' crashing!"
                                )
                                logger.error(error_message)
                                raise TerminateError(error_message)

                        # Get the current job in the queue (also ensures it empties out)\
                        job_count += monitor_task.set_current_job_and_count_completed_jobs()

                        # Still fetching a new job, or haven't got the first one yet (both have current_job of None)
                        if monitor_task.current_job is None:
                            continue

                        # A job is active because it has a current_job that isn't None
                        # and the subprocess hasn't terminated for any reason.
                        is_any_job_active = True

                        # Perform memory checks if enabled
                        if self._cfg.enable_mem_limits:
                            if not self._are_memory_limits_good(monitor_task):
                                # prevent temp file buildup.
                                self._kill_child_processes(concurrent_task_list)
                                self.delete_tempfiles()
                                child_process = self._create_and_start_child_process(
                                    start_child_process_func, job_limit, queue, logging_queue
                                )

                        # Check for timeout and raise heartbeat if required.
                        if not self._is_healthy_heartbeat_and_memory_checks(monitor_task):
                            self._kill_child_processes(concurrent_task_list)
                            # prevent temp file buildup.
                            self.delete_tempfiles()
                            child_process = self._create_and_start_child_process(
                                start_child_process_func, job_limit, queue, logging_queue
                            )

        finally:
            self._kill_child_processes(concurrent_task_list)
            # Close all the queues.
            for c_task in concurrent_task_list:
                c_task.queue.close()
            logging_queue.close()

    def _are_memory_limits_good(self, monitor_task: MonitorTask):
        """Check if the memory limit has been reached and kill the child process if required.

        Return True if the child process is healthy and False if it should be terminated.
        """
        # Assuming that if max memory exists and was an int cur_mem will be to (save processing time).
        with open(self._cfg.cur_mem_file_path, "r") as cur_mem_file:
            # Account for case when memory file is empty or unreadable.
            try:
                cur_mem_bytes = int(cur_mem_file.read())
            except ValueError as e:
                logger.warning(f"Couldn't read memory file with error {e}")
                return True
            if cur_mem_bytes >= self.error_memusage_bytes:
                multiplugin_text = self._plugin.NAME
                if monitor_task.current_job.multi_plugin_name:
                    multiplugin_text += f"-{monitor_task.current_job.multi_plugin_name}"
                # Ran out of memory kill child process.
                message = (
                    f"Plugin {multiplugin_text} failed to complete "
                    + f"job '{monitor_task.current_job.in_event.entity.sha256}' because it ran out of memory, "
                    + f"memory limit is {pydantic.ByteSize(self.max_memusage_bytes).human_readable()} and memory "
                    + f"usage was {pydantic.ByteSize(cur_mem_bytes).human_readable()} which is "
                    + f"{(cur_mem_bytes / self.max_memusage_bytes) * 100:.1f}% memory usage."
                )
                logger.error(message)

                result = JobResult(
                    state=State(
                        State.Label.ERROR_OOM,
                        failure_name="Out of Memory",
                        message=message,
                    ),
                )
                self._post_status(monitor_task, result)
                return False

            elif not monitor_task.job_low_memory_warning_raised and cur_mem_bytes >= self.warn_memusage_bytes:

                multiplugin_text = self._plugin.NAME
                if monitor_task.current_job.multi_plugin_name:
                    multiplugin_text += f"-{monitor_task.current_job.multi_plugin_name}"
                logger.warning(
                    f"The job '{monitor_task.current_job.in_event.entity.sha256}' is nearly out of memory it is at "
                    + f"{(cur_mem_bytes / self.max_memusage_bytes) * 100:.1f}% memory usage for the plugin "
                    + f"{multiplugin_text}."
                )
                monitor_task.job_low_memory_warning_raised = True
        return True

    def _is_healthy_heartbeat_and_memory_checks(self, monitor_task: MonitorTask) -> bool:
        """Check if heartbeats should be created and verify.

        Return True if the child process is healthy and False if it should be terminated.
        """
        # Thread is still running - Raise heartbeat and check if plugin should be forced to time out.
        runtime = time.time() - monitor_task.current_job.start_time_epoch

        if self._cfg.run_timeout and runtime > self._cfg.run_timeout:
            # Kill the plugin as it's run out of time.
            logger.warning(
                f"Execution of {self._plugin.NAME} timed out after {self._cfg.run_timeout} seconds "
                f"for job with id '{monitor_task.current_job.in_event.entity.sha256}'"
            )
            state = State(
                State.Label.ERROR_TIMEOUT,
                failure_name="PluginTimeout Error",
                message=f"{self._plugin.NAME} timed out on job '{monitor_task.current_job.in_event.entity.sha256}'",
            )
            result = JobResult(
                state=state,
                date_start=datetime.datetime.fromtimestamp(monitor_task.current_job.start_time_epoch),
                date_end=datetime.datetime.now(datetime.timezone.utc),
                runtime=int(runtime),
            )
            self._post_status(monitor_task, result)
            monitor_task.timeout_count += 1
            if monitor_task.timeout_count >= self._cfg.max_timeouts_before_exit:
                msg = f"Exiting due to hitting limit of {self._cfg.max_timeouts_before_exit} timeouts"
                logger.error(msg)
                raise PluginTimeoutError(msg)
            return False
        # Only check for heartbeats every heartbeat internal.
        elif (runtime // self._cfg.heartbeat_interval) > monitor_task.job_sent_heartbeats:
            # Send heartbeat.
            monitor_task.job_sent_heartbeats += 1
            result = JobResult(
                state=State(State.Label.HEARTBEAT),
                date_start=datetime.datetime.fromtimestamp(monitor_task.current_job.start_time_epoch),
                date_end=None,
                runtime=int(runtime),
            )
            if self.no_network:
                # Ignore heartbeats if there is no network connection.
                logger.info("Not sending heartbeat as networking is disabled.")
                return True
            self._post_status(monitor_task, result)

        return True

    def _post_status(self, monitor_task: MonitorTask, result: JobResult):
        """Post a status message to dispatcher."""
        if self.no_network:
            raise NoNetworkResultError(
                f"Attempting to post a {result.state} event and networking isn't enabled.", result
            )
        self._network.ack_job(monitor_task.current_job.in_event, result, monitor_task.current_job.multi_plugin_name)
