from __future__ import annotations

import contextlib
import ctypes
import multiprocessing
import os
import tempfile
import time
import unittest
from multiprocessing import Process
from queue import Empty
from typing import Any, ClassVar

import pytest

from azul_runner import coordinator, monitor, settings
from azul_runner.settings import WatchTypeEnum

from . import mock_dispatcher as md
from . import plugin_support as sup


def modify_file_in_background_causing_crash(filepath: str, queue: multiprocessing.Queue):
    # wait one second while emptying the queue and then modify the test file and exit.
    start_time = time.time()
    while start_time + 1 > time.time():
        # NOTE - without continually emptying the queue otherwise the plugin can get stuck putting elements into the
        # multiprocessing queue.
        # If there is no timeout on a multiprocessing queue put statement the application will un-expectedly hang.
        with contextlib.suppress(Empty):
            queue.get(block=False)
    with open(os.path.join(filepath, "tmp.txt"), "w") as f:
        f.write("2")


class TestPluginExecutionWrapper(unittest.TestCase):
    """
    Tests the handling of plugin execution by Plugin._exec_wrapper, using the sup.DummyPlugin class and TestPlugin template.
    """

    PLUGIN_TO_TEST = sup.DummyPlugin

    mock_server: ClassVar[md.MockDispatcher]
    server: ClassVar[str]  # Endpoint to the mock server, suitable for passing to a plugin's config['server']
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_server, cls.server = sup.setup_mock_dispatcher()

        # Dummy shared memory queue and ctype
        cls.dummy_queue: multiprocessing.Queue = multiprocessing.Queue()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()
        cls.dummy_queue.close()

    def _inner_test_watch(self, filepath: str, watch_type: str = None):
        """Common code between normal and git watchers."""
        config_dict = {"server": self.server + "/test_data", "watch_path": filepath, "watch_wait": 0}
        if watch_type:
            config_dict["watch_type"] = watch_type
        loop = coordinator.Coordinator(sup.DummyPlugin, settings.parse_config(sup.DummyPlugin, config_dict))

        p = Process(
            target=modify_file_in_background_causing_crash,
            args=(
                filepath,
                self.dummy_queue,
            ),
        )
        p.start()

        with self.assertRaises(coordinator.RecreateException):
            loop.run_loop(queue=self.dummy_queue, job_limit=None)
        # kill the watchdog threads to reduce chance of weird stuff during testing
        del loop

    @pytest.mark.timeout(10)
    def test_watch(self):
        """Tests that coordinator raises a recreation exception when a change occurs."""
        with tempfile.TemporaryDirectory() as filepath:
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("1")
            self._inner_test_watch(filepath)

    class DPWatchGitMissing(sup.DummyPlugin):
        def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
            super().__init__(config)
            with open(os.path.join(config["test_input_data"]["filepath"], "tmp.txt"), "r") as f:
                self.retval = f.read()

        def execute(self, job):
            self.add_feature_values("example_string", self.retval)

    def test_watch_git_missing(self):
        with tempfile.TemporaryDirectory() as filepath:
            self.assertRaisesRegex(
                coordinator.CriticalError,
                r"is git not installed or .* not a valid git checkout",
                coordinator.Coordinator,
                *(
                    self.DPWatchGitMissing,
                    settings.parse_config(
                        self.DPWatchGitMissing,
                        {
                            "watch_path": filepath,
                            "watch_wait": 0,
                            "watch_type": "git",
                            "test_input_data": {"filepath": filepath},
                        },
                    ),
                ),
            )

    class DPTestNoWatch(sup.DummyPlugin):
        def execute(self, job):
            # Sleep up to job limit (6) waiting up to 3 seconds for a Recreate error to be generated.
            time.sleep(0.5)

    @pytest.mark.timeout(10)
    def test_no_watch(self):
        """Test that no exception is raised when files are modified that would cause watch to trigger if configured."""

        with tempfile.TemporaryDirectory() as filepath:
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("1")

            loop = coordinator.Coordinator(
                self.DPTestNoWatch, settings.parse_config(self.DPTestNoWatch, {"server": self.server + "/test_data"})
            )

            p = Process(
                target=modify_file_in_background_causing_crash,
                args=(
                    filepath,
                    self.dummy_queue,
                ),
            )
            p.start()
            with self.assertRaises(SystemExit):
                loop.run_loop(queue=self.dummy_queue, job_limit=6)


def modify_watched_file_in_background(filepath: str, delay: float = 0.5):
    """Modify a watched file after a specified delay."""
    time.sleep(delay)
    watch_file = os.path.join(filepath, "tmp.txt")
    with open(watch_file, "w") as f:
        f.write("modified")


class TestMonitorWatch(unittest.TestCase):
    """
    Tests Monitor's watch functionality and coordinator recreation behavior.
    Verifies that Monitor properly detects file changes, handles coordinator exit codes,
    and recreates the coordinator instance.
    """

    mock_server: ClassVar[md.MockDispatcher]
    server: ClassVar[str]  # Endpoint to the mock server
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_server, cls.server = sup.setup_mock_dispatcher()
        cls.editor = md.Editor(cls.server)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()

    class DPQuickExecute(sup.DummyPlugin):
        """Plugin that executes quickly for testing."""

        def execute(self, job):
            # Execute quickly so we can test file watching and recreation
            time.sleep(0.1)

    @pytest.mark.timeout(20)
    def test_monitor_watch_detects_file_change_and_recreates_coordinator(self):
        """Test that Monitor properly detects file changes and recreates the coordinator.

        This test verifies that:
        1. Monitor detects when a watched file is modified
        2. The Coordinator exits with the correct RECREATE_PLUGIN exit code
        3. Monitor properly recreates a new Coordinator instance
        4. The monitor continues to function correctly after recreation
        """
        with tempfile.TemporaryDirectory(prefix="donotdelete_") as watch_dir:
            # Create initial file to watch
            watch_file = os.path.join(watch_dir, "tmp.txt")
            with open(watch_file, "w") as f:
                f.write("initial")

            # Create a Monitor instance with watch_path configured
            config_dict = {
                "server": self.server + "/test_data",
                "watch_path": watch_dir,
                "watch_type": WatchTypeEnum.PLAIN,  # Use plain file watching, not git
                "watch_wait": 0,  # No wait between detecting change and recreating
            }

            monitor_instance = monitor.Monitor(self.DPQuickExecute, config_dict)

            # Start a background process that will modify the watched file after a short delay
            file_modifier = Process(
                target=modify_watched_file_in_background,
                args=(watch_dir, 1),  # Modify after 1 second
            )
            file_modifier.start()

            # Create a temporary file that will be deleted when the monitor recreates the coordinator
            with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f_tmp_deleted:
                temp_deleted_path = f_tmp_deleted.name
                f_tmp_deleted.write(b"temporary file content")

            try:
                self.assertTrue(os.path.exists(temp_deleted_path))
                # Run the monitor with a job_limit to allow it to complete
                # If Monitor doesn't properly handle the RECREATE_PLUGIN exit code,
                # it will raise an error here instead of completing successfully
                monitor_instance.run_loop(job_limit=2)

                # If we reach here, Monitor successfully handled the recreation
                # (This validates both points 2 and 3: the exit code was RECREATE_PLUGIN
                # and Monitor properly recreated without crashing)

            finally:
                self.assertFalse(
                    os.path.exists(temp_deleted_path), "Temporary file should have been deleted on restart"
                )
                self.assertTrue(os.path.exists(watch_file), "Watch file should not have been deleted on restart")
                file_modifier.join(timeout=5)
                if file_modifier.is_alive():
                    file_modifier.terminate()
                    file_modifier.join()
