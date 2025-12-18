from __future__ import annotations

import contextlib
import ctypes
import multiprocessing
import os
import subprocess
import tempfile
import time
import unittest
import warnings
from multiprocessing import Process
from queue import Empty
from typing import Any, ClassVar

import httpx
import pytest

from azul_runner import coordinator, settings

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

        cls.mock_server = md.MockDispatcher()
        cls.mock_server.start()
        while not cls.mock_server.is_alive():
            time.sleep(0.2)  # Wait for server to start
        cls.server = "http://%s:%s" % (cls.mock_server.host, cls.mock_server.port)
        # Wait for server to be ready to respond
        tries = 0
        while True:
            time.sleep(0.2)
            tries += 1
            try:
                _ = httpx.get(cls.server + "/mock/get_var/fetch_count")
                break  # Exit loop if successful
            except (httpx.TimeoutException, httpx.ConnectError):
                if tries > 20:  # Time out after about 4 seconds
                    raise RuntimeError("Timed out waiting for mock server to be ready")

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

        mp_ctx = multiprocessing.get_context("fork")
        p = mp_ctx.Process(
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

    @pytest.mark.timeout(10)
    def test_watch_git(self):
        """Tests git watch."""

        with tempfile.TemporaryDirectory() as filepath:
            subprocess.call(["git", "init"], cwd=filepath)
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("1")
            subprocess.call(["git", "add", "tmp.txt"], cwd=filepath)
            subprocess.call(["git", "commit", "-m", '"add file"'], cwd=filepath)

            self._inner_test_watch(filepath, watch_type="git")

    def test_watch_git_missing(self):
        with tempfile.TemporaryDirectory() as filepath:

            class DP(sup.DummyPlugin):
                def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
                    super().__init__(config)
                    with open(os.path.join(filepath, "tmp.txt"), "r") as f:
                        self.retval = f.read()

                def execute(self, job):
                    self.add_feature_values("example_string", self.retval)

            self.assertRaisesRegex(
                coordinator.CriticalError,
                r"is git not installed or .* not a valid git checkout",
                coordinator.Coordinator,
                *(DP, settings.parse_config(DP, {"watch_path": filepath, "watch_wait": 0, "watch_type": "git"})),
            )

    @pytest.mark.timeout(10)
    def test_no_watch(self):
        """Test that no exception is raised when files are modified that would cause watch to trigger if configured."""

        with tempfile.TemporaryDirectory() as filepath:
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("1")

            class DP(sup.DummyPlugin):
                def execute(self, job):
                    # Sleep up to job limit (6) waiting up to 3 seconds for a Recreate error to be generated.
                    time.sleep(0.5)

            loop = coordinator.Coordinator(DP, settings.parse_config(DP, {"server": self.server + "/test_data"}))

            mp_ctx = multiprocessing.get_context("fork")
            p = mp_ctx.Process(
                target=modify_file_in_background_causing_crash,
                args=(
                    filepath,
                    self.dummy_queue,
                ),
            )
            p.start()
            with self.assertRaises(SystemExit):
                loop.run_loop(queue=self.dummy_queue, job_limit=6)
