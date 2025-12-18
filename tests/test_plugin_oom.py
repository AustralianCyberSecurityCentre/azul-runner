from __future__ import annotations

import datetime
import json
import multiprocessing
import tempfile
import time
import unittest
from typing import ClassVar

import httpx
import pytest
from azul_bedrock import models_network as azm

from azul_runner import State, monitor
from azul_runner.settings import add_settings
from tests import plugin_support as sup

from . import mock_dispatcher as md
from .test_plugin_timeout import TestPluginTimeouts


class CustomTestException(Exception):
    """Unique exception that is only raised in tests."""

    pass


def dump(x):
    return json.loads(x.model_dump_json(exclude_defaults=True))


class SleepMultiPlugin(sup.DummyPlugin):

    SETTINGS = add_settings(memory_file_path=(str, ""))

    def mp_callback(self, job):
        "Write 100% to memory usage and then sleep."
        time.sleep(1)
        with open(self.cfg.memory_file_path, "w") as f:
            f.write("100")
            f.flush()
            f.seek(0)
        time.sleep(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.cfg.memory_file_path, "w") as f:
            f.write("0")
            f.flush()
            f.seek(0)

        self.register_multiplugin("MultiPluginName", "1.0", self.mp_callback)

    def execute(self, entity):
        pass


class TestPluginOom(unittest.TestCase):
    """
    Tests the Out of Memory functionality of Plugin.run_loop(), using the mock server.
    """

    mock_server: ClassVar[md.MockDispatcher]
    server: ClassVar[str]  # Endpoint to the mock server, suitable for passing to a plugin's config['server']
    dummy_log_handler: sup.DummyLogHandler  # Set for each instance in setUp()
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()

    def setUp(self):
        # Add a prefix to avoid deletion of temporary files
        self.max_mem_file = tempfile.NamedTemporaryFile("w+", prefix="dontdelete")
        self.max_mem_file.write("100")
        self.max_mem_file.flush()
        self.max_mem_file.seek(0)
        # Add a prefix to avoid deletion of temporary files
        self.cur_mem_file = tempfile.NamedTemporaryFile("w+", prefix="dontdelete")
        self.cur_mem_file.write("0")
        self.cur_mem_file.flush()
        self.cur_mem_file.seek(0)
        self.default_config = {
            # Mem limit settings
            "enable_mem_limits": True,
            "used_mem_warning_frac": 0.8,
            "used_mem_force_exit_frac": 0.9,
            "max_mem_file_path": self.max_mem_file.name,
            "cur_mem_file_path": self.cur_mem_file.name,
            "mem_poll_frequency_milliseconds": 1000,
        }

        self.basic_input_event = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="test-dummy",
            dequeued="test-dummy-dequeued",
            action=azm.BinaryAction.Sourced,
            timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            source=azm.Source(
                name="source",
                path=[],
                timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            ),
            author=azm.Author(name="TestServer", category="blah"),
            entity=azm.BinaryEvent.Entity(sha256="1234", datastreams=[], features=[]),
        )

    # ############### #
    # #### Tests #### #
    # ############### #
    @pytest.mark.timeout(10)
    def test_normal_plugin_run(self):
        """Check that the plugin runs and returns successfully with a memory limit set when it stays in that limit."""
        loop = monitor.Monitor(TestPluginTimeouts.gen_sleep_plugin(2), self.default_config.copy())
        result = loop.run_once(self.basic_input_event)

        print(result[None])
        self.assertEqual(result[None].state, State(State.Label.COMPLETED_EMPTY))

    @pytest.mark.timeout(5)
    def test_oom_failure(self):
        """Tests that the plugin exits with an exception if memory usage is over the error threshold."""
        self.cur_mem_file.write("100")
        self.cur_mem_file.flush()
        self.cur_mem_file.seek(0)
        loop = monitor.Monitor(TestPluginTimeouts.gen_sleep_plugin(1.1), self.default_config.copy())
        with self.assertRaises(monitor.NoNetworkResultError) as e:
            loop.run_once(self.basic_input_event)
        job_oom_result = e.exception.result

        # standardise variable timestamps
        print(job_oom_result.state)
        self.assertEqual(
            job_oom_result.state,
            State(
                State.Label.ERROR_OOM,
                failure_name="Out of Memory",
                message="Plugin DummySleepPlugin failed to complete job '1234' because it ran out of memory, memory limit is 100B and memory usage was 100B which is 100.0% memory usage.",
            ),
        )

    @pytest.mark.timeout(7)
    def test_oom_failure_multiplugin(self):
        """Tests that the plugin exits with an exception if memory usage is over the error threshold."""
        self.cur_mem_file.write("0")
        self.cur_mem_file.flush()
        self.cur_mem_file.seek(0)
        config = self.default_config.copy()
        config["memory_file_path"] = self.cur_mem_file.name
        config["server"] = self.server + "/null"
        loop = monitor.Monitor(SleepMultiPlugin, config)
        loop.run_loop(1)

        r = httpx.get("%s/mock/get_var/all_requests" % self.server)
        # Get the OOM event which should occur after the two plugin registration events.
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[2][0])

        # standardise variable timestamps
        self.assertEqual(out_evt.entity.status, State.Label.ERROR_OOM.value)
        self.assertEqual(out_evt.entity.error, "Out of Memory")
        print(f"Actual Message: {out_evt.entity.message}")
        self.assertEqual(
            out_evt.entity.message,
            "Plugin SleepMultiPlugin-MultiPluginName failed to complete job '1234' because it ran out of memory, memory limit is 100B and memory usage was 100B which is 100.0% memory usage.",
        )
        self.assertEqual(
            out_evt.author,
            azm.Author(category="plugin", name="SleepMultiPlugin-MultiPluginName", version="1.0-1.0", security=None),
        )

    @pytest.mark.timeout(20)
    def test_high_memory_usage_warnings(self):
        """Tests that warnings are raised if the plugin has high memory usage but not error level usage."""
        self.cur_mem_file.write("85")
        self.cur_mem_file.flush()
        self.cur_mem_file.seek(0)
        loop = monitor.Monitor(TestPluginTimeouts.gen_sleep_plugin(1.1), self.default_config.copy())
        with self.assertLogs() as caught_logs:
            loop.run_once(self.basic_input_event)

        found = False
        for log_list in caught_logs:
            for log_record in log_list:
                print(log_record)
                if "'1234' is nearly out of memory it is at 85.0% memory usage" in str(log_record):
                    found = True
        self.assertTrue(found, "Failed to find warning that runner is at 85% memory usage.")

    @pytest.mark.timeout(4)
    def test_setup_fails_and_disables_mem_tracking(self):
        """Tests that setup errors when trying to read cgroup memory."""
        self.max_mem_file.write("MAXMEM")
        self.max_mem_file.flush()
        self.max_mem_file.seek(0)
        loop = monitor.Monitor(TestPluginTimeouts.gen_sleep_plugin(2), self.default_config.copy())
        result = loop.run_once(self.basic_input_event)

        # Ensure a result is acquired rather than an OOM error
        print(result[None])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[None].state, State(State.Label.COMPLETED_EMPTY))

    @pytest.mark.timeout(20)
    def test_kill_children_called_oom(self):
        """Test to see if child processes are killed on out of memory"""

        loop = monitor.Monitor(TestPluginTimeouts.gen_sleep_plugin(1.1), self.default_config.copy())

        called = False

        def raise_exception_instead(*args, **kwargs):
            nonlocal called
            called = True
            raise CustomTestException()

        loop._kill_child_processes = raise_exception_instead
        try:
            loop.run_once(self.basic_input_event)
        except CustomTestException:
            pass
        self.assertTrue(
            called, "kill_child_process wasn't called even though out of memory should have been detected."
        )
