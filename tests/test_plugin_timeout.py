from __future__ import annotations

import datetime
import json
import logging
import multiprocessing
import time
import unittest
from typing import ClassVar

import httpx
import pytest
from azul_bedrock import models_network as azm

from azul_runner import State, coordinator, monitor

from . import mock_dispatcher as md
from . import plugin_support as sup


def dump(x):
    return json.loads(x.model_dump_json(exclude_defaults=True))


class TestPluginTimeouts(unittest.TestCase):
    """
    Tests the run_timeout functionality of Plugin.run_loop(), using the mock server.
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
            time.sleep(0.5)
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
        lr = logging.getLogger("azul_runner")
        lr.setLevel(logging.WARNING)
        self.dummy_log_handler = sup.DummyLogHandler()
        lr.addHandler(self.dummy_log_handler)

    # #### Utility
    @staticmethod
    def gen_sleep_plugin(delay, delay_after_exception=0):
        """Returns a plugin class which will sleep <delay> seconds then return an empty result.

        If <delay_after_exception> is nonzero, the plugin will catch PluginTimeoutError and
        sleep again for <delay_after_exception> seconds before returning."""

        class DummySleepPlugin(sup.DummyPlugin):
            """Dummy plugin class that sleeps for 2 seconds before returning, to test timeouts with."""

            @staticmethod
            def interruptible_sleep(secs):
                """Sleeps in bursts of 250ms so that exceptions can still interrupt us."""
                start = datetime.datetime.now()
                tick = 0.25
                while True:
                    time.sleep(tick)
                    elapsed = (datetime.datetime.now() - start).total_seconds()
                    if elapsed >= secs:
                        return
                    elif secs - elapsed < tick:
                        tick = secs - elapsed

            def execute(self, entity):
                if delay_after_exception:
                    try:
                        self.interruptible_sleep(delay)
                    except coordinator.PluginTimeoutError:
                        self.interruptible_sleep(delay_after_exception)
                else:
                    self.interruptible_sleep(delay)
                return

        return DummySleepPlugin

    def _filter_error_from_logs(self, msgs: list):
        return msgs

    # ############### #
    # #### Tests #### #
    # ############### #

    def test_no_timeout(self):
        """Check that the plugin runs and returns successfully within the timeout."""
        loop = monitor.Monitor(
            self.gen_sleep_plugin(2),
            {
                "server": self.server + "/null",
                "run_timeout": 3,
                "heartbeat_interval": 3,
                "max_timeouts_before_exit": 0,
            },
        )
        loop.run_loop(1)

        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        # standardise variable timestamps
        dt = datetime.datetime.now(datetime.timezone.utc)
        out_event["timestamp"] = dt.isoformat()
        self.assertEqual(
            out_event,
            dump(
                azm.StatusEvent(
                    model_version=azm.CURRENT_MODEL_VERSION,
                    kafka_key="runner-placeholder",
                    timestamp=dt,
                    author=azm.Author(
                        category="plugin",
                        name="DummySleepPlugin",
                        version="1.0",
                        security=None,
                    ),
                    entity=azm.StatusEvent.Entity(
                        input=azm.BinaryEvent(**out_event["entity"]["input"]),  # Don't care about this
                        status=State.Label.COMPLETED_EMPTY,
                        runtime=out_event["entity"]["runtime"],
                        results=[],
                    ),
                ),
            ),
        )
        print("------ACTUAL------")
        print(self.dummy_log_handler.logs)
        # Check the log messages, should be empty
        self.assertEqual(
            self._filter_error_from_logs(self.dummy_log_handler.logs),
            [],
        )

    @pytest.mark.timeout(10)
    def test_simple_timeout_and_catch_with_return(self):
        """Check that the plugin times out after 1 second, both with and without catching the exception."""
        # p1 just runs for 2 seconds and is aborted by the exception
        loop1 = monitor.Monitor(
            self.gen_sleep_plugin(2),
            {
                "server": self.server + "/null",
                "run_timeout": 1,
                "heartbeat_interval": 1,
                "max_timeouts_before_exit": 0,
            },
        )
        # p2 catches the timeout exception and sleeps 2 more seconds before returning
        loop2 = monitor.Monitor(
            self.gen_sleep_plugin(2, 2),
            {
                "server": self.server + "/null",
                "run_timeout": 1,
                "heartbeat_interval": 1,
                "max_timeouts_before_exit": 0,
            },
        )

        for loop in (loop1, loop2):

            self.assertRaises(monitor.PluginTimeoutError, loop.run_loop, 1)

            r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
            r.raise_for_status()
            out_event: dict = r.json()[0]
            # standardise variable timestamps
            dt = datetime.datetime.now(datetime.timezone.utc)
            out_event["timestamp"] = dt.isoformat()
            self.assertEqual(
                out_event,
                dump(
                    azm.StatusEvent(
                        model_version=azm.CURRENT_MODEL_VERSION,
                        kafka_key="runner-placeholder",
                        timestamp=dt,
                        author=azm.Author(
                            category="plugin",
                            name="DummySleepPlugin",
                            version="1.0",
                            security=None,
                        ),
                        entity=azm.StatusEvent.Entity(
                            input=azm.BinaryEvent(**out_event["entity"]["input"]),  # Don't care about this
                            status=State.Label.ERROR_TIMEOUT,
                            error="PluginTimeout Error",
                            message="DummySleepPlugin timed out on job '1234'",
                            runtime=out_event["entity"]["runtime"],
                            results=[],
                        ),
                    ),
                ),
            )
        print("------ACTUAL------")
        print(self.dummy_log_handler.logs)
        # Check the log messages - a warning is expected since our timeout is less than the default heartbeat time
        self.assertEqual(
            self._filter_error_from_logs(self.dummy_log_handler.logs),
            [
                "WARNING: Execution of DummySleepPlugin timed out after 1 seconds for job with id '1234'",
                "ERROR: Exiting due to hitting limit of 0 timeouts",
                "WARNING: Execution of DummySleepPlugin timed out after 1 seconds for job with id '1234'",
                "ERROR: Exiting due to hitting limit of 0 timeouts",
            ],
        )

    @pytest.mark.timeout(10)
    def test_timeout_with_catch_no_return(self):
        """Tests that the expected errors occur when a plugin catches the timeout exception and doesn't terminate."""
        loop = monitor.Monitor(
            self.gen_sleep_plugin(2, 5),
            {
                "server": self.server + "/null",
                "run_timeout": 1,
                "heartbeat_interval": 1,
                "max_timeouts_before_exit": 0,
            },
        )
        self.assertRaises(
            monitor.PluginTimeoutError, loop.run_loop, 2
        )  # Set 2 jobs limit, but it should exit on error after the first one

        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        # standardise variable timestamps
        dt = datetime.datetime.now(datetime.timezone.utc)
        out_event["timestamp"] = dt.isoformat()
        self.assertEqual(
            out_event,
            dump(
                azm.StatusEvent(
                    model_version=azm.CURRENT_MODEL_VERSION,
                    kafka_key="runner-placeholder",
                    timestamp=dt,
                    author=azm.Author(
                        category="plugin",
                        name="DummySleepPlugin",
                        version="1.0",
                        security=None,
                    ),
                    entity=azm.StatusEvent.Entity(
                        input=azm.BinaryEvent(**out_event["entity"]["input"]),  # Don't care about this
                        status=State.Label.ERROR_TIMEOUT,
                        error="PluginTimeout Error",
                        message="DummySleepPlugin timed out on job '1234'",
                        runtime=out_event["entity"]["runtime"],
                        results=[],
                    ),
                ),
            ),
        )
        print("------ACTUAL------")
        print(self.dummy_log_handler.logs)
        # Check the log messages are as expected
        self.assertEqual(
            self._filter_error_from_logs(self.dummy_log_handler.logs),
            [
                "WARNING: Execution of DummySleepPlugin timed out after 1 seconds for job with id '1234'",
                "ERROR: Exiting due to hitting limit of 0 timeouts",
            ],
        )

    @pytest.mark.timeout(15)
    def test_timeout_count_limit(self):
        """Test that the runner exits after the specified number of timeouts."""
        loop = monitor.Monitor(
            self.gen_sleep_plugin(2),
            {
                "server": self.server + "/null",
                "run_timeout": 1,
                "heartbeat_interval": 1,
                "max_timeouts_before_exit": 3,
            },
        )

        self.assertRaises(
            monitor.PluginTimeoutError, loop.run_loop, job_limit=10
        )  # Should stop after 3 when they all time out
        print("------ACTUAL------")
        print(self.dummy_log_handler.logs)
        # Check the log messages are as expected
        self.assertEqual(
            self._filter_error_from_logs(self.dummy_log_handler.logs),
            [
                "WARNING: Execution of DummySleepPlugin timed out after 1 seconds for job with id '1234'",
                "WARNING: Execution of DummySleepPlugin timed out after 1 seconds for job with id '1234'",
                "WARNING: Execution of DummySleepPlugin timed out after 1 seconds for job with id '1234'",
                "ERROR: Exiting due to hitting limit of 3 timeouts",
            ],
        )
