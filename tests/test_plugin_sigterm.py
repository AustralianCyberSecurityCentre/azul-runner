from __future__ import annotations

import datetime
import json
import logging
import multiprocessing
import os
import signal
import tempfile
import time
import unittest
from typing import ClassVar
from unittest.mock import patch

import httpx
import psutil
import pytest
from azul_bedrock import models_network as azm

from azul_runner import State, coordinator, monitor, settings
from azul_runner.settings import add_settings
from tests import plugin_support as sup

from . import mock_dispatcher as md
from .test_plugin_timeout import TestPluginTimeouts


class CustomTestException(Exception):
    """Unique exception that is only raised in tests."""

    pass


def dump(x):
    return json.loads(x.model_dump_json(exclude_defaults=True))


class TestPluginTerminated(unittest.TestCase):
    """
    Tests a plugin stops when it recieves an appropriate signal.
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
        cls.editor = md.Editor(cls.server)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()

    def setUp(self):
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
        self.dummy_queue: multiprocessing.Queue = multiprocessing.Queue()

    @pytest.mark.timeout(20)
    def test_sigterm_monitor(self):
        """Test to see if child processes are killed when a sigterm is sent to the parent process (monitor)"""

        loop = monitor.Monitor(
            TestPluginTimeouts.gen_sleep_plugin(3),
            {"events_url": self.server + "/test_data", "data_url": self.server},
        )

        def proxy_run_loop(*args):
            """Raise a termination signal"""
            loop.run_loop()

        mp_ctx = multiprocessing.get_context("fork")
        process_ref = mp_ctx.Process(
            target=proxy_run_loop,
            args=(),
        )
        process_ref.start()
        time.sleep(1)
        parent = psutil.Process(process_ref.pid)
        all_pids = [parent.pid]
        for child_processes in parent.children(recursive=True):
            all_pids.append(child_processes.pid)

        # Should be at least 2 pids
        self.assertGreaterEqual(len(all_pids), 2)
        # Verify processes are running
        for p in all_pids:
            cur_process = psutil.Process(p)
            self.assertTrue(cur_process.is_running())

        # Send sigterm
        process_ref.terminate()
        # Wait up to 20 seconds for process to exit
        process_ref.join(20)

        # Verify all child processes are exited.
        for p in all_pids:
            with self.assertRaises(psutil.NoSuchProcess):
                psutil.Process(p)

    @pytest.mark.timeout(20)
    def test_sigterm_coordinator(self):
        """Test to see if coordinator completes it's last job and then exits when reciving a SIGTERM.

        As opposed to just accepting the SIGTERM and exiting immediately.
        """
        print(f"{os.getpid()},{os.getpid()},{os.getpid()},{os.getpid()}")
        loop = coordinator.Coordinator(
            TestPluginTimeouts.gen_sleep_plugin(1),
            settings.Settings(events_url=self.server + "/test_data", data_url=self.server),
        )

        def proxy_run_loop(*args):
            """Raise a termination signal"""
            with self.assertRaises(coordinator.SigTermExitError):
                loop.run_loop(queue=self.dummy_queue)

        mp_ctx = multiprocessing.get_context("fork")
        p = mp_ctx.Process(
            target=proxy_run_loop,
            args=(),
        )
        p.start()
        time.sleep(2)
        # Send sigterm
        p.terminate()
        # Wait up to 20 seconds for process to exit
        p.join(20)
        # Verify that jobs were processed and that the last message to the dummy_queue is None.
        # None should be the last value because this is placed on the queue during a fetch to dispatcher.
        # It indicates that the plugin didn't immediately exit and completed successfully before accepting the SIGTERM.
        # This is because None is only added to the queue just before the fetch from dispatcher.
        # This test would fail intermittently if that was happening by chance
        queue_val = 1
        num_none_vals = 0
        num_jobs = 0
        while not self.dummy_queue.empty():
            queue_val = self.dummy_queue.get()
            if queue_val:
                num_jobs += 1
            else:
                num_none_vals += 1
            # The queue has a None event, Job, None event, Job each iteration.
            # None events indicate a job is done, where Job events indicate a job is being processed.

        self.assertIsNone(queue_val)
        self.assertGreaterEqual(2, num_jobs)
        self.assertGreaterEqual(3, num_none_vals)
