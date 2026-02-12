from __future__ import annotations

import contextlib
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
from .test_plugin_timeout import DummySleepPlugin, TestPluginTimeouts


def _proxy_run_loop_monitor(server: str, *args):
    """Raise a termination signal"""
    loop = monitor.Monitor(
        DummySleepPlugin,
        {"events_url": server + "/test_data", "data_url": server, "delay": 3},
    )
    loop.run_loop()


def _proxy_run_loop_coordinator(server: str, dummy_queue):
    """Raise a termination signal"""
    loop = coordinator.Coordinator(
        DummySleepPlugin,
        settings.Settings(events_url=server + "/test_data", data_url=server, delay=1, delay_after_exception=0),
    )
    loop.run_loop(queue=dummy_queue)


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
        cls.mock_server, cls.server = sup.setup_mock_dispatcher()
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
        process_ref = multiprocessing.Process(
            target=_proxy_run_loop_monitor,
            args=(self.server),
        )
        process_ref.start()
        time.sleep(2)
        parent = psutil.Process(process_ref.pid)
        all_pids = [parent.pid]
        for child_processes in parent.children(recursive=True):
            all_pids.append(child_processes.pid)
            print(f"{child_processes.pid} - {child_processes}, {child_processes.parent().pid}")

        # Should be at least 1 child process
        self.assertGreaterEqual(len(all_pids), 1)
        # Verify processes are running
        for p in all_pids:
            cur_process = psutil.Process(p)
            self.assertTrue(cur_process.is_running())

        # Send sigterm
        process_ref.terminate()
        # Wait up to 20 seconds for process to exit
        process_ref.join(20)

        self.assertFalse(process_ref.is_alive())

        # Verify all child processes are exited.
        # (this takes slightly longer than the parent process, due to propagation time, so continually check.)
        one_alive = True
        while one_alive:
            one_alive = False
            for p in all_pids:
                with contextlib.suppress(Exception):
                    cur_process = psutil.Process(p)
                    time.sleep(0.5)
                    one_alive = True

        for p in all_pids:
            with self.assertRaises(psutil.NoSuchProcess):
                cur_process = psutil.Process(p)

    @pytest.mark.timeout(20)
    def test_sigterm_coordinator(self):
        """Test to see if coordinator completes it's last job and then exits when receiving a SIGTERM.

        As opposed to just accepting the SIGTERM and exiting immediately.
        """
        p = multiprocessing.Process(
            target=_proxy_run_loop_coordinator,
            args=(self.server, self.dummy_queue),
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
