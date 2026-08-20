from __future__ import annotations
import signal
from azul_runner.models import TaskExitCodeEnum
from azul_runner.log_setup import LogLevel
from azul_runner.coordinator import RESTART_SIGNAL

import contextlib
import datetime
import json
import multiprocessing
import time
import unittest
from typing import ClassVar
from unittest import mock
import psutil
import os
import pytest
from azul_bedrock import models_network as azm
from azul_runner import coordinator, monitor, settings
from tests import plugin_support as sup

from . import mock_dispatcher as md
from .test_plugin_timeout import DummySleepPlugin


def _proxy_run_loop_monitor(server: str, *args):
    """Raise a termination signal"""
    loop = monitor.Monitor(
        DummySleepPlugin,
        {"events_url": server + "/test_data", "data_url": server, "delay": 3, "graceful_shutdown": True},
    )
    loop.run_loop()


def _proxy_run_loop_monitor_with_immediate_git_sync(server: str, *args):
    """Raise a termination signal"""
    loop = monitor.Monitor(
        DummySleepPlugin,
        {"events_url": server + "/test_data", "data_url": server, "delay": 3},
    )

    def start_child_process_mock(self, *args, **kwargs):
        """Start a fake child process."""
        fake_child_process = mock.MagicMock()
        fake_child_process.is_alive.return_value = False
        fake_child_process.exitcode = -RESTART_SIGNAL.value
        fake_child_process.pid = -1

        return fake_child_process

    loop._create_and_start_child_process = start_child_process_mock
    loop.run_loop()


def _proxy_run_loop_coordinator(server: str, dummy_queue, dummy_log_handler):
    """Raise a termination signal"""
    monitor._start_loop_coordinator(
        DummySleepPlugin,
        settings.Settings(
            events_url=server + "/test_data", data_url=server, delay=1, delay_after_exception=0, graceful_shutdown=True
        ),
        None,
        "DEBUG",
        dummy_queue,
        dummy_log_handler,
    )


class CustomTestException(Exception):
    """Unique exception that is only raised in tests."""

    pass


def dump(x):
    return json.loads(x.model_dump_json(exclude_defaults=True))


class TestPluginTerminated(unittest.TestCase):
    """Tests a plugin stops when it receives an appropriate signal."""

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
        self.dummy_log_handler: multiprocessing.Queue = multiprocessing.Queue()

    def tearDown(self):
        self.dummy_queue.close()
        self.dummy_log_handler.close()

    @pytest.mark.timeout(20)
    def test_sigterm_monitor(self):
        """Test to see if child processes are killed when a sigterm is sent to the parent process (monitor)"""
        process_ref = multiprocessing.Process(
            target=_proxy_run_loop_monitor,
            args=(self.server),
        )
        process_ref.start()

        # Wait until parent id is available.
        parent = None
        for i in range(20):
            time.sleep(0.5)
            with contextlib.suppress(psutil.NoSuchProcess):
                parent = psutil.Process(process_ref.pid)
            if i >= 19:
                raise Exception("cannot find parent process after waiting, something is wrong!")
            if parent:
                break

        all_pids = [parent.pid]
        for child_processes in parent.children(recursive=True):
            all_pids.append(child_processes.pid)

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
        # Terminated via sigterm cleanly.
        self.assertEqual(process_ref.exitcode, -signal.SIGTERM)

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
            args=(self.server, self.dummy_queue, self.dummy_log_handler),
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
        self.assertEqual(p.exitcode, TaskExitCodeEnum.TERMINATE.value)
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

    @pytest.mark.timeout(20)
    def test_custom_signal_coordinator(self):
        """Test to see if the user based restart signal returns a recreate plugin code."""
        p = multiprocessing.Process(
            target=_proxy_run_loop_coordinator,
            args=(self.server, self.dummy_queue, self.dummy_log_handler),
        )
        p.start()
        time.sleep(2)
        # Send sigterm
        os.kill(p.pid, RESTART_SIGNAL)

        # Wait up to 20 seconds for process to exit
        p.join(20)
        # Verify that exit code is the expected recreate plugin code.
        self.assertEqual(
            p.exitcode, TaskExitCodeEnum.RECREATE_PLUGIN.value
        )  # Verify that jobs were processed and that the last message to the dummy_queue is None.
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

    @pytest.mark.timeout(20)
    def test_custom_signal_coordinator_raced(self):
        """Test to see if the user based restart signal returns a recreate plugin code but don't wait for startup.

        This causes the signal to be the sent RESTART_SIGNAL value because the signal reciever in coordinator never
        got a chance to setup.
        """
        p = multiprocessing.Process(
            target=_proxy_run_loop_coordinator,
            args=(self.server, self.dummy_queue, self.dummy_log_handler),
        )
        p.start()
        # Send sigterm
        os.kill(p.pid, RESTART_SIGNAL)

        # Wait up to 20 seconds for process to exit
        p.join(20)
        # Verify that exit code is the expected recreate plugin code.
        self.assertEqual(
            p.exitcode, -RESTART_SIGNAL.value
        )  # Verify that jobs were processed and that the last message to the dummy_queue is None.
        # None should be the last value because this is placed on the queue during a fetch to dispatcher.
        # It indicates that the plugin didn't immediately exit and completed successfully before accepting the SIGTERM.
        # This is because None is only added to the queue just before the fetch from dispatcher.
        # This test would fail intermittently if that was happening by chance
        queue_val = "queueNotTouched"
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

        self.assertEqual(queue_val, "queueNotTouched")
        self.assertEqual(0, num_jobs)
        self.assertEqual(0, num_none_vals)

    @pytest.mark.timeout(20)
    def test_custom_signal_monitor_raced(self):
        """Test to see if a child process that is continually triggering a custom user exit code -RESTART_SIGNAL
        causes a program to crash.

        If the process crashes the exit code will be something that isn't -signal.SIGTERM (-15).
        This means that monitor is not correctly handling the -RESTART_SIGNAL.value case.
        This occurs when a RESTART_SIGNAL is sent to coordinator before it can setup it's signal.signal() interceptors.

        This is a mocked case of that scenario.
        """
        process_ref = multiprocessing.Process(
            target=_proxy_run_loop_monitor_with_immediate_git_sync,
            args=(self.server),
        )
        process_ref.start()
        time.sleep(2)

        # Send sigterm
        process_ref.terminate()
        # Wait up to 20 seconds for process to exit
        process_ref.join(10)

        self.assertFalse(process_ref.is_alive())
        #
        self.assertEqual(process_ref.exitcode, -signal.SIGTERM)
