from __future__ import annotations

import datetime
import json
import multiprocessing
import os
import time
import unittest
from typing import ClassVar

import pytest
from azul_bedrock import models_network as azm

from azul_runner import coordinator, settings, models
from tests import plugin_support as sup

from . import mock_dispatcher as md
from .test_plugin_timeout import DummySleepPlugin


def _proxy_run_loop_coordinator(server: str, dummy_queue):
    """Exit with the correct code depending on if the coordinator raises a RecreateException or not."""
    loop = coordinator.Coordinator(
        DummySleepPlugin,
        settings.Settings(events_url=server + "/test_data", data_url=server, delay=1, delay_after_exception=0),
    )
    try:
        loop.run_loop(queue=dummy_queue)
        exit(models.TaskExitCodeEnum.COMPLETED.value)
    except coordinator.RecreateException:
        exit(models.TaskExitCodeEnum.RECREATE_PLUGIN.value)


def dump(x):
    return json.loads(x.model_dump_json(exclude_defaults=True))


class TestPluginRecreated(unittest.TestCase):
    """
    Tests that a plugin exits with the correct code to be restarted when it receives the appropriate signal.
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
    def test_request_recreation_coordinator(self):
        """Test to see if coordinator completes it's last job and then exits when receiving a RESTART_SIGNAL.

        As opposed to just accepting the RESTART_SIGNAL and exiting immediately.
        """
        p = multiprocessing.Process(
            target=_proxy_run_loop_coordinator,
            args=(self.server, self.dummy_queue),
        )

        p.start()
        time.sleep(2)
        # Send RESTART_SIGNAL
        os.kill(p.pid, coordinator.RESTART_SIGNAL)
        # Wait up to 20 seconds for process to exit
        p.join(20)

        self.assertEqual(models.TaskExitCodeEnum.RECREATE_PLUGIN.value, p.exitcode)

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
