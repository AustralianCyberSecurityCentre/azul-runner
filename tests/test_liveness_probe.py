"""Test cases for liveness probe functionality."""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
import time
import unittest
from typing import ClassVar

from azul_runner import monitor, network

from . import mock_dispatcher as md
from . import plugin_support as sup


KEEPALIVE_PATH = pathlib.Path(tempfile.gettempdir()) / network.KEEPALIVE_FILENAME


class TestLivenessProbe(unittest.TestCase):
    """Test cases for liveness probe functionality."""

    mock_server: ClassVar[md.MockDispatcher]
    server: ClassVar[str]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the mock dispatcher server for all tests."""
        cls.mock_server, cls.server = sup.setup_mock_dispatcher()
        cls.editor = md.Editor(cls.server)
        cls.editor.set_stream(
            "b4b389c849d799d9331d5937cde7f0dfd297d76083242366cbef53b498cd6051", 200, b"small content"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up the mock dispatcher server."""
        cls.mock_server.stop()
        cls.mock_server.kill()

    def setUp(self) -> None:
        """Clean up keepalive file before each test."""
        if KEEPALIVE_PATH.exists():
            KEEPALIVE_PATH.unlink()

    def tearDown(self) -> None:
        """Clean up keepalive file after each test."""
        if KEEPALIVE_PATH.exists():
            KEEPALIVE_PATH.unlink()

    def test_liveness_probe_disabled_never_creates_file(self) -> None:
        """Test that when enable_liveness_probe is False, the file is never created."""
        config = {
            "events_url": self.server + "/depth_1",
            "data_url": self.server + "/data",
            "enable_liveness_probe": False,
        }
        mon = monitor.Monitor(sup.DummyPlugin, config)

        net = network.Network(mon._plugin)
        net.post_registrations()

        mon.run_loop(job_limit=1)

        self.assertFalse(
            KEEPALIVE_PATH.exists(),
            f"Keepalive file should NOT be created when liveness_probe is False, but {KEEPALIVE_PATH} exists",
        )

    def test_liveness_probe_enabled_creates_file(self) -> None:
        """Test that when enable_liveness_probe is True, the keepalive file is created."""
        config = {
            "events_url": self.server + "/depth_1",
            "data_url": self.server + "/data",
            "enable_liveness_probe": True,
        }
        mon = monitor.Monitor(sup.DummyPlugin, config)

        net = network.Network(mon._plugin)
        net.post_registrations()

        mon.run_loop(job_limit=1)

        self.assertTrue(
            KEEPALIVE_PATH.exists(),
            f"Keepalive file should be created when enable_liveness_probe is True, but {KEEPALIVE_PATH} does not exist",
        )

    def test_liveness_probe_check_fresh_file(self) -> None:
        """Test that the liveness probe check used in Kubernetes passes when keepalive file is fresh."""

        KEEPALIVE_PATH.touch()

        # Act - Run the Kubernetes-style liveness probe check
        check_result = subprocess.run(
            [
                "/bin/sh",
                "-c",
                f"[ $(($(date +%s) - $(stat -c %Y {KEEPALIVE_PATH}))) -le 300 ]",
            ]
        )

        self.assertEqual(
            check_result.returncode,
            0,
            "Liveness probe check should pass (exit code 0) when file is recent",
        )

    def test_liveness_probe_check_stale_file(self) -> None:
        """Test that the liveness probe check used in Kubernetes fails when keepalive file is stale.

        This simulates the Kubernetes liveness probe detecting a stale file and triggering a pod restart.
        """
        # Create a stale keepalive file
        KEEPALIVE_PATH.touch()
        stale_mtime = time.time() - 400
        os.utime(KEEPALIVE_PATH, (stale_mtime, stale_mtime))

        check_result = subprocess.run(
            [
                "/bin/sh",
                "-c",
                f"[ $(($(date +%s) - $(stat -c %Y {KEEPALIVE_PATH}))) -le 300 ]",
            ]
        )

        self.assertNotEqual(
            check_result.returncode,
            0,
            "Liveness probe check should fail (exit code non-zero) when file is stale",
        )


if __name__ == "__main__":
    unittest.main()
