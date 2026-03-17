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
from azul_runner.monitor import GitSync, GitError

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


class TestGitSync(unittest.TestCase):
    """Test GitSync git monitoring functionality."""

    @staticmethod
    def _setup_bare_repo_with_content(remote_path: str, content: str = "v1"):
        """Helper to create a bare repo with initial content."""
        subprocess.run(["git", "init", "--bare"], cwd=remote_path, check=True, capture_output=True)
        with tempfile.TemporaryDirectory() as temp_clone:
            subprocess.run(["git", "clone", remote_path, "."], cwd=temp_clone, check=True, capture_output=True)
            with open(os.path.join(temp_clone, "test.txt"), "w") as f:
                f.write(content)
            subprocess.run(["git", "add", "."], cwd=temp_clone, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "initial"], cwd=temp_clone, check=True, capture_output=True)
            subprocess.run(["git", "push"], cwd=temp_clone, check=True, capture_output=True)

    @staticmethod
    def _push_update_to_remote(remote_path: str, new_content: str):
        """Helper to push an update to the remote repo."""
        with tempfile.TemporaryDirectory() as temp_clone:
            subprocess.run(["git", "clone", remote_path, "."], cwd=temp_clone, check=True, capture_output=True)
            with open(os.path.join(temp_clone, "test.txt"), "w") as f:
                f.write(new_content)
            subprocess.run(["git", "add", "test.txt"], cwd=temp_clone, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"update to {new_content}"], cwd=temp_clone, check=True, capture_output=True
            )
            subprocess.run(["git", "push"], cwd=temp_clone, check=True, capture_output=True)

    @pytest.mark.timeout(15)
    def test_watch_repo_initialization(self):
        """Test GitSync initializes with correct attributes."""
        watch = GitSync(
            repo="https://example.com/repo.git",
            branch="main",
            watch_path="/tmp/watch",
            period=60,
            username="user",
            password="pass",
            do_ssh_auth=True,
            ssh_key_path="/path/to/key",
        )
        self.assertEqual(watch.repo, "https://example.com/repo.git")
        self.assertEqual(watch.branch, "main")
        self.assertEqual(watch.watch_path, "/tmp/watch")
        self.assertEqual(watch.period, 60)
        self.assertEqual(watch.username, "user")
        self.assertEqual(watch.password, "pass")
        self.assertTrue(watch.do_ssh_auth)
        self.assertEqual(watch.ssh_key_path, "/path/to/key")
        self.assertFalse(watch.update_pending())

    @pytest.mark.timeout(15)
    def test_watch_repo_clone_and_initial_fetch(self):
        """Test GitSync clones repo and performs initial fetch on start."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                self._setup_bare_repo_with_content(remote_repo, "initial content")

                watch = GitSync(repo=remote_repo, watch_path=watch_path, period=1)
                watch.start()

                # Verify clone and checkout happened
                self.assertTrue(os.path.exists(os.path.join(watch_path, "test.txt")))
                with open(os.path.join(watch_path, "test.txt")) as f:
                    self.assertEqual(f.read(), "initial content")

                watch.stop()

    @pytest.mark.timeout(15)
    def test_watch_repo_skips_clone_if_exists(self):
        """Test GitSync does not reclone if local repo already exists."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                self._setup_bare_repo_with_content(remote_repo, "v1")

                # Pre-clone the repo locally
                subprocess.run(["git", "clone", remote_repo, "."], cwd=watch_path, check=True, capture_output=True)

                # Modify the local file to prove we don't reclone
                with open(os.path.join(watch_path, "test.txt"), "w") as f:
                    f.write("local modification")

                # Start watch - should not overwrite local modification
                watch = GitSync(repo=remote_repo, watch_path=watch_path, period=1)
                watch.start()

                with open(os.path.join(watch_path, "test.txt")) as f:
                    content = f.read()
                self.assertEqual(content, "local modification")

                watch.stop()

    @pytest.mark.timeout(15)
    def test_watch_repo_detects_remote_updates(self):
        """Test GitSync detects when remote has new commits."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                self._setup_bare_repo_with_content(remote_repo, "v1")

                watch = GitSync(repo=remote_repo, watch_path=watch_path, period=1)
                watch.start()

                # Should not have updates initially
                self.assertFalse(watch.update_pending())

                # Push an update to remote
                self._push_update_to_remote(remote_repo, "v2")

                # Wait for polling thread to detect update
                start_time = time.time()
                while time.time() - start_time < 5:
                    if watch.update_pending():
                        break
                    time.sleep(0.1)

                self.assertTrue(watch.update_pending())
                watch.stop()

    @pytest.mark.timeout(15)
    def test_watch_repo_fetch_clears_update_event(self):
        """Test GitSync fetch clears the update_pending flag."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                self._setup_bare_repo_with_content(remote_repo, "v1")

                watch = GitSync(repo=remote_repo, watch_path=watch_path, period=1)
                watch.start()

                # Push an update
                self._push_update_to_remote(remote_repo, "v2")

                # Wait for update to be detected
                start_time = time.time()
                while time.time() - start_time < 5:
                    if watch.update_pending():
                        break
                    time.sleep(0.1)

                self.assertTrue(watch.update_pending())

                # Fetch should clear the flag and pull new content
                watch.pull()
                self.assertFalse(watch.update_pending())

                # Verify new content was pulled
                with open(os.path.join(watch_path, "test.txt")) as f:
                    self.assertEqual(f.read(), "v2")

                watch.stop()

    @pytest.mark.timeout(15)
    def test_watch_repo_bad_url_raises_error(self):
        """Test GitSync raises GitError for invalid repo URL."""
        with tempfile.TemporaryDirectory() as watch_path:
            watch = GitSync(repo="/nonexistent/repo/path", watch_path=watch_path, period=1)
            with self.assertRaises(GitError):
                watch.start()

    @pytest.mark.timeout(15)
    def test_watch_repo_double_start_raises_error(self):
        """Test GitSync raises GitError if started twice."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                self._setup_bare_repo_with_content(remote_repo, "v1")

                watch = GitSync(repo=remote_repo, watch_path=watch_path, period=1)
                watch.start()

                # Attempting to start again should raise
                with self.assertRaises(GitError):
                    watch.start()

                watch.stop()

    @pytest.mark.timeout(15)
    def test_watch_repo_thread_lifecycle(self):
        """Test GitSync manages thread lifecycle correctly."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                self._setup_bare_repo_with_content(remote_repo, "v1")

                watch = GitSync(repo=remote_repo, watch_path=watch_path, period=1)

                # Thread should not be running initially
                self.assertFalse(watch._notify_thread.is_alive())

                watch.start()
                # Thread should be running after start
                self.assertTrue(watch._notify_thread.is_alive())

                watch.stop()
                # Give thread time to exit
                time.sleep(0.5)
                # Thread should be stopped after stop
                self.assertFalse(watch._notify_thread.is_alive())

    @pytest.mark.timeout(15)
    def test_watch_repo_fetch_with_no_changes(self):
        """Test GitSync fetch works when there are no remote changes."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                self._setup_bare_repo_with_content(remote_repo, "v1")

                watch = GitSync(repo=remote_repo, watch_path=watch_path, period=1)
                watch.start()

                # Fetch when nothing has changed should not raise
                watch.pull()

                with open(os.path.join(watch_path, "test.txt")) as f:
                    self.assertEqual(f.read(), "v1")

                watch.stop()

    @pytest.mark.timeout(15)
    def test_watch_repo_specified_branch(self):
        """Test GitSync can checkout a specific branch."""
        with tempfile.TemporaryDirectory() as remote_repo:
            with tempfile.TemporaryDirectory() as watch_path:
                # initialize a repo multiple branches
                subprocess.run(["git", "init", "--bare"], cwd=remote_repo, check=True, capture_output=True)

                with tempfile.TemporaryDirectory() as temp_clone:
                    subprocess.run(["git", "clone", remote_repo, "."], cwd=temp_clone, check=True, capture_output=True)

                    # Create main branch with content
                    with open(os.path.join(temp_clone, "test.txt"), "w") as f:
                        f.write("main content")
                    subprocess.run(["git", "add", "test.txt"], cwd=temp_clone, check=True, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "main"], cwd=temp_clone, check=True, capture_output=True)
                    subprocess.run(["git", "push", "-u", "origin"], cwd=temp_clone, check=True, capture_output=True)

                    # Create dev branch
                    subprocess.run(["git", "checkout", "-b", "dev"], cwd=temp_clone, check=True, capture_output=True)
                    with open(os.path.join(temp_clone, "test.txt"), "w") as f:
                        f.write("dev content")
                    subprocess.run(["git", "add", "test.txt"], cwd=temp_clone, check=True, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "dev"], cwd=temp_clone, check=True, capture_output=True)
                    subprocess.run(
                        ["git", "push", "-u", "origin", "dev"], cwd=temp_clone, check=True, capture_output=True
                    )

                # Watch dev branch
                watch = GitSync(repo=remote_repo, branch="dev", watch_path=watch_path, period=1)
                watch.start()

                # Should have dev content
                with open(os.path.join(watch_path, "test.txt")) as f:
                    self.assertEqual(f.read(), "dev content")

                watch.stop()
