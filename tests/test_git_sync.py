"""Unit tests for GitSync functionality in azul_runner.monitor."""

import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from azul_runner.git_sync import GitSync, GitError


# ============================================================================
# Helpers and Fixtures
# ============================================================================


def run_git(cmd: list[str], cwd: str, check: bool = True) -> None:
    """Run git command with standard options."""
    subprocess.run(["git"] + cmd, cwd=cwd, check=check, capture_output=True)


def setup_bare_repo_with_content(remote_path: str, content: str = "v1") -> None:
    """Create a bare repo with initial test content."""
    run_git(["init", "--bare"], remote_path)

    with tempfile.TemporaryDirectory() as temp_clone:
        run_git(["clone", remote_path, "."], temp_clone)
        test_file = Path(temp_clone) / "test.txt"
        test_file.write_text(content)
        run_git(["add", "."], temp_clone)
        run_git(["commit", "-m", "initial"], temp_clone)
        run_git(["push"], temp_clone)


def setup_multiword_branch_repo(remote_path: str, branches: dict[str, str]) -> None:
    """Create a repo with multiple branches and content.

    Args:
        remote_path: Path to bare repo
        branches: Dict of {branch_name: content}
    """
    run_git(["init", "--bare"], remote_path)

    with tempfile.TemporaryDirectory() as temp_clone:
        run_git(["clone", remote_path, "."], temp_clone)

        # Create first branch
        first_branch = next(iter(branches.keys()))
        first_content = branches[first_branch]
        test_file = Path(temp_clone) / "test.txt"
        test_file.write_text(first_content)
        run_git(["add", "test.txt"], temp_clone)
        run_git(["commit", "-m", first_branch], temp_clone)
        run_git(["push", "-u", "origin", first_branch], temp_clone)

        # Create additional branches
        for branch, content in list(branches.items())[1:]:
            run_git(["checkout", "-b", branch], temp_clone)
            test_file.write_text(content)
            run_git(["add", "test.txt"], temp_clone)
            run_git(["commit", "-m", branch], temp_clone)
            run_git(["push", "-u", "origin", branch], temp_clone)


def push_update_to_remote(remote_path: str, new_content: str) -> None:
    """Push an update to an existing remote repo."""
    with tempfile.TemporaryDirectory() as temp_clone:
        run_git(["clone", remote_path, "."], temp_clone)
        test_file = Path(temp_clone) / "test.txt"
        test_file.write_text(new_content)
        run_git(["add", "test.txt"], temp_clone)
        run_git(["commit", "-m", f"update to {new_content}"], temp_clone)
        run_git(["push"], temp_clone)


def wait_for_update_pending(watch: GitSync, timeout: float = 5) -> bool:
    """Wait for update to be detected, return True if found, raise TimeoutError if not."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if watch.update_pending():
            return True
        time.sleep(0.1)
    raise TimeoutError(f"Update not detected within {timeout}s")


@contextmanager
def git_sync_running(repo: str, watch_path: str, **kwargs):
    """Context manager for GitSync lifecycle (start/stop)."""
    watch = GitSync(repo=repo, watch_path=watch_path, period=1, **kwargs)
    watch.start_notify_thread()
    try:
        yield watch
    finally:
        watch.stop_notify_thread()
        time.sleep(0.5)


@pytest.fixture(scope="function")
def set_git_config():
    """Fixture to set up a temporary git config for testing."""
    with tempfile.NamedTemporaryFile(delete=False, prefix=".gitsync") as f:
        gitconfig = f.name

    original_git_config_global = os.environ.get("GIT_CONFIG_GLOBAL")
    os.environ["GIT_CONFIG_GLOBAL"] = gitconfig

    run_git(["config", "--global", "user.name", "Test User"])
    run_git(["config", "--global", "user.email", "test@example.com"])

    yield

    # Cleanup
    os.remove(gitconfig)
    if original_git_config_global is not None:
        os.environ["GIT_CONFIG_GLOBAL"] = original_git_config_global
    else:
        del os.environ["GIT_CONFIG_GLOBAL"]


@pytest.fixture
def tmp_git_repos(tmp_path):
    """Fixture providing remote and watch paths."""
    remote_path = tmp_path / "remote"
    remote_path.mkdir()
    watch_path = tmp_path / "watch"
    watch_path.mkdir()
    return str(remote_path), str(watch_path)


@pytest.fixture
def git_sync_with_initial_content(tmp_git_repos):
    """Fixture providing a started GitSync with initial repo content."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")
    with git_sync_running(remote_path, watch_path) as watch:
        yield watch


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.timeout(15)
def test_git_sync_clone_and_initial_fetch(tmp_git_repos):
    """Test GitSync clones repo and performs initial fetch on start."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "initial content")

    with git_sync_running(remote_path, watch_path) as watch:
        test_file = Path(watch_path) / "test.txt"
        assert test_file.exists()
        assert test_file.read_text() == "initial content"


@pytest.mark.timeout(15)
def test_git_sync_skips_clone_if_exists(tmp_git_repos):
    """Test GitSync does not reclone if local repo already exists."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    run_git(["clone", remote_path, "."], watch_path)
    test_file = Path(watch_path) / "test.txt"
    test_file.write_text("local modification")

    with git_sync_running(remote_path, watch_path):
        assert test_file.read_text() == "local modification"


@pytest.mark.timeout(15)
def test_git_sync_detects_remote_updates(tmp_git_repos):
    """Test GitSync detects when remote has new commits."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    with git_sync_running(remote_path, watch_path) as watch:
        assert not watch.update_pending()

        push_update_to_remote(remote_path, "v2")
        wait_for_update_pending(watch)
        assert watch.update_pending()


@pytest.mark.timeout(15)
def test_git_sync_fetch_clears_update_event(tmp_git_repos):
    """Test GitSync.pull() clears the update_pending flag and fetches content."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    with git_sync_running(remote_path, watch_path) as watch:
        push_update_to_remote(remote_path, "v2")
        wait_for_update_pending(watch)
        assert watch.update_pending()

        watch.pull()
        assert not watch.update_pending()
        assert (Path(watch_path) / "test.txt").read_text() == "v2"


@pytest.mark.timeout(15)
def test_git_sync_bad_url_raises_error(tmp_path):
    """Test GitSync raises GitError for invalid repo URL."""
    watch_path = tmp_path / "watch"
    watch_path.mkdir()

    with pytest.raises(GitError):
        GitSync(repo="/nonexistent/repo/path", watch_path=str(watch_path), period=1)


@pytest.mark.timeout(15)
def test_git_sync_double_start_raises_error(tmp_git_repos):
    """Test GitSync raises GitError if started twice."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    watch = GitSync(repo=remote_path, watch_path=watch_path, period=1)
    watch.start_notify_thread()

    try:
        with pytest.raises(GitError):
            watch.start_notify_thread()
    finally:
        watch.stop_notify_thread()


@pytest.mark.timeout(15)
def test_git_sync_thread_lifecycle(tmp_git_repos):
    """Test GitSync manages thread lifecycle correctly."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    watch = GitSync(repo=remote_path, watch_path=watch_path, period=1)

    watch.start_notify_thread()
    assert watch._notify_thread.is_alive()

    watch.stop_notify_thread()
    time.sleep(0.5)
    assert not watch._notify_thread.is_alive()


@pytest.mark.timeout(15)
def test_git_sync_fetch_with_no_changes(tmp_git_repos):
    """Test GitSync.pull() works when there are no remote changes."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    with git_sync_running(remote_path, watch_path) as watch:
        watch.pull()
        assert (Path(watch_path) / "test.txt").read_text() == "v1"


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "branch,expected_content",
    [
        ("master", "main content"),
        ("dev", "dev content"),
    ],
)
def test_git_sync_checkout_branch(tmp_path, branch, expected_content):
    """Test GitSync can checkout specific branches."""
    remote_path = tmp_path / "remote"
    remote_path.mkdir()
    watch_path = tmp_path / "watch"
    watch_path.mkdir()

    branches = {
        "master": "main content",
        "dev": "dev content",
    }
    setup_multiword_branch_repo(str(remote_path), branches)

    with git_sync_running(str(remote_path), str(watch_path), branch=branch) as watch:
        test_file = watch_path / "test.txt"
        assert test_file.read_text() == expected_content


@pytest.mark.timeout(15)
def test_git_sync_shallow_clone_with_depth(tmp_git_repos):
    """Test GitSync can perform shallow clone with specified depth."""
    remote_path, watch_path = tmp_git_repos

    run_git(["init", "--bare"], remote_path)

    with tempfile.TemporaryDirectory() as temp_clone:
        run_git(["clone", remote_path, "."], temp_clone)

        for i in range(5):
            test_file = Path(temp_clone) / "test.txt"
            test_file.write_text(f"commit {i}")
            run_git(["add", "test.txt"], temp_clone)
            run_git(["commit", "-m", f"commit {i}"], temp_clone)
        run_git(["push"], temp_clone)

    with git_sync_running(remote_path, watch_path, clone_depth=2) as watch:
        test_file = Path(watch_path) / "test.txt"
        assert test_file.exists()
        assert test_file.read_text() == "commit 4"


@pytest.mark.timeout(15)
def test_git_sync_max_sync_failures_raises_error(tmp_git_repos):
    """Test GitSync raises GitError when sync failures exceed max_sync_failures."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    watch = GitSync(repo=remote_path, watch_path=watch_path, period=1, max_sync_failures=2)
    watch.start_notify_thread()

    try:
        broken_repo_path = remote_path + "_broken"
        Path(remote_path).rename(broken_repo_path)
        time.sleep(4)

        with pytest.raises(GitError):
            watch.update_pending()
    finally:
        Path(broken_repo_path).rename(remote_path)
        watch.stop_notify_thread()


@pytest.mark.timeout(15)
def test_git_sync_stop_clears_resources(tmp_git_repos):
    """Test GitSync.stop() properly clears resources and allows restart."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    watch = GitSync(repo=remote_path, watch_path=watch_path, period=1)
    watch.start_notify_thread()

    assert watch._notify_thread.is_alive()

    watch.stop_notify_thread()
    time.sleep(1)

    assert not watch._notify_thread.is_alive()


@pytest.mark.timeout(15)
def test_git_sync_no_repo_url_raises_error(tmp_path):
    """Test GitSync raises error if repo URL is empty."""
    watch_path = tmp_path / "watch"
    watch_path.mkdir()

    with pytest.raises(ValueError):
        GitSync(repo="", watch_path=str(watch_path), period=1)


@pytest.mark.timeout(15)
def test_git_sync_update_event_detection_and_pull(tmp_git_repos):
    """Test GitSync detects update event and pull clears it."""
    remote_path, watch_path = tmp_git_repos
    setup_bare_repo_with_content(remote_path, "v1")

    with git_sync_running(remote_path, watch_path) as watch:
        assert not watch.update_pending()

        push_update_to_remote(remote_path, "v2")
        wait_for_update_pending(watch)

        assert watch.update_pending()

        watch.pull()
        assert not watch.update_pending()


@pytest.mark.timeout(15)
def test_git_sync_multiple_branch_tracking(tmp_path):
    """Test GitSync can track different branches in separate instances."""
    remote_path = tmp_path / "remote"
    remote_path.mkdir()
    watch_main = tmp_path / "watch_main"
    watch_main.mkdir()
    watch_dev = tmp_path / "watch_dev"
    watch_dev.mkdir()

    # Create repo with multiple branches
    branches = {"master": "main", "dev": "dev"}
    setup_multiword_branch_repo(str(remote_path), branches)

    watch_main_sync = GitSync(repo=str(remote_path), watch_path=str(watch_main), period=1, branch="master")
    watch_main_sync.start_notify_thread()

    watch_dev_sync = GitSync(repo=str(remote_path), watch_path=str(watch_dev), period=1, branch="dev")
    watch_dev_sync.start_notify_thread()

    try:
        # Verify content matches branch
        assert (watch_main / "test.txt").read_text() == "main"
        assert (watch_dev / "test.txt").read_text() == "dev"
    finally:
        watch_main_sync.stop_notify_thread()
        watch_dev_sync.stop_notify_thread()
