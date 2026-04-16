"""Module for syncing a local git repository with a remote and notifying the main thread when updates are available."""

import logging
import os
import subprocess
import tempfile
import threading

logger = logging.getLogger(__name__)


class GitError(RuntimeError):
    """Raised if the remote git server for the repository being watched cannot be reached or cannot be authenticated to."""

    pass


class GitSync:
    """Notifies main thread when a watched git repository has been updated.

    Security Note: When using HTTPS authentication with credential store, credentials are stored
    in plaintext in ~/.gitcredential. For sensitive credentials, consider using SSH authentication
    instead (do_ssh_auth=True with ssh_key_path).
    """

    def __init__(
        self,
        repo: str,
        watch_path: str,
        period: int,
        branch: str = "",
        username: str = "",
        password: str = "",
        do_ssh_auth: bool = False,
        ssh_key_path: str = "",
        max_sync_failures: int = 0,
        clone_depth: int = 0,
        submodules: str = "off",
        git_config: str = "",
    ):
        # Validate inputs
        if not repo:
            raise ValueError("repo URL must not be empty")
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        if do_ssh_auth and not ssh_key_path:
            raise ValueError("ssh_key_path must be provided when do_ssh_auth is True")
        if not do_ssh_auth and (username or password):
            if not username or not password:
                raise ValueError("Both username and password must be provided for HTTPS auth")
        if clone_depth < 0:
            raise ValueError(f"clone_depth must be >= 0, got {clone_depth}")
        if submodules not in ("off", "on", "recursive"):
            raise ValueError(f"submodules must be 'off', 'on', or 'recursive', got {submodules}")
        if git_config and ":" not in git_config:
            raise ValueError(f"git_config must be in 'key:value' format, got {git_config!r}")

        self.repo: str = repo
        self.branch: str = branch
        self.watch_path: str = watch_path
        self.period: int = period
        self.username: str = username
        self.password: str = password
        self.do_ssh_auth: bool = do_ssh_auth
        self.ssh_key_path: str = ssh_key_path
        self.max_sync_failures: int = max_sync_failures
        self.clone_depth: int = clone_depth
        self.submodules: str = submodules
        self.git_config: str = git_config

        with tempfile.NamedTemporaryFile(delete=False, prefix=".gitsync") as f:
            self._gitconfig: str = f.name
        with tempfile.NamedTemporaryFile(delete=False, prefix=".gitsync") as f:
            self._gitcredential: str = f.name

        # preserve old git config if it exists
        self._original_git_config_global: str | None = os.environ.get("GIT_CONFIG_GLOBAL")
        if self._original_git_config_global:
            with open(self._original_git_config_global, "r") as original, open(self._gitconfig, "w") as new:
                new.write(original.read())
        os.environ["GIT_CONFIG_GLOBAL"] = self._gitconfig

        self._sync_failures: int = 0
        self._sync_failures_lock: threading.Lock = threading.Lock()
        self._notify_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._update_event: threading.Event = threading.Event()

        self._init_repo()

    def _init_repo(self):
        """Initialize the git repository in the watch path if it doesn't exist."""
        logger.info(f"GitSync repo {self.repo} at {self.watch_path} with period {self.period} seconds")

        # create watch dir if necessary
        if not os.path.isdir(self.watch_path):
            os.makedirs(self.watch_path, exist_ok=True)
            logger.info(f"Created watch directory at {self.watch_path}")

        # clone if repo does not exist locally yet
        if not os.path.exists(os.path.join(self.watch_path, ".git")):
            clone_cmd = ["clone", "--verbose", self.repo, "."]
            if self.branch:
                clone_cmd.insert(2, f"--branch={self.branch}")
            if self.clone_depth > 0:
                clone_cmd.insert(2, f"--depth={self.clone_depth}")

            self._refresh_auth()
            logger.info(f"Cloning repository from {self.repo} to {self.watch_path}")
            clone_output = self._run_git(clone_cmd)
            logger.info(f"{self.repo} clone complete: {clone_output}")

            if self.submodules != "off":
                self._sync_submodules()

    def _refresh_auth(self):
        if not self.do_ssh_auth and (self.username or self.password):
            logger.debug(f"Refreshing HTTPS authentication for git repository at {self.repo}")
            cmd = ["config", "--global", "credential.helper"]
            # home directory may not be writable, but cache needs home directory to be writable
            if os.access(os.path.expanduser("~"), os.W_OK):
                cmd += ["cache"]
            else:
                cmd += [f"store --file={self._gitcredential}"]

            self._run_git(cmd)
            if "@" in self.repo:
                # if @ is in repo url, it contains a username and we need to use it instead of self.username
                input = f"url={self.repo}\npassword={self.password}\n"
            else:
                # if @ is not in repo url, we can use self.username
                input = f"url={self.repo}\nusername={self.username}\npassword={self.password}\n"
            self._run_git(
                ["credential", "approve"],
                input=input,
            )
        elif self.do_ssh_auth:
            logger.debug(f"Refreshing SSH authentication for git repository at {self.repo}")
            self._run_git(
                [
                    "config",
                    "--global",
                    "core.sshCommand",
                    f"ssh -i {self.ssh_key_path} -o StrictHostKeyChecking=No -o UserKnownHostsFile=/dev/null",
                ]
            )

    def _run_git(self, cmd: list[str], input: str = None) -> str:
        """Run a git command in the watch path and return the output."""
        config_args = []
        if self.git_config:
            key, _, value = self.git_config.partition(":")
            config_args = ["-c", f"{key}={value}"]
        try:
            output = subprocess.check_output(  # noqa: S603
                ["git"] + config_args + cmd,
                cwd=self.watch_path,
                text=True,
                input=input,
                stderr=subprocess.STDOUT,
            )

            if self.do_ssh_auth and "Warning: " in output:
                # Suppress warning about not being able to verify host key when using SSH auth
                logger.debug(f"Received host key warning when running git command '{' '.join(cmd)}': {output.strip()}")
                output = "\n".join(output.splitlines()[1:])
            return output
        except subprocess.CalledProcessError as e:
            msg = f"Git command '{' '.join(cmd)}' failed with code {e.returncode}: {e.output}"
            logger.error(msg)
            raise GitError(msg) from e

    def _sync_submodules(self):
        submodule_cmd = ["submodule", "update", "--init", "--no-progress"]
        if self.submodules == "recursive":
            submodule_cmd.append("--recursive")
        if self.clone_depth > 0:
            submodule_cmd.append(f"--depth={self.clone_depth}")
        self._run_git(submodule_cmd)
        logger.info(f"Synced {self.repo} submodules with option {self.submodules}")

    def update_pending(self) -> bool:
        """Return whether or not the notification thread has set the update_event flag, indicating the remote has new content."""
        with self._sync_failures_lock:
            sync_failures = self._sync_failures
        if sync_failures > self.max_sync_failures:
            msg = f"GitSync has failed to check for updates {sync_failures} times which is above the max_sync_failures threshold of {self.max_sync_failures}."
            logger.error(msg)
            self.stop_notify_thread()
            raise GitError(msg)
        return self._update_event.is_set()

    def pull(self):
        """Fetch updates from the remote, if available."""
        logger.info(f"Pulling updates from {self.repo} to {self.watch_path}")
        self._refresh_auth()
        pull_output = self._run_git(["pull", "origin", "--verbose", "--no-progress", "--prune"])
        logger.info(f"{self.repo} pull complete: {pull_output}")
        with self._sync_failures_lock:
            self._sync_failures = 0

        if self.submodules != "off":
            self._sync_submodules()

        self._update_event.clear()

    def start_notify_thread(self):
        """Start a thread that watches a remote repo for updates and notifies the main thread when updates are available."""
        if self._notify_thread is not None and self._notify_thread.is_alive():
            msg = "GitSync thread is already running"
            logger.error(msg)
            raise GitError(msg)

        self._stop_event.clear()
        self._notify_thread = threading.Thread(target=self._run_notify_thread)
        self._notify_thread.start()
        logger.info(f"Notification thread started for {self.repo} at {self.watch_path} with {self.period}s period")

    def stop_notify_thread(self):
        """Tell notify thread to exit then wait for it to join."""
        logger.info("Stopping GitSync notification thread")
        self._stop_event.set()
        if self._notify_thread is not None and self._notify_thread.is_alive():
            self._notify_thread.join(timeout=10)
            if self._notify_thread.is_alive():
                logger.warning("GitSync thread did not exit within 10 seconds")
        self._cleanup()

    def _run_notify_thread(self):
        """Thread function to check for updates from the remote repo and notify the main thread when updates are available."""
        while not self._stop_event.is_set():
            try:
                local = self._run_git(["rev-parse", "HEAD"]).strip()

                ls_cmd = ["ls-remote", "origin"] + ([self.branch] if self.branch else ["HEAD"])
                self._refresh_auth()

                remote = self._run_git(ls_cmd).strip().split()[0]

                # post to self.update_event if they are not equal (parent proc now knows to pull then restart the plugin)
                if local != remote:
                    logger.info(f"Remote repo has new updates (local HEAD: {local} remote HEAD: {remote}).")
                    self._update_event.set()
                else:
                    logger.debug(f"Remote repo has no new updates (local HEAD: {local} remote HEAD: {remote}).")

                # wait until it is either time to check remote again or the main thread tell this thread to stop with GitSync.stop()
                self._stop_event.wait(timeout=self.period)
                with self._sync_failures_lock:
                    self._sync_failures = 0
            except Exception as e:
                logger.error(f"Notify Thread: Error checking for updates from remote repo: {e}")
                with self._sync_failures_lock:
                    self._sync_failures += 1
                    sync_failures = self._sync_failures
                if sync_failures > self.max_sync_failures:
                    logger.error(f"Max sync failures reached ({self.max_sync_failures}); terminating notify thread.")
                    break

    def _cleanup(self):
        """Clean up temporary files created for git config/credentials."""
        for temp_file in [self._gitconfig, self._gitcredential]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
        # Restore original GIT_CONFIG_GLOBAL if it was set
        if self._original_git_config_global is not None:
            os.environ["GIT_CONFIG_GLOBAL"] = self._original_git_config_global
        elif "GIT_CONFIG_GLOBAL" in os.environ:
            del os.environ["GIT_CONFIG_GLOBAL"]
