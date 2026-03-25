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
    """Notifies main thread when a watched git repository has been updated."""

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
    ):
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

        self._gitconfig = tempfile.NamedTemporaryFile(delete=False, prefix=".git").name
        self._gitcredential = tempfile.NamedTemporaryFile(delete=False, prefix=".git").name
        self._sync_failures = 0
        self._notify_thread: threading.Thread = threading.Thread(target=self._run_loop)
        self._stop_event: threading.Event = threading.Event()
        self._update_event: threading.Event = threading.Event()

        logger.info(
            f"Setting up GitSync to watch repo {self.repo} at path {self.watch_path} with period {self.period} seconds"
        )
        self._init_repo()

    def start_notify_thread(self):
        """Start a thread that watches a remote repo for updates and notifies the main thread when updates are available."""
        if self._notify_thread.is_alive():
            msg = "GitSync thread is already running"
            logger.error(msg)
            raise GitError(msg)

        self._notify_thread.start()
        logger.info("GitSync thread started")

    def stop_notify_thread(self):
        """Tell notify thread to exit then wait for it to join."""
        logger.info("Stopping GitSync notification thread")
        self._stop_event.set()
        if self._notify_thread.is_alive():
            self._notify_thread.join(timeout=30.0)

    def update_pending(self) -> bool:
        """Return whether or not the notification thread has set the update_event flag, indicating the remote has new content."""
        if self._sync_failures > self.max_sync_failures:
            msg = f"GitSync has failed to check for updates {self._sync_failures} times which is above the max_sync_failures threshold of {self.max_sync_failures}."
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
        self._sync_failures = 0

        if self.submodules != "off":
            self._sync_submodules()

        if self._update_event.is_set():
            self._update_event.clear()

    def _init_repo(self):
        """Initialize the git repository in the watch path if it doesn't exist."""
        if not self.repo:
            msg = "GitSync repo URL is not set"
            logger.error(msg)
            raise GitError(msg)

        logger.info(f"Initializing git repository at {self.watch_path} with remote {self.repo}")

        # create watch dir if necessary
        if not os.path.isdir(self.watch_path):
            os.makedirs(self.watch_path, exist_ok=True)
            logger.info(f"Created watch directory at {self.watch_path}")

        # ~/.gitconfig may not be writable
        if not os.access(os.path.expanduser("~"), os.W_OK):
            os.environ["GIT_CONFIG_GLOBAL"] = self._gitconfig

        if not os.path.exists(os.path.join(self.watch_path, ".git")):
            # clone if repo does not exist
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

    def _sync_submodules(self):
        submodule_cmd = ["submodule", "update", "--init", "--no-progress"]
        if self.submodules == "recursive":
            submodule_cmd.append("--recursive")
        if self.clone_depth > 0:
            submodule_cmd.append(f"--depth={self.clone_depth}")
        self._run_git(submodule_cmd)
        logger.info(f"Synced {self.repo} submodules with option {self.submodules}")

    def _run_git(self, cmd: list[str], input: str = None) -> str:
        """Run a git command in the watch path and return the output."""
        try:
            return subprocess.check_output(  # noqa: S603
                ["git"] + cmd,
                cwd=self.watch_path,
                text=True,
                input=input,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            msg = f"Git command '{' '.join(cmd)}' failed: {e.output}:{e.returncode}"
            logger.error(msg)
            raise GitError(msg) from e

    def _refresh_auth(self):
        if not self.do_ssh_auth and (self.username or self.password):
            # Refresh creds in memory since we are using cache as the storage mechanism
            logger.debug("Refreshing HTTPS authentication for git")
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
        if self.do_ssh_auth:
            logger.debug(f"Refreshing SSH authentication for git repository at {self.repo}")
            self._run_git(["config", "--global", "core.sshCommand", f"ssh -i {self.ssh_key_path}"])

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                local = self._run_git(["rev-parse", "HEAD"]).strip()

                ls_cmd = ["ls-remote", "origin"] + ([self.branch] if self.branch else ["HEAD"])
                self._refresh_auth()
                remote = self._run_git(ls_cmd).split()[0].strip()

                # post to self.update_event if they are not equal (parent proc now knows to pull then restart the plugin)
                if local != remote:
                    logger.info(f"Remote repo has new updates (local HEAD: {local} remote HEAD: {remote}).")
                    self._update_event.set()
                else:
                    logger.debug(f"Remote repo has no new updates (local HEAD: {local} remote HEAD: {remote}).")

                # wait until it is either time to check remote again or the main thread tell this thread to stop with GitSync.stop()
                self._stop_event.wait(timeout=self.period)
                self._sync_failures = 0
            except GitError as e:
                logger.error(f"Error checking for updates from remote repo: {e}")
                self._sync_failures += 1
                if self._sync_failures > self.max_sync_failures:
                    logger.error(
                        f"Maximum sync failures reached ({self.max_sync_failures}); terminating watch thread."
                    )
                    break
