from __future__ import annotations

from dataclasses import dataclass
import datetime
import logging
from multiprocessing import shared_memory
import pickle
import time
from typing import Any, Callable

import httpx

from azul_runner import Feature, FeatureType, Job, Plugin, add_settings
from azul_runner.models import FV, State
from azul_bedrock import models_network as azm
from . import mock_dispatcher as md
from azul_runner.test_utils.test_template import TestPlugin

# Example instances of every feature type in azul.runner.models.VALID_FEATURE_TYPES
VALID_FEATURE_EXAMPLES = (101, 55.5, "string", b"BYTES1011", datetime.datetime.now(datetime.timezone.utc))


def setup_mock_dispatcher() -> tuple[md.MockDispatcher, str]:
    """Setup a mock dispatcher server and return the mock_server and the server url."""
    mock_server = md.RunnerMockDispatcher()
    mock_server.start()
    total_sleep = 0
    while not mock_server.is_alive() and total_sleep < 20:
        time.sleep(0.5)  # Wait for server to start
        total_sleep += 1
    server = "http://%s:%s" % (mock_server.host, mock_server.port)
    # Wait for server to be ready to respond
    tries = 0
    while True:
        time.sleep(0.2)
        tries += 1
        try:
            resp = httpx.get(server + "/mock/get_var/fetch_count")
            if resp.is_error:
                raise httpx.ConnectError(f"Failing to connect to dispatcher on {server}/mock/get_var/fetch_count")
            break  # Exit loop if successful
        except (httpx.TimeoutException, httpx.ConnectError):
            if tries > 20:  # Time out after about 4 seconds
                mock_server.stop()
                raise RuntimeError("Timed out waiting for mock server to be ready")
    return mock_server, server


class DummyLogHandler(logging.Handler):
    """Implements a handler that simply records any log messages generated in `self.logs`."""

    logs: list[str]

    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record: logging.LogRecord) -> None:
        self.logs.append("%s: %s" % (record.levelname, record.getMessage()))


# ################################################
# Dummy plugin definitions for use by test cases
# ################################################


class DummyPluginMinimum(Plugin):
    """
    Test class that will register successfully but has no execute method.
    """

    SECURITY = None
    VERSION = "none"
    FEATURES = []


# noinspection PyAbstractClass
class DummyPluginNoExecute(Plugin):
    """
    Test class that will register successfully but has no execute method.
    """

    SETTINGS = add_settings(request_retry_count=0)  # Don't retry failed requests when testing
    SECURITY = None
    VERSION = "none"
    FEATURES = []


class DummyPluginNotReady(DummyPluginNoExecute):
    """
    Test class that will always report the plugin is not ready for jobs and raise if given any.
    """

    def is_ready(self):
        return False

    def execute(self, entity):
        raise Exception("Never call me")


class DummyPlugin(Plugin):
    """Test class that passes various registration information, and returns a configurable value from execute()."""

    SETTINGS = add_settings(
        # test_input_data allows plugins to pass in data needed in the execute method easily.
        request_retry_count=0,
        test_input_data=(dict, {}),
    )  # Don't retry failed requests when testing
    # leave security property unset
    # SECURITY = None
    VERSION = "1.0"
    MULTI_STREAM_AWARE = True
    FEATURES = [
        Feature("example_string", "Example string feature", type=FeatureType.String),
        Feature("example_int", "Example int feature", type=FeatureType.Integer),
        Feature("example_raw", "Example raw bytes feature", type=FeatureType.Binary),
        Feature("example_date", "Example datetime feature", type=FeatureType.Datetime),
        Feature("example_unspec", "Example feature of unspecified type"),
        Feature("example_path", "Example Filepath feature", type=FeatureType.Filepath),
        Feature("example_uri", "Example URI feature", type=FeatureType.Uri),
    ]

    def execute(self, job: Job):
        pass


class DummyPluginDefaultSharedMem(DummyPlugin):
    def get_shared_memory_name(self):
        raise NotImplementedError(
            "You must define a get_shared_memory_name method that returned the shared memory name"
        )

    def _load_from_shared_memory(self):
        """Unpickle the value stored in shared memory."""
        shared_mem = shared_memory.SharedMemory(name=self.get_shared_memory_name(), create=False)
        return pickle.loads(shared_mem.buf[: shared_mem.size])


def cleanup_shared_memory(shared_memory_name):
    """Delete an old shared memory file if it exists.

    If shutdown happens in a bad way sometimes the shared memory file can get left over.
    """
    try:
        shared_mem_old = shared_memory.SharedMemory(name=shared_memory_name, create=False)
        shared_mem_old.close()
        shared_mem_old.unlink()
    except Exception:
        pass


class DummyPluginFeatureInheritance(DummyPlugin):
    """Test class to ensure that features accumulate between template plugins and their descendants."""

    VERSION = "2.0"  # Much better than 1.0. (Test to ensure it differs from the parent class)
    FEATURES = [
        Feature("descendant feature", "A feature added by the child plugin"),
        Feature("example_unspec", "Child class redefining feature", type=FeatureType.String),
    ]
    # All other class vars should be inherited


class TestPluginSharedMem(TestPlugin):
    def _set_shared_memory(self, shared_mem: shared_memory.SharedMemory, value: Any):
        """Set the shared memory to be equal to the provided value"""
        mem_view = shared_mem.buf
        if not mem_view:
            raise Exception("Memory view is unexpectedly None, can't use shared memory!")
        # Clear memory buffer
        mem_view[: shared_mem.size] = b"\x00" * shared_mem.size
        # Set value
        picked_value = pickle.dumps(value)
        mem_view[: len(picked_value)] = picked_value
