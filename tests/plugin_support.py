from __future__ import annotations

import datetime
import logging
import os

from azul_runner import Feature, FeatureType, Job, Plugin, add_settings

# Example instances of every feature type in azul.runner.models.VALID_FEATURE_TYPES
VALID_FEATURE_EXAMPLES = (101, 55.5, "string", b"BYTES1011", datetime.datetime.now(datetime.timezone.utc))


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
        request_retry_count=0, use_multiprocessing_fork=True
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


class DummyPluginFeatureInheritance(DummyPlugin):
    """Test class to ensure that features accumulate between template plugins and their descendants."""

    VERSION = "2.0"  # Much better than 1.0. (Test to ensure it differs from the parent class)
    FEATURES = [
        Feature("descendant feature", "A feature added by the child plugin"),
        Feature("example_unspec", "Child class redefining feature", type=FeatureType.String),
    ]
    # All other class vars should be inherited
