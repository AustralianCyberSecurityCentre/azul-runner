# flake8: noqa - Flake8 doesn't detect these exported functions and classes properly.
import contextlib

from azul_bedrock.models_network import DataLabel, FeatureType
from azul_bedrock.models_network import FeatureValue as APIFeatureValue

from .binary_plugin import BinaryPlugin
from .main import cmdline_run
from .models import (
    FV,
    Event,
    EventData,
    EventParent,
    Feature,
    FeatureValue,
    Filepath,
    Job,
    JobResult,
    State,
    Uri,
)
from .plugin import Plugin
from .settings import add_settings
from .storage import DATA_HASH, DATA_HASH_NAME, StorageProxyFile

append_all: list[str] = []
with contextlib.suppress(ImportError):
    from .test_utils import test_template
    from .test_utils.test_template import TestPlugin

    print("You have the azul-runner[test_utils] installed, this should only be used for development or testing.")
    append_all = ["TestPlugin", "test_template"]

EXPORTS = append_all + [
    "add_settings",
    "APIFeatureValue",
    "BinaryPlugin",
    "cmdline_run",
    "DATA_HASH_NAME",
    "DATA_HASH",
    "DataLabel",
    "Event",
    "EventData",
    "EventParent",
    "Feature",
    "FeatureType",
    "FeatureValue",
    "Filepath",
    "FV",
    "Job",
    "JobResult",
    "Plugin",
    "State",
    "StorageProxyFile",
    "Uri",
]

__all__ = list(EXPORTS)
