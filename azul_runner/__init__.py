# flake8: noqa - Ruff's linting rules doesn't detect these exported functions and classes properly.
import contextlib

from azul_bedrock.models_network import DataLabel, FeatureType
from azul_bedrock.models_network import FeatureValue as APIFeatureValue
from azul_bedrock.models_network import DownloadAction, DownloadEvent
from azul_runner.binary_plugin import BinaryPlugin
from azul_runner.download_plugin import DownloadPlugin
from azul_runner.main import cmdline_run
from azul_runner.models import (
    DownloadJob,
    Event,
    EventData,
    EventParent,
    FV,
    Feature,
    FeatureValue,
    Filepath,
    Job,
    JobResult,
    State,
    Uri,
)
from azul_runner.plugin import Plugin
from azul_runner.settings import add_settings
from azul_runner.storage import DATA_HASH, DATA_HASH_NAME, StorageProxyFile

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
    "DownloadAction",
    "DownloadEvent",
    "DownloadJob",
    "DownloadPlugin",
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
