from __future__ import annotations
from lib2to3.fixes.fix_input import context
from azul_runner.download_plugin import DownloadPlugin

import contextlib
import ctypes
import datetime
import json
import logging
import multiprocessing
import os
import tempfile
import time
import unittest
from multiprocessing import Process
import pendulum
from typing import Any, ClassVar
from unittest import mock

from azul_bedrock.exception_enums import ExceptionCodeEnum
import httpx
import yara_x
from azul_bedrock import exceptions_bedrock, models_network as azm

from azul_runner import (
    DATA_HASH,
    FV,
    Event,
    Feature,
    FeatureValue,
    Job,
    JobResult,
    State,
    StorageProxyFile,
    add_settings,
    coordinator,
    local,
    monitor,
    network,
    settings,
)
from azul_runner.models import TaskExitCodeEnum

from . import mock_dispatcher as md
from . import plugin_support as sup


def dump(x):
    return json.loads(x.model_dump_json(exclude_defaults=True))


class DummyDownloadPlugin(DownloadPlugin):
    """Hello world."""

    NAME = ""

    FEATURES = [
        Feature(name="feat1", desc="", type=azm.FeatureType.String),
        Feature(name="per_stream_feat", desc="", type=azm.FeatureType.String),
    ]

    SETTINGS = add_settings(
        # test_input_data allows plugins to pass in data needed in the execute method easily.
        request_retry_count=0,
        test_input_data=(dict, {}),
    )  # Don't retry failed requests when testing
    # leave security property unset
    # SECURITY = None
    VERSION = "1.0"

    def execute_download(self, job: DownloadJob) -> azm.DownloadAction | None:
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"dummy download/upload file, this file was nominally downloaded from some external source!")
            f.seek(0)
            with open(f.name, "rb") as raw_file:
                self.upload_sourced_file(raw_file, filename="dummy-download")

    def execute(self, job: Job):
        self.add_feature_values(FV("feat1", value="dummy-feat-1", label="dummy-label"))


class TestBaseDownloadPlugin(unittest.TestCase):
    """
    Test cases for base plugin class - cases that talk to the mock server
    """

    mock_server: ClassVar[md.MockDispatcher]
    server: ClassVar[str]  # Endpoint to the mock server, suitable for passing to a plugin's config['server']
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_server, cls.server = sup.setup_mock_dispatcher()
        # cls.editor = md.Editor(cls.server)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()

    # ############### #
    # #### Tests #### #
    # ############### #

    def _wait_for_is_alive_to_be_false(self, plugin_sub: Process, max_wait_sec: int = 20):
        """Sleep up to max_wait_sec waiting for a plugin subprocess to no longer be alive."""
        total_sleep = 0
        while plugin_sub.is_alive() and total_sleep < max_wait_sec:
            time.sleep(1)
            total_sleep += 1

    def _del(self, event):
        del (
            event["timestamp"],
            event["entity"]["config"]["server"],
            event["entity"]["config"]["events_url"],
            event["entity"]["config"]["data_url"],
        )

    def get_last_req_params(self) -> dict:
        r = httpx.get("%s/mock/get_var/last_req_params" % self.server)
        r.raise_for_status()
        return r.json()

    def test_download_plugin_registration(self):

        p = DummyDownloadPlugin(
            config={
                "events_url": self.server + "/download_event",
                "data_url": self.server,
            }
        )
        p.main_loop(0)

        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        self._del(out_event)
        print(out_event)
        self.assertEqual(
            out_event,
            {
                "model_version": 6,
                "kafka_key": "runner-placeholder",
                "author": {"category": "plugin", "name": "DummyDownloadPlugin", "version": "1.0"},
                "entity": {
                    "category": "plugin",
                    "name": "DummyDownloadPlugin",
                    "version": "1.0",
                    "description": "Hello world.",
                    "features": [
                        {"name": "feat1", "desc": "", "type": "string"},
                        {
                            "name": "file_extension",
                            "desc": "File extension of the 'content' stream.",
                            "type": "string",
                        },
                        {
                            "name": "file_format",
                            "desc": "Assemblyline file type of the 'content' stream.",
                            "type": "string",
                        },
                        {"name": "filename", "desc": "Name on disk of the 'content' stream.", "type": "filepath"},
                        {"name": "magic", "desc": "File magic found for the 'content' stream.", "type": "string"},
                        {"name": "malformed", "desc": "File is malformed in some way.", "type": "string"},
                        {"name": "mime", "desc": "Mimetype found for the 'content' stream.", "type": "string"},
                        {"name": "per_stream_feat", "desc": "", "type": "string"},
                    ],
                    "config": {
                        "assume_streams_available": "false",
                        "concurrent_plugin_instances": "1",
                        "cur_mem_file_path": '"/sys/fs/cgroup/memory.current"',
                        "cur_mem_summary_file_path": '"/sys/fs/cgroup/memory.stat"',
                        "deployment_key": '""',
                        "enable_liveness_probe": "false",
                        "enable_mem_limits": "false",
                        "filter_allow_event_types": "[]",
                        "filter_data_types": "{}",
                        "filter_max_content_size": "209715200",
                        "filter_min_content_size": "0",
                        "filter_self": "false",
                        "git_sync_clone_depth": "0",
                        "git_sync_git_config": '""',
                        "git_sync_max_sync_failures": "0",
                        "git_sync_password": '""',
                        "git_sync_period": "600",
                        "git_sync_ref": '""',
                        "git_sync_repo": '""',
                        "git_sync_ssh": "false",
                        "git_sync_ssh_key_path": '"/etc/git-secret/ssh/id_rsa"',
                        "git_sync_submodules": '"off"',
                        "git_sync_username": '""',
                        "heartbeat_interval": "30",
                        "is_processing_download_events": "false",
                        "max_mem_file_path": '"/sys/fs/cgroup/memory.max"',
                        "max_security": '""',
                        "max_timeouts_before_exit": "100",
                        "max_value_length": "4000",
                        "max_values_per_feature": "1000",
                        "mem_poll_frequency_milliseconds": "1000",
                        "name_remove_prefix": '"AzulPlugin"',
                        "name_suffix": '""',
                        "not_ready_backoff": "5",
                        "plugin_depth_limit": "10",
                        "request_retry_count": "0",
                        "request_timeout": "15",
                        "require_expedite": "true",
                        "require_historic": "true",
                        "require_live": "true",
                        "run_timeout": "600",
                        "security_override": '""',
                        "test_input_data": "{}",
                        "used_mem_force_exit_frac": "0.9",
                        "used_mem_warning_frac": "0.8",
                        "version_suffix": '""',
                        "watch_path": '""',
                        "watch_type": '""',
                        "watch_wait": "10",
                    },
                },
            },
        )

    def test_download_plugin_basic(self):

        p = DummyDownloadPlugin(
            config={
                "events_url": self.server + "/download_event",
                "data_url": self.server,
            }
        )
        p.main_loop(1)

        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        # self._del(out_event)
        with contextlib.suppress(Exception):
            out_event["timestamp"] = "1900-01-01T00:00:00+00:00"
        self.assertEqual(
            out_event,
            {
                "model_version": 6,
                "kafka_key": "runner-placeholder",
                "timestamp": "1900-01-01T00:00:00+00:00",
                "author": {"category": "plugin", "name": "DummyDownloadPlugin", "version": "1.0"},
                "entity": {"hash": "1234"},
                "source": {"name": "source", "timestamp": "1900-01-01T00:00:00+00:00", "path": []},
                "action": "success",
            },
        )

    @mock.patch("pendulum.now", lambda *args: pendulum.parse("2023-10-10T10:10:10Z"))
    def test_download_plugin_basic_inspect_sourced_event(self):
        p = DummyDownloadPlugin(
            config={
                "events_url": self.server + "/download_event",
                "data_url": self.server,
            }
        )
        p.main_loop(1)

        r = httpx.get("%s/mock/get_var/all_requests" % self.server)
        r.raise_for_status()
        out_events: dict = r.json()

        # Status event
        download_status_event = out_events[-1][0]
        with contextlib.suppress(Exception):
            download_status_event["timestamp"] = "1900-01-01T00:00:00+00:00"
        print("Download status event:")
        print(download_status_event)
        self.assertEqual(
            download_status_event,
            {
                "model_version": 6,
                "kafka_key": "runner-placeholder",
                "timestamp": "1900-01-01T00:00:00+00:00",
                "author": {"category": "plugin", "name": "DummyDownloadPlugin", "version": "1.0"},
                "entity": {"hash": "1234"},
                "source": {"name": "source", "timestamp": "1900-01-01T00:00:00+00:00", "path": []},
                "action": "success",
            },
        )

        # Event being sourced
        sourced_event = out_events[-2][0]
        print("Sourced event:")
        print(sourced_event)
        self.assertEqual(
            sourced_event,
            {
                "model_version": 6,
                "kafka_key": "DummyDownloadPlugin-placeholder",
                "timestamp": "2023-10-10T10:10:10+00:00",
                "author": {"category": "plugin", "name": "DummyDownloadPlugin", "version": "1.0"},
                "entity": {
                    "sha256": "fa79e7d0b1f22e945259b123f2169ca3a6c7d3dcf8dc40f123d146aec11a17ff",
                    "sha512": "0bdf7e5b6356b3791935da2db5b924ea8c8b95ecb01391736a543a289d0ecfe8df2901d7121a4efd064ca68722a53f305d7249adfafb97d29d3e53ecfa6b1c95",
                    "sha1": "a70bade3cc336425734de84d1fc52c2805ba8c91",
                    "md5": "bd1da1742bb553d977b7a4cc7524dfa7",
                    "size": 89,
                    "file_format": "#TEST/ONLY",
                    "file_extension": "tonly",
                    "mime": "#TESTONLY",
                    "magic": "#TESTONLY",
                    "features": [
                        {"name": "file_format", "type": "string", "value": "#TEST/ONLY"},
                        {"name": "file_extension", "type": "string", "value": "tonly"},
                        {"name": "magic", "type": "string", "value": "#TESTONLY"},
                        {"name": "mime", "type": "string", "value": "#TESTONLY"},
                        {"name": "filename", "type": "filepath", "value": "dummy-download"},
                    ],
                    "datastreams": [
                        {
                            "sha256": "fa79e7d0b1f22e945259b123f2169ca3a6c7d3dcf8dc40f123d146aec11a17ff",
                            "sha512": "0bdf7e5b6356b3791935da2db5b924ea8c8b95ecb01391736a543a289d0ecfe8df2901d7121a4efd064ca68722a53f305d7249adfafb97d29d3e53ecfa6b1c95",
                            "sha1": "a70bade3cc336425734de84d1fc52c2805ba8c91",
                            "md5": "bd1da1742bb553d977b7a4cc7524dfa7",
                            "size": 89,
                            "file_format": "#TEST/ONLY",
                            "file_extension": "tonly",
                            "mime": "#TESTONLY",
                            "magic": "#TESTONLY",
                            "identify_version": 1,
                            "label": "content",
                        }
                    ],
                },
                "action": "sourced",
                "source": {
                    "name": "source",
                    "timestamp": "2023-10-10T10:10:10+00:00",
                    "path": [
                        {
                            "sha256": "fa79e7d0b1f22e945259b123f2169ca3a6c7d3dcf8dc40f123d146aec11a17ff",
                            "action": "sourced",
                            "timestamp": "2023-10-10T10:10:10+00:00",
                            "author": {"category": "plugin", "name": "DummyDownloadPlugin", "version": "1.0"},
                            "file_format": "#TEST/ONLY",
                            "size": 89,
                            "filename": "dummy-download",
                        }
                    ],
                },
                "dequeued": "fa79e7d0b1f22e945259b123f2169ca3a6c7d3dcf8dc40f123d146aec11a17ff.DummyDownloadPlugin.1.0.2023-10-10T10:10:10Z",
            },
        )

        # Plugin registration event.
        plugin_registration = out_events[-3][0]
        with contextlib.suppress(Exception):
            del plugin_registration["entity"]["config"]
            plugin_registration["timestamp"] = "1900-01-01T00:00:00+00:00"
        print("Plugin Registration event:")
        print(plugin_registration)
        self.assertEqual(
            plugin_registration,
            {
                "model_version": 6,
                "kafka_key": "runner-placeholder",
                "timestamp": "1900-01-01T00:00:00+00:00",
                "author": {"category": "plugin", "name": "DummyDownloadPlugin", "version": "1.0"},
                "entity": {
                    "category": "plugin",
                    "name": "DummyDownloadPlugin",
                    "version": "1.0",
                    "description": "Hello world.",
                    "features": [
                        {"name": "feat1", "desc": "", "type": "string"},
                        {
                            "name": "file_extension",
                            "desc": "File extension of the 'content' stream.",
                            "type": "string",
                        },
                        {
                            "name": "file_format",
                            "desc": "Assemblyline file type of the 'content' stream.",
                            "type": "string",
                        },
                        {"name": "filename", "desc": "Name on disk of the 'content' stream.", "type": "filepath"},
                        {"name": "magic", "desc": "File magic found for the 'content' stream.", "type": "string"},
                        {"name": "malformed", "desc": "File is malformed in some way.", "type": "string"},
                        {"name": "mime", "desc": "Mimetype found for the 'content' stream.", "type": "string"},
                        {"name": "per_stream_feat", "desc": "", "type": "string"},
                    ],
                },
            },
        )

    class NotFoundDummyDownloadPlugin(DummyDownloadPlugin):
        def execute_download(self, job: DownloadJob) -> azm.DownloadAction | None:
            pass

        def execute(self, job: Job):
            pass

    def test_download_plugin_basic_file_not_found(self):
        """Attempt to download a file but the plugin doesn't download anything so it just fails."""
        p = self.NotFoundDummyDownloadPlugin(
            config={
                "events_url": self.server + "/download_event",
                "data_url": self.server,
            }
        )
        p.main_loop(1)

        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        # self._del(out_event)
        with contextlib.suppress(Exception):
            out_event["timestamp"] = "1900-01-01T00:00:00+00:00"
        print(out_event)
        self.assertEqual(
            out_event,
            {
                "model_version": 6,
                "kafka_key": "runner-placeholder",
                "timestamp": "1900-01-01T00:00:00+00:00",
                "author": {"category": "plugin", "name": "NotFoundDummyDownloadPlugin", "version": "1.0"},
                "entity": {"hash": "1234"},
                "source": {"name": "source", "timestamp": "1900-01-01T00:00:00+00:00", "path": []},
                "action": "failed-not-found",
            },
        )

    class BadExitCodeDummyDownloadPlugin(DummyDownloadPlugin):
        def execute_download(self, job: DownloadJob) -> azm.DownloadAction | None:
            return azm.DownloadAction.Requested

        def execute(self, job: Job):
            pass

    def test_download_plugin_bad_return_action(self):
        """Attempt to download a file but the plugin doesn't download anything so it just fails."""
        p = self.BadExitCodeDummyDownloadPlugin(
            config={
                "events_url": self.server + "/download_event",
                "data_url": self.server,
            }
        )
        with self.assertRaises(Exception):
            p.main_loop(1)
