from __future__ import annotations

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
from typing import Any, ClassVar

import httpx
import yara_x
from azul_bedrock import models_network as azm

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


class TestBasePluginLive(unittest.TestCase):
    """
    Test cases for base plugin class - cases that talk to the mock server
    """

    mock_server: ClassVar[md.MockDispatcher]
    server: ClassVar[str]  # Endpoint to the mock server, suitable for passing to a plugin's config['server']
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_server = md.MockDispatcher()
        cls.mock_server.start()
        total_sleep = 0
        while not cls.mock_server.is_alive() and total_sleep < 20:
            time.sleep(0.2)  # Wait for server to start
            total_sleep += 1
        cls.server = "http://%s:%s" % (cls.mock_server.host, cls.mock_server.port)
        # Wait for server to be ready to respond
        tries = 0
        while True:
            time.sleep(0.2)
            tries += 1
            try:
                _ = httpx.get(cls.server + "/mock/get_var/fetch_count")
                break  # Exit loop if successful
            except (httpx.TimeoutException, httpx.ConnectError):
                if tries > 20:  # Time out after about 4 seconds
                    raise RuntimeError("Timed out waiting for mock server to be ready")
        # Dummy shared memory queue and ctype
        cls.dummy_queue: multiprocessing.Queue = multiprocessing.Queue()

        cls.editor = md.Editor(cls.server)
        cls.editor.set_stream(
            "b4b389c849d799d9331d5937cde7f0dfd297d76083242366cbef53b498cd6051", 200, b"small content"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()
        cls.dummy_queue.close()

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

    def test_config_secret(self) -> None:
        p = sup.DummyPlugin(
            config={
                "events_url": self.server + "/depth_1",
                "password": "yolo",
                "secret_password": "pog",
                "data_url": self.server,
            }
        )
        net = network.Network(p)
        net.post_registrations()
        self.assertEqual(p.cfg.password, "yolo")
        self.assertEqual(p.cfg.secret_password, "pog")
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        self._del(out_event)

        self.assertIn("password", out_event["entity"]["config"])
        self.assertNotIn("secret_password", out_event["entity"]["config"])

    def test_registration_config_overrides(self):
        class DP(sup.DummyPlugin):
            """Hello world."""

            FEATURES = [
                Feature(name="feat1", desc="", type=azm.FeatureType.String),
                Feature(name="per_stream_feat", desc="", type=azm.FeatureType.String),
            ]

        p = DP(
            config={
                "server": self.server + "/depth_1",
                "name_suffix": "ASuffix",
                "name_remove_prefix": "DiffPrefix",
                "version_suffix": "Beta5",
                "security_override": "Alpha Beta",
                "filter_min_content_size": "100kib",  # ensure that bytesize is encoded ok
            }
        )
        net = network.Network(p)
        net.post_registrations()
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        self._del(out_event)
        print(out_event)
        self.assertEqual(
            out_event,
            {
                "kafka_key": "runner-placeholder",
                "model_version": 5,
                "author": {
                    "category": "plugin",
                    "name": "DP-ASuffix",
                    "version": "1.0-Beta5",
                    "security": "Alpha Beta",
                },
                "entity": {
                    "category": "plugin",
                    "name": "DP-ASuffix",
                    "version": "1.0-Beta5",
                    "security": "Alpha Beta",
                    "description": "Hello world.",
                    "features": [
                        {"name": "example_date", "desc": "Example datetime feature", "type": "datetime"},
                        {"name": "example_int", "desc": "Example int feature", "type": "integer"},
                        {"name": "example_path", "desc": "Example Filepath feature", "type": "filepath"},
                        {"name": "example_raw", "desc": "Example raw bytes feature", "type": "binary"},
                        {"name": "example_string", "desc": "Example string feature", "type": "string"},
                        {"name": "example_unspec", "desc": "Example feature of unspecified type", "type": "string"},
                        {"name": "example_uri", "desc": "Example URI feature", "type": "uri"},
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
                        {
                            "name": "file_format_legacy",
                            "desc": "Azul file type of the 'content' stream.",
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
                        "content_meta_cache_limit": "0",
                        "cur_mem_file_path": '"/sys/fs/cgroup/memory.current"',
                        "deployment_key": '""',
                        "enable_mem_limits": "false",
                        "filter_allow_event_types": "[]",
                        "filter_data_types": "{}",
                        "filter_max_content_size": "209715200",
                        "filter_min_content_size": "102400",
                        "filter_self": "false",
                        "heartbeat_interval": "30",
                        "max_mem_file_path": '"/sys/fs/cgroup/memory.max"',
                        "max_timeouts_before_exit": "100",
                        "max_value_length": "4000",
                        "max_values_per_feature": "1000",
                        "mem_poll_frequency_milliseconds": "1000",
                        "name_remove_prefix": '"DiffPrefix"',
                        "name_suffix": '"ASuffix"',
                        "not_ready_backoff": "5",
                        "plugin_depth_limit": "10",
                        "request_retry_count": "0",
                        "request_timeout": "15",
                        "require_expedite": "true",
                        "require_live": "true",
                        "require_historic": "true",
                        "run_timeout": "600",
                        "security_override": '"Alpha Beta"',
                        "use_multiprocessing_fork": "true",
                        "used_mem_force_exit_frac": "0.9",
                        "used_mem_warning_frac": "0.8",
                        "version_suffix": '"Beta5"',
                        "concurrent_plugin_instances": "1",
                        "watch_path": '""',
                        "watch_type": '""',
                        "watch_wait": "10",
                    },
                },
            },
        )

    def test_registration_inheritance(self):
        p = sup.DummyPluginFeatureInheritance(config={"events_url": self.server + "/depth_1", "data_url": self.server})
        net = network.Network(p)
        net.post_registrations()
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        self._del(out_event)
        print(out_event)
        self.assertEqual(
            out_event,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "author": {"category": "plugin", "name": "DummyPluginFeatureInheritance", "version": "2.0"},
                "entity": {
                    "category": "plugin",
                    "name": "DummyPluginFeatureInheritance",
                    "version": "2.0",
                    "description": "Test class to ensure that features accumulate between template plugins and their descendants.",
                    "features": [
                        {
                            "name": "descendant feature",
                            "desc": "A feature added by the child plugin",
                            "type": "string",
                        },
                        {"name": "example_date", "desc": "Example datetime feature", "type": "datetime"},
                        {"name": "example_int", "desc": "Example int feature", "type": "integer"},
                        {"name": "example_path", "desc": "Example Filepath feature", "type": "filepath"},
                        {"name": "example_raw", "desc": "Example raw bytes feature", "type": "binary"},
                        {"name": "example_string", "desc": "Example string feature", "type": "string"},
                        {"name": "example_unspec", "desc": "Child class redefining feature", "type": "string"},
                        {"name": "example_uri", "desc": "Example URI feature", "type": "uri"},
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
                        {
                            "name": "file_format_legacy",
                            "desc": "Azul file type of the 'content' stream.",
                            "type": "string",
                        },
                        {"name": "filename", "desc": "Name on disk of the 'content' stream.", "type": "filepath"},
                        {"name": "magic", "desc": "File magic found for the 'content' stream.", "type": "string"},
                        {"name": "malformed", "desc": "File is malformed in some way.", "type": "string"},
                        {"name": "mime", "desc": "Mimetype found for the 'content' stream.", "type": "string"},
                    ],
                    "config": {
                        "assume_streams_available": "false",
                        "concurrent_plugin_instances": "1",
                        "content_meta_cache_limit": "0",
                        "cur_mem_file_path": '"/sys/fs/cgroup/memory.current"',
                        "deployment_key": '""',
                        "enable_mem_limits": "false",
                        "filter_allow_event_types": "[]",
                        "filter_data_types": "{}",
                        "filter_max_content_size": "209715200",
                        "filter_min_content_size": "0",
                        "filter_self": "false",
                        "heartbeat_interval": "30",
                        "max_mem_file_path": '"/sys/fs/cgroup/memory.max"',
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
                        "use_multiprocessing_fork": "true",
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

    def test_registration_security_field(self):
        class DummyPluginSecurity(sup.DummyPlugin):
            """Test case plugin class for security field."""

            SECURITY = "security_1 other_security"

        p = DummyPluginSecurity(config={"events_url": self.server + "/depth_1", "data_url": self.server})
        net = network.Network(p)
        net.post_registrations()
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        self._del(out_event)
        print(out_event)
        self.assertEqual(
            out_event,
            {
                "kafka_key": "runner-placeholder",
                "model_version": 5,
                "author": {
                    "category": "plugin",
                    "name": "DummyPluginSecurity",
                    "version": "1.0",
                    "security": "security_1 other_security",
                },
                "entity": {
                    "category": "plugin",
                    "name": "DummyPluginSecurity",
                    "version": "1.0",
                    "security": "security_1 other_security",
                    "description": "Test case plugin class for security field.",
                    "features": [
                        {"name": "example_date", "desc": "Example datetime feature", "type": "datetime"},
                        {"name": "example_int", "desc": "Example int feature", "type": "integer"},
                        {"name": "example_path", "desc": "Example Filepath feature", "type": "filepath"},
                        {"name": "example_raw", "desc": "Example raw bytes feature", "type": "binary"},
                        {"name": "example_string", "desc": "Example string feature", "type": "string"},
                        {"name": "example_unspec", "desc": "Example feature of unspecified type", "type": "string"},
                        {"name": "example_uri", "desc": "Example URI feature", "type": "uri"},
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
                        {
                            "name": "file_format_legacy",
                            "desc": "Azul file type of the 'content' stream.",
                            "type": "string",
                        },
                        {"name": "filename", "desc": "Name on disk of the 'content' stream.", "type": "filepath"},
                        {"name": "magic", "desc": "File magic found for the 'content' stream.", "type": "string"},
                        {"name": "malformed", "desc": "File is malformed in some way.", "type": "string"},
                        {"name": "mime", "desc": "Mimetype found for the 'content' stream.", "type": "string"},
                    ],
                    "config": {
                        "assume_streams_available": "false",
                        "content_meta_cache_limit": "0",
                        "cur_mem_file_path": '"/sys/fs/cgroup/memory.current"',
                        "deployment_key": '""',
                        "enable_mem_limits": "false",
                        "filter_allow_event_types": "[]",
                        "filter_data_types": "{}",
                        "filter_max_content_size": "209715200",
                        "filter_min_content_size": "0",
                        "filter_self": "false",
                        "heartbeat_interval": "30",
                        "max_mem_file_path": '"/sys/fs/cgroup/memory.max"',
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
                        "require_live": "true",
                        "require_historic": "true",
                        "run_timeout": "600",
                        "security_override": '""',
                        "use_multiprocessing_fork": "true",
                        "used_mem_force_exit_frac": "0.9",
                        "used_mem_warning_frac": "0.8",
                        "version_suffix": '""',
                        "watch_path": '""',
                        "watch_type": '""',
                        "watch_wait": "10",
                        "concurrent_plugin_instances": "1",
                    },
                },
            },
        )

    def test_registration_security_dict(self):
        """Test registration using a security dict instead of list."""

        class DummyPluginSecurityDict(sup.DummyPlugin):
            """Test case plugin class for security field dict."""

            SECURITY = "security_1 inc_grp TLP:WHITE"

        p = DummyPluginSecurityDict(
            config={"events_url": self.server + "/depth_1", "deployment_key": "apple", "data_url": self.server}
        )
        net = network.Network(p)
        net.post_registrations()
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        self._del(out_event)
        print(out_event)
        self.assertEqual(
            out_event,
            {
                "kafka_key": "runner-placeholder",
                "model_version": 5,
                "author": {
                    "category": "plugin",
                    "name": "DummyPluginSecurityDict",
                    "version": "1.0",
                    "security": "security_1 inc_grp TLP:WHITE",
                },
                "entity": {
                    "category": "plugin",
                    "name": "DummyPluginSecurityDict",
                    "version": "1.0",
                    "security": "security_1 inc_grp TLP:WHITE",
                    "description": "Test case plugin class for security field dict.",
                    "features": [
                        {"name": "example_date", "desc": "Example datetime feature", "type": "datetime"},
                        {"name": "example_int", "desc": "Example int feature", "type": "integer"},
                        {"name": "example_path", "desc": "Example Filepath feature", "type": "filepath"},
                        {"name": "example_raw", "desc": "Example raw bytes feature", "type": "binary"},
                        {"name": "example_string", "desc": "Example string feature", "type": "string"},
                        {"name": "example_unspec", "desc": "Example feature of unspecified type", "type": "string"},
                        {"name": "example_uri", "desc": "Example URI feature", "type": "uri"},
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
                        {
                            "name": "file_format_legacy",
                            "desc": "Azul file type of the 'content' stream.",
                            "type": "string",
                        },
                        {"name": "filename", "desc": "Name on disk of the 'content' stream.", "type": "filepath"},
                        {"name": "magic", "desc": "File magic found for the 'content' stream.", "type": "string"},
                        {"name": "malformed", "desc": "File is malformed in some way.", "type": "string"},
                        {"name": "mime", "desc": "Mimetype found for the 'content' stream.", "type": "string"},
                    ],
                    "config": {
                        "assume_streams_available": "false",
                        "content_meta_cache_limit": "0",
                        "cur_mem_file_path": '"/sys/fs/cgroup/memory.current"',
                        "deployment_key": '"apple"',
                        "enable_mem_limits": "false",
                        "filter_allow_event_types": "[]",
                        "filter_data_types": "{}",
                        "filter_max_content_size": "209715200",
                        "filter_min_content_size": "0",
                        "filter_self": "false",
                        "heartbeat_interval": "30",
                        "max_mem_file_path": '"/sys/fs/cgroup/memory.max"',
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
                        "require_live": "true",
                        "require_historic": "true",
                        "run_timeout": "600",
                        "security_override": '""',
                        "use_multiprocessing_fork": "true",
                        "used_mem_force_exit_frac": "0.9",
                        "used_mem_warning_frac": "0.8",
                        "version_suffix": '""',
                        "watch_path": '""',
                        "watch_type": '""',
                        "watch_wait": "10",
                        "concurrent_plugin_instances": "1",
                    },
                },
            },
        )

    def test_fetch_job(self):
        net = network.Network(
            sup.DummyPlugin(
                config={"events_url": self.server + "/depth_1", "deployment_key": "apple", "data_url": self.server}
            )
        )
        r = net.fetch_job()
        self.assertTrue(isinstance(r, azm.BinaryEvent))

        net = network.Network(
            sup.DummyPlugin(
                config={"events_url": self.server + "/null", "deployment_key": "apple", "data_url": self.server}
            )
        )
        r = net.fetch_job()
        expected = dump(
            azm.BinaryEvent(
                model_version=azm.CURRENT_MODEL_VERSION,
                kafka_key="test-dummy",
                dequeued="test-dummy-dequeued",
                action=azm.BinaryAction.Enriched,
                timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                source=azm.Source(
                    name="source",
                    path=[],
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                ),
                author=azm.Author(name="TestServer", category="blah"),
                entity=azm.BinaryEvent.Entity(sha256="1234", datastreams=[], features=[]),
            )
        )
        self.assertEqual(dump(r), expected)
        # check query parameters
        print(self.get_last_req_params())
        self.assertEqual(
            {
                "name": ["DummyPlugin"],
                "version": ["1.0"],
                "deployment_key": ["apple"],
                "r-expedite": ["true"],
                "r-live": ["true"],
                "r-historic": ["true"],
                "r-under-content-size": ["209715200"],
                "count": ["1"],
                "deadline": ["10"],
            },
            self.get_last_req_params(),
        )

        # test with more advanced query parameters
        class AzulPluginCustom(sup.DummyPlugin):
            pass

        net = network.Network(
            AzulPluginCustom(
                config={
                    "server": self.server + "/null",
                    "require_historic": False,
                    "filter_max_content_size": "1000kb",
                    "filter_min_content_size": "1kb",
                    "filter_allow_event_types": ["extracted"],
                    "filter_self": True,
                    "filter_data_types": {"*": ["executable/windows/dos"], "blob": ["archive/gzip", "archive/bzip2"]},
                    "deployment_key": "apple",
                }
            )
        )
        r = net.fetch_job()
        print(self.get_last_req_params())
        self.assertEqual(
            {
                "name": ["Custom"],
                "version": ["1.0"],
                "count": ["1"],
                "deployment_key": ["apple"],
                "deadline": ["10"],
                "r-expedite": ["true"],
                "r-live": ["true"],
                "r-under-content-size": ["1000000"],
                "r-over-content-size": ["1000"],
                "r-action": ["extracted"],
                "d-self": ["true"],
                "r-streams": ["*,executable/windows/dos", "blob,archive/gzip,archive/bzip2"],
            },
            self.get_last_req_params(),
        )

    def test_post_and_get_new_content(self):
        # Tests the method where each item is posted individually and the server response with a ContentEntry
        p = sup.DummyPlugin(config={"events_url": self.server + "/generic", "data_url": self.server})
        sample_hashed = {
            "6a0e91ed5a21963490c9677934e6b99ae901f8eda7de3c1d9d52fa23d40bafb2": (["content"], b"fake data 1"),
            "f7d60cd635b71dc219222122a7bb5b14da90eb6b67c5bcd18234b7309582d2c2": (["content"], b"fake data 2"),
        }
        net = network.Network(p)
        net._post_data(
            "source",
            sample_hashed,
        )
        self.assertEqual(
            net._cached_file_data,
            {
                (
                    "source",
                    "content",
                    "6a0e91ed5a21963490c9677934e6b99ae901f8eda7de3c1d9d52fa23d40bafb2",
                ): azm.Datastream(
                    identify_version=1,
                    label=azm.DataLabel.CONTENT,
                    size=11,
                    magic="#TESTONLY",
                    mime="#TESTONLY",
                    file_format_legacy="#TESTONLY",
                    file_format="#TEST/ONLY",
                    file_extension="tonly",
                    md5="69ea00f9c3104cced2e97c4662ba8b5c",
                    sha1="ac3ee7a4206a3b98702d20c925ecb6242ab64383",
                    sha256="6a0e91ed5a21963490c9677934e6b99ae901f8eda7de3c1d9d52fa23d40bafb2",
                    sha512="3f7e39f41ae6204a51b7a0103a1cccc9afb36be98f3afe2faef3ab3da0f0bb29"
                    "7d6a635c1cc06641d1ae40cee12f71671c85c492ee5b07d28ce2e1ed1bb7f193",
                ),
                (
                    "source",
                    "content",
                    "f7d60cd635b71dc219222122a7bb5b14da90eb6b67c5bcd18234b7309582d2c2",
                ): azm.Datastream(
                    identify_version=1,
                    label=azm.DataLabel.CONTENT,
                    size=11,
                    magic="#TESTONLY",
                    mime="#TESTONLY",
                    file_format_legacy="#TESTONLY",
                    file_format="#TEST/ONLY",
                    file_extension="tonly",
                    md5="4af0be381e877439bc3b676519dea88f",
                    sha1="2ee5a439436eb9d5ee8c77eab994c854acbb7fab",
                    sha256="f7d60cd635b71dc219222122a7bb5b14da90eb6b67c5bcd18234b7309582d2c2",
                    sha512="85f2dc9828f5c359d22fa307405a5744f454c88a91f87b1d73ce97164869d9b5"
                    "18712abc36df09bba633a4f9143cb36c58b28a692c305a98a46ca38b23786a30",
                ),
            },
        )
        # Tests the newly posted content can be retrieved via a GET request.
        r = httpx.get(
            "%s/api/v3/stream/source/content/6a0e91ed5a21963490c9677934e6b99ae901f8eda7de3c1d9d52fa23d40bafb2"
            % self.server
        )
        r.raise_for_status()
        self.assertEqual(r.content, b"fake data 1")
        r = httpx.get(
            "%s/api/v3/stream/source/content/f7d60cd635b71dc219222122a7bb5b14da90eb6b67c5bcd18234b7309582d2c2"
            % self.server
        )
        r.raise_for_status()
        self.assertEqual(r.content, b"fake data 2")

    def test_ack_job(self):
        class DP(sup.DummyPlugin):
            FEATURES = [
                Feature(name="feat1", desc="", type=azm.FeatureType.String),
                Feature(name="per_stream_feat", desc="", type=azm.FeatureType.String),
            ]
            SECURITY = "test_security_level"  # should be passed in result

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        _ = loop.run_once(local.gen_event(entity))
        dt = datetime.datetime.now(datetime.timezone.utc)
        event = loop._network.fetch_job()
        text_stream = b"Text data stream"
        text_id = DATA_HASH(text_stream).hexdigest()
        loop._network.ack_job(
            src=event,
            result=JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="1",
                        features={"feat1": {FeatureValue("This is a string value", label="label1")}},
                        info={
                            "list_field": [1, 2, 3],
                            "dict_field": {"foo": "bar", "int": 5},
                        },
                        data=[{"label": "text", "hash": text_id, "language": "english"}],
                    ),
                    Event(
                        sha256="1",
                        parent_sha256="b4b389c849d799d9331d5937cde7f0dfd297d76083242366cbef53b498cd6051",
                        features={"per_stream_feat": {FeatureValue('Set on "small content" stream')}},
                    ),
                ],
                data={
                    text_id: text_stream,
                },
                runtime=3,
                date_start=dt,
                date_end=dt + datetime.timedelta(seconds=3),
            ),
        )
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        print(out_event)
        # Override timestamps to make comparison easier
        out_event["timestamp"] = dt.isoformat()
        self.assertEqual(len(out_event["entity"]["results"]), 2, "There should be two events generated.")
        out_event["entity"]["results"][0]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][0]["source"]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][0]["source"]["path"][-1]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][1]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][1]["source"]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][1]["source"]["path"][-1]["timestamp"] = dt.isoformat()

        prepath = event.source.path
        event2 = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="runner-placeholder",
            action=azm.BinaryAction.Enriched,
            timestamp=dt,
            source=azm.Source(
                name="source",
                timestamp=dt,
                path=prepath
                + [
                    azm.PathNode(
                        action=azm.BinaryAction.Enriched,
                        timestamp=dt,
                        author=azm.Author(
                            category="plugin",
                            name="DP",
                            version="1.0",
                            security="test_security_level",
                        ),
                        sha256="1234",
                    )
                ],
            ),
            author=azm.Author(
                category="plugin",
                name="DP",
                version="1.0",
                security="test_security_level",
            ),
            entity=azm.BinaryEvent.Entity(
                sha256="1234",
                features=[
                    azm.FeatureValue(name="per_stream_feat", type="string", value='Set on "small content" stream'),
                ],
            ),
        )

        event1 = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="runner-placeholder",
            action=azm.BinaryAction.Augmented,
            timestamp=dt,
            source=azm.Source(
                name="source",
                timestamp=dt,
                path=prepath
                + [
                    azm.PathNode(
                        action=azm.BinaryAction.Augmented,
                        timestamp=dt,
                        author=azm.Author(
                            category="plugin",
                            name="DP",
                            version="1.0",
                            security="test_security_level",
                        ),
                        sha256="1234",
                    )
                ],
            ),
            author=azm.Author(
                category="plugin",
                name="DP",
                version="1.0",
                security="test_security_level",
            ),
            entity=azm.BinaryEvent.Entity(
                sha256="1234",
                datastreams=[
                    azm.Datastream(
                        identify_version=1,
                        file_format_legacy="ft",
                        file_format="#TEST/ONLY",
                        label=azm.DataLabel.CONTENT,
                        md5="5",
                        magic="mm",
                        mime="mt",
                        sha1="1",
                        sha256="b4b389c849d799d9331d5937cde7f0dfd297d76083242366cbef53b498cd6051",
                        sha512="512",
                        size=13,
                    ),
                    # alt-stream content won't be included
                    azm.Datastream(
                        identify_version=1,
                        label=azm.DataLabel.TEXT,
                        size=16,
                        language="english",
                        file_format_legacy="#TESTONLY",
                        file_format="#TEST/ONLY",
                        file_extension="tonly",
                        mime="#TESTONLY",
                        magic="#TESTONLY",
                        md5="2fae0d09ab333884009fa812b6c4d097",
                        sha1="8805b0694e0128f2c7c22adb92d5515957161e87",
                        sha256="903e0722b8ebdfc5547af8ca1a9c8a76d0a7b067a7b1a561dfff13593e76d5a8",
                        sha512="718e7bc44d2e1396646fe014812c9af52d1b59c2858efbcc2d3bc096eda27289"
                        "2eac2b7bb2ac37bb9fc2f49b54b68f969f03d73e66e03a67c0a1a69ced48bce7",
                    ),
                ],
                features=[
                    azm.FeatureValue(name="feat1", type="string", value="This is a string value", label="label1"),
                ],
                info={
                    "list_field": [1, 2, 3],
                    "dict_field": {"foo": "bar", "int": 5},
                },
            ),
        )
        event.entity.features = []
        event.entity.datastreams = []
        event.entity.info = {}
        expected = dump(
            azm.StatusEvent(
                model_version=azm.CURRENT_MODEL_VERSION,
                kafka_key="runner-placeholder",
                timestamp=dt,
                author=azm.Author(
                    category="plugin",
                    name="DP",
                    version="1.0",
                    security="test_security_level",
                ),
                entity=azm.StatusEvent.Entity(
                    input=event,
                    status=State.Label.COMPLETED,
                    results=[event1, event2],
                ),
            )
        )
        expected["entity"]["runtime"] = out_event["entity"]["runtime"]

        self.assertEqual(out_event["entity"]["results"][0], dump(event1))
        self.assertEqual(out_event["entity"]["results"][1], dump(event2))
        self.assertEqual(out_event, expected)

    def test_ack_job_security_string_passthrough(self):
        class DP(sup.DummyPlugin):
            FEATURES = [
                Feature(name="feat1", desc="", type=azm.FeatureType.String),
                Feature(name="per_stream_feat", desc="", type=azm.FeatureType.String),
            ]
            SECURITY = "test_security_level"  # should be passed in result

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        _ = loop.run_once(local.gen_event(entity))[None]
        dt = datetime.datetime.now(datetime.timezone.utc)
        event = loop._network.fetch_job()
        event.author.security = "FISH"
        event.source.security = "FISH"
        # event.source.path[0].author.security = "FISH"
        text_stream = b"Text data stream"
        text_id = DATA_HASH(text_stream).hexdigest()
        loop._network.ack_job(
            src=event,
            result=JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="1",
                        parent_sha256="b4b389c849d799d9331d5937cde7f0dfd297d76083242366cbef53b498cd6051",
                        features={"per_stream_feat": {FeatureValue('Set on "small content" stream')}},
                    ),
                ],
                runtime=3,
                date_start=dt,
                date_end=dt + datetime.timedelta(seconds=3),
            ),
        )
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        # Override timestamps to make comparison easier
        out_event["timestamp"] = dt.isoformat()
        self.assertEqual(len(out_event["entity"]["results"]), 1)
        out_event["entity"]["results"][0]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][0]["source"]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][0]["source"]["path"][-1]["timestamp"] = dt.isoformat()

        event.entity.features = []
        event.entity.datastreams = []
        event.entity.info = {}
        prepath = event.source.path
        event1 = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="runner-placeholder",
            action=azm.BinaryAction.Enriched,
            timestamp=dt,
            source=azm.Source(
                name="source",
                timestamp=dt,
                path=prepath
                + [
                    azm.PathNode(
                        action=azm.BinaryAction.Enriched,
                        timestamp=dt,
                        author=azm.Author(
                            category="plugin",
                            name="DP",
                            version="1.0",
                            security="test_security_level",
                        ),
                        sha256="1234",
                    )
                ],
                security="FISH",
            ),
            author=azm.Author(
                category="plugin",
                name="DP",
                version="1.0",
                security="test_security_level",
            ),
            entity=azm.BinaryEvent.Entity(
                sha256="1234",
                features=[
                    azm.FeatureValue(name="per_stream_feat", type="string", value='Set on "small content" stream'),
                ],
            ),
        )
        expected = dump(
            azm.StatusEvent(
                model_version=azm.CURRENT_MODEL_VERSION,
                kafka_key="runner-placeholder",
                timestamp=dt,
                author=azm.Author(
                    category="plugin",
                    name="DP",
                    version="1.0",
                    security="test_security_level",
                ),
                entity=azm.StatusEvent.Entity(
                    input=event,
                    status=State.Label.COMPLETED,
                    results=[event1],
                ),
            )
        )
        expected["entity"]["runtime"] = out_event["entity"]["runtime"]

        self.assertEqual(out_event["entity"]["results"][0], dump(event1))
        self.assertEqual(out_event, expected)

    def test_ack_job_children_and_security_dict(self):
        """Test the API post output of an event with children and grandchildren, using a dict value for SECURITY"""

        class DP(sup.DummyPlugin):
            FEATURES = [
                Feature(name="sample_feature", desc="", type=azm.FeatureType.String),
            ]
            SECURITY = "ex_value TLP:WHITE"  # should be passed in result

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        _ = loop.run_once(local.gen_event(entity))[None]
        dt = datetime.datetime.now(datetime.timezone.utc)
        event = loop._network.fetch_job()
        child_data = b"foo"
        child_id = DATA_HASH(child_data).hexdigest()
        gc_data = b"bar"
        gc_id = DATA_HASH(gc_data).hexdigest()
        # Should not need to test 'per-stream' result output in combination with child/grandchildren,
        #  as the per-stream ack is tested above and the ack loop runs the same code over both (per-stream/normal)
        loop._network.ack_job(
            src=event,
            result=JobResult(
                **{
                    "state": State(State.Label.COMPLETED),
                    "events": [
                        {
                            "parent": {"sha256": ""},
                            "sha256": child_id,
                            "relationship": {"r": "v"},
                            "data": [{"label": "content", "hash": child_id}],
                        },
                        {
                            "parent": {
                                "parent": {"sha256": ""},
                                "sha256": child_id,
                                "relationship": {"r": "v"},
                            },
                            "sha256": gc_id,
                            "features": {"sample_feature": {FeatureValue("completed ok")}},
                            "relationship": {"r": "v2"},
                            "data": [{"label": "content", "language": "foo", "hash": gc_id}],
                        },
                    ],
                    "data": {child_id: child_data, gc_id: gc_data},
                    "runtime": 21,
                    "date_start": dt,
                    "date_end": dt + datetime.timedelta(seconds=21),
                }
            ),
        )
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        # Override timestamps to make comparison easier
        out_event["timestamp"] = dt.isoformat()
        self.assertEqual(len(out_event["entity"]["results"]), 2, "There should be two events generated.")
        out_event["entity"]["results"][0]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][0]["source"]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][0]["source"]["path"][-1]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][1]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][1]["source"]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][1]["source"]["path"][-1]["timestamp"] = dt.isoformat()
        out_event["entity"]["results"][1]["source"]["path"][-2]["timestamp"] = dt.isoformat()

        event.entity.features = []
        event.entity.datastreams = []
        event.entity.info = {}
        prepath = event.source.path
        expected = dump(
            azm.StatusEvent(
                model_version=azm.CURRENT_MODEL_VERSION,
                kafka_key="runner-placeholder",
                timestamp=dt,
                author=azm.Author(
                    category="plugin",
                    name="DP",
                    version="1.0",
                    security="ex_value TLP:WHITE",
                ),
                entity=azm.StatusEvent.Entity(
                    input=event,
                    status=State.Label.COMPLETED,
                    results=[
                        azm.BinaryEvent(
                            model_version=azm.CURRENT_MODEL_VERSION,
                            kafka_key="runner-placeholder",
                            action=azm.BinaryAction.Extracted,
                            timestamp=dt,
                            source=azm.Source(
                                name="source",
                                timestamp=dt,
                                path=prepath
                                + [
                                    azm.PathNode(
                                        action=azm.BinaryAction.Extracted,
                                        relationship={"r": "v"},
                                        timestamp=dt,
                                        author=azm.Author(
                                            category="plugin",
                                            name="DP",
                                            version="1.0",
                                            security="ex_value TLP:WHITE",
                                        ),
                                        sha256=child_id,
                                        size=3,
                                        file_format_legacy="#TESTONLY",
                                        file_format="#TEST/ONLY",
                                    )
                                ],
                            ),
                            author=azm.Author(
                                category="plugin",
                                name="DP",
                                version="1.0",
                                security="ex_value TLP:WHITE",
                            ),
                            entity=azm.BinaryEvent.Entity(
                                sha256=child_id,
                                datastreams=[
                                    azm.Datastream(
                                        identify_version=1,
                                        file_format_legacy="#TESTONLY",
                                        file_format="#TEST/ONLY",
                                        file_extension="tonly",
                                        label=azm.DataLabel.CONTENT,
                                        md5="acbd18db4cc2f85cedef654fccc4a4d8",
                                        magic="#TESTONLY",
                                        mime="#TESTONLY",
                                        sha1="0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                                        sha256="2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
                                        sha512="f7fbba6e0636f890e56fbbf3283e524c6fa3204ae298382d624741d0dc663832"
                                        "6e282c41be5e4254d8820772c5518a2c5a8c0c7f7eda19594a7eb539453e1ed7",
                                        size=3,
                                    ),
                                ],
                                features=[
                                    azm.FeatureValue(name="file_format", type="string", value="#TEST/ONLY"),
                                    azm.FeatureValue(name="file_format_legacy", type="string", value="#TESTONLY"),
                                    azm.FeatureValue(name="file_extension", type="string", value="tonly"),
                                    azm.FeatureValue(name="magic", type="string", value="#TESTONLY"),
                                    azm.FeatureValue(name="mime", type="string", value="#TESTONLY"),
                                ],
                                file_format_legacy="#TESTONLY",
                                file_format="#TEST/ONLY",
                                file_extension="tonly",
                                mime="#TESTONLY",
                                magic="#TESTONLY",
                                md5="acbd18db4cc2f85cedef654fccc4a4d8",
                                sha1="0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33",
                                sha512="f7fbba6e0636f890e56fbbf3283e524c6fa3204ae298382d624741d0dc663832"
                                "6e282c41be5e4254d8820772c5518a2c5a8c0c7f7eda19594a7eb539453e1ed7",
                                size=3,
                            ),
                        ),
                        azm.BinaryEvent(
                            model_version=azm.CURRENT_MODEL_VERSION,
                            kafka_key="runner-placeholder",
                            action=azm.BinaryAction.Extracted,
                            timestamp=dt,
                            source=azm.Source(
                                name="source",
                                timestamp=dt,
                                path=prepath
                                + [
                                    azm.PathNode(
                                        action=azm.BinaryAction.Extracted,
                                        relationship={"r": "v"},
                                        timestamp=dt,
                                        author=azm.Author(
                                            category="plugin",
                                            name="DP",
                                            version="1.0",
                                            security="ex_value TLP:WHITE",
                                        ),
                                        sha256=child_id,
                                        size=3,
                                        file_format_legacy="#TESTONLY",
                                        file_format="#TEST/ONLY",
                                    ),
                                    azm.PathNode(
                                        action=azm.BinaryAction.Extracted,
                                        relationship={"r": "v2"},
                                        timestamp=dt,
                                        author=azm.Author(
                                            category="plugin",
                                            name="DP",
                                            version="1.0",
                                            security="ex_value TLP:WHITE",
                                        ),
                                        sha256=gc_id,
                                        size=3,
                                        file_format_legacy="#TESTONLY",
                                        file_format="#TEST/ONLY",
                                        language="foo",
                                    ),
                                ],
                            ),
                            author=azm.Author(
                                category="plugin",
                                name="DP",
                                version="1.0",
                                security="ex_value TLP:WHITE",
                            ),
                            entity=azm.BinaryEvent.Entity(
                                sha256=gc_id,
                                datastreams=[
                                    azm.Datastream(
                                        identify_version=1,
                                        file_format_legacy="#TESTONLY",
                                        file_format="#TEST/ONLY",
                                        file_extension="tonly",
                                        label=azm.DataLabel.CONTENT,
                                        language="foo",
                                        magic="#TESTONLY",
                                        mime="#TESTONLY",
                                        md5="37b51d194a7513e45b56f6524f2d51f2",
                                        sha1="62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                                        sha256="fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9",
                                        sha512="d82c4eb5261cb9c8aa9855edd67d1bd10482f41529858d925094d173fa662aa9"
                                        "1ff39bc5b188615273484021dfb16fd8284cf684ccf0fc795be3aa2fc1e6c181",
                                        size=3,
                                    ),
                                ],
                                features=[
                                    azm.FeatureValue(name="file_format", type="string", value="#TEST/ONLY"),
                                    azm.FeatureValue(name="file_format_legacy", type="string", value="#TESTONLY"),
                                    azm.FeatureValue(name="file_extension", type="string", value="tonly"),
                                    azm.FeatureValue(name="magic", type="string", value="#TESTONLY"),
                                    azm.FeatureValue(name="mime", type="string", value="#TESTONLY"),
                                    azm.FeatureValue(name="sample_feature", type="string", value="completed ok"),
                                ],
                                file_format_legacy="#TESTONLY",
                                file_format="#TEST/ONLY",
                                file_extension="tonly",
                                md5="37b51d194a7513e45b56f6524f2d51f2",
                                sha1="62cdb7020ff920e5aa642c3d4066950dd1f01f4d",
                                sha512="d82c4eb5261cb9c8aa9855edd67d1bd10482f41529858d925094d173fa662aa9"
                                "1ff39bc5b188615273484021dfb16fd8284cf684ccf0fc795be3aa2fc1e6c181",
                                size=3,
                                mime="#TESTONLY",
                                magic="#TESTONLY",
                            ),
                        ),
                    ],
                ),
            )
        )
        expected["entity"]["runtime"] = out_event["entity"]["runtime"]
        self.assertEqual(out_event, expected)

    def test_ack_job_new_stream_only(self):
        """Ensure if plugin only produces a new stream, it is still treated as an output result."""

        class DP(sup.DummyPlugin):
            SECURITY = "test_security_level"  # should be passed in result

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        _ = loop.run_once(local.gen_event(entity))[None]
        dt = datetime.datetime.now(datetime.timezone.utc)
        event = loop._network.fetch_job()
        text_stream = b"Text data stream"
        text_id = DATA_HASH(text_stream).hexdigest()
        loop._network.ack_job(
            src=event,
            result=JobResult(
                **{
                    "state": State(State.Label.COMPLETED),
                    "data": {
                        text_id: text_stream,
                    },
                    "events": [
                        dict(
                            sha256="5",
                            data=[
                                dict(
                                    label=azm.DataLabel.TEXT,
                                    language="english",
                                    hash=text_id,
                                )
                            ],
                        )
                    ],
                    "runtime": 3,
                    "date_start": dt,
                    "date_end": dt + datetime.timedelta(seconds=3),
                }
            ),
        )
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        self.assertTrue(out_event["entity"].get("results"))

    def test_ack_job_status_id(self):
        class DP(sup.DummyPlugin):
            SECURITY = "test_security_level"  # should be passed in result

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_path", "data_url": self.server})
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        _ = loop.run_once(local.gen_event(entity))[None]
        dt = datetime.datetime.now(datetime.timezone.utc)
        event = loop._network.fetch_job()
        loop._network.ack_job(
            src=event,
            result=JobResult(
                **{
                    "state": State(State.Label.COMPLETED),
                    "data": {},
                    "events": [],
                    "runtime": 3,
                    "date_start": dt,
                    "date_end": dt + datetime.timedelta(seconds=3),
                }
            ),
        )
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_event: dict = r.json()[0]
        # Ensure that this hash matches what is tested against in the dispatcher
        self.assertEqual("generated-dummy-dequeued-id", out_event["entity"]["input"]["dequeued"])

    def test_runloop_posting(self):
        """Directly test monitoring process managing loop"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                time.sleep(0.1)

        logging_queue = multiprocessing.Queue()

        p = Process(
            target=monitor._start_loop_coordinator,
            kwargs=dict(
                plugin=DP,
                config=settings.parse_config(DP, {"events_url": self.server + "/null_sleep", "data_url": self.server}),
                job_limit=None,
                log_level=logging.INFO,
                queue=self.dummy_queue,
                logging_queue=logging_queue,
            ),
        )
        p.start()
        while not p.is_alive():
            time.sleep(0.2)
        # Wait to make sure plugin enters main loop and executes a couple of times
        time.sleep(1.0)
        # Request a graceful shutdown
        p.terminate()
        self._wait_for_is_alive_to_be_false(p)
        # Expect that the plugin has terminated.
        self.assertFalse(p.is_alive())
        # Check that the last post was an ack job
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        j = r.json()[0]
        expected = dump(
            azm.StatusEvent(
                model_version=azm.CURRENT_MODEL_VERSION,
                kafka_key="runner-placeholder",
                timestamp=datetime.datetime.now(tz=datetime.timezone.utc),  # Will be replaced with correct value below
                author=azm.Author(
                    category="plugin",
                    name="DP",
                    version="1.0",
                ),
                entity=azm.StatusEvent.Entity(
                    input=azm.BinaryEvent(
                        model_version=azm.CURRENT_MODEL_VERSION,
                        kafka_key="test-dummy",
                        dequeued="test-dummy-dequeued",
                        action=azm.BinaryAction.Enriched,
                        timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                        # hash="dbb2466f01612c0e632e3ca8bfc1de02",
                        source=azm.Source(
                            name="source",
                            path=[],
                            timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                        ),
                        author=azm.Author(name="TestServer", category="blah"),
                        entity=azm.BinaryEvent.Entity(sha256="1234", datastreams=[], features=[]),
                    ),
                    status=State.Label.COMPLETED_EMPTY,
                    results=[],  # null_sleep does not produce features or children
                ),
            )
        )
        print(json.dumps(j))
        # Copy the values that vary each run
        expected["timestamp"] = j["timestamp"]
        expected["entity"]["runtime"] = j.get("entity", {}).get("runtime")
        self.assertEqual(j, expected)

    def test_runloop_with_heartbeat(self):
        class DP(sup.DummyPlugin):
            def execute(self, job: Job):
                time.sleep(2.5)
                self.add_feature_values("example_string", "test")

        # Clean any requests
        r = httpx.get("%s/mock/get_var/all_requests" % self.server)

        # Send heartbeat ever 1 sec
        loop = monitor.Monitor(
            DP, {"events_url": self.server + "/test_data", "heartbeat_interval": 1, "data_url": self.server}
        )
        test_data = b"This is test data that should be fetched by the runner"
        loop._network._post_data(
            "source",
            {DATA_HASH(test_data).hexdigest(): (["content"], test_data)},
        )
        loop.run_loop(1)

        # Check that heartbeats have been sent
        r = httpx.get("%s/mock/get_var/all_requests" % self.server)

        r.raise_for_status()
        out_evts: list[azm.StatusEvent] = [
            azm.StatusEvent(**ev[0]) for ev in r.json() if ev[0].get("entity", {}).get("input")
        ]
        print("---------START-------------")
        print(out_evts)
        print("---------START-------------")
        # Allow for either 1 or 2 heartbeats
        self.assertLessEqual(2, len(out_evts))
        self.assertGreaterEqual(3, len(out_evts))

        for out_evt in out_evts:
            self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        for out_evt in out_evts[:-1]:
            self.assertEqual(State.Label.HEARTBEAT, out_evt.entity.status)

        self.assertEqual(out_evts[-1].entity.status, State.Label.COMPLETED)
        self.assertEqual(
            out_evts[-1].entity.results[0].entity.features,
            [azm.FeatureValue(name="example_string", type="string", value="test")],
        )

    def test_runloop_with_content(self):
        class DP(sup.DummyPlugin):
            def execute(self, job: Job):
                if job.get_all_data():
                    pass
                else:
                    # Alternative way of failing test to avoid serialising self
                    raise coordinator.CriticalError("get_all_data is not truthy!")
                lengths = []
                for ds in job.get_all_data():
                    if not isinstance(ds, StorageProxyFile):
                        # Alternative way of failing test to avoid serialising self
                        raise coordinator.CriticalError("Provided data is not a StorageProxyFile!")
                    lengths.append(len(ds.read()))
                self.add_feature_values("example_int", lengths)

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        test_data = b"This is test data that should be fetched by the runner"
        loop._network._post_data(
            "source",
            {DATA_HASH(test_data).hexdigest(): (["content"], test_data)},
        )
        loop.run_loop(1)
        # Check that the ack contains the correct features
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
        self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        self.assertEqual(
            out_evt.entity.status,
            State.Label.COMPLETED,
            "Unexpected result: %s:%s (%s)" % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
        )
        self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
        self.assertEqual(
            out_evt.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_int", type="integer", value="13"),
                azm.FeatureValue(name="example_int", type="integer", value="54"),
            ],
        )

    def test_runloop_multiple_iterations(self):
        """Ensure the plugin can run in a loop and post more than one response."""

        class DP(sup.DummyPlugin):
            def execute(self, job: Job):
                if job.get_all_data():
                    pass
                else:
                    # Alternative way of failing test to avoid serialising self
                    raise coordinator.CriticalError("get_all_data is not truthy!")
                lengths = []
                for ds in job.get_all_data():
                    if not isinstance(ds, StorageProxyFile):
                        # Alternative way of failing test to avoid serialising self
                        raise coordinator.CriticalError("Provided data is not a StorageProxyFile!")
                    lengths.append(len(ds.read()))
                self.add_feature_values("example_int", lengths)

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        loop.time_to_wait_between_checks = 0.5
        test_data = b"This is test data that should be fetched by the runner"
        loop._network._post_data(
            "source",
            {DATA_HASH(test_data).hexdigest(): (["content"], test_data)},
        )
        request_count = httpx.get("%s/mock/get_var/fetch_count" % self.server)
        inital_request_count = int(request_count.content.decode())
        loop.run_loop(5)
        request_count = httpx.get("%s/mock/get_var/fetch_count" % self.server)
        requests_in_interval = int(request_count.content.decode()) - inital_request_count
        # Ensure there was at least five requests.
        self.assertGreaterEqual(requests_in_interval, 5)

        # Check that the ack contains the correct features
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
        self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        self.assertEqual(
            out_evt.entity.status,
            State.Label.COMPLETED,
            "Unexpected result: %s:%s (%s)" % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
        )
        self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
        self.assertEqual(
            out_evt.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_int", type="integer", value="13"),
                azm.FeatureValue(name="example_int", type="integer", value="54"),
            ],
        )

    def test_runloop_depth_limit_reached(self):
        # No specific authors to filter but depth has reached config
        loop = monitor.Monitor(
            sup.DummyPlugin, {"events_url": self.server + "/depth_2", "plugin_depth_limit": 2, "data_url": self.server}
        )
        loop.run_loop(1)
        # Check that the ack contains the correct features
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
        self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        self.assertEqual(
            out_evt.entity.status,
            State.Label.OPT_OUT,
            "Unexpected result: %s:%s (%s)" % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
        )

        # No specific authors to filter but depth has reached config
        loop = monitor.Monitor(
            sup.DummyPlugin,
            {
                "server": self.server + "/depth_dupe",
                "plugin_depth_limit": 2,
            },
        )
        loop.run_loop(1)
        # Check that the ack contains the correct features
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
        self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        self.assertEqual(
            out_evt.entity.status,
            State.Label.OPT_OUT,
            "Unexpected result: %s:%s (%s)" % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
        )

    def test_runloop_with_watch(self):
        with tempfile.TemporaryDirectory() as filepath:
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("1")

            class DP(sup.DummyPlugin):
                def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
                    super().__init__(config)
                    with open(os.path.join(filepath, "tmp.txt"), "r") as f:
                        self.retval = f.read()

                def execute(self, job):
                    self.add_feature_values("example_string", self.retval)

            loop = monitor.Monitor(
                DP,
                {
                    "events_url": self.server + "/test_data",
                    "watch_path": filepath,
                    "watch_wait": 0,
                    "data_url": self.server,
                },
            )
            # test_data = b"This is test data that should be fetched by the runner"
            # loop._network._post_data({DATA_HASH(test_data).hexdigest(): test_data})
            loop.run_loop(1)
            # Check that the ack contains the correct features
            r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
            r.raise_for_status()
            out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
            self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
            self.assertEqual(
                out_evt.entity.status,
                State.Label.COMPLETED,
                "Unexpected result: %s:%s (%s)"
                % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
            )
            self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
            self.assertEqual(
                out_evt.entity.results[0].entity.features,
                [
                    azm.FeatureValue(name="example_string", type="string", value="1"),
                ],
            )

            # second run with new values
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("2")

            # idk how long it takes to pick up changes
            time.sleep(1)

            loop.run_loop(1)
            # Check that the ack contains the correct features
            r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
            r.raise_for_status()
            out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
            self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
            self.assertEqual(
                out_evt.entity.status,
                State.Label.COMPLETED,
                "Unexpected result: %s:%s (%s)"
                % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
            )
            self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
            # must have updated value here
            self.assertEqual(
                out_evt.entity.results[0].entity.features,
                [
                    azm.FeatureValue(name="example_string", type="string", value="2"),
                ],
            )
        # kill the watchdog threads to reduce chance of weird stuff during testing
        del loop

    def test_runloop_with_no_watch(self):
        with tempfile.TemporaryDirectory() as filepath:
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("1")

            class DP(sup.DummyPlugin):
                def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
                    super().__init__(config)
                    with open(os.path.join(filepath, "tmp.txt"), "r") as f:
                        self.retval = f.read()

                def execute(self, job):
                    self.add_feature_values("example_string", self.retval)

            # Have to run with coordinator, because monitor would restart the coordinator between iterations.
            loop = coordinator.Coordinator(
                DP,
                settings.parse_config(DP, {"events_url": self.server + "/test_data", "data_url": self.server}),
            )
            # test_data = b"This is test data that should be fetched by the runner"
            # loop._network._post_data({DATA_HASH(test_data).hexdigest(): test_data})
            with self.assertRaises(SystemExit) as e:
                loop.run_loop(queue=self.dummy_queue, job_limit=1)

            # Verify clean exit
            self.assertEqual(e.exception.code, 0)

            # Check that the ack contains the correct features
            r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
            r.raise_for_status()
            out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
            self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
            self.assertEqual(
                out_evt.entity.status,
                State.Label.COMPLETED,
                "Unexpected result: %s:%s (%s)"
                % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
            )
            self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
            self.assertEqual(
                out_evt.entity.results[0].entity.features,
                [
                    azm.FeatureValue(name="example_string", type="string", value="1"),
                ],
            )

            # second run with new values
            with open(os.path.join(filepath, "tmp.txt"), "w") as f:
                f.write("2")

            # idk how long it takes to pick up changes
            time.sleep(1)
            with self.assertRaises(SystemExit) as e:
                loop.run_loop(queue=self.dummy_queue, job_limit=1)

            # Verify clean exit
            self.assertEqual(e.exception.code, 0)

            # Check that the ack contains the correct features
            r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
            r.raise_for_status()
            out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
            self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
            self.assertEqual(
                out_evt.entity.status,
                State.Label.COMPLETED,
                "Unexpected result: %s:%s (%s)"
                % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
            )
            self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
            print("_______________________________________A_______________________________________")
            print(out_evt.entity.results[0].entity.features)
            print("_______________________________________B_______________________________________")
            # must not have updated value here
            self.assertEqual(
                out_evt.entity.results[0].entity.features,
                [
                    azm.FeatureValue(name="example_string", type="string", value="1"),
                ],
            )

    def test_runloop_utf8(self):
        # ensure that utf8 is encoded and decoded correctly
        # different length when encoded
        text = "中英文均可"
        self.assertEqual(len(text), 5)
        self.assertEqual(len(text.encode()), 15)

        class DP(sup.DummyPlugin):
            def execute(self, job: Job):
                lengths = []
                if job.get_all_data():
                    pass
                else:
                    # A way of crashing the test without assertions
                    raise coordinator.CriticalError("get_all_data is not truthy!")
                for ds in job.get_all_data():
                    if not isinstance(ds, StorageProxyFile):
                        raise coordinator.CriticalError("Provided data is not a StorageProxyFile!")
                    lengths.append(len(ds.read()))
                self.add_feature_values("example_string", text)

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        test_data = b"This is test data that should be fetched by the runner"
        loop._network._post_data(
            "source",
            {DATA_HASH(test_data).hexdigest(): (["content"], test_data)},
        )
        loop.run_loop(1)
        # Check that the ack contains the correct features
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
        self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        self.assertEqual(
            out_evt.entity.status,
            State.Label.COMPLETED,
            "Unexpected result: %s:%s (%s)" % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
        )
        self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
        self.assertEqual(
            out_evt.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_string", type="string", value=text),
            ],
        )

    def test_runloop_surrogate_characters(self):
        # ensure that utf8 is encoded and decoded correctly
        # different length when encoded
        text = "\ud83d\ude4f"
        expected_fv_text = "\\ud83d\\ude4f"
        self.assertEqual(len(text), 2)
        self.assertEqual(len(text.encode(errors="backslashreplace")), 12)

        class DP(sup.DummyPlugin):
            def execute(self, job: Job):
                lengths = []
                if job.get_all_data():
                    pass
                else:
                    # A way of crashing the test without assertions
                    raise coordinator.CriticalError("get_all_data is not truthy!")
                for ds in job.get_all_data():
                    if not isinstance(ds, StorageProxyFile):
                        raise coordinator.CriticalError("Provided data is not a StorageProxyFile!")
                    lengths.append(len(ds.read()))
                self.add_feature_values("example_string", text)

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        test_data = b"This is test data that should be fetched by the runner"
        loop._network._post_data(
            "source",
            {DATA_HASH(test_data).hexdigest(): (["content"], test_data)},
        )
        loop.run_loop(1)
        # Check that the ack contains the correct features
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
        self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        self.assertEqual(
            out_evt.entity.status,
            State.Label.COMPLETED,
            "Unexpected result: %s:%s (%s)" % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
        )
        self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
        self.assertEqual(
            out_evt.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_string", type="string", value=expected_fv_text),
            ],
        )

    def test_too_many_events(self):
        class DP(sup.DummyPlugin):
            pass

        loop = monitor.Monitor(DP, {"events_url": self.server + "/too_many", "data_url": self.server})
        self.assertRaisesRegex(ValueError, "2 events fetched by dispatcher, only 1 allowed", loop._network.fetch_job)

    def test_use_spawn_mp_plugin(self):
        """Test using spawn multiprocessing setting works with yarax."""
        in_event = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="test-dummy",
            dequeued="test-dummy-dequeued",
            action=azm.BinaryAction.Sourced,
            timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            source=azm.Source(
                name="source",
                path=[],
                timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            ),
            author=azm.Author(name="TestServer", category="blah"),
            entity=azm.BinaryEvent.Entity(sha256="1234", datastreams=[], features=[]),
        )

        loop = monitor.Monitor(YaraxDP, {})
        final_result = loop.run_once(in_event)
        print(final_result)
        self.assertEqual(
            final_result[None].events[0].features,
            {"example_string": [FV(value="test")]},
        )

    def test_run_multiple_plugin_instances_concurrently(self):
        """Run 5 instances of a plugin concurrently and ensure they get at most 2 jobs.

        There should be 5 instances running a total of 5 jobs.
        The plugins should all run at least once and may run twice.
        But must not run more than twice. (or else there might just be one instance of the plugin.)
        The reason for this is, the plugin instances will all process their first job at roughly the same time.
        They may not ALL be done by the time coordinator checks if the job limit is reached.
        Once the job limit is reached all plugins will have one at least once.
        """

        class DP(sup.DummyPlugin):
            SETTINGS = add_settings(
                request_retry_count=0,
                use_multiprocessing_fork=True,
                concurrent_plugin_instances=5,
            )  # Don't retry failed requests when testing

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.received_jobs = 0

            def execute(self, job: Job):
                if self.received_jobs and self.received_jobs == 1:
                    # Sleep before completing second job.
                    time.sleep(1)
                elif self.received_jobs and self.received_jobs > 1:
                    # Ran more than 2 jobs where only a max of 2 is allowed.
                    raise coordinator.CriticalError("Instance received more than one job and this is unexpected!")

                self.received_jobs += 1
                if job.get_all_data():
                    pass
                else:
                    # Alternative way of failing test to avoid serialising self
                    raise coordinator.CriticalError("get_all_data is not truthy!")
                lengths = []
                for ds in job.get_all_data():
                    if not isinstance(ds, StorageProxyFile):
                        # Alternative way of failing test to avoid serialising self
                        raise coordinator.CriticalError("Provided data is not a StorageProxyFile!")
                    lengths.append(len(ds.read()))
                self.add_feature_values("example_int", lengths)

        loop = monitor.Monitor(DP, {"events_url": self.server + "/test_data", "data_url": self.server})
        loop.time_to_wait_between_checks = 0.5
        test_data = b"This is test data that should be fetched by the runner"
        loop._network._post_data(
            "source",
            {DATA_HASH(test_data).hexdigest(): (["content"], test_data)},
        )
        request_count = httpx.get("%s/mock/get_var/fetch_count" % self.server)
        inital_request_count = int(request_count.content.decode())
        loop.run_loop(5)
        request_count = httpx.get("%s/mock/get_var/fetch_count" % self.server)
        requests_in_interval = int(request_count.content.decode()) - inital_request_count
        # Ensure there was at least five requests.
        self.assertGreaterEqual(requests_in_interval, 5)

        # Check that the ack contains the correct features
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        out_evt: azm.StatusEvent = azm.StatusEvent(**r.json()[0])
        self.assertIsInstance(out_evt.entity, azm.StatusEvent.Entity)
        self.assertEqual(
            out_evt.entity.status,
            State.Label.COMPLETED,
            "Unexpected result: %s:%s (%s)" % (out_evt.entity.status, out_evt.entity.error, out_evt.entity.message),
        )
        self.assertEqual(len(out_evt.entity.results), 1, "Expected one result entity; got:\n%s" % out_evt)
        self.assertEqual(
            out_evt.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_int", type="integer", value="13"),
                azm.FeatureValue(name="example_int", type="integer", value="54"),
            ],
        )


class YaraxDP(sup.DummyPlugin):
    """Create a plugin that uses yara-x to ensure it works in azul-runner.

    This is because yara-x fails to function when using certain multiprocessing options
    and always fails when using threading.
    """

    SETTINGS = add_settings(use_multiprocessing_fork=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules = yara_x.compile(
            """
        rule test { 
            strings: 
            $a = "fetched" 
            condition: 
            $a
        }"""
        )

    def execute(self, job: Job):
        result = self.rules.scan(b"This is test data that should be fetched by the runner")
        for rule_hit in result.matching_rules:
            self.add_feature_values("example_string", rule_hit.identifier)
