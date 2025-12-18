from __future__ import annotations

import datetime
import hashlib
import time
import unittest
from typing import ClassVar
from unittest import mock

import httpx
import pendulum
from azul_bedrock import models_network as azm

from azul_runner import FeatureType, Job, network_transform, storage
from azul_runner.plugin import Multiplugin
from azul_runner.pusher import Pusher
from tests import plugin_support as sup

from . import mock_dispatcher as md
from . import plugin_support as sup


class TestPusherLive(unittest.TestCase):
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
        while not cls.mock_server.is_alive():
            time.sleep(0.2)  # Wait for server to start
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()

    def get_last_req_body(self) -> dict:
        r = httpx.get("%s/mock/get_var/last_request_body" % self.server)
        r.raise_for_status()
        return r.json()

    @mock.patch("pendulum.now", lambda *args: pendulum.parse("2023-10-10T10:10:10Z"))
    def test_push_once_sourced(self):
        class DummyPluginWithFeats(sup.DummyPlugin):
            """Plugin that adds features based on input"""

            _IS_USING_PUSHER = True

            def execute(self, job: Job):
                data = job.get_data()
                self.add_feature_values("example_string", "dummy string")
                if data:
                    self.add_feature_values("example_raw", data.read())

        test_source = "source"
        # First attempt at using pusher
        pusher = Pusher(
            DummyPluginWithFeats,
            {
                "events_url": self.server + "/generic",
                "data_url": self.server,
            },
        )

        pusher.push_once_sourced(
            content=b"test-file1",
            security="OFFICIAL",
            source_label=test_source,
            references={"ref1": "val1"},
            filename="random-file-name1",
        )
        result = azm.StatusEvent.model_validate(self.get_last_req_body()[0])
        print(result.entity.results[0].entity.features)
        self.assertCountEqual(
            result.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_raw", type=azm.FeatureType.Binary, value="dGVzdC1maWxlMQ=="),
                azm.FeatureValue(name="example_string", type=azm.FeatureType.String, value="dummy string"),
                azm.FeatureValue(name="file_format", type=azm.FeatureType.String, value="#TEST/ONLY"),
                azm.FeatureValue(name="file_format_legacy", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="file_extension", type=azm.FeatureType.String, value="tonly"),
                azm.FeatureValue(name="magic", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="mime", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value="random-file-name1"),
            ],
        )

        # More data to verify that state isn't going to break anything.
        pusher.push_once_sourced(
            content=b"random-different-data",
            security="OFFICIAL",
            source_label=test_source,
            references={"ref1": "val1"},
            filename="random-diff-data",
        )
        status = self.get_last_req_body()[0]
        result = azm.StatusEvent.model_validate(status)
        print(result.entity.results[0].entity.features)
        self.assertCountEqual(
            result.entity.results[0].entity.features,
            [
                azm.FeatureValue(
                    name="example_raw", type=azm.FeatureType.Binary, value="cmFuZG9tLWRpZmZlcmVudC1kYXRh"
                ),
                azm.FeatureValue(name="example_string", type=azm.FeatureType.String, value="dummy string"),
                azm.FeatureValue(name="file_format", type=azm.FeatureType.String, value="#TEST/ONLY"),
                azm.FeatureValue(name="file_format_legacy", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="file_extension", type=azm.FeatureType.String, value="tonly"),
                azm.FeatureValue(name="magic", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="mime", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value="random-diff-data"),
            ],
        )

        status["timestamp"] = "2023-10-10T10:10:10+00:00"
        status["entity"]["results"][0]["timestamp"] = "2023-10-10T10:10:10+00:00"
        # Should only be a single event.
        self.assertEqual(len(result.entity.results[0].source.path), 1)
        print(f"actual\n\n{status}\n\nend actual")
        self.assertEqual(
            status,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "timestamp": "2023-10-10T10:10:10+00:00",
                "author": {"category": "plugin", "name": "DummyPluginWithFeats", "version": "1.0"},
                "entity": {
                    "input": {
                        "model_version": 5,
                        "kafka_key": "DummyPluginWithFeats-placeholder",
                        "timestamp": "2023-10-10T10:10:10+00:00",
                        "author": {
                            "category": "plugin",
                            "name": "DummyPluginWithFeats",
                            "version": "1.0",
                            "security": "OFFICIAL",
                        },
                        "entity": {
                            "sha256": "2f094deac91260d23d3a9a9c0c9e59448342f346fa2927f0ec10af1d72cf55d0",
                            "sha512": "ee543f098404b991962a3072ff0038b1eaf0fbd6d957fdbe6074f9c57d88da9f3546925aedeff4a1917a6cf104aa72e7201069c1202a6e77d0ead590dbb053ea",
                            "sha1": "b8f17c427f04225815f1e2fa1050e3d24c96e550",
                            "md5": "a548bfb3e683426097217764db88e07a",
                            "size": 21,
                            "file_format_legacy": "#TESTONLY",
                            "file_format": "#TEST/ONLY",
                            "file_extension": "tonly",
                            "mime": "#TESTONLY",
                            "magic": "#TESTONLY",
                        },
                        "action": "sourced",
                        "source": {
                            "name": "source",
                            "path": [
                                {
                                    "sha256": "2f094deac91260d23d3a9a9c0c9e59448342f346fa2927f0ec10af1d72cf55d0",
                                    "action": "sourced",
                                    "timestamp": "2023-10-10T10:10:10+00:00",
                                    "author": {
                                        "category": "plugin",
                                        "name": "DummyPluginWithFeats",
                                        "version": "1.0",
                                        "security": "OFFICIAL",
                                    },
                                    "file_format_legacy": "#TESTONLY",
                                    "file_format": "#TEST/ONLY",
                                    "size": 21,
                                    "filename": "random-diff-data",
                                }
                            ],
                            "timestamp": "2023-10-10T10:10:10+00:00",
                            "security": "OFFICIAL",
                            "references": {"ref1": "val1"},
                        },
                        "dequeued": "2f094deac91260d23d3a9a9c0c9e59448342f346fa2927f0ec10af1d72cf55d0.DummyPluginWithFeats.1.0.2023-10-10T10:10:10Z",
                    },
                    "status": "completed",
                    "runtime": 0.0,
                    "results": [
                        {
                            "model_version": 5,
                            "kafka_key": "runner-placeholder",
                            "timestamp": "2023-10-10T10:10:10+00:00",
                            "author": {"category": "plugin", "name": "DummyPluginWithFeats", "version": "1.0"},
                            "entity": {
                                "sha256": "2f094deac91260d23d3a9a9c0c9e59448342f346fa2927f0ec10af1d72cf55d0",
                                "sha512": "ee543f098404b991962a3072ff0038b1eaf0fbd6d957fdbe6074f9c57d88da9f3546925aedeff4a1917a6cf104aa72e7201069c1202a6e77d0ead590dbb053ea",
                                "sha1": "b8f17c427f04225815f1e2fa1050e3d24c96e550",
                                "md5": "a548bfb3e683426097217764db88e07a",
                                "size": 21,
                                "file_format_legacy": "#TESTONLY",
                                "file_format": "#TEST/ONLY",
                                "file_extension": "tonly",
                                "mime": "#TESTONLY",
                                "magic": "#TESTONLY",
                                "features": [
                                    {"name": "example_raw", "type": "binary", "value": "cmFuZG9tLWRpZmZlcmVudC1kYXRh"},
                                    {"name": "example_string", "type": "string", "value": "dummy string"},
                                    {"name": "file_format", "type": "string", "value": "#TEST/ONLY"},
                                    {"name": "file_format_legacy", "type": "string", "value": "#TESTONLY"},
                                    {"name": "file_extension", "type": "string", "value": "tonly"},
                                    {"name": "magic", "type": "string", "value": "#TESTONLY"},
                                    {"name": "mime", "type": "string", "value": "#TESTONLY"},
                                    {"name": "filename", "type": "filepath", "value": "random-diff-data"},
                                ],
                                "datastreams": [
                                    {
                                        "sha256": "2f094deac91260d23d3a9a9c0c9e59448342f346fa2927f0ec10af1d72cf55d0",
                                        "sha512": "ee543f098404b991962a3072ff0038b1eaf0fbd6d957fdbe6074f9c57d88da9f3546925aedeff4a1917a6cf104aa72e7201069c1202a6e77d0ead590dbb053ea",
                                        "sha1": "b8f17c427f04225815f1e2fa1050e3d24c96e550",
                                        "md5": "a548bfb3e683426097217764db88e07a",
                                        "size": 21,
                                        "file_format_legacy": "#TESTONLY",
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
                                "path": [
                                    {
                                        "sha256": "2f094deac91260d23d3a9a9c0c9e59448342f346fa2927f0ec10af1d72cf55d0",
                                        "action": "sourced",
                                        "timestamp": "2023-10-10T10:10:10+00:00",
                                        "author": {
                                            "category": "plugin",
                                            "name": "DummyPluginWithFeats",
                                            "version": "1.0",
                                            "security": "OFFICIAL",
                                        },
                                        "file_format_legacy": "#TESTONLY",
                                        "file_format": "#TEST/ONLY",
                                        "size": 21,
                                        "filename": "random-diff-data",
                                    }
                                ],
                                "timestamp": "2023-10-10T10:10:10+00:00",
                                "security": "OFFICIAL",
                                "references": {"ref1": "val1"},
                            },
                        }
                    ],
                },
            },
        )

    @mock.patch("pendulum.now", lambda *args: pendulum.parse("2023-10-10T10:10:10Z"))
    def test_push_once_mapped(self):
        class DummyPluginWithFeats(sup.DummyPlugin):
            """Plugin that adds features based on input"""

            _IS_USING_PUSHER = True

            def execute(self, job: Job):
                data = job.get_data()
                self.add_feature_values("example_string", "dummy string")
                if data:
                    self.add_feature_values("example_raw", data.read())

        test_source = "source"
        # First attempt at using pusher
        pusher = Pusher(
            DummyPluginWithFeats,
            {
                "events_url": self.server + "/generic",
                "data_url": self.server,
            },
        )

        source_info = pusher.generate_base_mapped_source(
            source_label=test_source, references={"ref1": "val1"}, security="OFFICIAL"
        )
        pusher.push_once_mapped(
            source_file_info=azm.FileInfo(
                sha256="dummy-sha256",
                sha512="dummy-sha512",
                size=1,
                file_format="text/plain",
                file_format_legacy="Text",
            ),
            source_info=source_info,
            filename="random-file-name1",
            security="OFFICIAL",
        )
        result = azm.StatusEvent.model_validate(self.get_last_req_body()[0])
        print(result.entity.results[0].entity.features)
        self.assertCountEqual(
            result.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_string", type=azm.FeatureType.String, value="dummy string"),
                azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value="random-file-name1"),
                azm.FeatureValue(name="file_format", type=azm.FeatureType.String, value="text/plain"),
                azm.FeatureValue(name="file_format_legacy", type=azm.FeatureType.String, value="Text"),
            ],
        )

        # Again generating a mapped event instead of a sourced event by providing no data
        # Results in less features in the sourced event and one less from the plugin itself.
        source_info = pusher.generate_base_mapped_source(
            source_label=test_source, references={"ref1": "val1"}, security="OFFICIAL"
        )
        source_info.path.append(
            azm.PathNode(
                sha256="random-other-sha256",
                action=azm.BinaryAction.Sourced,
                timestamp=pendulum.now(),
                author=azm.Author(
                    name="user1",
                    category="User",
                    security="OFFICIAL",
                ),
                file_format="text/plain",
                file_format_legacy="TEXT",
            )
        )
        pusher.push_once_mapped(
            source_file_info=azm.FileInfo(
                sha256="dummy-sha256",
                sha512="dummy-sha512",
                size=1,
                file_format="text/plain",
                file_format_legacy="Text",
            ),
            source_info=source_info,
            security="OFFICIAL",
            filename="random-file-name1",
            relationship={"Asssemblyline_reingest": "Reingested from Assemblyline"},
        )
        status = self.get_last_req_body()[0]
        result = azm.StatusEvent.model_validate(status)
        print(result.entity.results[0].entity.features)
        self.assertCountEqual(
            result.entity.results[0].entity.features,
            [
                azm.FeatureValue(name="example_string", type=azm.FeatureType.String, value="dummy string"),
                azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value="random-file-name1"),
                azm.FeatureValue(name="file_format", type=azm.FeatureType.String, value="text/plain"),
                azm.FeatureValue(name="file_format_legacy", type=azm.FeatureType.String, value="Text"),
            ],
        )

        status["timestamp"] = "2023-10-10T10:10:10+00:00"
        status["entity"]["results"][0]["timestamp"] = "2023-10-10T10:10:10+00:00"
        # Should be attached to provided source info so should have depth of two.
        self.assertEqual(len(result.entity.results[0].source.path), 2)

        print(f"actual\n\n{status}\n\nend actual")
        self.assertEqual(
            status,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "timestamp": "2023-10-10T10:10:10+00:00",
                "author": {"category": "plugin", "name": "DummyPluginWithFeats", "version": "1.0"},
                "entity": {
                    "input": {
                        "model_version": 5,
                        "kafka_key": "DummyPluginWithFeats-placeholder",
                        "timestamp": "2023-10-10T10:10:10+00:00",
                        "author": {
                            "category": "plugin",
                            "name": "DummyPluginWithFeats",
                            "version": "1.0",
                            "security": "OFFICIAL",
                        },
                        "entity": {
                            "sha256": "dummy-sha256",
                            "sha512": "dummy-sha512",
                            "size": 1,
                            "file_format_legacy": "Text",
                            "file_format": "text/plain",
                        },
                        "action": "mapped",
                        "source": {
                            "security": "OFFICIAL",
                            "name": "source",
                            "timestamp": "2023-10-10T10:10:10+00:00",
                            "references": {"ref1": "val1"},
                            "path": [
                                {
                                    "sha256": "random-other-sha256",
                                    "action": "sourced",
                                    "timestamp": "2023-10-10T10:10:10+00:00",
                                    "author": {"category": "User", "name": "user1", "security": "OFFICIAL"},
                                    "file_format_legacy": "TEXT",
                                    "file_format": "text/plain",
                                },
                                {
                                    "sha256": "dummy-sha256",
                                    "action": "mapped",
                                    "timestamp": "2023-10-10T10:10:10+00:00",
                                    "author": {
                                        "category": "plugin",
                                        "name": "DummyPluginWithFeats",
                                        "version": "1.0",
                                        "security": "OFFICIAL",
                                    },
                                    "relationship": {"Asssemblyline_reingest": "Reingested from Assemblyline"},
                                    "file_format_legacy": "Text",
                                    "file_format": "text/plain",
                                    "size": 1,
                                    "filename": "random-file-name1",
                                },
                            ],
                        },
                        "dequeued": "dummy-sha256.DummyPluginWithFeats.1.0.2023-10-10T10:10:10Z",
                    },
                    "status": "completed",
                    "runtime": 0.0,
                    "results": [
                        {
                            "model_version": 5,
                            "kafka_key": "runner-placeholder",
                            "timestamp": "2023-10-10T10:10:10+00:00",
                            "author": {"category": "plugin", "name": "DummyPluginWithFeats", "version": "1.0"},
                            "entity": {
                                "sha256": "dummy-sha256",
                                "sha512": "dummy-sha512",
                                "size": 1,
                                "file_format_legacy": "Text",
                                "file_format": "text/plain",
                                "features": [
                                    {"name": "example_string", "type": "string", "value": "dummy string"},
                                    {"name": "file_format", "type": "string", "value": "text/plain"},
                                    {"name": "file_format_legacy", "type": "string", "value": "Text"},
                                    {"name": "filename", "type": "filepath", "value": "random-file-name1"},
                                ],
                            },
                            "action": "mapped",
                            "source": {
                                "security": "OFFICIAL",
                                "name": "source",
                                "timestamp": "2023-10-10T10:10:10+00:00",
                                "references": {"ref1": "val1"},
                                "path": [
                                    {
                                        "sha256": "random-other-sha256",
                                        "action": "sourced",
                                        "timestamp": "2023-10-10T10:10:10+00:00",
                                        "author": {"category": "User", "name": "user1", "security": "OFFICIAL"},
                                        "file_format_legacy": "TEXT",
                                        "file_format": "text/plain",
                                    },
                                    {
                                        "sha256": "dummy-sha256",
                                        "action": "mapped",
                                        "timestamp": "2023-10-10T10:10:10+00:00",
                                        "author": {
                                            "category": "plugin",
                                            "name": "DummyPluginWithFeats",
                                            "version": "1.0",
                                            "security": "OFFICIAL",
                                        },
                                        "relationship": {"Asssemblyline_reingest": "Reingested from Assemblyline"},
                                        "file_format_legacy": "Text",
                                        "file_format": "text/plain",
                                        "size": 1,
                                        "filename": "random-file-name1",
                                    },
                                ],
                            },
                        }
                    ],
                },
            },
        )

    @mock.patch("pendulum.now", lambda *args: pendulum.parse("2023-10-10T10:10:10Z"))
    def test_push_once_alt_stream(self):
        class DummyPluginWithAlt(sup.DummyPlugin):
            """Plugin that adds features based on input"""

            _IS_USING_PUSHER = True

            def execute(self, job: Job):
                data = job.get_data()
                data_alt = job.get_data(azm.DataLabel.TEST)
                self.add_feature_values("example_raw", [data.read(), data_alt.read()])

        test_source = "source"
        pusher = Pusher(
            DummyPluginWithAlt,
            {
                "events_url": self.server + "/generic",
                "data_url": self.server,
            },
        )

        binary_data = b"test-file1"
        alt_content = b"alt-stream-data-is-here-and-available."
        with storage.StorageProxyFile(
            source="alt-source",
            label=azm.DataLabel.TEST,
            hash=hashlib.sha256(alt_content).hexdigest(),
            init_data=alt_content,
            file_info=azm.Datastream(label=azm.DataLabel.TEST, file_format_legacy="Text", file_format="text/plain"),
            allow_unbounded_read=True,
        ) as meta_spf:
            with storage.StorageProxyFile(
                source=test_source,
                label=azm.DataLabel.CONTENT,
                hash=hashlib.sha256(binary_data).hexdigest(),
                init_data=binary_data,
                file_info=azm.Datastream(
                    label=azm.DataLabel.CONTENT, file_format_legacy="Text", file_format="text/plain"
                ),
                allow_unbounded_read=True,
            ) as spf:
                pusher.push_once_sourced(
                    security="OFFICIAL",
                    source_label=test_source,
                    references={"ref1": "val1"},
                    filename="random-file-name1",
                    content=binary_data,
                    local=[spf, meta_spf],
                )
        result = azm.StatusEvent.model_validate(self.get_last_req_body()[0])
        print(result.entity.results[0].entity.features)
        self.assertCountEqual(
            result.entity.results[0].entity.features,
            [
                azm.FeatureValue(
                    name="example_raw",
                    type=azm.FeatureType.Binary,
                    value="YWx0LXN0cmVhbS1kYXRhLWlzLWhlcmUtYW5kLWF2YWlsYWJsZS4=",
                ),
                azm.FeatureValue(name="example_raw", type=azm.FeatureType.Binary, value="dGVzdC1maWxlMQ=="),
                azm.FeatureValue(name="file_format", type=azm.FeatureType.String, value="#TEST/ONLY"),
                azm.FeatureValue(name="file_format_legacy", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="file_extension", type=azm.FeatureType.String, value="tonly"),
                azm.FeatureValue(name="magic", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="mime", type=azm.FeatureType.String, value="#TESTONLY"),
                azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value="random-file-name1"),
            ],
        )
