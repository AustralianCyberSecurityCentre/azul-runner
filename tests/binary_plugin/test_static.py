from __future__ import annotations

import datetime
import io
import unittest

from azul_bedrock import models_network as azm

from azul_runner import (
    BinaryPlugin,
    Feature,
    FeatureType,
    State,
    StorageProxyFile,
    add_settings,
    local,
    plugin_executor,
)
from azul_runner.models import Job
from azul_runner.storage import ProxyFileNotFoundError


class PluginNoData(BinaryPlugin):
    """
    Test class that will register successfully but has no execute method.
    """

    FEATURES = [
        Feature("sample_feature", "An output feature for the test plugin", type=FeatureType.String),
    ]
    SETTINGS = add_settings(assume_streams_available=False)

    def execute(self, job: Job) -> dict:
        digest = job.event.entity.sha256
        self.add_feature_values("sample_feature", digest)


class TestBasePluginStatic(unittest.TestCase):
    """Static test cases for binary template."""

    def test_generate_filter_string(self):
        """Test that the size limit filter string is generated correctly"""

        class P(BinaryPlugin):
            SETTINGS = add_settings(
                filter_data_types={"example": ["content-type-1", "content-type-2"]},
                filter_max_content_size=500,
            )

        p = P()
        self.assertEqual(p.cfg.filter_max_content_size, 500)
        self.assertEqual(p.cfg.filter_data_types, {"example": ["content-type-1", "content-type-2"]})

    def test_generate_filter_string_cmdline_config(self):
        """Test that the size limit filter string is generated correctly when passed via cmdline."""

        class P(BinaryPlugin):
            SETTINGS = add_settings(
                filter_data_types={"example": ["content-type-1", "content-type-2"]},
                filter_max_content_size=100,
            )

        p = P({"filter_max_content_size": "500"})
        self.assertEqual(p.cfg.filter_max_content_size, 500)
        self.assertEqual(p.cfg.filter_data_types, {"example": ["content-type-1", "content-type-2"]})

    def test_handle_no_data(self):
        """Test that the template correctly handles binaries with data unavailable"""

        # Test with a plugin that expects data
        class Test404(BinaryPlugin):
            def execute(self, job) -> dict:
                data = job.get_data()
                print(data)
                # Mimic the exception raised by StorageProxyFile on a 404
                raise ProxyFileNotFoundError(2, "Got 404 requesting (nothing)")

        p = Test404()

        streams = [
            StorageProxyFile(
                source="local",
                label=azm.DataLabel.CONTENT,
                hash="test_data",
                init_data=b"test",
                file_info=local.gen_api_content(
                    io.BytesIO(b"test"),
                ),
            )
        ]

        event = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="runner-placeholder",
            action=azm.BinaryAction.Enriched,
            timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            source=azm.Source(
                name="source",
                path=[
                    azm.PathNode(
                        author=azm.Author(name="foo"),
                        action=azm.BinaryAction.Enriched,
                        timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                        sha256="test_entity",
                    ),
                ],
                timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            ),
            author=azm.Author(name="TestServer", category="blah"),
            entity=azm.BinaryEvent.Entity(sha256="test_entity", datastreams=[], features=[], info={}),
        )
        job = Job(event=event)
        job.load_streams(local=streams)
        res = plugin_executor.run_plugin_with_job(p, job, None)
        self.assertEqual(
            res.state,
            State(State.Label.OPT_OUT, message="Plugin requires binary content but none exists"),
        )
        # Test that it still raises if the plugin doesn't require data
        # Test404.REQUIRES_DATA = False
        p.cfg.assume_streams_available = False
        self.assertRaisesRegex(ProxyFileNotFoundError, "Got 404 requesting", p.execute, job)

        # validate you can test plugins that don't need data
        self.assertEqual(PluginNoData().cfg.assume_streams_available, False)
        p = PluginNoData()
        entity = azm.BinaryEvent.Entity(sha256="test_entity", datastreams=[], features=[], info={})
        event = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="runner-placeholder",
            action=azm.BinaryAction.Enriched,
            timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            source=azm.Source(
                name="source",
                path=[
                    azm.PathNode(
                        author=azm.Author(name="foo"),
                        action=azm.BinaryAction.Enriched,
                        timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                        sha256="test_entity",
                    ),
                ],
                timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            ),
            author=azm.Author(name="TestServer", category="blah"),
            entity=entity,
        )
        job = Job(event=event)
        res = plugin_executor.run_plugin_with_job(p, job, None)
        self.assertEqual(res.state, State())

        # Test that it treats regular FileNotFound correctly
        # Test with a plugin that expects data
        class TestFNF(BinaryPlugin):
            def execute(self, entity):
                data = job.get_data()
                with open(".nonexistent_file_3284767538574862745", "rb"):
                    raise AssertionError("Didn't get a 'file not found' ...")
                return

        self.assertEqual(TestFNF().cfg.assume_streams_available, True)
        p = TestFNF()
        self.assertRaises(FileNotFoundError, p.execute, job)
