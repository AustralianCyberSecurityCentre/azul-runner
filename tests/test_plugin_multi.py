import datetime
import hashlib
import io
import json
import unittest
from typing import Any, Dict
from unittest import mock

from azul_bedrock import models_network as azm
from pydantic import TypeAdapter

from azul_runner import (
    FV,
    Event,
    EventData,
    EventParent,
    Feature,
    FeatureType,
    FeatureValue,
    Job,
    JobResult,
    Plugin,
    State,
    StorageProxyFile,
    TestPlugin,
    add_settings,
    coordinator,
    local,
    network,
    network_transform,
    settings,
)
from azul_runner.plugin import Multiplugin


class DP(Plugin):
    SETTINGS = add_settings(
        request_retry_count=0,
        server="https://localhost:9876",
        filter_data_types={},
    )  # Don't retry failed requests when testing

    # leave security property unset
    # SECURITY = None
    VERSION = "1.0"
    MULTI_STREAM_AWARE = True
    FEATURES = [
        Feature("apples", "Example feature", type=FeatureType.Integer),
    ]

    def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.register_multiplugin(
            "decode_rot", "1.0.0", self.decode_rot, description="Description decode_rot", security="LEMON RUSH"
        )
        self.register_multiplugin("decode_xor", "21-02-2025", self.decode_xor, description="Description decode_xor")
        self.register_multiplugin("decode_nor", "v1-beta", self.decode_nor)

    def execute(self, job) -> dict:
        return

    def decode_rot(self, job) -> dict:
        return

    def decode_xor(self, job) -> dict:
        return

    def decode_nor(self, job) -> dict:
        return


class TestBasic(unittest.TestCase):
    def test_basic(self):
        class DummyPlugin(DP):
            """Dummy plugin description."""

            def decode_rot(self, job):
                self.add_feature_values("apples", [FV(1)])

            def decode_xor(self, job):
                raise Exception("invalid")

            def decode_nor(self, job):
                c = self._add_child("1", {"a": 1})
                c.add_data("content", {}, b"5")
                self.add_info({"k": "1234567890"})

        loop = coordinator.Coordinator(DummyPlugin, settings.parse_config(DummyPlugin, {}))

        # test registration events
        self.assertEqual(len(network_transform.get_registrations(loop._plugin)), 4)
        self.assertEqual(network_transform.get_registrations(loop._plugin)[0].author.name, "DummyPlugin")
        self.assertEqual(network_transform.get_registrations(loop._plugin)[1].author.name, "DummyPlugin-decode_rot")
        self.assertEqual(network_transform.get_registrations(loop._plugin)[2].author.name, "DummyPlugin-decode_xor")
        self.assertEqual(network_transform.get_registrations(loop._plugin)[3].author.name, "DummyPlugin-decode_nor")

        self.assertEqual(
            network_transform.get_registrations(loop._plugin)[0].entity.description, "Dummy plugin description."
        )
        self.assertEqual(
            network_transform.get_registrations(loop._plugin)[1].entity.description, "Description decode_rot"
        )
        self.assertEqual(
            network_transform.get_registrations(loop._plugin)[2].entity.description, "Description decode_xor"
        )
        self.assertEqual(
            network_transform.get_registrations(loop._plugin)[3].entity.description, "Dummy plugin description."
        )

        self.assertSetEqual(
            set(loop._plugin.get_registered_multiplugins()), set(["decode_rot", "decode_xor", "decode_nor", None])
        )
        self.assertEqual(
            loop._plugin.get_multiplugin("decode_rot"),
            Multiplugin(
                name="decode_rot",
                version="1.0.0",
                callback=loop._plugin.decode_rot,
                description="Description decode_rot",
                security="LEMON RUSH",
            ),
        )

        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        results = loop.run_once(local.gen_event(entity))
        self.assertEqual(set(results.keys()), {None, "decode_rot", "decode_xor", "decode_nor"})

        sub = results["decode_rot"]
        sub.runtime = None
        sub.date_start = None
        sub.date_end = None
        self.assertEqual(sub.state, State())
        self.assertEqual(
            sub.feature_types, [Feature(name="apples", desc="Example feature", type=azm.FeatureType.Integer)]
        )

        self.assertEqual(
            TypeAdapter(list[Event]).dump_python(sub.events),
            TypeAdapter(list[Event]).dump_python(
                [
                    Event(
                        sha256="id",
                        features={"apples": {FeatureValue(value=1, offset=None, size=None)}},
                    )
                ]
            ),
        )

        sub = results["decode_xor"]
        sub.runtime = None
        sub.date_start = None
        sub.date_end = None
        self.assertEqual(sub.state.label, State.Label.ERROR_EXCEPTION)

        sub = results["decode_nor"]
        sub.runtime = None
        sub.date_start = None
        sub.date_end = None
        self.assertEqual(sub.state, State())
        self.assertEqual(sub.feature_types, [])
        self.assertIn("ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d", sub.data)
        self.assertEqual(
            TypeAdapter(list[Event]).dump_python(sub.events),
            TypeAdapter(list[Event]).dump_python(
                [
                    Event(sha256="id", info={"k": "1234567890"}),
                    Event(
                        parent=EventParent(sha256="id"),
                        sha256="1",
                        data=[
                            {
                                "hash": "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
                                "label": "content",
                            }
                        ],
                        relationship={"a": 1},
                    ),
                ]
            ),
        )


class TestRegistration(unittest.TestCase):
    @mock.patch("azul_bedrock.dispatcher.DispatcherAPI.submit_events")
    def test_registration(self, reqw):
        class DummyPlugin(DP):
            pass

        p = DummyPlugin()
        net = network.Network(p)
        net.post_registrations()

        author_events: list[azm.PluginEvent] = [x[0][0][0] for x in reqw.call_args_list]

        self.assertEqual(len(author_events), 4)

        self.assertEqual(author_events[0].author, azm.Author(category="plugin", version="1.0", name="DummyPlugin"))
        self.assertEqual(
            author_events[1].author,
            azm.Author(category="plugin", version="1.0-1.0.0", name="DummyPlugin-decode_rot", security="LEMON RUSH"),
        )
        self.assertEqual(
            author_events[2].author,
            azm.Author(category="plugin", version="1.0-21-02-2025", name="DummyPlugin-decode_xor"),
        )
        self.assertEqual(
            author_events[3].author,
            azm.Author(category="plugin", version="1.0-v1-beta", name="DummyPlugin-decode_nor"),
        )

    @mock.patch("azul_bedrock.dispatcher.DispatcherAPI.submit_events")
    def test_registration_no_version(self, reqw):
        class DummyPlugin(DP):
            def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
                super().__init__(config)
                self._multiplugins = {}  # remove parent plugins
                self.register_multiplugin(None, None, lambda x: None)
                self.register_multiplugin("DecodeRot", "", self.decode_rot, description="Custom description here")

        p = DummyPlugin()
        net = network.Network(p)
        net.post_registrations()
        author_events: list[azm.PluginEvent] = [x[0][0][0] for x in reqw.call_args_list]
        self.assertEqual(len(author_events), 2)

        self.assertEqual(author_events[0].author, azm.Author(category="plugin", version="1.0", name="DummyPlugin"))
        self.assertEqual(
            author_events[1].author, azm.Author(category="plugin", version="1.0", name="DummyPlugin-DecodeRot")
        )
        self.assertEqual(author_events[1].entity.description, "Custom description here")


class TestLoop(unittest.TestCase):
    def setUp(self) -> None:
        class DummyPlugin(DP):
            def decode_rot(self, job):
                self.add_feature_values("apples", [FV(1)])

            def decode_xor(self, job):
                raise Exception("invalid")

            def decode_nor(self, job):
                c = self._add_child("1", {"a": 1})
                c.add_data("content", {}, b"5")
                c.add_feature_values("apples", [FV(1)])
                self.add_info({"k": "1234567890"})

        self.ploop = coordinator.Coordinator(DummyPlugin, settings.parse_config(DummyPlugin, {}))
        content = b"small content"
        content_sha = hashlib.sha256(content).hexdigest()
        self.streams = [
            StorageProxyFile(
                source="local",
                label="",
                hash="",
                init_data=content,
                file_info=local.gen_api_content(io.BytesIO(content)),
            )
        ]
        self.event = azm.BinaryEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="gregre",
            action=azm.BinaryAction.Enriched,
            timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            source=azm.Source(
                name="source",
                path=[],
                timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
            ),
            author=azm.Author(name="TestServer", category="blah"),
            entity=azm.BinaryEvent.Entity(
                sha256="1234",
                datastreams=[
                    azm.Datastream(
                        identify_version=1,
                        label=azm.DataLabel.CONTENT,
                        size=13,
                        sha256=content_sha,
                        sha1="1",
                        md5="5",
                        sha512="512",
                        mime="mt",
                        magic="mm",
                        file_format_legacy="ft",
                    ),
                ],
            ),
        )

        self.job = Job(event=self.event)
        self.job.load_streams(local=self.streams)
        return super().setUp()

    def test_run_once_multiplugin(self):
        """Test that individual multiplugins can be run for event."""
        resp_result = self.ploop._run_job_with_multiplugin(self.job, None)
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED_EMPTY)

        resp_result = self.ploop._run_job_with_multiplugin(self.job, "decode_rot")
        self.assertEqual(resp_result.events[0].features, {"apples": [FV(1)]})
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED)

        resp_result = self.ploop._run_job_with_multiplugin(self.job, "decode_xor")
        self.assertEqual(resp_result.state.label, State.Label.ERROR_EXCEPTION)

        resp_result = self.ploop._run_job_with_multiplugin(self.job, "decode_nor")
        self.assertEqual(
            TypeAdapter(list[Event]).dump_python(resp_result.events),
            TypeAdapter(list[Event]).dump_python(
                [
                    Event(sha256="1234", info={"k": "1234567890"}),
                    Event(
                        parent=EventParent(sha256="1234"),
                        sha256="1",
                        data=[
                            {
                                "hash": "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
                                "label": "content",
                            }
                        ],
                        relationship={"a": 1},
                        features={"apples": {FeatureValue(value=1, offset=None, size=None)}},
                    ),
                ]
            ),
        )
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED)

    def test_run_once_ack(self):
        calls = [x for x in self.ploop._run_job(self.job.event)]
        self.assertEqual(len(calls), 4)

        resp_result = calls[0][0]
        self.assertEqual(resp_result.events[0].features, {"apples": [FV(1)]})
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED)
        multiplugin = calls[0][1]
        self.assertEqual(multiplugin, "decode_rot")

        resp_result = calls[1][0]
        self.assertEqual(resp_result.state.label, State.Label.ERROR_EXCEPTION)
        multiplugin = calls[1][1]
        self.assertEqual(multiplugin, "decode_xor")

        resp_result = calls[2][0]
        self.assertEqual(
            TypeAdapter(list[Event]).dump_python(resp_result.events),
            TypeAdapter(list[Event]).dump_python(
                [
                    Event(sha256="1234", info={"k": "1234567890"}),
                    Event(
                        parent=EventParent(
                            sha256="1234",
                        ),
                        sha256="1",
                        data=[
                            {
                                "hash": "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
                                "label": "content",
                            }
                        ],
                        relationship={"a": 1},
                        features={"apples": {FeatureValue(value=1, offset=None, size=None)}},
                    ),
                ]
            ),
        )
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED)
        multiplugin = calls[2][1]
        self.assertEqual(multiplugin, "decode_nor")

        resp_result = calls[3][0]
        self.assertFalse(resp_result.events)
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED_EMPTY)

    def test_run_once_ack_optout(self):
        class DummyPlugin(DP):
            def execute(self, job):
                return State.Label.OPT_OUT

        self.ploop = coordinator.Coordinator(DummyPlugin, settings.parse_config(DummyPlugin, {}))

        calls = [x for x in self.ploop._run_job(self.job.event)]
        self.assertEqual(len(calls), 1)

        resp_result = calls[0][0]
        self.assertNotIn("features", resp_result)
        self.assertEqual(resp_result.state.label, State.Label.OPT_OUT)

    def test_run_completed_with_errors(self):
        class DummyPlugin(DP):
            def execute(self, job):
                return State.Label.COMPLETED_WITH_ERRORS

        self.ploop = coordinator.Coordinator(DummyPlugin, settings.parse_config(DummyPlugin, {}))

        calls = [x for x in self.ploop._run_job(self.job.event)]
        self.assertEqual(len(calls), 4)
        # Sub-plugins completed with no data
        resp_result = calls[0][0]
        self.assertNotIn("features", resp_result)
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED_EMPTY)

        # Parent plugin completed with errors.
        resp_result = calls[-1][0]
        self.assertNotIn("features", resp_result)
        self.assertEqual(resp_result.state.label, State.Label.COMPLETED_WITH_ERRORS)


class TestPluginExecute(TestPlugin):
    """Tests for the handling of add_entity_content."""

    class DummyPlugin(DP):
        def decode_rot(self, job):
            self.add_feature_values("apples", [FV(1)])

        def decode_xor(self, job):
            raise Exception("invalid")

        def decode_nor(self, job):
            c = self._add_child("1", {"a": 1})
            c.add_data("content", {}, b"5")
            c.add_feature_values("apples", [FV(1)])
            self.add_info({"k": "1234567890"})

    PLUGIN_TO_TEST = DummyPlugin

    def test_multi_output(self):
        """Test that multiplugin results are passed through to test cases."""
        result = self.do_execution()

        self.assertEqual(set(result.keys()), {None, "decode_rot", "decode_xor", "decode_nor"})
        self.assertJobResult(
            result["decode_rot"],
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"apples": [FV(1)]})],
            ),
            inspect_data=True,
        )
        self.assertEqual(result["decode_xor"].state.label, State.Label.ERROR_EXCEPTION)
        self.assertJobResult(
            result["decode_nor"],
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(sha256="test_entity", info={"k": "1234567890"}),
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="1",
                        relationship={"a": 1},
                        data=[
                            EventData(
                                hash="ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
                                label=azm.DataLabel.CONTENT,
                            )
                        ],
                        features={"apples": [FV(1)]},
                    ),
                ],
                data={"ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d": b"5"},
            ),
            inspect_data=True,
        )


class TestPluginExecuteOptout(TestPlugin):
    """
    Tests for the handling of add_entity_content
    """

    class DummyPlugin(DP):
        def execute(self, job):
            return State.Label.OPT_OUT

        def decode_rot(self, job):
            self.add_feature_values("apples", [FV(1)])

        def decode_xor(self, job):
            raise Exception("invalid")

        def decode_nor(self, job):
            self._add_child("1", {"a": 1}, child_data=[("content", b"5")], child_features={"apples": [FV(1)]})
            self.add_info({"k": "1234567890"})

    PLUGIN_TO_TEST = DummyPlugin

    def test_multi_output(self):
        """Test that multiplugin results are passed through to test cases."""
        result = self.do_execution()
        # check that only one result was returned
        self.assertEqual(result.state.label, State.Label.OPT_OUT)
