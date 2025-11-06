from __future__ import annotations

import datetime
import os
import unittest

from azul_bedrock import models_network as azm

from azul_runner import (
    EventData,
    Feature,
    FeatureValue,
    Job,
    JobResult,
    State,
    TestPlugin,
    add_settings,
    local,
    monitor,
    network,
    network_transform,
)

from . import plugin_support as sup


class TestLoadFile(TestPlugin):
    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_load_cart(self):
        """Ensure the unwrap helper function works as expected on a CaRT file."""
        data = self.load_cart("u_testfile.cart", description="test_file")
        self.assertEqual(data, b"file contents")

    def test_load_test_file_bytes(self):
        """Ensure the unwrap helper function works as expected on a CaRT file."""
        data = self.load_test_file_bytes(
            "7bb6f9f7a47a63e684925af3608c059edcc371eb81188c48c9714896fb1091fd",
            "Small carted test file to verify a file can be loaded.",
        )
        self.assertEqual(data, b"file contents")

    def test_load_file_path(self):
        """Ensure the unwrap helper function works as expected on a CaRT file."""
        path = self.load_test_file_path(
            "7bb6f9f7a47a63e684925af3608c059edcc371eb81188c48c9714896fb1091fd",
            "Small carted test file to verify a file can be loaded.",
        )
        with path.open("rb") as f:
            self.assertEqual(f.read(), b"file contents")


class TestAssertReprEqual(TestPlugin):
    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_formatted(self):
        try:
            self.assertReprEqual(JobResult(state=State(State.Label.OPT_OUT)), JobResult(state=State()))
        except AssertionError as e:
            print(e.args[0])
            self.assertIn(
                r"""'JobResult(state=State(State.Label.OPT_OUT, message="No opt-out reason was provided."))\n' != 'JobResult(state=State(State.Label.COMPLETED))\n'
- JobResult(state=State(State.Label.OPT_OUT, message="No opt-out reason was provided."))
+ JobResult(state=State(State.Label.COMPLETED))
""",
                e.args[0],
            )


class QuickFailPlugin(sup.DummyPlugin):
    """Class definied at module level because add_settings creates a dynamic class which is harder for dill to work."""

    SETTINGS = add_settings(heartbeat_interval=1)


class TestBasePluginStatic(unittest.TestCase):
    """
    Test cases for base plugin class - cases that don't require mock server
    """

    def test_feature_inheritance_instance(self):
        """Tests that feature inheritance functions correctly on instantiated classes"""

        class DP(sup.DummyPluginFeatureInheritance):
            def execute(self, job):
                self.add_feature_values("example_string", ["String value"])
                self.add_feature_values("descendant feature", ["Whatever"])

        loop = monitor.Monitor(DP, {})
        self.assertEqual(loop._plugin.VERSION, "2.0")
        self.assertEqual(loop._plugin.cfg.filter_data_types, {})
        self.assertEqual(
            loop._plugin.FEATURES,
            sorted(
                set(sup.DummyPlugin().FEATURES).union(
                    [Feature("descendant feature", "A feature added by the child plugin")]
                )
            ),
        )

        # Check that features give correct output on execution
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        res = loop.run_once(local.gen_event(entity))[None]
        self.assertEqual(res.state, State(State.Label.COMPLETED))
        # Ensure that it returns the features correctly (which means it has passed validation/processing post-run)
        # One feature each from parent and child.
        self.assertEqual(
            res.events[0].features,
            {
                "example_string": [FeatureValue("String value")],
                "descendant feature": [FeatureValue("Whatever")],
            },
        )

    def test_add_no_features(self):
        """Tests that feature inheritance functions correctly on instantiated classes"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                pass

        loop = monitor.Monitor(DP, {})
        self.assertEqual(loop._plugin.cfg.filter_data_types, {})

        # Check that features give correct output on execution
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={})
        res = loop.run_once(local.gen_event(entity))[None]
        self.assertEqual(res.state, State(State.Label.COMPLETED_EMPTY))

    def test_config_checks(self) -> None:
        self.maxDiff = None
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            sup.DummyPluginMinimum(config=42)
        p = sup.DummyPluginMinimum()
        print(p.cfg.model_dump())
        self.assertEqual(
            p.cfg.model_dump(),
            {
                "heartbeat_interval": 30,
                "run_timeout": 600,
                "max_timeouts_before_exit": 100,
                "server": "",
                "events_url": "",
                "data_url": "",
                "request_timeout": 15,
                "require_expedite": True,
                "require_live": True,
                "require_historic": True,
                "request_retry_count": 3,
                "content_meta_cache_limit": 0,
                "max_values_per_feature": 1000,
                "max_value_length": 4000,
                "plugin_depth_limit": 10,
                "not_ready_backoff": 5,
                "name_remove_prefix": "AzulPlugin",
                "name_suffix": "",
                "version_suffix": "",
                "security_override": "",
                "assume_streams_available": False,
                "deployment_key": "",
                "watch_path": "",
                "watch_type": "",
                "watch_wait": 10,
                "enable_mem_limits": False,
                "used_mem_warning_frac": 0.8,
                "used_mem_force_exit_frac": 0.9,
                "max_mem_file_path": "/sys/fs/cgroup/memory.max",
                "cur_mem_file_path": "/sys/fs/cgroup/memory.current",
                "mem_poll_frequency_milliseconds": 1000,
                "filter_max_content_size": 209715200,
                "filter_min_content_size": 0,
                "filter_allow_event_types": [],
                "filter_self": False,
                "filter_data_types": {},
                "concurrent_plugin_instances": 1,
                "use_multiprocessing_fork": False,
            },
        )

    def test_config_env(self) -> None:
        os.environ["plugin_heartbeat_interval"] = "50"
        # custom plugin option
        os.environ["plugin_myvalue2"] = "100"
        os.environ["plugin_myvalue3"] = "100"

        class CustomPlugin(sup.DummyPluginMinimum):
            SETTINGS = add_settings(myvalue2=(str, "100"), myvalue3=(int, 10))

        p = CustomPlugin(config={"name_suffix": "test", "custom1": 999})
        os.environ.pop("plugin_heartbeat_interval")
        os.environ.pop("plugin_myvalue2")
        print("Actual", p.cfg.model_dump())
        self.assertEqual(
            p.cfg.model_dump(),
            {
                "heartbeat_interval": 50,
                "run_timeout": 600,
                "max_timeouts_before_exit": 100,
                "server": "",
                "events_url": "",
                "data_url": "",
                "request_timeout": 15,
                "require_expedite": True,
                "require_live": True,
                "require_historic": True,
                "request_retry_count": 3,
                "content_meta_cache_limit": 0,
                "max_values_per_feature": 1000,
                "max_value_length": 4000,
                "plugin_depth_limit": 10,
                "not_ready_backoff": 5,
                "name_remove_prefix": "AzulPlugin",
                "name_suffix": "test",
                "version_suffix": "",
                "security_override": "",
                "assume_streams_available": False,
                "deployment_key": "",
                "watch_path": "",
                "watch_type": "",
                "watch_wait": 10,
                "enable_mem_limits": False,
                "used_mem_warning_frac": 0.8,
                "used_mem_force_exit_frac": 0.9,
                "max_mem_file_path": "/sys/fs/cgroup/memory.max",
                "cur_mem_file_path": "/sys/fs/cgroup/memory.current",
                "mem_poll_frequency_milliseconds": 1000,
                "filter_max_content_size": 209715200,
                "filter_min_content_size": 0,
                "filter_allow_event_types": [],
                "filter_self": False,
                "filter_data_types": {},
                "concurrent_plugin_instances": 1,
                "use_multiprocessing_fork": False,
                "custom1": 999,
                "myvalue2": "100",
                "myvalue3": 100,
            },
        )

    def test_config_inheritance(self) -> None:
        # Test that parent/template plugins' default configs are picked up, but overridden by children
        ppt = type(
            "ParentPlugin",
            (sup.DummyPluginMinimum,),
            {"SETTINGS": add_settings(conf1=(int, 1), conf2=(str, "parent"))},
        )
        cpt = type("ChildPlugin", (ppt,), {"SETTINGS": add_settings(conf2=(str, "child"))})
        inst = cpt()
        self.assertEqual(inst.cfg.conf1, 1)
        self.assertEqual(inst.cfg.conf2, "child")
        # Test that explicitly given values override defaults
        inst = cpt(config={"conf2": "override"})
        self.assertEqual(inst.cfg.conf2, "override")

    def test_server_none_errors(self) -> None:
        loop = monitor.Monitor(QuickFailPlugin, {"server": ""})
        net = network.Network(loop._plugin)
        self.assertRaises(ValueError, net.fetch_job)
        self.assertRaises(monitor.TerminateError, loop.run_loop)

    def test_base_class_no_execute(self) -> None:
        p = sup.DummyPluginNoExecute()
        with self.assertRaises(NotImplementedError):
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
            p.execute(Job(event=event))

    def test_content_to_API_format(self):
        p = sup.DummyPluginMinimum()
        net = network.Network(p)
        net._cached_file_data = {
            "dummy_hash": azm.Datastream(
                identify_version=1,
                label=azm.DataLabel.CONTENT,
                size=0,
                sha256="256",
                sha1="1",
                md5="5",
                sha512="512",
                mime="mt",
                magic="mm",
                file_format_legacy="ft",
            )
        }
        p.data = {"dummy_hash": b"data"}
        self.assertEqual(
            network_transform._to_api_content(
                net._cached_file_data, [EventData(**{"label": azm.DataLabel.TEST, "hash": "dummy_hash"})]
            ),
            [
                azm.Datastream(
                    identify_version=1,
                    label=azm.DataLabel.TEST,
                    size=0,
                    sha256="256",
                    sha1="1",
                    md5="5",
                    sha512="512",
                    mime="mt",
                    magic="mm",
                    file_format_legacy="ft",
                )
            ],
        )

    def test_features_to_API_format(self):
        class DP(sup.DummyPlugin):
            FEATURES = [
                Feature(name="another feature", desc="", type=azm.FeatureType.String),
                Feature(name="feature1", desc="", type=azm.FeatureType.String),
            ]

        p = DP()
        self.assertEqual(
            network_transform._to_api_features(
                p,
                {
                    "feature1": {
                        FeatureValue("foo", label="label1"),
                        FeatureValue("bar", label="label2", offset=99),
                        FeatureValue("bar", label="label"),
                    },
                    "another feature": {
                        FeatureValue("whatever"),
                        FeatureValue("404"),
                    },
                },
            ),
            [
                azm.FeatureValue(name="another feature", type="string", value="404"),
                azm.FeatureValue(name="another feature", type="string", value="whatever"),
                azm.FeatureValue(name="feature1", type="string", value="bar", label="label"),
                azm.FeatureValue(name="feature1", type="string", value="bar", label="label2", offset=99),
                azm.FeatureValue(name="feature1", type="string", value="foo", label="label1"),
            ],
        )

    def test_generate_filter_string(self):
        class P(sup.DummyPlugin):
            SETTINGS = add_settings(
                filter_data_types={"example": ["content-type-1", "content-type-2"]},
                filter_max_content_size=500,
                filter_min_content_size=1,
            )

        p = P()
        self.assertEqual(p.cfg.filter_max_content_size, 500)
        self.assertEqual(p.cfg.filter_min_content_size, 1)
        self.assertEqual(p.cfg.filter_data_types, {"example": ["content-type-1", "content-type-2"]})

    def test_generate_wildcard_filter_string(self):
        class P(sup.DummyPlugin):
            SETTINGS = add_settings(
                filter_data_types={"*": ["content-type-1", "content-type-2"]},
                filter_max_content_size=500,
                filter_min_content_size=1,
            )

        p = P()
        self.assertEqual(p.cfg.filter_max_content_size, 500)
        self.assertEqual(p.cfg.filter_min_content_size, 1)
        self.assertEqual(p.cfg.filter_data_types, {"*": ["content-type-1", "content-type-2"]})
