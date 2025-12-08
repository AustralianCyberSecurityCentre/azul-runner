from __future__ import annotations

import unittest
import uuid
from typing import Any

from azul_bedrock import exceptions as azbe
from azul_bedrock import models_network as azm

from azul_runner import (
    FV,
    Event,
    EventData,
    Feature,
    FeatureType,
    JobResult,
    State,
    TestPlugin,
    add_settings,
    settings,
)
from azul_runner.models import EventParent

from . import plugin_support as sup


class BaseInputContentNoFilter(sup.DummyPlugin):
    SETTINGS = add_settings(
        filter_data_types={azm.DataLabel.TEST: []},
        use_multiprocessing_fork=True,
    )


class BaseInputContentAssertionPluginV1(sup.DummyPluginMinimum):
    SETTINGS = add_settings(
        filter_data_types={azm.DataLabel.TEST: ["text/plain"]},
        use_multiprocessing_fork=True,
    )


class BaseInputContentAssertionPluginV2(sup.DummyPluginMinimum):
    SETTINGS = add_settings(
        filter_data_types={azm.DataLabel.TEST: ["text/plain", "unknown", "network/tcpdump"]},
        use_multiprocessing_fork=True,
    )


class BaseInputContentAssertionPluginV3(sup.DummyPluginMinimum):
    SETTINGS = add_settings(
        filter_data_types={azm.DataLabel.TEST: ["network/tcpdump"]},
        use_multiprocessing_fork=True,
    )


# New file format checks


class TestExecutionWrapperInputChecking(TestPlugin):
    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_input_content_assertions(self):
        """Checks that filter_data_types is working correctly."""
        self.assertRaisesRegex(
            AssertionError,
            r"Provided file format \{\} does not meet plugin requirements \{'test': \['text/plain'\]\}\n"
            r"magic: \[\]\n"
            r"mime types: \[\]",
            self.do_execution,
            plugin_class=BaseInputContentAssertionPluginV1,
        )

        self.assertRaisesRegex(
            AssertionError,
            r"Provided file format \{'test': \{'unknown'\}, '\*': \{'unknown'\}\} does not meet plugin requirements \{'test': \['text/plain'\]\}\n"
            r"magic: \['data'\]\n"
            r"mime types: \['application/octet-stream'\]",
            self.do_execution,
            data_in=[(azm.DataLabel.TEST, b"\xff\xff\xe8\x00\xfd")],
            plugin_class=BaseInputContentAssertionPluginV1,
        )
        # Should not raise
        self.do_execution(data_in=[(azm.DataLabel.TEST, b"plain text data")])
        self.do_execution(verify_input_content=False)
        self.do_execution(data_in=[(azm.DataLabel.TEST, b"\xff\xff\xe8\x00\xfd")], verify_input_content=False)

        # Should not raise with no requirements
        self.do_execution(data_in=[(azm.DataLabel.TEST, b"plain text data")], plugin_class=BaseInputContentNoFilter)
        self.do_execution(
            data_in=[(azm.DataLabel.TEST, b"\xff\xff\xe8\x00\xfd")], plugin_class=BaseInputContentNoFilter
        )

        self.do_execution(
            data_in=[(azm.DataLabel.TEST, b"plain text data")], plugin_class=BaseInputContentAssertionPluginV2
        )
        # Should not raise
        self.do_execution(
            data_in=[(azm.DataLabel.TEST, b"plain text data")], plugin_class=BaseInputContentAssertionPluginV2
        )
        self.do_execution(verify_input_content=False, plugin_class=BaseInputContentAssertionPluginV2)
        self.do_execution(
            data_in=[(azm.DataLabel.TEST, b"\xff\xff\xe8\x00\xfd")],
            verify_input_content=False,
            plugin_class=BaseInputContentAssertionPluginV2,
        )

        self.assertRaisesRegex(
            AssertionError,
            r"Provided file format \{'test': \{'text/plain'\}, '\*': \{'text/plain'\}\} does not meet plugin requirements \{'test': \['network/tcpdump'\]\}\n"
            r"magic: \['ASCII text, with no line terminators'\]\n"
            r"mime types: \['text/plain'\]",
            self.do_execution,
            data_in=[(azm.DataLabel.TEST.value, b"plain text data")],
            plugin_class=BaseInputContentAssertionPluginV3,
        )


class TestEventLimits(TestPlugin):
    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_too_large(self):
        """Limit total size of status message."""

        # Generate a dynamic class with input content requirements
        class DP(sup.DummyPlugin):
            def execute(self, job) -> dict:
                tmp = "a" * (1024 * 1024 * 2 + 512)
                print(len(tmp))
                print(tmp[:20])
                self.add_info({"large": tmp})

        self.assertRaisesRegex(
            azbe.NetworkDataException,
            "event produced by plugin was too large: ",
            self.do_execution,
            plugin_class=DP,
        )


class TestAlterConfig(TestPlugin):
    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_too_large(self):
        """Limit total size of status message."""

        # Generate a dynamic class with input content requirements
        class DP(sup.DummyPlugin):
            def execute(self, job) -> dict:
                tmp = "a" * (1024 * 1024 * 2 + 512)
                print(len(tmp))
                print(tmp[:20])
                self.add_info({"large": tmp})

            def _alter_config(self, config: settings.Settings) -> settings.Settings:
                config.filter_data_types = {"*": ["text/plain"]}
                return config

        p = DP()
        self.assertEqual(p.cfg.filter_data_types, {"*": ["text/plain"]})


class TestRequireConfig(TestPlugin):
    """Verify that a plugin can enforce config requirements.
    The embedded tests will fail if PLUGIN_TO_TEST_CONFIG is not set.
    """

    class PluginRequireConfig(sup.DummyPlugin):
        SETTINGS = add_settings(custom_config=(str, ""))

        def __init__(self, config: settings.Settings | dict = None) -> None:
            super().__init__(config)
            if not self.cfg.custom_config:
                raise Exception("plugin requires custom_config property to be set")

    PLUGIN_TO_TEST = PluginRequireConfig
    PLUGIN_TO_TEST_CONFIG = {"custom_config": "yeah"}


class TestMultiStreamResults(TestPlugin):
    """
    Tests correct handling of per-stream results by the runner
    """

    class DP(sup.DummyPlugin):
        def execute(self, job) -> dict:
            e = self.get_data_event("nonexistent")
            e.add_feature_values("example_int", 42)

    PLUGIN_TO_TEST = DP

    def test_invalid_stream_error(self):
        """Tests that an error is raised when a result is returned for a non-existent input stream"""
        result = self.do_execution(
            data_in=[(azm.DataLabel.TEST, b"foo")],
        )
        self.assertRegex(result.state.message, "ResultError: results for non-existent stream")
        self.assertEqual(
            result.state,
            State(
                label=State.Label.ERROR_OUTPUT,
                failure_name="Invalid Plugin Output",
                message=result.state.message,
            ),
        )


class DummyMappingPlugin(sup.DummyPlugin):
    _IS_USING_PUSHER = True
    FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

    def execute(self, job):
        self.add_feature_values("value", 20)


class TestPluginMappedOutput(TestPlugin):
    """Test custom output action types doesn't negatively effect plugin output."""

    PLUGIN_TO_TEST = DummyMappingPlugin

    def test_mapped_output(self):
        result = self.do_execution()
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"value": [FV("20")]})],
            ),
        )


class TestPluginOutputStream(TestPlugin):
    """
    Tests for the handling of add_entity_content
    """

    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_add_alt_stream(self):
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_data("text", {"language": "value"}, b"Text data stream")

        # Test a plugin successfully adding an additional stream result
        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        data=[
                            EventData(
                                hash="903e0722b8ebdfc5547af8ca1a9c8a76d0a7b067a7b1a561dfff13593e76d5a8",
                                label=azm.DataLabel.TEXT,
                                language="value",
                            )
                        ],
                    )
                ],
                data={"903e0722b8ebdfc5547af8ca1a9c8a76d0a7b067a7b1a561dfff13593e76d5a8": b"Text data stream"},
            ),
            inspect_data=True,
        )

    def test_addition_common_feature_malformed(self):
        class DP(sup.DummyPlugin):
            def execute(self, job):
                return self.is_malformed("This file is seriously malformed")

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED_WITH_ERRORS, message="This file is seriously malformed"),
                events=[
                    Event(
                        sha256="test_entity",
                        features={"malformed": [FV("This file is seriously malformed")]},
                    ),
                ],
            ),
        )

    def test_mutating_augmented_stream(self):
        """Test that the test case fails if you mutate the augmented streams.

        Important to limit data produced by plugins.
        """

        # Plugin that generates random sha256's
        class DP(sup.DummyPlugin):

            def execute(self, job):
                unique_id_1 = str(uuid.uuid4())
                self.add_data(azm.DataLabel.TEST, {}, unique_id_1.encode())

        # Verify assertion error is raised with inconsistent outputs
        with self.assertRaises(AssertionError):
            self.do_execution(
                plugin_class=DP,
            )

        # Verify bypassing the check is possible.
        result = self.do_execution(
            plugin_class=DP,
            check_consistent_augmented_stream=False,
        )
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", data=[EventData(hash="0", label="test")])],
                data={"0": b""},
            ),
            strip_hash=True,
        )

    def test_mutating_child_stream(self):
        """Test that the test case fails if you mutate the child streams.

        Important to limit data produced by plugins.
        """

        # Plugin that generates random sha256's
        class DP(sup.DummyPlugin):

            def execute(self, job):
                unique_id_1 = str(uuid.uuid4())
                self.add_child_with_data({"action": "extracted"}, unique_id_1.encode())

        # Verify assertion error is raised with inconsistent outputs
        with self.assertRaises(AssertionError):
            self.do_execution(
                plugin_class=DP,
            )

        # Verify bypassing the check is possible.
        result = self.do_execution(
            plugin_class=DP,
            check_consistent_augmented_stream=False,
        )
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="0",
                        parent=EventParent(sha256="test_entity"),
                        relationship={"action": "extracted"},
                        data=[EventData(hash="0", label="content")],
                    )
                ],
                data={"0": b""},
            ),
            strip_hash=True,
        )

    def test_mutating_multiplugin_stream(self):
        """Test that the test case fails if you mutate the augmented/child streams with a multi-plugin.

        Important to limit data produced by plugins.
        """

        # Plugin that generates random sha256's
        class DP(sup.DummyPlugin):
            def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
                super().__init__(config)
                self._multiplugins = {}  # remove parent plugins
                self.register_multiplugin(None, None, lambda x: self.execute(x))
                self.register_multiplugin(None, None, lambda x: self.execute(x))

            def execute(self, job):
                unique_id_1 = str(uuid.uuid4())
                self.add_child_with_data({"action": "extracted"}, unique_id_1.encode())
                unique_id_2 = str(uuid.uuid4())
                self.add_data(azm.DataLabel.TEST, {}, unique_id_2.encode())

        # Verify assertion error is raised with inconsistent outputs
        with self.assertRaises(AssertionError):
            self.do_execution(
                plugin_class=DP,
            )

        # Verify bypassing the check is possible.
        result = self.do_execution(
            plugin_class=DP,
            check_consistent_augmented_stream=False,
        )
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(sha256="test_entity", data=[EventData(hash="1", label="test")]),
                    Event(
                        sha256="0",
                        parent=EventParent(sha256="test_entity"),
                        relationship={"action": "extracted"},
                        data=[EventData(hash="0", label="content")],
                    ),
                ],
                data={"0": b"", "1": b""},
            ),
            strip_hash=True,
        )


if __name__ == "__main__":
    unittest.main()
