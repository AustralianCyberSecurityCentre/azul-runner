from __future__ import annotations

import tempfile

from azul_bedrock import models_network as azm

from azul_runner import (
    DATA_HASH,
    FV,
    Event,
    EventData,
    EventParent,
    FeatureValue,
    JobResult,
    State,
    TestPlugin,
)

from . import plugin_support as sup


class TestPluginChildHandling(TestPlugin):
    """
    Tests for the handling of add_child, add_child_data, and adding children in run_once
    """

    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_add_none(self):
        # Test plugin adding child feature entries that are not a valid type for a feature
        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self._add_child("foo", {"rt": "rv"})
                c.add_feature_values("example_unspec", None)

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="foo",
                        relationship={"rt": "rv"},
                        features={"example_unspec": []},
                    )
                ],
            ),
        )

        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self._add_child("foo", {"rt": "rv"})
                c.add_feature_values("example_unspec", ["1", "2", None])

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="foo",
                        relationship={"rt": "rv"},
                        features={"example_unspec": [FV("1"), FV("2")]},
                    )
                ],
            ),
        )

    def test_add_empty(self):
        """Tests that raises the expected errors when empty children are added."""

        # Test plugin adding a child feature not defined in FEATURES
        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self.add_child_with_data({"rt": "rv"}, b"")
                c.add_feature_values("example_string", [None])

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, r"tried to add data file with 0 bytes")
        self.assertEqual(result.state, State(State.Label.ERROR_EXCEPTION, "ValueError", m))

    def test_add_child_feature_errors(self):
        """Tests that raises the expected errors when child features are malformed"""

        # Test plugin adding a child feature not defined in FEATURES
        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self._add_child("foo", {"rt": "rv"})
                c.add_feature_values("feat", [None])

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, r"ResultError: Plugin tried to set undeclared features")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

        # Test plugin adding child feature entries that are not a valid type for a feature
        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self._add_child("foo", {"rt": "rv"})
                c.add_feature_values("example_unspec", 555)

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, "Plugin returned a value with incorrect type")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

        # Test plugin adding a child feature item list with invalid value types in it
        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self._add_child("foo", {"rt": "rv"})
                c.add_feature_values("example_unspec", ["1", "2", "3", 4])

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, "Plugin returned a value with incorrect type")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

        # Test that labels can be added to a child feature
        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self._add_child("ID", {"action": "blah"})
                c.add_feature_values(
                    "example_string", [FeatureValue("a", label="foo"), FeatureValue("b", label="bar")]
                )

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="ID",
                        relationship={"action": "blah"},
                        features={"example_string": [FV("a", label="foo"), FV("b", label="bar")]},
                    )
                ],
            ),
        )

    def test_add_child_simple_feature_value(self):
        # Test a plugin successfully adding a simple contentless child and encapsulating single-value feature
        for val in sup.VALID_FEATURE_EXAMPLES:

            class DP(sup.DummyPlugin):
                def execute(self, job):
                    c = self._add_child("cid", {"reltype": "dummy"})
                    c.add_feature_values("example_unspec", str(val))

            result = self.do_execution(plugin_class=DP)
            # Should pick up the plugin's default entity type since it's set to None in the add_child call
            self.assertJobResult(
                result,
                JobResult(
                    state=State(State.Label.COMPLETED),
                    events=[
                        Event(
                            parent=EventParent(sha256="test_entity"),
                            sha256="cid",
                            relationship={"reltype": "dummy"},
                            features={"example_unspec": [FV(str(val))]},
                        )
                    ],
                ),
            )
        # Same test but with a FeatureValue object
        for val in sup.VALID_FEATURE_EXAMPLES:

            class DP(sup.DummyPlugin):
                def execute(self, job):
                    c = self._add_child("cid", {"example": "foo"})
                    c.add_feature_values("example_unspec", FeatureValue(str(val)))

            result = self.do_execution(plugin_class=DP)
            self.assertJobResult(
                result,
                JobResult(
                    state=State(State.Label.COMPLETED),
                    events=[
                        Event(
                            parent=EventParent(sha256="test_entity"),
                            sha256="cid",
                            relationship={"example": "foo"},
                            features={"example_unspec": [FV(str(val))]},
                        )
                    ],
                ),
            )

    def test_add_child_results(self):
        """Tests that added children return the expected result"""

        # Check the expected child and features are returned
        class DP(sup.DummyPlugin):
            def execute(self, job):
                c = self._add_child("foo", {})
                c.add_feature_values("example_unspec", ["3", "horse", "1", "2"])
                c.add_data(azm.DataLabel.TEST, {}, b"some content")

        result = self.do_execution(plugin_class=DP)
        data_hash = DATA_HASH(b"some content").hexdigest()
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="foo",
                        data=[EventData(hash=data_hash, label=azm.DataLabel.TEST)],
                        features={"example_unspec": [FV("1"), FV("2"), FV("3"), FV("horse")]},
                    )
                ],
                data={data_hash: b"some content"},
            ),
            inspect_data=True,
        )

        # Ensure that children are not returned when the result is 'not complete'
        class DP(sup.DummyPlugin):
            def execute(self, *args):
                c = self._add_child("foo", {"rel": "1"})
                c.add_feature_values("example_unspec", ["3", "horse", "1", "2"])
                c.add_data(azm.DataLabel.TEST, {}, b"some content")
                return State(State.Label.ERROR_INPUT, "Generic Error")

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(result, JobResult(state=State(State.Label.ERROR_INPUT, failure_name="Generic Error")))

        # Add multiple children, and additional data for some
        class DP(sup.DummyPlugin):

            def execute(self, *args):
                self.add_feature_values("example_unspec", ["1337"])
                self.add_feature_values("filename", ["bad.exe"])
                c = self.add_child_with_data({"rel": "1"}, b"some content")
                c.add_feature_values("example_unspec", ["3", "horse", "1", "2"])
                c2 = self.add_child_with_data({"rel": 2}, b"other content")
                c2.add_feature_values("example_string", ["value", "other value"])
                c2.add_data(azm.DataLabel.REPORT, {"language": "value2"}, b"more content")
                # duplicate of earlier child but with a different feature
                c3 = self.add_child_with_data({"rel": "1"}, b"some content")
                c3.add_feature_values("example_unspec", ["some new value"])
                c3.add_data(azm.DataLabel.TEST, {"language": "value"}, b"some more content")

        some_content = DATA_HASH(b"some content").hexdigest()
        other_content = DATA_HASH(b"other content").hexdigest()
        more_content = DATA_HASH(b"more content").hexdigest()
        some_more_content = DATA_HASH(b"some more content").hexdigest()
        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={"example_unspec": [FV("1337")], "filename": [FV("bad.exe")]},
                    ),
                    Event(
                        parent=EventParent(sha256="test_entity", filename="bad.exe"),
                        sha256=some_content,
                        relationship={"rel": "1"},
                        data=[
                            EventData(hash=some_content, label=azm.DataLabel.CONTENT),
                            EventData(hash=some_more_content, label=azm.DataLabel.TEST, language="value"),
                        ],
                        features={"example_unspec": [FV("1"), FV("2"), FV("3"), FV("horse"), FV("some new value")]},
                    ),
                    Event(
                        parent=EventParent(sha256="test_entity", filename="bad.exe"),
                        sha256=other_content,
                        relationship={"rel": 2},
                        data=[
                            EventData(hash=other_content, label=azm.DataLabel.CONTENT),
                            EventData(hash=more_content, label=azm.DataLabel.TEST, language="value2"),
                        ],
                        features={"example_string": [FV("other value"), FV("value")]},
                    ),
                ],
                data={
                    some_content: b"some content",
                    other_content: b"other content",
                    more_content: b"more content",
                    some_more_content: b"some more content",
                },
            ),
            inspect_data=True,
        )

    def test_add_child_results_file(self):
        """Tests that added children return the expected result"""

        # Check the expected child and features are returned
        class DP(sup.DummyPlugin):
            def execute(self, job):
                with tempfile.TemporaryFile("r+b") as tf1, tempfile.TemporaryFile("r+b") as tf2:
                    # 100mb of 'a'/'b'
                    for _ in range(100):
                        tf1.write(b"a" * 1_000_000)
                        tf2.write(b"b" * 1_000_000)
                    c = self.add_child_with_data_file({"thing": "yeah"}, tf1)
                    c.add_data_file(azm.DataLabel.TEST, {"language": "english"}, tf2)

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="83d30385a4a11980275dc23de3fb49ff37b906cc841efa048a96c62d90ff3b5f",
                        relationship={"thing": "yeah"},
                        data=[
                            EventData(
                                hash="83d30385a4a11980275dc23de3fb49ff37b906cc841efa048a96c62d90ff3b5f",
                                label=azm.DataLabel.CONTENT,
                            ),
                            EventData(
                                hash="1854ac434080022f8c7addd0d7d79199ad38a8c551b950f3b37a76aee3c08da7",
                                label=azm.DataLabel.TEST,
                                language="english",
                            ),
                        ],
                    )
                ],
                data={
                    "83d30385a4a11980275dc23de3fb49ff37b906cc841efa048a96c62d90ff3b5f": b"",
                    "1854ac434080022f8c7addd0d7d79199ad38a8c551b950f3b37a76aee3c08da7": b"",
                },
            ),
        )
