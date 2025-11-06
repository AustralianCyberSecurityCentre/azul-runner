from __future__ import annotations

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
    JobResult,
    State,
    TestPlugin,
)
from azul_runner.models import Job
from tests import plugin_support as sup


class TestPluginExecutionWrapper(TestPlugin):
    """
    Tests the handling of plugin execution by Plugin._exec_wrapper, using the sup.DummyPlugin class and TestPlugin template.
    """

    PLUGIN_TO_TEST = sup.DummyPlugin

    def test_basic_run(self):
        """The wrapper should convert the empty dict return to a valid status and empty features and children."""
        result = self.do_execution()
        self.assertEqual(result.state, State(label=State.Label.COMPLETED_EMPTY))

    def test_nothing_on_failure(self):
        """Check that any features or children are deleted when returning a non-completed status"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self._add_child("foo", {})
                self.add_feature_values("example_string", ["should not be returned to caller"])
                self.add_info({"key": "value"})
                return State.Label.OPT_OUT

        result = self.do_execution(plugin_class=DP)
        self.assertEqual(result.state, State(State.Label.OPT_OUT))
        self.assertEqual(result.main, None)
        self.assertEqual(len(result.events), 0)

    def test_exception_handling(self):
        """Check for correct result if the plugin raises an exception"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                raise RuntimeError("foo")

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, ".*\nRuntimeError: foo\n")
        self.assertEqual(result.state, State(State.Label.ERROR_EXCEPTION, "RuntimeError", m))

    def test_return_nondict(self):
        """Verifies error result when the plugin doesn't return a dict"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                return [1, 2, 3, 4]

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, "Returned data cannot be processed as a State instance")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

    def test_return_bad_status(self):
        """
        Verifies error result when the plugin returns result['status'] != State(...).
        Don't check further State verification, as this is tested in azul_runner.structures' unit tests.
        """

        class DP(sup.DummyPlugin):
            def execute(self, job):
                return "Not a State"

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, "Returned data cannot be processed as a State instance")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

    def test_return_bare_result(self):
        """
        Verifies that runner wraps a bare State / State.Label return value into a dict.
        """

        class DP(sup.DummyPlugin):
            def execute(self, job):
                return State.Label.OPT_OUT

        result = self.do_execution(plugin_class=DP)
        self.assertEqual(result.state, State(State.Label.OPT_OUT))

        class DP(sup.DummyPlugin):
            def execute(self, job):
                return State(State.Label.ERROR_INPUT, "Plugin had an error")

        result = self.do_execution(plugin_class=DP)
        self.assertEqual(result.state, State(State.Label.ERROR_INPUT, "Plugin had an error"))

    def test_return_feature_nonexistent(self):
        """Tests a plugin returning a feature that's not in FEATURES"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("nonexistent_feature", [1])

        # 'Primary' feature test
        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, "Plugin tried to set undeclared features: \\['nonexistent_feature'\\]\n")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

        # Multiple invalid features
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", ["String value"])
                self.add_feature_values("nonexistent_subfeature", ["String value", "49"])
                self.add_feature_values("nonexistent_other", [b"something", b"another"])

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(
            m,
            "Plugin tried to set undeclared features: \\['nonexistent_other', 'nonexistent_subfeature'\\]\n",
        )
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

    def test_return_feature_wrapping(self):
        """
        Tests that returned values get wrapped in a FeatureValue instance, while leaving returned FeatureValues alone.
        """

        # Check feature is wrapped and accepted if correct type
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", ["String but not a FeatureValue instance"])

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={"example_string": [FV("String but not a FeatureValue instance")]},
                    )
                ],
            ),
        )

        # Check feature is accepted if returned already wrapped in FeatureValue
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", [FeatureValue("String")])

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_string": [FV("String")]})],
            ),
        )

    def test_return_str_for_specials(self):
        """Tests that returning a simple str value for Filepath or Uri features is correctly handled."""

        # Check str is accepted for Filepath and Uri
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_path", ["~/not/a/Filepath/instance"])
                self.add_feature_values("example_uri", ["http://foo/", "https://bar/"])

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={
                            "example_path": [FV("~/not/a/Filepath/instance")],
                            "example_uri": [FV("http://foo/"), FV("https://bar/")],
                        },
                    )
                ],
            ),
        )

    def test_return_feature_wrong_type(self):
        """Tests a plugin returning a feature whose data type doesn't match the type specified in FEATURES."""

        # 'Primary' feature test
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", ["String value"])

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(
            m,
            "Plugin returned a value with incorrect type "
            "\\(example_int should be <class 'int'>, not <class 'str'>\\)\n",
        )
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

    def test_return_feature_none(self):
        """Tests a plugin fails instantly when adding None feature value."""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", None)

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_string": []})],
            ),
        )

        # check falsy value is ok
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", "")

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_string": [FV("")]})],
            ),
        )

    def test_return_feature_bare_value(self):
        """Tests that _exec_wrapper accepts single bare values of the correct type and converts them to lists"""

        # Simple case
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", 300)

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_int": [FV(300)]})],
            ),
        )

        # Simple case inside a FeatureValue
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", FeatureValue(300))

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_int": [FV(300)]})],
            ),
        )

        # Bare strings should be accepted for Filepath (and Uri, etc)
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_path", r"c:\foo\bar.txt")

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={"example_path": [FV("c:\\foo\\bar.txt")]},
                    )
                ],
            ),
        )

        # Check that incorrect type is rejected/not autoconverted
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", 5.0)

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(m, "Plugin returned a value with incorrect type")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

        # Check that an unspecified type accepts any valid feature type
        for testval in sup.VALID_FEATURE_EXAMPLES:

            class DP(sup.DummyPlugin):
                def execute(self, job):
                    self.add_feature_values("example_unspec", str(testval))

            result = self.do_execution(plugin_class=DP)
            self.assertEqual(result.state, State())
            self.assertEqual(
                TypeAdapter(list[Event]).dump_python(result.events),
                TypeAdapter(list[Event]).dump_python(
                    [
                        Event(
                            sha256="test_entity",
                            features={"example_unspec": {FeatureValue(str(testval))}},
                        )
                    ]
                ),
            )

        # Check that unspec feature rejects an invalid feature type
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_unspec", AssertionError)

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(
            m,
            "Input should be a valid",
        )
        self.assertEqual(result.state, State(State.Label.ERROR_EXCEPTION, "ValidationError", m))

        # Check that a FeatureValue of incorrect type is wrapped but rejected
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", FeatureValue(5))

        result = self.do_execution(plugin_class=DP)
        m: str = result.state.message
        self.assertRegex(
            m,
            "Plugin returned a value with incorrect type \\(example_string should be <class "
            "'str'>, not <class 'int'>\\)",
        )
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

    def test_return_duplicate_values(self):
        """Tests that a plugin returning multiple copies of the same value no longer causes an error"""

        # Simple list
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", [1, 2, 3, 2])

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_int": [FV(1), FV(2), FV(3)]})],
            ),
        )

        # Duplicate value is a FeatureValue
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", [1, 2, 3, FeatureValue(2)])

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_int": [FV(1), FV(2), FV(3)]})],
            ),
        )

        # Labelled feat should be merged with standalone feat for a single result value
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values(
                    "example_string",
                    [
                        FeatureValue("foo"),
                        FeatureValue("foo", label="bar"),
                    ],
                )
                self.add_feature_values(
                    "example_int",
                    [
                        FeatureValue(5, label="baz"),
                        FeatureValue(5, label="bar"),
                        FeatureValue(5, label="bam"),
                    ],
                )

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={
                            "example_int": [FV(5, label="bam"), FV(5, label="bar"), FV(5, label="baz")],
                            "example_string": [FV("foo"), FV("foo", label="bar")],
                        },
                    )
                ],
            ),
        )

    def test_return_invalid_labels(self):
        """Tests the error checking when returned labels are wrong type"""
        # Check FeatureValue refuses non-str labels
        self.assertRaises(
            TypeError,
            FeatureValue,
            5,
            b"not a str",
        )

    def test_return_labelled_featurevalues(self):
        """Tests that plugins returning FeatureValue fields with labels works correctly"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values(
                    "example_string",
                    [
                        FeatureValue("a string that has labels", label="label1 set in FV object"),
                        FeatureValue("a string that has labels"),
                    ],
                )

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={
                            "example_string": [
                                FV("a string that has labels"),
                                FV("a string that has labels", label="label1 set in FV object"),
                            ]
                        },
                    )
                ],
            ),
        )

    def test_completed_with_error_plugin_return(self):
        """Tests that returning a simple str value for Filepath or Uri features is correctly handled."""

        # Check str is accepted for Filepath and Uri
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", [1])
                return State(
                    State.Label.COMPLETED_WITH_ERRORS, message="Plugin had an error but still has useful data"
                )

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(
                    State.Label.COMPLETED_WITH_ERRORS, message="Plugin had an error but still has useful data"
                ),
                events=[
                    Event(
                        sha256="test_entity",
                        features={
                            "example_int": [FV(1)],
                        },
                    )
                ],
            ),
        )

    def test_feature_value_limit(self):
        """Tests that value counts are properly limited"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", [9, 8, 7, 6, 5, 4, 3, 2, 1])

        result = self.do_execution(plugin_class=DP, config={"max_values_per_feature": 5})
        self.assertEqual(result.state.label, State.Label.COMPLETED_WITH_ERRORS)
        self.assertIn("too many values for feature example_int (9)", result.state.message)
        print(result.events[0])
        self.assertEqual(len(result.events[0].features["example_int"]), 5)

    def test_feature_value_length_limit(self):
        """Tests that value counts are properly limited"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", "wow such a long string")

        result = self.do_execution(plugin_class=DP, config={"max_value_length": 5})
        self.assertEqual(result.state.label, State.Label.ERROR_OUTPUT)
        self.assertEqual(result.state.failure_name, "Invalid Plugin Output")
        self.assertIn(
            "feature example_string has value that is too long (wow such a long string...)", result.state.message
        )

    def test_feature_value_limit_total(self):
        """Tests that value counts are properly limited"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", list(x for x in range(10010)))

        result = self.do_execution(plugin_class=DP, config={"max_values_per_feature": 100000})
        self.assertEqual(result.state.label, State.Label.COMPLETED_WITH_ERRORS)
        self.assertIn("too many values for plugin (10010)", result.state.message)
        self.assertEqual(len(result.events[0].features["example_int"]), 9500)

    def test_feature_value_limit_total2(self):
        """Tests that value counts are properly limited"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_path", list(f"{x}" for x in range(10010)))
                self.add_feature_values("example_int", list(x for x in range(10008)))
                self.add_feature_values("example_string", list(f"{x}" for x in range(10009)))

        result = self.do_execution(plugin_class=DP, config={"max_values_per_feature": 100000})
        self.assertEqual(result.state.label, State.Label.COMPLETED_WITH_ERRORS)
        self.assertIn("too many values for plugin (30027) only returning", result.state.message)
        self.assertIn("dropping 508/10008 values from example_int", result.state.message)
        self.assertIn("dropping 10009/10009 values from example_string", result.state.message)
        self.assertIn("dropping 10010/10010 values from example_path", result.state.message)
        self.assertEqual(len(result.events[0].features["example_path"]), 0)
        self.assertEqual(len(result.events[0].features["example_int"]), 9500)
        self.assertEqual(len(result.events[0].features["example_string"]), 0)

    def test_return_info(self):
        """Tests that plugin can successfully return 'extra' values (unspecified structured data)"""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_info({"key": "value", "list": [1, 2, 5]})

        result = self.do_execution(plugin_class=DP)
        self.assertEqual(result.state, State())
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", info={"key": "value", "list": [1, 2, 5]})],
            ),
        )

    def test_invalid_info(self):
        """Test that an error is returned when the 'extra' value isn't JSON-serialisable."""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_info({"key": "value", "invalid": b"bytes"})

        result = self.do_execution(plugin_class=DP)
        print(result.state)
        self.assertEqual(result.state.label, State.Label.ERROR_OUTPUT)
        self.assertEqual(result.state.failure_name, "Invalid Plugin Output")
        self.assertIn("TypeError: Object of type bytes is not JSON serializable", result.state.message)
        self.assertRegex(result.state.message, "info was not JSON-serialisable")

    def test_event_properties_set(self):
        """Test that an error is returned when the 'extra' value isn't JSON-serialisable."""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                data = job.event.entity.datastreams[0]
                assert data.sha256 == "1669594220a92d73d62727293e988b4213b5b4829de36c3afe43c9b4f3ddf35e"
                assert data.size == 17
                assert data.file_format_legacy == "Text"
                assert data.magic == "ASCII text, with no line terminators"
                assert data.mime == "text/plain"
                assert job.event.entity.sha256 == "1669594220a92d73d62727293e988b4213b5b4829de36c3afe43c9b4f3ddf35e"
                assert job.event.entity.size == 17
                assert job.event.entity.file_format_legacy == "Text"

        result = self.do_execution(plugin_class=DP, data_in=[("content", b"this is some text")])
        self.assertJobResult(result, JobResult(state=State(State.Label.COMPLETED_EMPTY)))

    def test_feats_in(self):
        """Test that feature values can be accessed.

        Important for plugins which use filepath for things, yara, dynamic, etc.
        """

        class DP(sup.DummyPlugin):
            def execute(self, job):
                f = job.event.entity.features[0]
                self.add_feature_values("example_string", f.value)

        result = self.do_execution(
            plugin_class=DP,
            feats_in=[azm.FeatureValue(name="filename", value="walrus.exe", type=azm.FeatureType.String)],
        )
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[Event(sha256="test_entity", features={"example_string": [FV("walrus.exe")]})],
            ),
        )

    def test_feature_order(self):
        """Test that feature and feature value ordering is correct (re-sorted in assertJobResult)."""

        class DP(sup.DummyPlugin):
            FEATURES = [
                Feature("a", "demo", type=FeatureType.String),
                Feature("b", "demo", type=FeatureType.String),
                Feature("c", "demo", type=FeatureType.String),
                Feature("d", "demo", type=FeatureType.String),
                Feature("e", "demo", type=FeatureType.String),
                Feature("num", "demo", type=FeatureType.Integer),
            ]

            def execute(self, job):
                self.add_feature_values("e", "1")
                self.add_feature_values("c", "1")
                self.add_feature_values("a", "99")
                self.add_feature_values("b", "1")
                self.add_feature_values("d", "1")
                self.add_feature_values("a", "9")
                self.add_feature_values("a", "1")
                self.add_feature_values("a", "999")
                self.add_feature_values("num", 1)
                self.add_feature_values("num", 9)
                self.add_feature_values("num", 10)
                self.add_feature_values("num", 90)
                self.add_feature_values("num", 100)
                self.add_feature_values("num", 900)

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={
                            "b": [FV("1")],
                            "c": [FV("1")],
                            "d": [FV("1")],
                            "e": [FV("1")],
                            "a": [FV("1"), FV("9"), FV("99"), FV("999")],
                            "num": [FV(900), FV(1), FV(9), FV(10), FV(90), FV(100)],
                        },
                    )
                ],
            ),
        )

        self.assertEqual(["a", "b", "c", "d", "e", "num"], list(result.events[0].features.keys()))
        self.assertEqual([FV("1"), FV("9"), FV("99"), FV("999")], result.events[0].features["a"])

    def test_inspect_data(self):
        """Test that data can be stripped."""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_data("text", {"language": "numerals"}, b"12345")

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
                                hash="5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",
                                label=azm.DataLabel.TEXT,
                                language="numerals",
                            )
                        ],
                    )
                ],
                data={"5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5": b"12345"},
            ),
            inspect_data=True,
        )
        # test can still read data
        self.assertEqual(
            result.data["5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5"].read(), b"12345"
        )

    def test_strip_data(self):
        """Test that data can be stripped."""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_data("text", {"language": "numerals"}, b"12345")

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
                                hash="5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",
                                label=azm.DataLabel.TEXT,
                                language="numerals",
                            )
                        ],
                    )
                ],
                data={"5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5": b""},
            ),
        )
        # test can still read data
        self.assertEqual(
            result.data["5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5"].read(), b"12345"
        )

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_child_with_data({"action": "barbeque"}, b"12345")

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",
                        relationship={"action": "barbeque"},
                        data=[
                            EventData(
                                hash="5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",
                                label=azm.DataLabel.CONTENT,
                            )
                        ],
                    )
                ],
                data={"5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5": b""},
            ),
        )
        self.assertEqual(
            result.data["5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5"].read(), b"12345"
        )

    def test_strip_hash(self):
        """Test that hash can be stripped."""

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_child_with_data({"action": "barbeque"}, b"12345")

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="0",
                        relationship={"action": "barbeque"},
                        data=[EventData(hash="0", label=azm.DataLabel.CONTENT)],
                    )
                ],
                data={"0": b""},
            ),
            strip_hash=True,
        )

        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_string", "binary_bbq")
                self.add_child_with_data({"action": "barbeque"}, b"12345")
                self.add_data("text", {"language": "c++"}, b"123456")
                self.add_child_with_data({"action": "bbq3"}, b"centigrade")

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        data=[EventData(hash="1", label=azm.DataLabel.TEXT, language="c++")],
                        features={"example_string": [FV("binary_bbq")]},
                    ),
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="0",
                        relationship={"action": "barbeque"},
                        data=[EventData(hash="0", label=azm.DataLabel.CONTENT)],
                    ),
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="2",
                        relationship={"action": "bbq3"},
                        data=[EventData(hash="2", label=azm.DataLabel.CONTENT)],
                    ),
                ],
                data={"0": b"", "1": b"", "2": b""},
            ),
            strip_hash=True,
        )

    def test_old_event_properties(self):
        class DP(sup.DummyPlugin):
            def execute(self, job):
                self.add_feature_values("example_int", [4])

        result = self.do_execution(plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        entity_id="test_entity",
                        entity_type="binary",
                        features={"example_int": [FV("4")]},
                    )
                ],
            ),
        )

    def test_add_settings(self):
        """
        Test adding settings to test case run.
        """

        # Check feature is wrapped and accepted if correct type
        TEST_SETTING_KEY = "test_setting"

        class DP(sup.DummyPlugin):
            def execute(self, job: Job):
                test_val = job.event.source.settings.get(TEST_SETTING_KEY)
                self.add_feature_values("example_string", test_val)

        result = self.do_execution(
            plugin_class=DP, submission_settings={TEST_SETTING_KEY: "Random test setting value that seems fine."}
        )
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="test_entity",
                        features={"example_string": [FV("Random test setting value that seems fine.")]},
                    )
                ],
            ),
        )
