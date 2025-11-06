from __future__ import annotations

from azul_bedrock import models_network as azm

from azul_runner import (
    DATA_HASH,
    FV,
    BinaryPlugin,
    Event,
    EventData,
    EventParent,
    Feature,
    FeatureType,
    FeatureValue,
    Filepath,
    Job,
    JobResult,
    State,
    TestPlugin,
    Uri,
    add_settings,
)


class BinaryTestPlugin(BinaryPlugin):
    VERSION = "1.0"
    SETTINGS = add_settings(use_multiprocessing_fork=True)
    FEATURES = [
        Feature("sample_feature", "An output feature for the test plugin", type=FeatureType.String),
    ]

    def execute(self, job: Job) -> dict:
        pass


class TestBinaryTemplateDynamic(TestPlugin):
    PLUGIN_TO_TEST = BinaryTestPlugin

    def test_simple_run(self):
        """Check that the test plugin runs and returns a basic feature."""

        class DP(BinaryTestPlugin):
            def execute(self, job):
                self.add_feature_values("sample_feature", ["completed ok"])

        result = self.do_execution(data_in=[("content", b"dummy data")], plugin_class=DP)
        self.assertEqual(result.state, State())
        self.assertEqual(result.main.features, {"sample_feature": [FeatureValue("completed ok")]})

    def test_legacy_stuff(self):
        """Check that legacy stuff works."""

        # legacy feature typing
        # legacy Filepath and Uri usage in plugin and tests
        class DP(BinaryTestPlugin):
            FEATURES = [
                Feature("a_filepath", "", type=Filepath),
                Feature("a_uri", "", type=Uri),
            ]

            def execute(self, job):
                self.add_feature_values("a_filepath", [FV(Filepath("/rootf/"))])
                self.add_feature_values("a_uri", [FV(Uri("http://abc.com"))])

        result = self.do_execution(data_in=[("content", b"dummy data")], plugin_class=DP)
        self.assertEqual(result.state, State())
        self.assertEqual(
            result.main.features,
            {
                "a_filepath": [FV(Uri("/rootf/"))],
                "a_uri": [FV(Uri("http://abc.com"))],
            },
        )

    def test_add_text(self):
        """Check that binary.add_text functions correctly."""

        class DP(BinaryTestPlugin):
            def execute(self, job):
                self.add_text('print("Hello, world!")', "python")

        result = self.do_execution(data_in=[("content", b"dummy data")], plugin_class=DP)
        stream_id = DATA_HASH(b'print("Hello, world!")').hexdigest()
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="797bb0abff798d7200af7685dca7901edffc52bf26500d5bd97282658ee24152",
                        data=[
                            EventData(
                                hash=stream_id,
                                label=azm.DataLabel.TEXT,
                                language="python",
                            )
                        ],
                    )
                ],
                data={stream_id: b'print("Hello, world!")'},
            ),
            inspect_data=True,
        )

    def test_add_child_invalid_data(self):
        """Check that add_binary_grandchild correctly tests for the child/parent to be present."""

        class DP(BinaryTestPlugin):
            def execute(self, job):
                self.add_child_with_data({"r": "v"}, 45)

        result = self.do_execution(data_in=[("content", b"dummy data")], plugin_class=DP)
        self.assertEqual(result.state, State(State.Label.ERROR_EXCEPTION, "TypeError", result.state.message))
        self.assertIn("a bytes-like object is required", result.state.message)

    def test_add_grandchild_invalid_data(self):
        """Check that add_binary_grandchild correctly tests for the child/parent to be present."""

        class DP(BinaryTestPlugin):
            def execute(self, job):
                data = job.get_data()
                c = self.add_child_with_data({"r": "v"}, b"1")
                gc = c.add_child_with_data({"r": "v2"}, 45)

        result = self.do_execution(data_in=[("content", b"dummy data")], plugin_class=DP)
        self.assertEqual(result.state, State(State.Label.ERROR_EXCEPTION, "TypeError", result.state.message))
        self.assertIn("a bytes-like object is required", result.state.message)

    def test_add_grandchild(self):
        """Check that add_binary_grandchild correctly tests for the child/parent to be present."""
        child_data = b"foo"
        child_id = DATA_HASH(child_data).hexdigest()
        gc_data = b"bar"
        gc_id = DATA_HASH(gc_data).hexdigest()

        class DP(BinaryTestPlugin):
            def execute(self, job):
                data = job.get_data()
                c = self.add_child_with_data({"r": "v"}, child_data)
                gc = c.add_child_with_data({"r": "v2"}, gc_data)
                gc.add_feature_values("sample_feature", ["completed ok"])

        result = self.do_execution(data_in=[("content", b"dummy data")], plugin_class=DP)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(
                            sha256="797bb0abff798d7200af7685dca7901edffc52bf26500d5bd97282658ee24152",
                        ),
                        sha256=child_id,
                        relationship={"r": "v"},
                        data=[
                            EventData(
                                hash=child_id,
                                label=azm.DataLabel.CONTENT,
                            )
                        ],
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(
                                sha256="797bb0abff798d7200af7685dca7901edffc52bf26500d5bd97282658ee24152",
                            ),
                            sha256=child_id,
                            relationship={"r": "v"},
                        ),
                        sha256=gc_id,
                        relationship={"r": "v2"},
                        data=[
                            EventData(
                                hash=gc_id,
                                label=azm.DataLabel.CONTENT,
                            )
                        ],
                        features={"sample_feature": [FV("completed ok")]},
                    ),
                ],
                data={
                    "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae": b"foo",
                    "fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9": b"bar",
                },
            ),
            inspect_data=True,
        )
