from __future__ import annotations

from azul_runner import (
    BinaryPlugin,
    Feature,
    FeatureType,
    FeatureValue,
    Job,
    State,
    TestPlugin,
    add_settings,
)


class BinaryTestDatalessPlugin(BinaryPlugin):
    VERSION = "1.0"
    FEATURES = [
        Feature("sample_feature", "An output feature for the test plugin", type=FeatureType.String),
    ]

    SETTINGS = add_settings(
        assume_streams_available=False,
        # any data types
        filter_data_types={},
    )

    def execute(self, job: Job) -> dict:
        digest = job.event.entity.sha512
        self.add_feature_values("sample_feature", digest)


class TestBinaryTemplateDataless(TestPlugin):
    PLUGIN_TO_TEST = BinaryTestDatalessPlugin

    def test_simple_run(self):
        """Check that the dataless test plugin runs and returns a basic feature."""

        result = self.do_execution(entity_attrs={"sha512": "a" * 64})
        self.assertEqual(result.state, State())
        self.assertEqual(result.main.features, {"sample_feature": [FeatureValue("a" * 64)]})
