import unittest

from azul_runner import FV, Event, Feature, FeatureType, JobResult, State, TestPlugin, Plugin, add_settings


class TestGenEvents(unittest.TestCase):
    def test_feature(self):
        self.assertTrue(Feature(name="f1", desc="", type=FeatureType.Integer))
        self.assertTrue(Feature(name="f1", desc="", type=FeatureType.Float))
        self.assertTrue(Feature(name="f1", desc="", type=FeatureType.String))
        self.assertTrue(Feature(name="f1", desc="", type=FeatureType.Binary))
        self.assertTrue(Feature(name="f1", desc="", type=FeatureType.Datetime))
        self.assertTrue(Feature(name="f1", desc="", type=FeatureType.Filepath))
        self.assertTrue(Feature(name="f1", desc="", type=FeatureType.Uri))
        self.assertRaises(ValueError, Feature, *("f1", "", dict))

    def test_event(self):
        # check ordinary JobResult also sorts automatically
        tmp = JobResult(
            state=State(State.Label.COMPLETED),
            events=[
                Event(
                    sha256="test_entity",
                    features={
                        "b": [FV("1")],
                        "c": [FV("1")],
                        "d": [FV("1")],
                        "e": [FV("1")],
                        "a": [FV("999"), FV("1"), FV("9"), FV("99")],
                    },
                )
            ],
        )
        self.assertEqual(["a", "b", "c", "d", "e"], list(tmp.events[0].features.keys()))
        self.assertEqual([FV("1"), FV("9"), FV("99"), FV("999")], tmp.events[0].features["a"])


class TestLabelValidation(TestPlugin):
    """Tests for catching bad labels."""

    class DummyPlugin(Plugin):
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
            Feature("bananas", "Example feature", type=FeatureType.Integer),
            Feature("oranges", "Example feature", type=FeatureType.Integer),
        ]

        def execute(self, job):
            features = dict()
            apples = list()
            bananas = list()
            oranges = list()

            apples.append(FV(10, label=chr(0x1F34E)))  # Valid label: utf-8 emoji (RED APPLE)
            apples.append(FV(10, label="Red apple"))  # Valid label: str
            apples.append(FV(10, label="\n\r"))  # Invalid label: newline

            bananas.append(FV(10, label=chr(0x1F34C)))  # Valid label: utf-8 emoji (banana)
            bananas.append(FV(10, label="Yellow banana"))  # Valid label: str
            bananas.append(FV(10, label="\ud83d\ude4f"))  # Invalid label: surrogates

            oranges.append(FV(10, label=chr(0x1F7E0)))  # Valid label:  utf-8 emoji (LARGE ORANGE CIRCLE)
            for i in range(10):
                oranges.append(FV(i, label="\udfff"))  # Invalid label: surrogates

            features["apples"] = apples
            features["bananas"] = bananas
            features["oranges"] = oranges
            self.add_many_feature_values(features)
            return State.Label.COMPLETED

    PLUGIN_TO_TEST = DummyPlugin

    def test_fv_with_bad_label(self):
        """Check we handle bad label values."""
        result = self.do_execution()
        self.assertJobResult(
            result,
            JobResult(
                state=State(
                    State.Label.COMPLETED_WITH_ERRORS,
                    message="Partial completion occurred with the following errors: Invalid labels (1) for feature 'apples': {'\\n\\r'}\nInvalid labels (1) for feature 'bananas': {'\\\\ud83d\\\\ude4f'}\nInvalid labels (1) for feature 'oranges': {'\\\\udfff'}",
                ),
                events=[
                    Event(
                        sha256="test_entity",
                        features={
                            "apples": [
                                FV("10", label="\n\r"),
                                FV("10", label="Red apple"),
                                FV("10", label=chr(0x1F34E)),
                            ],
                            "bananas": [
                                FV("10", label="Yellow banana"),
                                FV("10", label="\\ud83d\\ude4f"),
                                FV("10", label=chr(0x1F34C)),
                            ],
                            "oranges": [
                                FV("0", label="\\udfff"),
                                FV("1", label="\\udfff"),
                                FV("10", label=chr(0x1F7E0)),
                                FV("2", label="\\udfff"),
                                FV("3", label="\\udfff"),
                                FV("4", label="\\udfff"),
                                FV("5", label="\\udfff"),
                                FV("6", label="\\udfff"),
                                FV("7", label="\\udfff"),
                                FV("8", label="\\udfff"),
                                FV("9", label="\\udfff"),
                            ],
                        },
                    )
                ],
            ),
        )
