import unittest

from azul_runner import FV, Event, Feature, FeatureType, JobResult, State


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
