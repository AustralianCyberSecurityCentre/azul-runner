import datetime
import unittest

from azul_bedrock import models_network as azm


class TestFeatureValue(unittest.TestCase):
    def test_value_decode(self):
        v = azm.FeatureValue(name="", type=azm.FeatureType.Integer, value="100")
        self.assertEqual(v.decode_value(), 100)
        v = azm.FeatureValue(name="", type=azm.FeatureType.Float, value="10.0")
        self.assertEqual(v.decode_value(), 10.0)
        v = azm.FeatureValue(name="", type=azm.FeatureType.String, value="test")
        self.assertEqual(v.decode_value(), "test")
        v = azm.FeatureValue(name="", type=azm.FeatureType.Binary, value="dGVzdA==")
        self.assertEqual(v.decode_value(), b"test")
        v = azm.FeatureValue(name="", type=azm.FeatureType.Datetime, value="2022-01-01T01:00:00+00:00")
        self.assertEqual(v.decode_value(), datetime.datetime(2022, 1, 1, 1, 0, tzinfo=datetime.timezone.utc))
        v = azm.FeatureValue(name="", type=azm.FeatureType.Filepath, value="a/b/c")
        self.assertEqual(v.decode_value(), "a/b/c")
        v = azm.FeatureValue(name="", type=azm.FeatureType.Uri, value="http://test.com")
        self.assertEqual(v.decode_value(), "http://test.com")
        v = azm.FeatureValue(name="", type=azm.FeatureType.Uri, value="http://test.com")
        self.assertEqual(v.decode_value(), "http://test.com")
        v = azm.FeatureValue(name="", type=azm.FeatureType.Uri, value="meow")
        v.type = "cat"
        self.assertRaises(ValueError, v.decode_value)
