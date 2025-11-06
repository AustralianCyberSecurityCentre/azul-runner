from __future__ import annotations

import datetime
import unittest

from pydantic import ValidationError

from azul_runner.models import Feature, FeatureValue

# ##################################
# FeatureValue and related classes


class TestFeatureComparison(unittest.TestCase):
    """Ensure that Feature class behaves as expected - comparison by name only"""

    def test_comparison(self):
        self.assertEqual(Feature("foo", "First description"), Feature("foo", "Other description"))
        self.assertLess(Feature("a feature", ""), Feature("other feature", ""))

    def test_set(self):
        # Same-named features are considered identical so sets should not update with them
        s = {Feature("a", "Feature a"), Feature("b", "Feature b")}
        s.add(Feature("a", "New description"))
        self.assertEqual(s, {Feature("a", "Feature a"), Feature("b", "Feature b")})
        s.update([Feature("a", "Another description")])
        self.assertEqual(s, {Feature("a", "Feature a"), Feature("b", "Feature b")})


class TestFeatureValue(unittest.TestCase):
    """Tests the FeatureValue class, including its label values"""

    def test_comparison(self):
        """Test that the __eq__/__lt__ functions work correctly"""
        self.assertEqual(FeatureValue(1), FeatureValue(1, offset=None, size=None))
        self.assertLess(FeatureValue(5), FeatureValue(10))
        self.assertGreater(FeatureValue("Zoo"), FeatureValue("Alphabet"))
        self.assertLessEqual(FeatureValue(5), FeatureValue(10))
        self.assertLess(FeatureValue(b"bytes"), FeatureValue("a string"))  # 'bytes' < 'str'
        self.assertGreater(
            FeatureValue(56), FeatureValue(datetime.datetime.now(tz=datetime.timezone.utc))
        )  # 'int' > 'datetime'
        # 'None' offset (and size) should be 'less than' any value
        self.assertLess(FeatureValue(0), FeatureValue(0, offset=-9001))
        # Labels should affect comparison
        self.assertNotEqual(FeatureValue(b"foo", label="x"), FeatureValue(b"foo", label="y"))
        self.assertGreater(FeatureValue(1, label="l1", offset=5), FeatureValue(1, label="l1"))
        self.assertLess(FeatureValue(1, label="l1", offset=3, size=2), FeatureValue(1, label="l1", offset=3, size=5))

    def test_sorting(self):
        self.assertEqual(
            sorted(
                [
                    FeatureValue(19),
                    FeatureValue(5),
                    FeatureValue("string"),
                    FeatureValue(b"blob"),
                    FeatureValue("apple"),
                ]
            ),
            [FeatureValue(b"blob"), FeatureValue(5), FeatureValue(19), FeatureValue("apple"), FeatureValue("string")],
        )
        # Test combination of labels and tags
        self.assertEqual(
            sorted(
                [
                    FeatureValue(5, label="two", size=9),
                    FeatureValue(5),
                    FeatureValue(5, label="one"),
                    FeatureValue(5, label="one", offset=55),
                ]
            ),
            [
                FeatureValue(5),
                FeatureValue(5, label="one"),
                FeatureValue(5, label="one", offset=55),
                FeatureValue(5, label="two", size=9),
            ],
        )

    def test_value_validated(self):
        """Tests that FeatureValue.value is forced to be a valid feature data type"""
        for v in (500, "string", b"bytes", datetime.datetime.now(tz=datetime.timezone.utc)):
            # Should raise no error
            _ = FeatureValue(v)
        self.assertRaisesRegex(ValidationError, "Input should be a valid", FeatureValue, None)
        self.assertRaisesRegex(ValidationError, "validation errors", FeatureValue, [1, 2])
        self.assertRaisesRegex(ValidationError, "validation errors", FeatureValue, (3 + 2j))

    def test_hashing(self):
        """Test that FeatureValues work as dict keys / set members"""
        s = {FeatureValue(5), FeatureValue(5, label="l"), FeatureValue("cat"), FeatureValue(b"foo", label="l")}
        self.assertIn(FeatureValue(5), s)
        self.assertIn(FeatureValue(5, label="l"), s)
        self.assertIn(FeatureValue("cat"), s)
        # Value should not match with different tags and labels, unless using match_value_only
        self.assertNotIn(FeatureValue("cat", label="some label"), s)
        self.assertNotIn(FeatureValue(b"foo"), s)
        # Labels etc are distinguished from each other
        self.assertNotEqual(
            {FeatureValue(5), FeatureValue(5, label="f")},
            {FeatureValue(5)},
        )
        # Tags/labels contribute to comparison but not hashing
        # self.assertEqual(
        #     {FeatureValue(5), FeatureValue(5, label='f')},
        #     {FeatureValue(5).match_value_only},
        # )
