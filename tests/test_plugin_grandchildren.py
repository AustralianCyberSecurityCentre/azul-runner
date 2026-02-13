from __future__ import annotations
from pydantic import BaseModel
from typing import Any

from azul_bedrock import models_network as azm
from multiprocessing import shared_memory
from azul_runner import (
    DATA_HASH,
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
from tests import plugin_support as sup


SHARED_MEM_NAME = "test_plugin_grandchildren"
_add_grandchild_result_parent_sha256 = DATA_HASH(b"input stream content").hexdigest()


class GrandchildrenSharableInput(BaseModel):
    """Sharable Pydantic BaseModel to allow default dict values."""

    add_child_key: str = "foo"
    add_child_val: dict = {"rt": "rv"}
    add_grandchild_key: str = "bar"
    add_grandchild_val: dict = {"rt": "rv"}
    feat_key: str = "example_unspec"
    feat_val: Any = None


class DPGrandChildrenSharedMem(sup.DummyPluginDefaultSharedMem):
    def get_shared_memory_name(self):
        return SHARED_MEM_NAME

    def execute(self, job):
        shared_mem_val: GrandchildrenSharableInput = self._load_from_shared_memory()
        c = self._add_child(shared_mem_val.add_child_key, shared_mem_val.add_child_val)
        gc = c._add_child(shared_mem_val.add_grandchild_key, shared_mem_val.add_grandchild_val)
        gc.add_feature_values(shared_mem_val.feat_key, shared_mem_val.feat_val)


class TestPluginGrandchildHandling(sup.TestPluginSharedMem):
    """
    Tests for the handling of add_grandchild and related errors
    """

    PLUGIN_TO_TEST = sup.DummyPlugin

    @classmethod
    def setUpClass(cls) -> None:
        sup.cleanup_shared_memory(SHARED_MEM_NAME)
        cls.shared_mem = shared_memory.SharedMemory(name=SHARED_MEM_NAME, create=True, size=10240)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.shared_mem.close()
        cls.shared_mem.unlink()
        return super().tearDownClass()

    def test_add_none(self):
        # Test plugin adding grandchild feature entries that are not a valid type for a feature
        self._set_shared_memory(
            self.shared_mem,
            GrandchildrenSharableInput(
                feat_val=None,
            ),
        )
        result = self.do_execution(plugin_class=DPGrandChildrenSharedMem)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="foo",
                        relationship={"rt": "rv"},
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(sha256="test_entity"),
                            sha256="foo",
                            relationship={"rt": "rv"},
                        ),
                        sha256="bar",
                        relationship={"rt": "rv"},
                        features={"example_unspec": []},
                    ),
                ],
            ),
        )

        # test list of values with a None
        self._set_shared_memory(
            self.shared_mem,
            GrandchildrenSharableInput(
                feat_val=["1", "2", None],
            ),
        )
        result = self.do_execution(plugin_class=DPGrandChildrenSharedMem)
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="foo",
                        relationship={"rt": "rv"},
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(sha256="test_entity"),
                            sha256="foo",
                            relationship={"rt": "rv"},
                        ),
                        sha256="bar",
                        relationship={"rt": "rv"},
                        features={"example_unspec": [FV("1"), FV("2")]},
                    ),
                ],
            ),
        )

    def test_add_grandchild_feature_errors(self):
        """Tests that raises the expected errors when child features are malformed"""

        # Test plugin adding a grandchild feature not defined in FEATURES
        self._set_shared_memory(
            self.shared_mem,
            GrandchildrenSharableInput(
                feat_key="feat",
                feat_val=[None],
            ),
        )
        result = self.do_execution(plugin_class=DPGrandChildrenSharedMem)
        m: str = result.state.message
        self.assertRegex(m, r"ResultError: Plugin tried to set undeclared features")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

        # Test plugin adding grandchild feature entries that are not a valid type for a feature
        self._set_shared_memory(
            self.shared_mem,
            GrandchildrenSharableInput(
                feat_val=555,
            ),
        )
        result = self.do_execution(plugin_class=DPGrandChildrenSharedMem)
        m: str = result.state.message
        self.assertRegex(m, "Plugin returned a value with incorrect type")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

        # Test plugin adding a grandchild feature item list with invalid value types in it
        self._set_shared_memory(
            self.shared_mem,
            GrandchildrenSharableInput(
                add_child_key="bar",
                add_child_val={},
                add_grandchild_key="baz",
                add_grandchild_val={},
                feat_key="example_unspec",
                feat_val=["1", "2", "3", 4],
            ),
        )
        result = self.do_execution(plugin_class=DPGrandChildrenSharedMem)
        m: str = result.state.message
        self.assertRegex(m, "Plugin returned a value with incorrect type")
        self.assertEqual(result.state, State(State.Label.ERROR_OUTPUT, "Invalid Plugin Output", m))

    def test_add_grandchild_simple_feature_value(self):
        # Test a plugin successfully adding a simple contentless grandchild and a single-value feature
        for val in sup.VALID_FEATURE_EXAMPLES:
            self._set_shared_memory(
                self.shared_mem,
                GrandchildrenSharableInput(
                    add_child_key="cid",
                    add_child_val={"reltype": "foo"},
                    add_grandchild_key="gid",
                    add_grandchild_val={"r": "t"},
                    feat_key="example_unspec",
                    feat_val=str(val),
                ),
            )
            result = self.do_execution(plugin_class=DPGrandChildrenSharedMem)

            self.assertJobResult(
                result,
                JobResult(
                    state=State(State.Label.COMPLETED),
                    events=[
                        Event(
                            parent=EventParent(sha256="test_entity"),
                            sha256="cid",
                            relationship={"reltype": "foo"},
                        ),
                        Event(
                            parent=EventParent(
                                parent=EventParent(sha256="test_entity"),
                                sha256="cid",
                                relationship={"reltype": "foo"},
                            ),
                            sha256="gid",
                            relationship={"r": "t"},
                            features={"example_unspec": [FV(str(val))]},
                        ),
                    ],
                ),
            )

        # Same test but with a FeatureValue object
        for val in sup.VALID_FEATURE_EXAMPLES:
            self._set_shared_memory(
                self.shared_mem,
                GrandchildrenSharableInput(
                    add_child_key="cid",
                    add_child_val={"reltype": "foo"},
                    add_grandchild_key="gid",
                    add_grandchild_val={},
                    feat_key="example_unspec",
                    feat_val=FeatureValue(str(val)),
                ),
            )
            result = self.do_execution(plugin_class=DPGrandChildrenSharedMem)
            self.assertJobResult(
                result,
                JobResult(
                    state=State(State.Label.COMPLETED),
                    events=[
                        Event(
                            parent=EventParent(sha256="test_entity"),
                            sha256="cid",
                            relationship={"reltype": "foo"},
                        ),
                        Event(
                            parent=EventParent(
                                parent=EventParent(sha256="test_entity"),
                                sha256="cid",
                                relationship={"reltype": "foo"},
                            ),
                            sha256="gid",
                            features={"example_unspec": [FV(str(val))]},
                        ),
                    ],
                ),
            )

    class DPAddGrandChildResults1(sup.DummyPlugin):
        """Add a child and a grandchild, and check that they are returned correctly in result['children']"""

        def execute(self, job):
            self.add_feature_values("filename", "bad.exe")
            c = self._add_child("foo", {})
            c.add_feature_values("example_unspec", ["3", "horse", "1", "2"])
            c.add_data(azm.DataLabel.TEST, {}, b"some content")
            gc = c._add_child("bar", {"rel": "extracted"})
            gc.add_feature_values("example_int", [8, 7, 6])
            gc.add_data(azm.DataLabel.CONTENT, {}, b"Grandchild binary content")

    class DPAddGrandChildResults2(sup.DummyPlugin):
        def execute(self, job):
            c1 = self._add_child("child", {})
            c2 = self._add_child("child2", {}, parent_sha256=_add_grandchild_result_parent_sha256)
            gc1 = c1._add_child("foo", {"rel": "1"})
            gc1.add_feature_values("example_unspec", ["3", "horse", "1", "2"])
            gc1.add_data(azm.DataLabel.TEXT, {}, b"some content")

            gc2 = c1.add_child_with_data({}, b"2")
            gc2.add_feature_values("example_string", ["value", "other value"])
            gc2.add_data(azm.DataLabel.ASSEMBLYLINE, {}, b"other content")
            gc2.add_data(azm.DataLabel.REPORT, {}, b"more content")

            gc3 = c2.add_child_with_data({"rel": "3"}, b"3")
            gc3.add_feature_values("example_int", [42])

    def test_add_grandchild_results(self):
        """Tests that added children return the expected result"""
        result = self.do_execution(plugin_class=self.DPAddGrandChildResults1)

        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(sha256="test_entity", features={"filename": [FV("bad.exe")]}),
                    Event(
                        parent=EventParent(sha256="test_entity", filename="bad.exe"),
                        sha256="foo",
                        data=[EventData(hash=DATA_HASH(b"some content").hexdigest(), label=azm.DataLabel.TEST)],
                        features={"example_unspec": [FV("1"), FV("2"), FV("3"), FV("horse")]},
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(sha256="test_entity", filename="bad.exe"),
                            sha256="foo",
                        ),
                        sha256="bar",
                        relationship={"rel": "extracted"},
                        data=[
                            EventData(
                                hash=DATA_HASH(b"Grandchild binary content").hexdigest(),
                                label=azm.DataLabel.CONTENT,
                            )
                        ],
                        features={"example_int": [FV(6), FV(7), FV(8)]},
                    ),
                ],
                data={
                    DATA_HASH(b"some content").hexdigest(): b"some content",
                    DATA_HASH(b"Grandchild binary content").hexdigest(): b"Grandchild binary content",
                },
            ),
            inspect_data=True,
        )

        # Add multiple grandchildren, one under a child in a stream-specific result
        some_content = DATA_HASH(b"some content").hexdigest()
        other_content = DATA_HASH(b"other content").hexdigest()
        more_content = DATA_HASH(b"more content").hexdigest()
        content_2 = DATA_HASH(b"2").hexdigest()
        content_3 = DATA_HASH(b"3").hexdigest()
        result = self.do_execution(
            plugin_class=self.DPAddGrandChildResults2, data_in=[(azm.DataLabel.TEST, b"input stream content")]
        )
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        sha256="child",
                    ),
                    Event(
                        parent=EventParent(sha256="test_entity"),
                        parent_sha256=_add_grandchild_result_parent_sha256,
                        sha256="child2",
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(sha256="test_entity"),
                            sha256="child",
                        ),
                        sha256="foo",
                        relationship={"rel": "1"},
                        data=[EventData(hash=some_content, label=azm.DataLabel.TEST)],
                        features={"example_unspec": [FV("1"), FV("2"), FV("3"), FV("horse")]},
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(sha256="test_entity"),
                            sha256="child",
                        ),
                        sha256=content_2,
                        data=[
                            EventData(hash=content_2, label=azm.DataLabel.CONTENT),
                            EventData(hash=other_content, label=azm.DataLabel.TEXT),
                            EventData(hash=more_content, label=azm.DataLabel.TEST),
                        ],
                        features={"example_string": [FV("other value"), FV("value")]},
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(sha256="test_entity"),
                            parent_sha256=_add_grandchild_result_parent_sha256,
                            sha256="child2",
                        ),
                        sha256=content_3,
                        relationship={"rel": "3"},
                        data=[EventData(hash=content_3, label=azm.DataLabel.CONTENT)],
                        features={"example_int": [FV(42)]},
                    ),
                ],
                data={
                    some_content: b"some content",
                    content_2: b"2",
                    other_content: b"other content",
                    more_content: b"more content",
                    content_3: b"3",
                },
            ),
            inspect_data=True,
        )

    class DPAddFilenames(sup.DummyPlugin):
        FEATURES = [
            Feature("filename", "Example string feature", type=FeatureType.String),
        ]

        def execute(self, job):
            c = self._add_child("cid", {"reltype": "dummy"})
            c.add_feature_values("filename", "alpha.3")
            c.add_feature_values("filename", "alpha.1")
            c2 = c._add_child("cid2", {"reltype": "dummy2"})

    def test_add_filenames(self):
        """Test that filename feature values for EventParent() are sorted before applying - for consistency."""
        result = self.do_execution(plugin_class=self.DPAddFilenames)
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
                        features={"filename": [FV("alpha.1"), FV("alpha.3")]},
                    ),
                    Event(
                        parent=EventParent(
                            parent=EventParent(sha256="test_entity"),
                            sha256="cid",
                            relationship={"reltype": "dummy"},
                            filename="alpha.1",
                        ),
                        sha256="cid2",
                        relationship={"reltype": "dummy2"},
                    ),
                ],
            ),
        )
