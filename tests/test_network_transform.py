import datetime
import hashlib
import json
import unittest
from unittest import mock

from azul_bedrock import models_network as azm

from azul_runner import (
    FV,
    Event,
    EventData,
    EventParent,
    Feature,
    JobResult,
    State,
    network_transform,
)

from . import plugin_support as sup


def dump(x):
    return json.loads(x.model_dump_json(exclude_defaults=True))


def gen_api_content(x, label: azm.DataLabel = azm.DataLabel.CONTENT):
    return azm.Datastream(
        identify_version=1,
        label=label,
        size=5,
        md5=x,
        sha1=x,
        sha256=x,
        sha512=x,
        mime="mime",
        magic="magic",
        file_format_legacy="filetype",
        file_format="file/type",
        file_extension="ft",
        ssdeep="12:some:thing",
        tlsh="T1xxxxxxx",
    )


def gen_src_event():
    timestamp = datetime.datetime(year=2000, month=1, day=1, tzinfo=datetime.timezone.utc)
    return azm.BinaryEvent(
        model_version=azm.CURRENT_MODEL_VERSION,
        kafka_key="dummy-id",
        dequeued="dummy-dequeued-id",
        action=azm.BinaryAction.Extracted,
        timestamp=timestamp,
        source=azm.Source(
            name="sauce",
            path=[
                azm.PathNode(
                    action=azm.BinaryAction.Extracted,
                    timestamp=timestamp,
                    author=azm.Author(category="plugin", name="system", version="1", security=None),
                    sha256="hash",
                )
            ],
            timestamp=timestamp,
            security=None,
            references={"keen": "eye"},
        ),
        author=azm.Author(category="plugin", name="system", version="1", security=None),
        entity=azm.BinaryEvent.Entity(
            sha256="hash",
            datastreams=[gen_api_content("hash"), gen_api_content("hash5", azm.DataLabel.TEXT)],
        ),
    )


class TestGenEvents(unittest.TestCase):
    maxDiff = None

    class P(sup.DummyPluginMinimum):
        FEATURES = [
            Feature(name="feature1", desc="", type=azm.FeatureType.Integer),
            Feature(name="feature2", desc="", type=azm.FeatureType.String),
        ]

    p = P()

    def test_gen_event_enriched(self):
        src = gen_src_event()
        posted = {"hash2": gen_api_content("hash2", azm.DataLabel.TEXT)}
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256=src.entity.sha256,
            info={"test": "data"},
            features={"feature1": [FV(1), FV(2)], "feature2": [FV("1"), FV("2")]},
            data=[],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        res = dump(network_transform._gen_event_output(self.p, posted, author, src, event, now))
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "source": {
                    "references": {"keen": "eye"},
                    "path": [
                        {
                            "sha256": "hash",
                            "author": {"category": "plugin", "name": "system", "version": "1"},
                            "action": "extracted",
                            "timestamp": "2000-01-01T00:00:00+00:00",
                        },
                        {
                            "sha256": "hash",
                            "author": {"category": "plugin", "name": "generic", "version": "1"},
                            "action": "enriched",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                        },
                    ],
                    "timestamp": "2000-01-01T00:00:00+00:00",
                    "name": "sauce",
                },
                "author": {"category": "plugin", "name": "generic", "version": "1"},
                "action": "enriched",
                "timestamp": "2010-01-01T00:00:00+00:00",
                "entity": {
                    "info": {"test": "data"},
                    "features": [
                        {"type": "integer", "name": "feature1", "value": "1"},
                        {"type": "integer", "name": "feature1", "value": "2"},
                        {"type": "string", "name": "feature2", "value": "1"},
                        {"type": "string", "name": "feature2", "value": "2"},
                    ],
                    "sha256": "hash",
                },
            },
        )

    def test_using_pusher_generation(self):
        """Test generating an event using the pusher event generation."""
        src = gen_src_event()
        posted = {"hash2": gen_api_content("hash2", azm.DataLabel.TEXT)}
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256=src.entity.sha256,
            info={"test": "data"},
            features={"feature1": [FV(1), FV(2)], "feature2": [FV("1"), FV("2")]},
            data=[],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        custom_p = self.P()
        custom_p._IS_USING_PUSHER = True
        res = dump(network_transform._gen_event_output(custom_p, posted, author, src, event, now))
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "timestamp": "2010-01-01T00:00:00+00:00",
                "author": {"category": "plugin", "name": "generic", "version": "1"},
                "entity": {
                    "sha256": "hash",
                    "features": [
                        {"name": "feature1", "type": "integer", "value": "1"},
                        {"name": "feature1", "type": "integer", "value": "2"},
                        {"name": "feature2", "type": "string", "value": "1"},
                        {"name": "feature2", "type": "string", "value": "2"},
                    ],
                    "datastreams": [
                        {
                            "sha256": "hash",
                            "sha512": "hash",
                            "sha1": "hash",
                            "md5": "hash",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "content",
                        },
                        {
                            "sha256": "hash5",
                            "sha512": "hash5",
                            "sha1": "hash5",
                            "md5": "hash5",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "text",
                        },
                    ],
                    "info": {"test": "data"},
                },
                "action": "extracted",
                "source": {
                    "name": "sauce",
                    "path": [
                        {
                            "sha256": "hash",
                            "action": "extracted",
                            "timestamp": "2000-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "system", "version": "1"},
                        }
                    ],
                    "timestamp": "2000-01-01T00:00:00+00:00",
                    "references": {"keen": "eye"},
                },
            },
        )

    def test_using_pusher_generation_both_events_have_features(self):
        """Test generating an event using the pusher event generation, where the source event has features to."""
        src = gen_src_event()
        src.action = azm.BinaryAction.Mapped
        src.source.path[0].action = azm.BinaryAction.Mapped
        src.entity.features.append(
            azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value="SPECIAL_SOURCE_FILENAME!")
        )
        posted = {"hash2": gen_api_content("hash2", azm.DataLabel.TEXT)}
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256=src.entity.sha256,
            info={"test": "data"},
            features={"feature1": [FV(1), FV(2)], "feature2": [FV("1"), FV("2")]},
            data=[],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        custom_p = self.P()
        custom_p._IS_USING_PUSHER = True
        res = dump(network_transform._gen_event_output(custom_p, posted, author, src, event, now))
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "timestamp": "2010-01-01T00:00:00+00:00",
                "author": {"category": "plugin", "name": "generic", "version": "1"},
                "entity": {
                    "sha256": "hash",
                    "features": [
                        {"name": "feature1", "type": "integer", "value": "1"},
                        {"name": "feature1", "type": "integer", "value": "2"},
                        {"name": "feature2", "type": "string", "value": "1"},
                        {"name": "feature2", "type": "string", "value": "2"},
                        {"name": "filename", "type": "filepath", "value": "SPECIAL_SOURCE_FILENAME!"},
                    ],
                    "datastreams": [
                        {
                            "sha256": "hash",
                            "sha512": "hash",
                            "sha1": "hash",
                            "md5": "hash",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "content",
                        },
                        {
                            "sha256": "hash5",
                            "sha512": "hash5",
                            "sha1": "hash5",
                            "md5": "hash5",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "text",
                        },
                    ],
                    "info": {"test": "data"},
                },
                "action": "mapped",
                "source": {
                    "name": "sauce",
                    "path": [
                        {
                            "sha256": "hash",
                            "action": "mapped",
                            "timestamp": "2000-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "system", "version": "1"},
                        }
                    ],
                    "timestamp": "2000-01-01T00:00:00+00:00",
                    "references": {"keen": "eye"},
                },
            },
        )

    def test_using_pusher_generation_only_source_event_has_features(self):
        """Test generating an event using the pusher event generation,
        where the source event has features and the child event doesn't.

        This accounts for an edge case where the parent features could be duplicated.
        """
        src = gen_src_event()
        src.action = azm.BinaryAction.Sourced
        src.source.path[0].action = azm.BinaryAction.Sourced
        src.entity.features.append(
            azm.FeatureValue(name="filename", type=azm.FeatureType.Filepath, value="SPECIAL_SOURCE_FILENAME!")
        )
        posted = {"hash2": gen_api_content("hash2", azm.DataLabel.TEXT)}
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256=src.entity.sha256,
            info={"test": "data"},
            data=[],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        custom_p = self.P()
        custom_p._IS_USING_PUSHER = True
        res = dump(network_transform._gen_event_output(custom_p, posted, author, src, event, now))
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "timestamp": "2010-01-01T00:00:00+00:00",
                "author": {"category": "plugin", "name": "generic", "version": "1"},
                "entity": {
                    "sha256": "hash",
                    "features": [{"name": "filename", "type": "filepath", "value": "SPECIAL_SOURCE_FILENAME!"}],
                    "datastreams": [
                        {
                            "sha256": "hash",
                            "sha512": "hash",
                            "sha1": "hash",
                            "md5": "hash",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "content",
                        },
                        {
                            "sha256": "hash5",
                            "sha512": "hash5",
                            "sha1": "hash5",
                            "md5": "hash5",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "text",
                        },
                    ],
                    "info": {"test": "data"},
                },
                "action": "sourced",
                "source": {
                    "name": "sauce",
                    "path": [
                        {
                            "sha256": "hash",
                            "action": "sourced",
                            "timestamp": "2000-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "system", "version": "1"},
                        }
                    ],
                    "timestamp": "2000-01-01T00:00:00+00:00",
                    "references": {"keen": "eye"},
                },
            },
        )

    def test_gen_event_augmented(self):
        src = gen_src_event()
        posted = {"hash2": gen_api_content("hash2", azm.DataLabel.TEXT)}
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256=src.entity.sha256,
            info={"test": "data"},
            features={"feature1": [FV(1), FV(2)], "feature2": [FV("1"), FV("2")]},
            data=[EventData(hash="hash2", label=azm.DataLabel.TEXT, language="javascript")],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        res = dump(network_transform._gen_event_output(self.p, posted, author, src, event, now))
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "source": {
                    "references": {"keen": "eye"},
                    "path": [
                        {
                            "sha256": "hash",
                            "author": {"category": "plugin", "name": "system", "version": "1"},
                            "action": "extracted",
                            "timestamp": "2000-01-01T00:00:00+00:00",
                        },
                        {
                            "sha256": "hash",
                            "author": {"category": "plugin", "name": "generic", "version": "1"},
                            "action": "augmented",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                        },
                    ],
                    "timestamp": "2000-01-01T00:00:00+00:00",
                    "name": "sauce",
                },
                "author": {"category": "plugin", "name": "generic", "version": "1"},
                "action": "augmented",
                "timestamp": "2010-01-01T00:00:00+00:00",
                "entity": {
                    "datastreams": [
                        {
                            "identify_version": 1,
                            "label": "content",
                            "size": 5,
                            "sha1": "hash",
                            "sha256": "hash",
                            "file_format_legacy": "filetype",
                            "md5": "hash",
                            "magic": "magic",
                            "mime": "mime",
                            "sha512": "hash",
                            "file_extension": "ft",
                            "file_format": "file/type",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                        },
                        {
                            "identify_version": 1,
                            "language": "javascript",
                            "label": "text",
                            "size": 5,
                            "sha1": "hash2",
                            "sha256": "hash2",
                            "file_format_legacy": "filetype",
                            "md5": "hash2",
                            "magic": "magic",
                            "mime": "mime",
                            "sha512": "hash2",
                            "file_extension": "ft",
                            "file_format": "file/type",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                        },
                    ],
                    "info": {"test": "data"},
                    "features": [
                        {"type": "integer", "name": "feature1", "value": "1"},
                        {"type": "integer", "name": "feature1", "value": "2"},
                        {"type": "string", "name": "feature2", "value": "1"},
                        {"type": "string", "name": "feature2", "value": "2"},
                    ],
                    "sha256": "hash",
                },
            },
        )

    def test_gen_event_extracted_1deep(self):
        src = gen_src_event()
        posted = {"hash2": gen_api_content("hash2", azm.DataLabel.TEXT), "hash3": gen_api_content("hash3")}
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256="hash3",
            parent=EventParent(
                sha256=src.entity.sha256,
            ),  # parent is top level binary
            relationship={"decoded": "rot26"},
            info={"test": "data"},
            features={
                "feature1": [FV(1), FV(2)],
                "feature2": [FV("1"), FV("2")],
                "filename": [FV("a.exe"), FV("b.exe")],
            },
            data=[
                EventData(hash="hash3", label=azm.DataLabel.CONTENT),
                EventData(hash="hash2", label=azm.DataLabel.TEXT, language="javascript"),
            ],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        res = dump(network_transform._gen_event_extracted(self.p, posted, author, src, event, now))
        print(res)
        self.assertEqual(
            res,
            {
                "kafka_key": "runner-placeholder",
                "action": "extracted",
                "model_version": 5,
                "timestamp": "2010-01-01T00:00:00+00:00",
                "source": {
                    "name": "sauce",
                    "path": [
                        {
                            "sha256": "hash",
                            "action": "extracted",
                            "timestamp": "2000-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "system", "version": "1"},
                        },
                        {
                            "sha256": "hash3",
                            "action": "extracted",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "generic", "version": "1"},
                            "relationship": {"decoded": "rot26"},
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "size": 5,
                            "filename": "a.exe",
                        },
                    ],
                    "timestamp": "2000-01-01T00:00:00+00:00",
                    "references": {"keen": "eye"},
                },
                "author": {"category": "plugin", "name": "generic", "version": "1"},
                "entity": {
                    "sha256": "hash3",
                    "sha512": "hash3",
                    "sha1": "hash3",
                    "md5": "hash3",
                    "ssdeep": "12:some:thing",
                    "tlsh": "T1xxxxxxx",
                    "size": 5,
                    "file_format_legacy": "filetype",
                    "file_format": "file/type",
                    "file_extension": "ft",
                    "mime": "mime",
                    "magic": "magic",
                    "info": {"test": "data"},
                    "features": [
                        {"name": "file_format", "type": "string", "value": "file/type"},
                        {"name": "file_format_legacy", "type": "string", "value": "filetype"},
                        {"name": "file_extension", "type": "string", "value": "ft"},
                        {"name": "magic", "type": "string", "value": "magic"},
                        {"name": "mime", "type": "string", "value": "mime"},
                        {"name": "feature1", "type": "integer", "value": "1"},
                        {"name": "feature1", "type": "integer", "value": "2"},
                        {"name": "feature2", "type": "string", "value": "1"},
                        {"name": "feature2", "type": "string", "value": "2"},
                        {"name": "filename", "type": "filepath", "value": "a.exe"},
                        {"name": "filename", "type": "filepath", "value": "b.exe"},
                    ],
                    "datastreams": [
                        {
                            "sha256": "hash2",
                            "sha512": "hash2",
                            "sha1": "hash2",
                            "md5": "hash2",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "text",
                            "language": "javascript",
                        },
                        {
                            "sha256": "hash3",
                            "sha512": "hash3",
                            "sha1": "hash3",
                            "md5": "hash3",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "content",
                        },
                    ],
                },
            },
        )

    def test_gen_event_extracted_4deep(self):
        src = gen_src_event()
        posted = {
            "hash2": gen_api_content("hash2", azm.DataLabel.TEXT),
            "hash3": gen_api_content("hash3"),
            "parent1": gen_api_content("parent1"),
            "parent2": gen_api_content("parent2"),
        }
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256="hash3",
            parent=EventParent(
                sha256="parent1",
                filename="double.exe",
                parent=EventParent(
                    sha256="parent2",
                    filename="trouble.exe",
                    parent=EventParent(
                        sha256=src.entity.sha256,
                    ),
                ),
            ),  # parent is top level binary
            relationship={"decoded": "rot26"},
            info={"test": "data"},
            features={
                "feature1": [FV(1), FV(2)],
                "feature2": [FV("1"), FV("2")],
                "filename": [FV("a.exe"), FV("b.exe")],
            },
            data=[
                EventData(hash="hash3", label=azm.DataLabel.CONTENT),
                EventData(hash="hash2", label=azm.DataLabel.TEXT, language="javascript"),
            ],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        res = dump(network_transform._gen_event_extracted(self.p, posted, author, src, event, now))
        print(res)
        self.assertEqual(
            res,
            {
                "kafka_key": "runner-placeholder",
                "action": "extracted",
                "model_version": 5,
                "timestamp": "2010-01-01T00:00:00+00:00",
                "source": {
                    "name": "sauce",
                    "path": [
                        {
                            "sha256": "hash",
                            "action": "extracted",
                            "timestamp": "2000-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "system", "version": "1"},
                        },
                        {
                            "sha256": "parent2",
                            "action": "extracted",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "generic", "version": "1"},
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "size": 5,
                            "filename": "trouble.exe",
                        },
                        {
                            "sha256": "parent1",
                            "action": "extracted",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "generic", "version": "1"},
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "size": 5,
                            "filename": "double.exe",
                        },
                        {
                            "sha256": "hash3",
                            "action": "extracted",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                            "author": {"category": "plugin", "name": "generic", "version": "1"},
                            "relationship": {"decoded": "rot26"},
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "size": 5,
                            "filename": "a.exe",
                        },
                    ],
                    "timestamp": "2000-01-01T00:00:00+00:00",
                    "references": {"keen": "eye"},
                },
                "author": {"category": "plugin", "name": "generic", "version": "1"},
                "entity": {
                    "sha256": "hash3",
                    "sha512": "hash3",
                    "sha1": "hash3",
                    "md5": "hash3",
                    "ssdeep": "12:some:thing",
                    "tlsh": "T1xxxxxxx",
                    "size": 5,
                    "file_format_legacy": "filetype",
                    "file_format": "file/type",
                    "file_extension": "ft",
                    "mime": "mime",
                    "magic": "magic",
                    "info": {"test": "data"},
                    "features": [
                        {"name": "file_format", "type": "string", "value": "file/type"},
                        {"name": "file_format_legacy", "type": "string", "value": "filetype"},
                        {"name": "file_extension", "type": "string", "value": "ft"},
                        {"name": "magic", "type": "string", "value": "magic"},
                        {"name": "mime", "type": "string", "value": "mime"},
                        {"name": "feature1", "type": "integer", "value": "1"},
                        {"name": "feature1", "type": "integer", "value": "2"},
                        {"name": "feature2", "type": "string", "value": "1"},
                        {"name": "feature2", "type": "string", "value": "2"},
                        {"name": "filename", "type": "filepath", "value": "a.exe"},
                        {"name": "filename", "type": "filepath", "value": "b.exe"},
                    ],
                    "datastreams": [
                        {
                            "sha256": "hash2",
                            "sha512": "hash2",
                            "sha1": "hash2",
                            "md5": "hash2",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "text",
                            "language": "javascript",
                        },
                        {
                            "sha256": "hash3",
                            "sha512": "hash3",
                            "sha1": "hash3",
                            "md5": "hash3",
                            "ssdeep": "12:some:thing",
                            "tlsh": "T1xxxxxxx",
                            "size": 5,
                            "file_format_legacy": "filetype",
                            "file_format": "file/type",
                            "file_extension": "ft",
                            "mime": "mime",
                            "magic": "magic",
                            "identify_version": 1,
                            "label": "content",
                        },
                    ],
                },
            },
        )

    def test_gen_processing_events_enriched(self):
        src = gen_src_event()
        posted = {
            "hash2": gen_api_content("hash2", azm.DataLabel.TEXT),
            "hash3": gen_api_content("hash3"),
            "parent1": gen_api_content("parent1"),
            "parent2": gen_api_content("parent2"),
        }
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(sha256=src.entity.sha256, info={"test": "data"})
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        result = JobResult(state=State(), events=[event], runtime=5)
        with mock.patch("azul_runner.network_transform._get_now", lambda: now):
            res = network_transform.gen_processing_events(self.p, posted, author, src, result)
        res = dump(res)
        res["entity"].pop("input")  # dont care about this
        print(res)
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "author": {"category": "plugin", "version": "1", "name": "generic"},
                "entity": {
                    "status": "completed",
                    "results": [
                        {
                            "model_version": 5,
                            "kafka_key": "runner-placeholder",
                            "source": {
                                "path": [
                                    {
                                        "author": {"category": "plugin", "version": "1", "name": "system"},
                                        "sha256": "hash",
                                        "action": "extracted",
                                        "timestamp": "2000-01-01T00:00:00+00:00",
                                    },
                                    {
                                        "author": {"category": "plugin", "version": "1", "name": "generic"},
                                        "sha256": "hash",
                                        "action": "enriched",
                                        "timestamp": "2010-01-01T00:00:00+00:00",
                                    },
                                ],
                                "references": {"keen": "eye"},
                                "name": "sauce",
                                "timestamp": "2000-01-01T00:00:00+00:00",
                            },
                            "author": {"category": "plugin", "version": "1", "name": "generic"},
                            "action": "enriched",
                            "entity": {
                                "sha256": "hash",
                                "info": {"test": "data"},
                            },
                            "timestamp": "2010-01-01T00:00:00+00:00",
                        }
                    ],
                    "runtime": 5.0,
                },
                "timestamp": "2010-01-01T00:00:00+00:00",
            },
        )

    def test_gen_processing_events_extracted(self):
        src = gen_src_event()
        posted = {
            "hash2": gen_api_content("hash2", azm.DataLabel.TEXT),
            "hash3": gen_api_content("hash3"),
            "parent1": gen_api_content("parent1"),
            "parent2": gen_api_content("parent2"),
        }
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256="hash3",
            parent=EventParent(sha256=src.entity.sha256),  # parent is top level binary
            relationship={"decoded": "rot26"},
            info={"test": "data"},
            data=[
                EventData(hash="hash3", label=azm.DataLabel.CONTENT),
            ],
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        result = JobResult(state=State(), events=[event], runtime=5)
        with mock.patch("azul_runner.network_transform._get_now", lambda: now):
            res = network_transform.gen_processing_events(self.p, posted, author, src, result)
        res = dump(res)
        res["entity"].pop("input")  # dont care about this
        print(res)
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "author": {"version": "1", "category": "plugin", "name": "generic"},
                "entity": {
                    "results": [
                        {
                            "author": {"version": "1", "category": "plugin", "name": "generic"},
                            "entity": {
                                "file_format_legacy": "filetype",
                                "datastreams": [
                                    {
                                        "identify_version": 1,
                                        "file_format_legacy": "filetype",
                                        "sha256": "hash3",
                                        "mime": "mime",
                                        "sha512": "hash3",
                                        "sha1": "hash3",
                                        "size": 5,
                                        "magic": "magic",
                                        "label": "content",
                                        "md5": "hash3",
                                        "file_extension": "ft",
                                        "file_format": "file/type",
                                        "ssdeep": "12:some:thing",
                                        "tlsh": "T1xxxxxxx",
                                    }
                                ],
                                "sha256": "hash3",
                                "info": {"test": "data"},
                                "features": [
                                    {"type": "string", "value": "mime", "name": "mime"},
                                    {"type": "string", "value": "magic", "name": "magic"},
                                    {"type": "string", "value": "filetype", "name": "file_format_legacy"},
                                ],
                                "sha512": "hash3",
                                "sha1": "hash3",
                                "size": 5,
                                "md5": "hash3",
                                "file_extension": "ft",
                                "file_format_legacy": "filetype",
                                "file_format": "file/type",
                                "ssdeep": "12:some:thing",
                                "tlsh": "T1xxxxxxx",
                            },
                            "action": "extracted",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                            "model_version": 5,
                            "kafka_key": "",
                            "source": {
                                "timestamp": "2000-01-01T00:00:00+00:00",
                                "name": "sauce",
                                "path": [
                                    {
                                        "author": {"version": "1", "category": "plugin", "name": "system"},
                                        "sha256": "hash",
                                        "action": "extracted",
                                        "timestamp": "2000-01-01T00:00:00+00:00",
                                    },
                                    {
                                        "author": {"version": "1", "category": "plugin", "name": "generic"},
                                        "sha256": "hash3",
                                        "action": "extracted",
                                        "timestamp": "2010-01-01T00:00:00+00:00",
                                        "file_format_legacy": "filetype",
                                        "size": 5,
                                        "relationship": {"decoded": "rot26"},
                                    },
                                ],
                                "references": {"keen": "eye"},
                            },
                        }
                    ],
                    "status": "completed",
                    "runtime": 5.0,
                    "kafka_key": "-generic-1",
                },
                "timestamp": "2010-01-01T00:00:00+00:00",
                "kafka_key": "",
            },
        )

    def test_gen_processing_events_extracted(self):
        src = gen_src_event()
        posted = {}
        author = azm.Author(category="plugin", name="generic", version="1", security=None)
        event = Event(
            sha256="hash3",
            parent=EventParent(sha256=src.entity.sha256),  # parent is top level binary
            relationship={"decoded": "rot26"},
            info={"test": "data"},
            data=[
                EventData(hash="hash3", label=azm.DataLabel.CONTENT),
            ],
            md5="md5",
            sha512="sha512",
            size=5454,
            file_format_legacy="octets",
        )
        now = datetime.datetime(year=2010, month=1, day=1, tzinfo=datetime.timezone.utc)
        result = JobResult(state=State(), events=[event], runtime=5)
        with mock.patch("azul_runner.network_transform._get_now", lambda: now):
            res = network_transform.gen_processing_events(self.p, posted, author, src, result)
        res = dump(res)
        res["entity"].pop("input")  # dont care about this
        print(res)
        self.assertEqual(
            res,
            {
                "model_version": 5,
                "kafka_key": "runner-placeholder",
                "timestamp": "2010-01-01T00:00:00+00:00",
                "author": {"version": "1", "category": "plugin", "name": "generic"},
                "entity": {
                    "runtime": 5.0,
                    "status": "completed",
                    "results": [
                        {
                            "model_version": 5,
                            "kafka_key": "runner-placeholder",
                            "timestamp": "2010-01-01T00:00:00+00:00",
                            "source": {
                                "name": "sauce",
                                "timestamp": "2000-01-01T00:00:00+00:00",
                                "path": [
                                    {
                                        "timestamp": "2000-01-01T00:00:00+00:00",
                                        "author": {"version": "1", "category": "plugin", "name": "system"},
                                        "sha256": "hash",
                                        "action": "extracted",
                                    },
                                    {
                                        "relationship": {"decoded": "rot26"},
                                        "timestamp": "2010-01-01T00:00:00+00:00",
                                        "author": {"version": "1", "category": "plugin", "name": "generic"},
                                        "sha256": "hash3",
                                        "action": "extracted",
                                        "file_format_legacy": "octets",
                                        "size": 5454,
                                    },
                                ],
                                "references": {"keen": "eye"},
                            },
                            "author": {"version": "1", "category": "plugin", "name": "generic"},
                            "entity": {
                                "sha256": "hash3",
                                "sha512": "sha512",
                                "info": {"test": "data"},
                                "features": [{"type": "string", "value": "octets", "name": "file_format_legacy"}],
                                "md5": "md5",
                                "size": 5454,
                                "file_format_legacy": "octets",
                            },
                            "action": "extracted",
                        }
                    ],
                },
            },
        )
