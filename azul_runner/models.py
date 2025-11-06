"""Data structures used by plugins to record job results, such as features, info, data."""

from __future__ import annotations

import datetime
import io
import logging
import tempfile
import typing
from enum import Enum
from functools import total_ordering
from math import inf
from typing import Annotated, Any, ClassVar, Type

from azul_bedrock import dispatcher
from azul_bedrock import models_network as azm
from pydantic import (
    ConfigDict,
    PlainSerializer,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    WrapSerializer,
    field_validator,
)

from . import storage

logger = logging.getLogger(__name__)


class BaseModelStrict(azm.BaseModelStrict):
    """Alters bedrock base model to have no default import namespace."""

    _DEFAULT_IMPORT = None


class ModelError(TypeError):
    """Error when validating that data matches the model."""

    pass


class Filepath(str):
    """A string value that will be parsed by Elastic as a file path."""

    pass


class Uri(str):
    """A string value that will be parsed by Elastic as a URI."""

    pass


# As inheritance for enums is impossible, this is used to override repr() and str()
# to render a valid reference to a state, for ease-of-use of the testing harness.
# WARNING - This directly alters the modules azm.StatusEventEnum so side effects will occur if you
# use that directly.
_status_event_enum_override = azm.StatusEnum
_status_event_enum_override.__repr__ = lambda self: f"State.Label.{self.name}"
_status_event_enum_override.__str__ = lambda self: f"State.Label.{self.name}"


class State(BaseModelStrict):
    """Data class recording the result of a plugin run."""

    Label: ClassVar[Type[azm.StatusEnum]] = _status_event_enum_override
    label: "State.Label"  # COMPLETED, or category of reason for non-execution
    failure_name: str | None = None  # Brief title of problem, eg 'OSError' or 'opt-out'
    message: str | None = None  # Any other output messages (eg traceback for exception)

    def __init__(self, label: Label = Label.COMPLETED, failure_name: str = None, message: str = None):
        if label == State.Label.OPT_OUT:
            if not message:
                message = "No opt-out reason was provided."
        super().__init__(**dict(label=label, failure_name=failure_name, message=message))

    def __repr__(self):
        """Custom pydantic repr."""
        return azm.repr_reproduce(self, required=["label"])

    __str__ = __repr__


State.model_rebuild()

# Lookup table for converting types to type functions.
feat_type_lookup_table = {
    "integer": int,
    "float": float,
    "string": str,
    "binary": bytes,
    "datetime": datetime.datetime,
    "filepath": str,
    "uri": str,
}


@total_ordering
class Feature(BaseModelStrict):
    """Data class to hold the definition of an output feature for a plugin."""

    name: str
    desc: str
    type: str
    # accepted class for values of the feature
    _typeref: type = PrivateAttr()

    def __init__(self, name, desc, type: azm.FeatureType = azm.FeatureType.String) -> None:
        # FUTURE rename parameter 'type' to 'kind' due to shadowing builtin 'type' function
        kind = type
        if kind not in azm.FeatureType:
            orig_kind = kind
            logger.warning(f"feature {name} should use a FeatureType.<option> enum entry for type, not {orig_kind}")
            # convert legacy types to enum
            try:
                kind = {
                    int: azm.FeatureType.Integer,
                    float: azm.FeatureType.Float,
                    str: azm.FeatureType.String,
                    bytes: azm.FeatureType.Binary,
                    datetime.datetime: azm.FeatureType.Datetime,
                    Filepath: azm.FeatureType.Filepath,
                    Uri: azm.FeatureType.Uri,
                }[kind]
            except KeyError:
                raise ValueError(
                    f"feature {name} should use a FeatureType.<option> enum entry for type, not {orig_kind}"
                )

        super().__init__(**dict(name=name, desc=desc, type=kind))

        # Has to be after super()
        self._typeref = feat_type_lookup_table[kind]

    def __hash__(self):
        """Hash function."""
        return hash(self.name)

    def __eq__(self, other: Feature) -> bool:
        """Equal function."""
        return NotImplemented if not isinstance(other, Feature) else (self.name == other.name)

    def __lt__(self, other: Feature) -> bool:
        """Less than function."""
        return NotImplemented if not isinstance(other, Feature) else (self.name < other.name)

    @property
    def typeref(self):
        """Return a class reference to the features type."""
        return self._typeref


@total_ordering
class FeatureValue(BaseModelStrict):
    """Data class to store the value of a feature, possibly with a label, offset, and/or size."""

    value: Annotated[azm.VALUE_DECODED, PlainSerializer(lambda v: azm.value_encode(v), return_type=azm.VALUE_DECODED)]
    label: str | None = None
    offset: int | None = None
    size: int | None = None

    def __init__(self, value=None, **kwargs):
        # convert legacy types to strings
        if isinstance(value, (Filepath, Uri)):
            value = str(value)
            logging.warning("Filepath(x) and Uri(x) are deprecated, use string instead.")
        kwargs["value"] = value
        super().__init__(**kwargs)

    def __repr__(self):
        """Custom pydantic repr."""
        # Print 'FV' instead of 'FeatureValue' because it is so common and takes up too much space otherwise.
        return azm.repr_reproduce(self, "FV", ["value"])

    __str__ = __repr__

    @property
    def _comparison_tuple(self):
        return (
            str(type(self.value)),
            self.value,
            "" if self.label is None else self.label,
            -inf if self.offset is None else self.offset,  # 'None' sorts before all values
            -inf if self.size is None else self.size,
        )

    def __eq__(self, other: FeatureValue) -> bool:
        """Explicitly implement equality only for FeatureValue to FeatureValue instances."""
        return (
            NotImplemented
            if not isinstance(other, FeatureValue)
            else (self._comparison_tuple == other._comparison_tuple)
        )

    def __lt__(self, other: FeatureValue) -> bool:
        """Explicitly implement less than for FeatureValue to FeatureValue instances."""
        return (
            NotImplemented
            if not isinstance(other, FeatureValue)
            else (self._comparison_tuple < other._comparison_tuple)
        )

    def value_encoded(self) -> str:
        """Return encoded value."""
        return azm.value_encode(self.value)

    model_config = ConfigDict(frozen=True)


# shortcut to commonly used class
FV = FeatureValue


@total_ordering
class EventData(BaseModelStrict):
    """Binary data and metadata."""

    # Ensure enum values are made into strings
    model_config = ConfigDict(use_enum_values=True)

    hash: str
    label: azm.DataLabel
    language: str | None = None

    def __eq__(self, other: EventData) -> bool:
        """Explicitly implement equality only for EventData to EventData instances."""
        return NotImplemented if not isinstance(other, EventData) else (self.hash == other.hash)

    def __lt__(self, other: EventData) -> bool:
        """Explicitly implement less than for EventData to EventData instances."""
        return NotImplemented if not isinstance(other, EventData) else (self.hash < other.hash)


class EventBase(azm.FileInfo):
    """Simple base for partial event."""

    _DEFAULT_IMPORT = None

    parent: EventParent | None = None
    parent_sha256: str | None = None
    sha256: str
    relationship: dict = {}

    def __init__(self, *args, **kwargs):
        # Backwards compatibility mapping for old plugin unittests to avoid having to update all the old tests.
        # (Maps the old name of fields to the new name).
        renames = [
            ("entity_id", "sha256"),
            ("parent_hash", "parent_sha256"),
            ("entity_type", "model"),
            ("file_type", "file_format_legacy"),
            ("file_type_al", "file_format"),
            ("mime_type", "mime"),
            ("mime_magic", "magic"),
        ]
        did = []
        for old, new in renames:
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
                did.append((old, new))

        # model is implicitly binary so drop this
        kwargs.pop("model", None)
        if did:
            logging.warning(f"Test Event() creation should replace args: {did}")

        super().__init__(*args, **kwargs)


class EventParent(EventBase):
    """Minified event preventing the need for events to track parent events."""

    filename: str | None = None
    language: str | None = None


class Event(EventBase):
    """Details of a partial event generated by a plugin.

    Note that a lot of properties may be unset, the original job event should be used to fill in these details.
    """

    data: list[EventData] = []
    features: dict[str, list[FeatureValue]] = {}
    info: dict = {}

    # track binary data in shared dictionary
    _shared_data: dict[str, typing.BinaryIO] = PrivateAttr()
    # track events in shared list
    _events: list[Event] = PrivateAttr()
    # track children of the current event so duplicate children are not added
    _children: dict[tuple[str, str, str], Event] = PrivateAttr({})

    @field_validator("features")
    @classmethod
    def _sort_features(cls, f):
        """Sort features on creation."""
        # sort feature values
        for v in f.values():
            v.sort()

        # sort feature names
        return dict(sorted(f.items()))

    def sort(self):
        """Sort contents of event."""
        # sort feature values
        for k in self.features.keys():
            self.features[k].sort()

        # sort feature names
        self.features = dict(sorted(self.features.items()))

    def set_stores(self, binaries: dict, events: list):
        """Update stores shared between all events."""
        self._shared_data = binaries
        self._events = events

    def as_parent(self) -> EventParent:
        """Generate an EventParent instance."""
        filenames = self.features.get("filename", [])
        # must sort before getting value, for consistency
        filename = sorted(filenames)[0].value if filenames else None
        lang = [x.language for x in self.data if x.label == "content" and x.language]
        # must sort before getting value, for consistency
        lang = sorted(lang)[0] if lang else None
        raw = self.model_dump()
        for k in ["data", "features", "info"]:
            # drop extra properties
            raw.pop(k, None)
        return EventParent(
            **raw,
            filename=filename,
            language=lang,
        )

    def keep(self) -> bool:
        """True if event is worth emitting from plugin."""
        # keep all child events
        # keep top level events if features, data or info are set
        return self.parent or (self.features or self.data or self.info)

    def add_data(self, label: azm.DataLabel, tags: dict[str, str], data: bytes) -> str:
        """Add data to the event and return data hash.

        Results in memory issues if data is large.
        """
        if len(data) > 2**24:
            logger.warning(f"adding large file as bytestring, which is bad for memory usage {label=}")
        return self.add_data_file(label, tags, io.BytesIO(data))

    def add_data_file(self, label: azm.DataLabel, tags: dict[str, str], data_file: typing.BinaryIO) -> str:
        """Add data file to the event and return data hash.

        File handle may be closed afterward.
        """
        # copy data file to temp file so that caller may close file handle and/or cleanup
        # FUTURE send to dispatcher straight away as this wastes disk space and time
        #   would require a harness for local execution and testing
        wrote_bytes = 0
        tmpfile = tempfile.TemporaryFile("r+b")
        data_file.seek(0)
        while True:
            # 100kb at a time
            buf = data_file.read(100_000)
            if not buf:
                break
            tmpfile.write(buf)
            wrote_bytes += len(buf)
        tmpfile.seek(0)

        if not wrote_bytes:
            raise ValueError("tried to add data file with 0 bytes")

        data_hash = storage.calc_stream_hash(data_file)
        if data_hash in [x.hash for x in self.data]:
            return
        data_file.seek(0)
        self.data.append(EventData(hash=data_hash, label=label, language=tags.pop("language", None)))
        if tags:
            raise ValueError(f"unknown tags for data {tags}")
        self._shared_data[data_hash] = tmpfile
        return data_hash

    def add_text(self, text: str, language: str = None) -> str:
        """Add a text stream to the current binary.

        Used for a report or other textual artifact related to the current plugin run. The text will be viewable
         in the UI.

        Optionally, you can specify the 'language' property to facilitate syntax highlighting in the Azul UI.
         This should be a language name supported by prism.js - eg, 'html', 'js'/'javascript', 'bash'/'shell',
         'c', 'dotnet', 'php', 'go', 'powershell', 'python', 'regex', 'vb'/'vba'
        """
        return self.add_data(azm.DataLabel.TEXT, {"language": language} if language else {}, text.encode("utf-8"))

    def add_feature_values(self, key: str, values: Any):
        """Add feature values to plugin result."""
        if not isinstance(values, (list, set)):
            values = [values]
        elif len(values) == 0:
            # Ignore empty lists and sets for feature values.
            logger.warning(f"ignoring empty list for feature {key}")
            return

        # normalise types
        values_fixed = []
        for value in values:
            if value is None:
                # ignore None values, likely the result of something like {}.get("myvalue"), which is common pattern
                logger.warning(f"ignoring None value for feature {key}")
                continue
            if not isinstance(value, FeatureValue):
                value = FeatureValue(value)
            values_fixed.append(value)
        self.features.setdefault(key, []).extend(values_fixed)

    def add_many_feature_values(self, features: dict[str, Any]):
        """Add a dictionary of feature values to the main event."""
        for k, v in features.items():
            self.add_feature_values(k, v)

    def add_info(self, info: dict):
        """Add info to event, overwriting previous info."""
        self.info = info

    def _add_child(self, sha256: str, relationship: dict, *, parent_sha256: str = None) -> Event:
        """Add child event to current event."""
        uniq = (parent_sha256, sha256)
        if uniq not in self._children:
            ret = Event(
                parent=self.as_parent(),
                parent_sha256=parent_sha256,
                sha256=sha256,
                relationship=relationship,
            )
            ret.set_stores(self._shared_data, self._events)
            self._events.append(ret)
            self._children[uniq] = ret
        return self._children[uniq]

    def add_child_with_data(self, relationship: dict, data: bytes, *, label: azm.DataLabel = azm.DataLabel.CONTENT):
        """Add child of current event using data hash as sha256.

        Results in memory issues if data is large.
        """
        return self.add_child_with_data_file(relationship, io.BytesIO(data), label=label)

    def add_child_with_data_file(
        self, relationship: dict, data_file: typing.BinaryIO, *, label: azm.DataLabel = azm.DataLabel.CONTENT
    ):
        """Add child of current event using data hash as sha256."""
        # data hash is child id
        data_hash = storage.calc_stream_hash(data_file)
        c = self._add_child(data_hash, relationship)
        # note - calculates hash again (inefficient)
        c.add_data_file(label, {}, data_file)
        return c


def custom_binary_serialize(data: dict[str, typing.BinaryIO | bytes], orig_serializer: SerializerFunctionWrapHandler):
    """Reads binary data before serialization to prevent data loss."""
    result = dict()
    for row in data:
        if issubclass(type(data[row]), io.IOBase):
            # FUTURE will overflow memory if large
            result[row] = data[row].read()
            data[row].seek(0)
        else:
            result[row] = data[row]
    return orig_serializer(result)


class JobResult(BaseModelStrict):
    """Contains results of a run of a plugin."""

    state: State
    events: list[Event] = []
    # data values are only 'bytes' during testing
    data: Annotated[dict[str, typing.BinaryIO | bytes], WrapSerializer(custom_binary_serialize)] = {}
    feature_types: list[Feature] = []
    runtime: int | None = None
    date_start: datetime.datetime | None = None
    date_end: datetime.datetime | None = None

    @property
    def main(self) -> Event:
        """Return main event."""
        return self.events[0] if len(self.events) else None

    def close(self):
        """Close any JobResult data handles."""
        for d in self.data:
            if issubclass(type(d), io.IOBase) and not self.data[d].closed:
                self.data[d].close()

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Job(BaseModelStrict):
    """Contains event and data streams for plugin to process."""

    # event for the job
    event: azm.BinaryEvent

    @property
    def id(self) -> str:
        """Entity id."""
        return self.event.entity.sha256

    # Input content streams as file-like objects
    _streams: list[storage.StorageProxyFile] = PrivateAttr(None)

    # helpers
    def load_streams(self, *, dp: dispatcher.DispatcherAPI = None, local: list[storage.StorageProxyFile] = None):
        """Load streams from network or local files."""
        self._streams = []
        if local:
            self._streams = local
        elif dp:
            # initialise data streams
            if not self.event.entity.datastreams:
                return
            for stream in self.event.entity.datastreams:
                self._streams.append(
                    storage.StorageProxyFile(
                        source=self.event.source.name,
                        label=stream.label,
                        hash=stream.sha256,
                        dp=dp,
                        allow_unbounded_read=True,
                        file_info=stream,
                    )
                )

    def get_data(self, label: azm.DataLabel = azm.DataLabel.CONTENT) -> storage.StorageProxyFile:
        """Return and ensure only one stream with label."""
        streams = self.get_all_data(label)
        if len(streams) > 1:
            raise ValueError(f'more than one "{label}" stream for entity {self.id}')
        return streams[0] if streams else None

    def get_all_data(
        self, label: azm.DataLabel = None, file_format: str = None, file_format_legacy: str = None
    ) -> list[storage.StorageProxyFile]:
        """Return data streams, optionally filtered by label and/or file_format."""
        if not self._streams:
            return []
        streams = self._streams
        if label:
            streams = [ds for ds in streams if ds.file_info.label == label]
        if file_format:
            streams = [ds for ds in streams if ds.file_info.file_format == file_format]
        if file_format_legacy:
            streams = [ds for ds in streams if ds.file_info.file_format_legacy == file_format_legacy]
        return streams


class TaskExitCodeEnum(Enum):
    """Exit code enum."""

    COMPLETED = 0  # Process has completed nicely (probably job_limit reached).
    TERMINATE = 10  # An error has occurred that isn't fixing and the process should fully exit.
    RECREATE_PLUGIN = 20  # Process needs to be re-created (probably a git change detected).


class TaskModel(BaseModelStrict):
    """Model that holds the current BinaryEvent being processed by a plugin."""

    in_event: azm.BinaryEvent
    # Time since start of epoch (e.g output of time.time())
    start_time_epoch: float
    multi_plugin_name: str | None = None
