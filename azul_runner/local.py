"""Local event generation."""

import datetime
import hashlib
import typing
import unittest

from azul_bedrock import identify
from azul_bedrock import models_network as azm

from . import storage
from .settings import Settings
from .storage import StorageProxyFile

unittest.util._MAX_LENGTH = 2000


def validate_streams(datastreams: list[StorageProxyFile], plugin_settings: Settings):
    """Verify that a set of data streams matches input reqs specified by plugin, or raise AssertionError."""
    # calculate what data the job provided in same format as INPUT_DATA
    available: dict[str, set[str]] = {}
    magics = set()
    mimes = set()

    for ds in datastreams:
        types = ds.file_info.file_format.split(";")[0]
        available.setdefault(ds.file_info.label, set()).add(types)
        available.setdefault("*", set()).add(types)
        magics.add(ds.file_info.magic)
        mimes.add(ds.file_info.mime)

    # check INPUT_DATA against job provided
    is_matching_type = False
    for content_label, plugin_allowed_formats in plugin_settings.filter_data_types.items():
        if content_label not in available:
            break
        # if no required type is available
        if plugin_allowed_formats:
            for allowed_type in available[content_label]:
                for allowed_format in plugin_allowed_formats:
                    if allowed_type.startswith(allowed_format):
                        is_matching_type = True
                        break
        # Type found
        if is_matching_type:
            return

    if any(len(list_of_values) > 0 for list_of_values in plugin_settings.filter_data_types.values()):
        raise AssertionError(
            f"Provided file format {available} does not meet plugin requirements {plugin_settings.filter_data_types}\n"
            f"magic: {sorted(magics)}\n"
            f"mime types: {sorted(mimes)}"
        )


def gen_api_content(binaryio: typing.BinaryIO, label: azm.DataLabel = azm.DataLabel.CONTENT):
    """Generate api content from bytes."""
    # get last offset and assume that is correct size
    binaryio.seek(0, 2)
    size = binaryio.tell()
    binaryio.seek(0)

    magic, mime, file_format, file_format_legacy, file_extension = identify.from_buffer(binaryio.read())
    binaryio.seek(0)
    return azm.Datastream(
        label=label,
        size=size,
        md5=storage.calc_stream_hash(binaryio, hashlib.md5),
        sha1=storage.calc_stream_hash(binaryio, hashlib.sha1),
        sha256=storage.calc_stream_hash(binaryio, hashlib.sha256),
        sha512=storage.calc_stream_hash(binaryio, hashlib.sha512),
        mime=mime,
        magic=magic,
        file_format_legacy=file_format_legacy,
        file_format=file_format,
        file_extension=file_extension,
    )


def gen_event(entity: azm.BinaryEvent.Entity):
    """Generate a simple network event."""
    return azm.BinaryEvent(
        model_version=azm.CURRENT_MODEL_VERSION,
        kafka_key="test",
        dequeued="test-response",
        action=azm.BinaryAction.Extracted,
        timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
        source=azm.Source(
            name="source",
            path=[
                azm.PathNode(
                    author=azm.Author(name="TestServer", category="plugin"),
                    action=azm.BinaryAction.Extracted,
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                    sha256=entity.sha256,
                ),
            ],
            timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
        ),
        author=azm.Author(name="TestServer", category="plugin"),
        entity=entity,
    )
