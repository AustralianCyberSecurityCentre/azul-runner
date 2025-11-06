"""CLI based execution of plugins for local testing or remote deployment."""

import argparse
import base64
import datetime
import hashlib
import io
import json
import logging
import os
import struct
import sys
from io import BytesIO
from typing import Any, BinaryIO, Type

import cart
from azul_bedrock import identify
from azul_bedrock import models_network as azm
from pydantic import BaseModel

from azul_runner.log_setup import setup_logger

from . import local, settings
from .models import JobResult
from .monitor import Monitor
from .plugin import Plugin, State
from .storage import StorageProxyFile

logger = logging.getLogger(__name__)
MANDATORY_CART_HEADER_LEN = struct.calcsize(cart.MANDATORY_HEADER_FMT)


class Args(BaseModel):
    """Hold command line arguments."""

    config: list[tuple[str, str]] = []
    entity_id: str = ""
    files: str = ""
    job_limit: int | None = None
    output_folder: str = ""
    output_json: bool = False
    quiet: bool = False
    server: str = ""
    stream: Any = None
    verbose: int = 0


def cmdline_run(plugin: type[Plugin] = None):
    """Run from command-line."""
    if not plugin:
        print("Error: not run with a plugin", file=sys.stderr)
        exit(1)
    return execute(plugin, parse_args())


def args_to_config(args: Args) -> dict[str, Any]:
    """Convert input arguments into co-ordinator config settings."""
    config = {}
    if args.server:
        config["events_url"] = args.server
        config["data_url"] = args.server
    if args.config:
        # Update with `-c NAME VALUE` args
        config.update({n: v for n, v in args.config})
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run an Azul plugin.")
    parser.add_argument(
        "-c",
        "--config",
        nargs=2,
        metavar=("NAME", "VALUE"),
        action="append",
        help="Provides config values for the plugin (and overrides values in config files). "
        "Only simple values are supported. Can be used multiple times.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Gives more verbose logging (pass once for info, twice for debug)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress warnings (log errors only). Overrides -v."
    )
    parser.add_argument(
        "--server",
        help="Fetch and execute jobs from the specified server. "
        "This is mutually exclusive with any of the local options below.",
        default=None,
    )
    parser.add_argument(
        "--job-limit",
        type=int,
        metavar="NUM",
        help="Exit after processing this many jobs. Only meaningful when connecting to a server.",
    )
    parser.add_argument(
        "-e",
        "--entity-id",
        default=None,
        help="Specify an entity-id for the entity to be processed. "
        "Most plugins should not require this to be explicitly set.",
    )
    parser.add_argument(
        "-o", "--output-json", action="store_true", help="Print JSON instead of human-readable summary."
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Write generated streams to this folder. Warning - Files are not neutered.",
    )
    parser.add_argument(
        "-s",
        "--stream",
        nargs=2,
        metavar=("TYPE", "FILENAME"),
        action="append",
        help="Read an input data stream of TYPE from the given FILENAME. Can be used multiple times.",
    )
    parser.add_argument(
        "files",
        nargs="?",
        help="Run plugin over file/folder recursively. Use - for stdin. Files will be loaded as 'content' streams.",
    )
    raw_args = parser.parse_args()

    # parse args
    setted = {}
    for k, v in raw_args.__dict__.items():
        if v is not None:
            setted[k] = v
    args = Args(**setted)

    # access log level from environment variable
    s = settings.RunnerSettings()

    # Log level settings
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose > 1:
        log_level = logging.DEBUG
    elif args.verbose > 0:
        log_level = logging.INFO
    else:
        # default to environment setting
        # logging library is incompatible between python versions 3.10 and 3.12 so this is being manually mapped.
        _nameToLevel = {
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
            "ERROR": logging.ERROR,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
        # fall back to warning level
        log_level = _nameToLevel.get(s.log_level.upper(), logging.WARNING)

    # Setup Logger
    setup_logger(log_level)

    # print runtime options
    logger.info("command line run options:")
    for k, v in args.model_dump().items():
        logger.info(f"{k:20}: {v}")

    return args


def execute(plugin_class: type[Plugin], args: Args):
    """Execute the plugin with the provided arguments."""
    config = args_to_config(args)
    if not any((args.stream, args.files)):
        # FUTURE SIGINT handler for ^C to exit
        m_loop = Monitor(plugin_class, config)
        m_loop.run_loop()
    else:
        # local execution
        # Input / output JSON
        m_loop = Monitor(plugin_class, config)

        if not args.stream and not args.files:
            print("Error: No files/streams specified", file=sys.stderr)
            exit(1)

        if args.files and os.path.isdir(args.files):
            for path, _dir, files in os.walk(args.files):
                for file in sorted(files):
                    path_file = os.path.join(path, file)
                    print(path_file)
                    process_file(plugin_class, m_loop, path_file, args)
        else:
            process_file(plugin_class, m_loop, args.files, args)


def process_file(plugin_class: Type[Plugin], m_loop: Monitor, filepath, args: Args):
    """Run plugin once over streams."""
    sha256 = args.entity_id

    ent_sha512 = None
    ent_sha1 = None
    ent_md5 = None
    file_format_legacy = None
    file_format = None
    file_extension = None
    size = None

    # Read files in
    spf_in: list[StorageProxyFile] = []
    fileinfo_in = []
    if filepath:
        if filepath == "-":
            # open stdin for reading
            data = sys.stdin.buffer.read()
        else:
            with open(filepath, "rb") as f:
                data = unpack_cart_and_read(f)
        if not sha256:
            sha256 = hashlib.sha256(data).hexdigest()
        ent_sha512 = hashlib.sha512(data).hexdigest()
        ent_sha1 = hashlib.sha1(data).hexdigest()  # nosec B324
        ent_md5 = hashlib.md5(data).hexdigest()  # nosec B324
        _, _, file_format, file_format_legacy, file_extension = identify.from_buffer(data)

        size = len(data)

        if len(data) == 0:
            logger.warning("Input content has zero length; did you mean to do this?")
        fileinfo = local.gen_api_content(io.BytesIO(data), label=azm.DataLabel.CONTENT)
        fileinfo_in.append(fileinfo)
        spf_in.append(
            StorageProxyFile(
                source="local",
                label=azm.DataLabel.CONTENT,
                hash="<content>",
                init_data=data,
                file_info=fileinfo,
                allow_unbounded_read=True,
            )
        )

    # Data streams
    for stype, sfile in args.stream or []:
        if stype == azm.DataLabel.CONTENT:
            raise Exception("use filepath for 'content' type, only one content binary is allowed")

        with open(sfile, "rb") as f:
            data = f.read()

        if len(data) == 0:
            logger.warning("Input stream (%s, %s) has zero length; did you mean to do this?" % (stype, sfile))
        fileinfo = local.gen_api_content(io.BytesIO(data), label=stype)
        fileinfo_in.append(fileinfo)
        spf_in.append(
            StorageProxyFile(
                source="local",
                label=stype,
                hash=sfile,
                init_data=data,
                file_info=fileinfo,
                allow_unbounded_read=True,
            )
        )

    # Use commandline provided file to calculate sha256 of file.
    if not sha256:
        sha256 = "cmdline_entity"

    # Validate input streams against plugin expectations
    try:
        local.validate_streams(spf_in, m_loop._plugin.cfg)
    except AssertionError as e:
        logger.warning("Input streams do not match plugin requirements (%s) - plugin may not run correctly" % e.args)

    entity = azm.BinaryEvent.Entity(
        size=size,
        datastreams=fileinfo_in,
        file_format_legacy=file_format_legacy,
        file_format=file_format,
        file_extension=file_extension,
        sha256=sha256,
        sha512=ent_sha512,
        sha1=ent_sha1,
        md5=ent_md5,
    )
    event = local.gen_event(entity)

    # Do execution
    res = m_loop.run_once(event, spf_in)

    [s.close() for s in spf_in if not s.closed]
    logger.info("Processing complete")
    if args.output_json:
        for result in res.values():
            # stick with jsonlines output in case of multiple results
            try:
                print(generate_json(result))
            except Exception as e:
                raise Exception(result) from e
    else:
        for subplugin, result in res.items():
            print_result(plugin_class, subplugin, result)

    # save plugin results to specific folder
    # not encoded, so raw malware may be written to disk
    if args.output_folder:
        try:
            os.mkdir(args.output_folder)
        except FileExistsError:
            pass

        # save all streams (may duplicate files)
        streams = []
        for jresult in res.values():
            for event in jresult.events:
                for edata in event.data:
                    streams.append((edata.label, edata.hash, jresult.data[edata.hash]))
        for stream in streams:
            with open(os.path.join(args.output_folder, f"{stream[1]}_{stream[0]}.data"), "wb") as f:
                f.write(stream[2].read())
                stream[2].seek(0)


def generate_json(result: JobResult, indent=None):
    """Return the supplied JobResult as json string."""
    result = result.model_dump(exclude_defaults=True)

    # Dump the JSON
    def object_to_json(obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode("ascii")
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        else:
            raise TypeError(f"Unimplemented JSON encode called for {type(obj)} object: \n{obj}")

    # stick with jsonlines output in case of multiple results
    return json.dumps(result, default=object_to_json, indent=indent)


def print_result(plugin_class: Type[Plugin], subplugin: str, result: JobResult):
    """Print result of plugin to stdout."""
    # only mention multiplugin if one is present
    if subplugin and subplugin is not None:
        print("----- %s-%s results -----" % (plugin_class.__name__, subplugin))
    else:
        print("----- %s results -----" % plugin_class.__name__)

    if getattr(plugin_class, "SECURITY", False):
        print("** SECURITY: %s **" % plugin_class.SECURITY)
    status: State = result.state
    if status.failure_name:
        print("%s (%s)" % (status.label.name, status.failure_name))
    else:
        print(status.label.name)
    if status.message:
        print(status.message)
    print()

    if result.events:
        print(f"events ({len(result.events)})")
        print()
        for event in result.events:
            # print event info basic
            if not event.sha256 and not event.parent_sha256:
                print("event for primary")
            elif event.parent_sha256:
                print(f"event for data {event.parent_sha256}")
            else:
                print(f"event for {event.sha256}:{event.parent_sha256}")
            print(f"  {event.relationship}")
            if (p := event.parent) is not None:
                if not p.sha256:
                    print("")
                else:
                    print(f"  child of {p.sha256}")
            # print child info
            if event.data:
                print(f"  output data streams ({len(event.data)}):")
                for data in event.data:
                    bin = result.data[data.hash]
                    # get size of result file
                    bin.seek(0, 2)
                    size = bin.tell()
                    bin.seek(0)
                    print(f"    {size} bytes - {data}")
                    bin.seek(0)
            if event.features:
                print("  output features:")
                fmt = "    %" + str(max([len(f) for f in event.features])) + "s: %s"
                fmt2 = " " * max([len(f) for f in event.features]) + "      %s"
                for fn, fvs in event.features.items():
                    for i, fv in enumerate(sorted(fvs)):
                        # Build value sting: <label> - <value> @ <offset> (offset)
                        values_str = f"{fv.value}"
                        if fv.label:
                            values_str = f"{fv.label} - " + values_str
                        if fv.offset:
                            values_str = values_str + f" @ {hex(fv.offset)} (offset)"

                        if not i:
                            print(fmt % (fn, values_str))
                        else:
                            print(fmt2 % values_str)
            if event.info:
                print("  info:")
                for k, v in event.info.items():
                    print("    %s: %s" % (k, v))
            print()

    if result.feature_types:
        print("Feature key:")
        for f in result.feature_types:
            print("  %s:  %s" % (f.name, f.desc))
        print()


def unpack_cart_and_read(input_stream: BinaryIO) -> bytes:
    """Conditionally unpacks CaRT files and reads bytes from provided filepath."""
    # Ensure read from start of the file
    input_stream.seek(0)
    header = input_stream.read(MANDATORY_CART_HEADER_LEN)
    # Reset the read head, incase it is not a CaRT file, the function can just return
    input_stream.seek(0)
    if cart.is_cart(header):
        unpacked = BytesIO()
        cart.unpack_stream(input_stream, unpacked)
        unpacked.seek(0)
        return unpacked.getvalue()
    else:
        return input_stream.read()


if __name__ == "__main__":
    cmdline_run()
