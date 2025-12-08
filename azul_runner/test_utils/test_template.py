"""Contains a template test case class to simplify testing of Azul plugins.

Usage:
  from azul_runner import test_template
  from plugins_package.my_plugin import MyPlugin

  class TestMyPlugin(test_template.TestPlugin):
      PLUGIN_TO_TEST = MyPlugin

      def test_foo_case(self):
          result = self.do_execution(feats_in={...}, files_in=[('stream-name', '<filename>'), ...])
          self.assertJobResult(result, JobResult(...))
          ...
"""

import datetime
import importlib
import io
import os
import pathlib
import struct
import tempfile
import typing
import unittest
import warnings
from sys import gettrace
from typing import Any, ClassVar, Iterable, Optional, Type
from unittest import mock

import black
import cart
from azul_bedrock import dispatcher
from azul_bedrock import exceptions as azbe
from azul_bedrock import models_network as azm
from azul_bedrock.test_utils import file_manager

from azul_runner import coordinator, settings
from azul_runner.coordinator import Coordinator

from .. import local, monitor
from ..models import EventBase, Feature, FeatureValue, JobResult
from ..plugin import Plugin
from ..storage import DATA_HASH, StorageProxyFile

unittest.util._MAX_LENGTH = 2000
MANDATORY_CART_HEADER_LEN = struct.calcsize(cart.MANDATORY_HEADER_FMT)


class TestPlugin(unittest.TestCase):
    """Parent class to simplify testing of concrete plugins.

    Handles setting up and execution of plugin using a provided sample file/data bytes
    and input features.

    Usage:
        class TestFooPlugin(TestPlugin):
            PLUGIN_TO_TEST = FooPlugin

            def test_normal_execution(self):
                result = self.do_execution(
                    feats_in = {
                        'feature1': [val1, val2, val3, ...],
                        'feature2': [
                            FeatureValue('blah', tag='foo', ...),
                            ...
                        ]
                    },
                    file_in = [('content', 'path/to/file'), ... ],
                    data_in = [('stream_label', b'<bytes>'), ...],
                )
                ... Make assertions about results ...
    """

    PLUGIN_TO_TEST: ClassVar[Type[Plugin]]  # Override in subclass
    PLUGIN_TO_TEST_CONFIG: ClassVar[dict] = {}  # Override in subclass where necessary
    maxDiff = None

    def without_data(self, result: JobResult):
        """Deprecated. Return copy of result with cleared data content strings.

        Use self.assertJobResult(x, y) instead.
        """
        warnings.warn("deprecated call to without_data - this does nothing.", DeprecationWarning, stacklevel=2)
        return result

    def _strip_hash(self, result: JobResult):
        """Replace data blob hashes and child ids with numbers.

        Some libraries may produce inconsistent output data so this allows for unit tests
        to still succeed on an equality check.

        Crucially, this allows for checking equality of metadata the plugin generated,

        Usecase: In a test, a generated png file may contain a date resulting in a new
        hash every time the test is run.
        """
        result = result.model_copy()
        replaces = {}
        for i, k in list(enumerate(result.data)):
            replaces[k] = str(i)
            result.data[str(i)] = result.data.pop(k)

        def _nested(e: EventBase):
            """Replace data hash wherever it is found."""
            if e.parent:
                # parents have nested parents which need to be processed
                _nested(e.parent)
            if e.parent_sha256 in replaces:
                e.parent_sha256 = replaces[e.parent_sha256]
            if e.sha256 in replaces:
                if e.md5:
                    e.md5 = replaces[e.sha256]
                if e.sha512:
                    e.sha512 = replaces[e.sha256]
                if e.sha1:
                    e.sha1 = replaces[e.sha256]
                e.sha256 = replaces[e.sha256]

        for e in result.events:
            _nested(e)
            # parent entities don't have a data block
            for d in e.data:
                if d.hash in replaces:
                    d.hash = replaces[d.hash]

        return result

    @classmethod
    def setUpClass(cls) -> None:
        """Unittest method."""
        if cls is TestPlugin:
            raise unittest.SkipTest("Template class for plugin tests cannot be run directly")
        # Create a temporary directory for storing results that are transferred by monitor back to the parent process.
        cls.test_multi_process_temp_dir = tempfile.TemporaryDirectory(
            prefix="test_multi_process_temp_dir", ignore_cleanup_errors=True
        )
        cls._file_manager = file_manager.FileManager()

    @classmethod
    def tearDownClass(cls) -> None:
        """Teardown the multiprocess test directory."""
        # Cleanup the temporary directory used for transferring data from child process.
        cls.test_multi_process_temp_dir.cleanup()

        return super().tearDownClass()

    def setUp(self) -> None:
        """Unittest method."""
        # Tried to do this with just a setUpClass method, but for some reason unittest doesn't seem to run
        #  setUpClass when the class is imported to another file... but still tries to run the tests.
        if self.__class__ is TestPlugin:
            raise unittest.SkipTest("Template class for plugin tests cannot be run directly")

        self._results = []

    def tearDown(self) -> None:
        """Clean up resources."""
        # try to close handles
        for res in self._results:
            res.close()

    def load_test_file_bytes(self, sha256: str, description: str) -> bytes:
        """Download a file from a cache or virustotal and provide the raw bytes.

        Description is intended to give a description of what the file you are downloading is.
        This is added to the cache and useful for inline documentation about the file.
        """
        return self._file_manager.download_file_bytes(sha256)

    def load_test_file_path(self, sha256: str, description: str) -> pathlib.Path:
        """Download a file from a cache or virustotal and provide the path to a copy in temp.

        Description is intended to give a description of what the file you are downloading is.
        This is added to the cache and useful for inline documentation about the file.
        """
        return self._file_manager.download_file_path(sha256)

    def _get_location(self) -> str:
        """Return path to child class that implements this class."""
        # import child module
        module = type(self).__module__
        i = importlib.import_module(module)
        # get location to child module
        return i.__file__

    def _get_data(self, sub1: str, sub2: str = None) -> typing.BinaryIO:
        """Return expected path to test data."""
        # get folder where test is
        folder = os.path.split(self._get_location())[0]
        path = os.path.join(sub1, sub2) if sub2 else sub1
        path = os.path.join(folder, "data", path)
        return open(path, "rb")

    def load_local_raw(
        self,
        path: str,
        subpath: str = None,
        *,
        description: str,
    ) -> bytes:
        """Return contents of a local file that is known to be good from the data directory.

        Description is intended to give a description of what the file you are loading is.
        This is added to the cache and useful for inline documentation about the file.
        """
        with self._get_data(path, subpath) as istream:
            data = istream.read()
        return data

    def _unpack_cart_and_read(self, input_stream: typing.BinaryIO) -> bytes:
        """Conditionally unpacks CaRT files and reads bytes from provided filepath."""
        # Ensure read from start of the file
        input_stream.seek(0)
        header = input_stream.read(MANDATORY_CART_HEADER_LEN)
        # Reset the read head, incase it is not a CaRT file, the function can just return
        input_stream.seek(0)
        if cart.is_cart(header):
            unpacked = io.BytesIO()
            cart.unpack_stream(input_stream, unpacked)
            unpacked.seek(0)
            return unpacked.getvalue()
        else:
            return input_stream.read()

    def load_cart(self, path: str, subpath: str = None, *, description: str) -> bytes:
        """Unwrap and return a cart file from the data directory.

        Description is intended to give a description of what the file you are loading is.
        This is added to the cache and useful for inline documentation about the file.
        """
        with self._get_data(path, subpath) as istream:
            data = self._unpack_cart_and_read(istream)
        return data

    def assertJobResultsDict(
        self,
        actual: dict[str, JobResult],
        expected: dict[str, JobResult],
        *,
        inspect_data: bool = False,
        strip_hash: bool = False,
    ):
        """Compare two dictionaries of JobResult objects for compatibility.

        Will strip out data blobs as they are large and sha256 comparison should
        meet the needs of most testing. Deeper testing will require manual inspection
        of the blobs in the job result or call assertFormatted instead.

        If strip_hash is set, 'in1' data stream hashes will be replaced with a number.
        This is useful when libraries generate inconsistent output due to versioning
        if you don't care for exact matches.
        """
        actual, expected = self._baseAssertJobResult(
            actual, expected, inspect_data=inspect_data, strip_hash=strip_hash
        )
        self.assertReprEqual(actual, expected)

    def assertJobResult(
        self, actual: JobResult, expected: JobResult, *, inspect_data: bool = False, strip_hash: bool = False
    ):
        """Compare two JobResult objects for compatibility.

        Will strip out data blobs as they are large and sha256 comparison should
        meet the needs of most testing. Deeper testing will require manual inspection
        of the blobs in the job result or call assertFormatted instead.

        If strip_hash is set, 'in1' data stream hashes will be replaced with a number.
        This is useful when libraries generate inconsistent output due to versioning
        if you don't care for exact matches.
        """
        actual, expected = self._baseAssertJobResult(
            {None: actual}, {None: expected}, inspect_data=inspect_data, strip_hash=strip_hash
        )
        self.assertReprEqual(actual[None], expected[None])

    def _baseAssertJobResult(
        self,
        actual_dict: dict[str, JobResult],
        expected_dict: dict[str, JobResult],
        *,
        inspect_data: bool = False,
        strip_hash: bool = False,
    ) -> dict[str, JobResult, dict[str, JobResult]]:
        # must not modify data block of original JobResult
        for actual_key in actual_dict:
            actual_dict[actual_key] = actual_dict[actual_key].model_copy()
            # process bytesio, in either case do not close these handles
            data2 = {}
            if inspect_data:
                # convert data to bytes
                for k, v in actual_dict[actual_key].data.items():
                    v.seek(0, 2)
                    size = v.tell()
                    if size > 10_000:
                        raise Exception(f"result data {k=} is too large to inspect {size=}")
                    v.seek(0)
                    data2[k] = v.read()
                    # reset pointer in case another thing wants to read the file
                    v.seek(0)
            else:
                # strip data
                for k in actual_dict[actual_key].data:
                    data2[k] = b""
            actual_dict[actual_key].data = data2

            if strip_hash:
                actual_dict[actual_key] = self._strip_hash(actual_dict[actual_key])

        # Normalise both data sources for simpler comparison.
        # Converts all the expected outputs feature values to their encoded equivalent and sorts them again.
        # Needed as supplied data could be something like a raw datetime or integer and it needs to test consistently.
        sources = [expected_dict, actual_dict]
        for source in sources:
            for expected in source.values():
                for evnt in expected.events:
                    for feat_val_list in evnt.features.values():
                        for i, feat_val in enumerate(feat_val_list):
                            feat_val_list[i] = FeatureValue(
                                feat_val.value_encoded(),
                                label=feat_val.label,
                                offset=feat_val.offset,
                                size=feat_val.size,
                            )
                        # ensure consistent sorting
                        feat_val_list.sort(key=lambda x: x.value)

        return actual_dict, expected_dict

    def assertReprEqual(self, actual, expected):
        """Format input repr() as multiline strings so unittest diff works on objects.

        This requires the input repr() to be valid python code and is intended for use with
        model Run, Event, FeatureValue, etc.
        """
        try:
            self.assertEqual(actual, expected)
        except AssertionError:
            print(
                "The below printout can form the base of your test, "
                "but you must double check the output is as expected.\n"
            )
            print(repr(actual))
            print("")
            # use black to render the difference between the two
            # exploiting the fact that the repr() is valid python code
            string1 = black.format_str(repr(actual), mode=black.Mode())
            string2 = black.format_str(repr(expected), mode=black.Mode())
            self.assertEqual(string1, string2)

    def test_00_testplugin_set(self):
        """Assert that PLUGIN_TO_TEST is set to something."""
        self.assertTrue(hasattr(self, "PLUGIN_TO_TEST"))
        self.assertIsNotNone(self.PLUGIN_TO_TEST)

    def test_00_execute_present(self):
        """Check that the execute() method is present."""
        self.assertTrue(hasattr(self.PLUGIN_TO_TEST, "execute"))
        self.assertTrue(callable(self.PLUGIN_TO_TEST.execute))

    def test_00_registration_info(self):
        """Check that the required plugin metadata (registration info) is present."""
        found_features: set[str] = set()  # Record of feature names seen, to track duplicates

        def make_feature_assertions(f: Feature):
            self.assertIsInstance(f, Feature)
            self.assertIsInstance(f.name, str)
            self.assertNotEqual(f.name, "")
            self.assertNotIn(f.name, found_features, "Duplicate feature defined: %s" % f.name)
            found_features.add(f.name)
            self.assertIsInstance(f.desc, str)
            self.assertNotEqual(f.desc, "")

        for a in ("VERSION", "FEATURES"):
            self.assertTrue(hasattr(self.PLUGIN_TO_TEST, a), "Plugin requires %s attribute" % a)

        try:
            # Use object getattribute for this one to bypass the parent-class accumulation in Plugin's getattr
            output_feats = object.__getattribute__(self.PLUGIN_TO_TEST, "FEATURES")
        except AttributeError:
            # If the plugin doesn't define FEATURES, ensure that one of its parents does, then
            #  set output_feats to an empty list (the parents output features don't need to be validated.)
            self.assertTrue(
                self.PLUGIN_TO_TEST.FEATURES,
                "Neither %s not its parents defines FEATURES!" % self.PLUGIN_TO_TEST,
            )
            output_feats = []

        plugin = self.PLUGIN_TO_TEST(self.PLUGIN_TO_TEST_CONFIG)
        req_content = plugin.cfg.filter_data_types

        # SECURITY_LEVEL doesn't have to be present, but if it is it must be None or a string
        self.assertIsInstance(getattr(self.PLUGIN_TO_TEST, "SECURITY_LEVEL", None), (type(None), str, list, dict))

        self.assertIsInstance(self.PLUGIN_TO_TEST.VERSION, str)

        self.assertIsInstance(req_content, dict)
        for k, items in req_content.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(items, list)
            for i in items:
                self.assertIsInstance(i, str)

        self.assertIsInstance(output_feats, Iterable)

        for item in output_feats:
            make_feature_assertions(item)

    def do_execution_multi(self, *args, **kwargs) -> dict[str, JobResult]:
        """Typing helper."""
        return self.do_execution(*args, **kwargs)

    def do_execution(
        self,
        *,
        feats_in: Optional[list[azm.FeatureValue]] = None,
        files_in: Optional[list[tuple[str, str]]] = None,
        data_in: Optional[list[tuple[str, bytes]]] = None,
        ent_id: Optional[str] = "test_entity",
        config: Optional[dict[str, dict[str, Any]]] = None,
        keep_times: Optional[bool] = False,
        keep_feature_types: Optional[bool] = False,
        verify_input_content: Optional[bool] = True,
        plugin_class: Optional[Type[Plugin]] = None,
        entity_attrs: Optional[dict] = None,
        submission_settings: Optional[dict] = None,
        provided_coordinator: Optional[Coordinator] = None,
        no_multiprocessing: Optional[bool] = False,
        check_consistent_augmented_stream: bool = True,
    ) -> JobResult:
        """Calls the plugin's execute method and returns the results.

        If plugin is specified, that plugin is used instead.

        Uses either the given input features and either a (local) input file or a specified byte sequence.

        `files_in` and `data_in` may both be None if the plugin doesn't care about the input entity content.
        If present, they should contain tuples of ('stream_type', ['filename'|b'content']) respectively.

        If `ent_id` is specified, it will be passed to the plugin, otherwise a hash of the data is used.

        If `config` is specified, it is passed directly to the plugin init.

        If `keep_times` is True, the results include date_start, date_end and runtime; otherwise, these are removed.

        If `return_plugin_instance` is True, result['_plugin_instance'] will be set to the plugin that just ran,
        so the caller can make assertions about the plugin state. Default is to not return _plugin_instance.

        If `verify_input_content` is True (the default), an assertion will be raised if the input test content
        does not match the plugin's data restriction.

        `entity_attrs`: Optionally specify additional attrs for the entity model.

        `provided_coordinator`: provide a coordinator instance instead of the default monitor which tracks memory,
        and timeouts and runs in a subprocess. (doesn't apply config if provided!)
        This is typically used in mocking situations where you need more state out of the plugin then you can get from
        the default running method, because it's occurring in a subprocess.
        NOTE - use as a last resort!

        `no_multiprocessing`: bypass running as a multiprocess, this should be used sparingly and is useful
        when mocking restapi's with libraries like respx.

        `check_consistent_augmented_stream`: re-runs the plugin if streams (augmented or child) are produced.
        This is to ensure a stable interface but disabling this will half the run time for those tests.

        """
        if feats_in is None:
            feats_in = []
        if files_in is None:
            files_in = []
        if data_in is None:
            data_in = []
        if config is None:
            config = self.PLUGIN_TO_TEST_CONFIG
        if entity_attrs is None:
            entity_attrs = {}
        if plugin_class is None:
            plugin_class = self.PLUGIN_TO_TEST

        # Default to raising errors on comms failure instead of retrying.
        # This can be overridden by setting 'request_retry_count'.
        config.setdefault("request_retry_count", 0)

        # Default to not having a timeout when debugging to prevent interruption.
        # This can be overridden by setting 'run_timeout'.
        if gettrace() is not None:
            config.setdefault("run_timeout", 0)

        input_data = []
        if (not ent_id or ent_id == "test_entity") and data_in:
            # use data hash if content is present.
            for k, v in data_in:
                if k == azm.DataLabel.CONTENT:
                    ent_id = DATA_HASH(v).hexdigest()
                    input_data = [local.gen_api_content(io.BytesIO(v))]
                    break

        # generate the event
        entity = azm.BinaryEvent.Entity(sha256=ent_id, datastreams=input_data, features=feats_in, **entity_attrs)

        # fill in properties from data
        if input_data:
            d = input_data[0]
            entity.datastreams = input_data
            entity.size = input_data[0].size
            entity.sha1 = d.sha1
            entity.sha256 = d.sha256
            entity.sha512 = d.sha512
            entity.md5 = d.md5
            entity.file_format_legacy = d.file_format_legacy

        event = local.gen_event(entity=entity)
        if submission_settings:
            event.source.settings = submission_settings

        # Set up data streams
        # Constraint: input files are read into memory, so you cannot test with massive sample files.
        datastreams: list[StorageProxyFile] = []

        for label, fname in files_in:
            with open(fname, "rb") as f:
                data = f.read()
            datastreams.append(
                StorageProxyFile(
                    source="local",
                    label=label,
                    hash=fname,
                    init_data=data,
                    file_info=local.gen_api_content(io.BytesIO(data), label=label),
                )
            )

        for label, data in data_in:
            datastreams.append(
                StorageProxyFile(
                    source="local",
                    label=label,
                    hash="test-data",
                    init_data=data,
                    file_info=local.gen_api_content(io.BytesIO(data), label=label),
                )
            )
        try:
            # if no_timeout_or_heartbeat_coordinator:
            if provided_coordinator:
                monitor_m_or_coord = provided_coordinator
            elif no_multiprocessing:
                monitor_m_or_coord = coordinator.Coordinator(plugin_class, settings.parse_config(plugin_class, config))
            else:
                monitor_m_or_coord = monitor.Monitor(plugin_class, config)
            # Verify that the specified data actually matches the provided test input
            if verify_input_content:
                local.validate_streams(datastreams, monitor_m_or_coord._plugin.cfg)

            try:
                test_dir_name = self.test_multi_process_temp_dir.name
            except AttributeError as e:
                raise AttributeError(
                    "Have you overridden setUpClass or tearDownClass without calling 'super().setUpClass()'."
                    + f" Missing test_multi_process_temp_dir with error - {e}"
                )

            if isinstance(monitor_m_or_coord, Coordinator):
                results = monitor_m_or_coord.run_once(event, datastreams)
            else:
                results = monitor_m_or_coord.run_once(event, datastreams, test_dir_name)

            # generate status event that would be sent over network
            # during plugin execution errors at this step cause the plugin to crash
            for multiplugin, run in results.items():
                monitor_m_or_coord._network.api.submit_binary = lambda x, y, z: local.gen_api_content(z, label=y)
                mm = monitor_m_or_coord._network.api.submit_events = mock.MagicMock()
                monitor_m_or_coord._network.ack_job(event, run, multiplugin)
                # append result to tearDown list
                self._results.append(run)
                status = mm.call_args[0][0][0]
                if (
                    eventlen := len(status.model_dump_json(exclude_defaults=True).encode())
                ) > dispatcher.MAX_MESSAGE_SIZE:
                    raise azbe.NetworkDataException(
                        f"event produced by plugin was too large: {eventlen}b" + f" > {dispatcher.MAX_MESSAGE_SIZE}b"
                    )

                self.assertTrue(status)

            # Checks that the time-sensitive values of results are within tolerance, then discards them.
            # End date of overall plugin should be within a few seconds of now (upped from 3 to 5 seconds for timeouts)
            self.assertLess(
                (datetime.datetime.now(datetime.timezone.utc) - results[None].date_end).total_seconds(), 10
            )
            for res in results.values():
                # Runtime should be within 1s due to int rounding
                self.assertAlmostEqual(res.runtime, (res.date_end - res.date_start).total_seconds(), delta=1)
                if not keep_times:
                    res.runtime = None
                    res.date_start = None
                    res.date_end = None
                if not keep_feature_types:
                    res.feature_types = []

            # Verify that the plugin gives consistent and child/augmented streams on repetitive runs.
            if check_consistent_augmented_stream:
                does_result_have_data = any([len(r.data) > 0 for r in results.values()])
                if does_result_have_data:
                    # reset all input streams
                    for stream in datastreams:
                        stream.seek(0)
                    if isinstance(monitor_m_or_coord, Coordinator):
                        results2 = monitor_m_or_coord.run_once(event, datastreams)
                    else:
                        results2 = monitor_m_or_coord.run_once(event, datastreams, test_dir_name)

                    self.assertIsNotNone(
                        results2, "Result 2 should not be none as there is augmented or child streams present."
                    )
                    # Verify all the streams for each multiplugin are equal.
                    for m_plugin, result in results.items():
                        self.assertCountEqual(
                            list(results2[m_plugin].data.keys()),
                            list(result.data.keys()),
                            "Inconsistent data streams, your plugin should return the same results when it's re-run.",
                        )
        finally:
            for ds in datastreams:
                if not ds.closed:
                    ds.close()

        # handle simple plugin results (backwards compatibility)
        if len(results) == 1:
            # return single results dict
            return list(results.values())[0]
        else:
            # return dictionary of multiplugin results dicts
            return results
