from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from multiprocessing import managers, shared_memory
from typing import Any, Type
from unittest import mock

from azul_bedrock import models_network as azm

from azul_runner import (
    Feature,
    FeatureType,
    FeatureValue,
    Job,
    Plugin,
    add_settings,
    cmdline_run,
    main,
    monitor,
)


class MainBaseTestPlugin(Plugin):
    SETTINGS = add_settings(use_multiprocessing_fork=True)


class TestCmdlineRun(unittest.TestCase):
    maxDiff = None

    # ####################### Utility method
    @staticmethod
    def _stdout_result(plugin: Type[Plugin]) -> str:
        saved_stdout = sys.stdout
        stdbuf = sys.stdout = io.StringIO()
        cmdline_run(plugin=plugin)
        sys.stdout = saved_stdout
        return stdbuf.getvalue()

    @staticmethod
    def _stdout_stderr_result(plugin: Type[Plugin]) -> tuple[str, str, str]:
        """Capture stdout, stderr and logs."""
        logbuf = io.StringIO()

        logger = logging.getLogger("azul.plugin")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(logbuf)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        stdbuf = sys.stdout = io.StringIO()
        errbuf = sys.stderr = io.StringIO()
        try:
            cmdline_run(plugin=plugin)
        except SystemExit:
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        return stdbuf.getvalue(), errbuf.getvalue(), logbuf.getvalue()

    # ######################## Test methods

    def test_simple_plugin(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.add_feature_values("value", 97)

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            res = self._stdout_result(TestPlugin)
        self.assertEqual(
            res,
            f"""----- TestPlugin results -----
COMPLETED

events (1)

event for {sha256}:None
  {{}}
  output features:
    value: 97

Feature key:
  value:  A return value

""",
        )

    def test_simple_plugin_empty_feature_value(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.add_feature_values("value", [])
                self.add_feature_values("value", set())

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            res = self._stdout_result(TestPlugin)
        self.assertEqual(res, "----- TestPlugin results -----\nCOMPLETED_EMPTY\n\n")

    def test_simple_plugin_offset_feature_values(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.add_feature_values("value", FeatureValue(value=97, label="label1", offset=10, size=1))
                self.add_feature_values("value", FeatureValue(value=97, offset=20))
                self.add_feature_values("value", FeatureValue(value=97, label="labelOnly"))
                self.add_feature_values("value", FeatureValue(value=97, size=99))

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            res = self._stdout_result(TestPlugin)
        self.assertEqual(
            res,
            f"""----- TestPlugin results -----
COMPLETED

events (1)

event for {sha256}:None
  {{}}
  output features:
    value: 97
           97 @ 0x14 (offset)
           label1 - 97 @ 0xa (offset)
           labelOnly - 97

Feature key:
  value:  A return value

""",
        )

    def test_simple_plugin_log_debug(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.logger.error("a error thing happened")
                self.logger.warning("a warning thing happened")
                self.logger.info("a info thing happened")
                self.logger.debug("a debug thing happened")
                self.add_feature_values("value", 97)

                mylog = logging.getLogger("external.flatlib")
                mylog.error("this is just a simple error")
                mylog.warning("this is just a simple warning")
                mylog.info("this is just a simple info")
                mylog.debug("this is just a simple debug")

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name, "--verbose", "--verbose"]
            res, err, log = self._stdout_stderr_result(TestPlugin)
        self.assertEqual(
            res,
            f"""----- TestPlugin results -----
COMPLETED

events (1)

event for {sha256}:None
  {{}}
  output features:
    value: 97

Feature key:
  value:  A return value

""",
        )
        # Show logs incase of errors
        print(f"--- error ---\n{err}\n--- End error ---")
        print(f"+++ log +++\n{log}\n+++ End log +++")
        # runner logs
        self.assertRegex(err, r"azul.plugin.testplugin.*a error thing happened")
        self.assertRegex(err, r"azul.plugin.testplugin.*a warning thing happened")
        self.assertRegex(err, r"azul.plugin.testplugin.*a info thing happened")
        self.assertRegex(err, r"azul.plugin.testplugin.*a debug thing happened")
        # external tool logs
        self.assertRegex(err, r"external.flatlib.*this is just a simple error")
        self.assertRegex(err, r"external.flatlib.*this is just a simple warning")
        self.assertRegex(err, r"external.flatlib.*this is just a simple info")
        self.assertRegex(err, r"external.flatlib.*this is just a simple debug")

    def test_simple_plugin_log_info(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.logger.error("a error thing happened")
                self.logger.warning("a warning thing happened")
                self.logger.info("a info thing happened")
                self.logger.debug("a debug thing happened")
                self.add_feature_values("value", 97)

                mylog = logging.getLogger("external.flatlib")
                mylog.error("this is just a simple error")
                mylog.warning("this is just a simple warning")
                mylog.info("this is just a simple info")
                mylog.debug("this is just a simple debug")

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name, "--verbose"]
            res, err, log = self._stdout_stderr_result(TestPlugin)
        # Show logs incase of errors
        print(f"--- error ---\n{err}\n--- End error ---")
        print(f"+++ log +++\n{log}\n+++ End log +++")
        # runner logs
        self.assertRegex(err, r"azul.plugin.testplugin.*a error thing happened")
        self.assertRegex(err, r"azul.plugin.testplugin.*a warning thing happened")
        self.assertRegex(err, r"azul.plugin.testplugin.*a info thing happened")
        self.assertNotRegex(err, r"azul.plugin.testplugin.*a debug thing happened")
        # external tool logs
        self.assertRegex(err, r"external.flatlib.*this is just a simple error")
        self.assertRegex(err, r"external.flatlib.*this is just a simple warning")
        self.assertRegex(err, r"external.flatlib.*this is just a simple info")
        self.assertNotRegex(err, r"external.flatlib.*this is just a simple debug")

    def test_simple_plugin_log_warn(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.logger.error("a error thing happened")
                self.logger.warning("a warning thing happened")
                self.logger.info("a info thing happened")
                self.logger.debug("a debug thing happened")
                self.add_feature_values("value", 97)

                mylog = logging.getLogger("external.flatlib")
                mylog.error("this is just a simple error")
                mylog.warning("this is just a simple warning")
                mylog.info("this is just a simple info")
                mylog.debug("this is just a simple debug")

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            res, err, log = self._stdout_stderr_result(TestPlugin)
        # Show logs incase of errors
        print(f"--- error ---\n{err}\n--- End error ---")
        print(f"+++ log +++\n{log}\n+++ End log +++")
        # runner logs
        self.assertRegex(err, r"azul.plugin.testplugin.*a error thing happened")
        self.assertRegex(err, r"azul.plugin.testplugin.*a warning thing happened")
        self.assertNotRegex(err, r"azul.plugin.testplugin.*a info thing happened")
        self.assertNotRegex(err, r"azul.plugin.testplugin.*a debug thing happened")
        # external tool logs
        self.assertRegex(err, r"external.flatlib.*this is just a simple error")
        self.assertRegex(err, r"external.flatlib.*this is just a simple warning")
        self.assertNotRegex(err, r"external.flatlib.*this is just a simple info")
        self.assertNotRegex(err, r"external.flatlib.*this is just a simple debug")

    def test_simple_plugin_log_err(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.logger.error("a error thing happened")
                self.logger.warning("a warning thing happened")
                self.logger.info("a info thing happened")
                self.logger.debug("a debug thing happened")
                self.add_feature_values("value", 97)

                mylog = logging.getLogger("external.flatlib")
                mylog.error("this is just a simple error")
                mylog.warning("this is just a simple warning")
                mylog.info("this is just a simple info")
                mylog.debug("this is just a simple debug")

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name, "--quiet"]
            res, err, log = self._stdout_stderr_result(TestPlugin)
        # Show logs incase of errors
        print(f"--- error ---\n{err}\n--- End error ---")
        print(f"+++ log +++\n{log}\n+++ End log +++")
        # runner logs
        self.assertRegex(err, r"azul.plugin.testplugin.*a error thing happened")
        self.assertNotRegex(err, r"azul.plugin.testplugin.*a warning thing happened")
        self.assertNotRegex(err, r"azul.plugin.testplugin.*a info thing happened")
        self.assertNotRegex(err, r"azul.plugin.testplugin.*a debug thing happened")
        # external tool logs
        self.assertRegex(err, r"external.flatlib.*this is just a simple error")
        self.assertNotRegex(err, r"external.flatlib.*this is just a simple warning")
        self.assertNotRegex(err, r"external.flatlib.*this is just a simple info")
        self.assertNotRegex(err, r"external.flatlib.*this is just a simple debug")

    def test_binary_input(self):
        shared_list = shared_memory.ShareableList([500 * "a" for _ in range(7)])

        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("job_event_id", "Getting Job details back", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                shared_list[0] = job.event.entity.sha256
                shared_list[1] = job.event.entity.sha512
                shared_list[2] = job.event.entity.sha1
                shared_list[3] = job.event.entity.md5
                shared_list[4] = job.event.entity.size
                shared_list[5] = job.event.entity.file_format_legacy

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            result_data = self._stdout_result(TestPlugin)
            # Logs errors if something is wrong with the shared list.
            print(result_data)
            result_dict = {
                "job.event.entity.sha256": shared_list[0],
                "job.event.entity.sha512": shared_list[1],
                "job.event.entity.sha1": shared_list[2],
                "job.event.entity.md5": shared_list[3],
                "job.event.entity.size": shared_list[4],
                "job.event.entity.file_format_legacy": shared_list[5],
            }
            print(result_dict)
            self.assertEqual(
                result_dict["job.event.entity.sha256"],
                "8b7df143d91c716ecfa5fc1730022f6b421b05cedee8fd52b1fc65a96030ad52",
            )
            self.assertEqual(
                result_dict["job.event.entity.sha512"],
                "39ca2b1f97c7d1d223dcb2b22cbe20c36f920aeefd201d0bf68ffc08db6d9ac608a0a202fb536d944c9d1f50cf9bd61b5bc84217212f0727a8db8a01c2fa54b7",
            )
            self.assertEqual(result_dict["job.event.entity.sha1"], "5bf1fd927dfb8679496a2e6cf00cbe50c1c87145")
            self.assertEqual(result_dict["job.event.entity.md5"], "6f1ed002ab5595859014ebf0951522d9")
            self.assertEqual(result_dict["job.event.entity.size"], 4)
            self.assertEqual(result_dict["job.event.entity.file_format_legacy"], "Text")
            # Close shared memory
            shared_list.shm.close()

    def test_simple_plugin_only_stream(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", FeatureType.String)]

            def execute(self, job: Job) -> dict:
                data = job.get_data(azm.DataLabel.TEST).read().decode("utf-8", errors="ignore")
                self.add_feature_values("value", data)

        sha256 = "test_stream_only"
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah" * 4)
            f.seek(0)
            sys.argv = ["test_main.py", "--entity-id", sha256, "--stream", azm.DataLabel.TEST, f.name]
            res = self._stdout_result(TestPlugin)
        self.assertEqual(
            res,
            f"""----- TestPlugin results -----
COMPLETED

events (1)

event for {sha256}:None
  {{}}
  output features:
    value: blahblahblahblah

Feature key:
  value:  A return value

""",
        )

    def test_multiple_streams(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                text = job.get_data(azm.DataLabel.TEST).file_info.sha256
                event = self.get_data_event(text)
                event.add_feature_values("value", FeatureValue(95))
                self.add_feature_values("value", 97)

        # execute again with two input data streams - expect a result for each stream
        saved_argv = sys.argv
        tf1 = tempfile.NamedTemporaryFile()
        tf1.write(b"Not a Content stream")
        tf1.seek(0)
        sha256 = hashlib.sha256(tf1.read()).hexdigest()
        tf1.seek(0)
        tf1.flush()
        tf2 = tempfile.NamedTemporaryFile()
        tf2.write(b"Text stream")
        tf2.seek(0)
        tf2.flush()
        sys.argv = ["test_main.py", "-s", azm.DataLabel.TEXT, tf1.name, "-s", azm.DataLabel.TEST, tf2.name]
        res = self._stdout_result(TestPlugin)
        print(res)
        self.assertEqual(
            res,
            """----- TestPlugin results -----
COMPLETED

events (2)

event for cmdline_entity:None
  {}
  output features:
    value: 97

event for data 7b238c15c251a28c21976f22d47d9a7f31c3d8cff922f53c4129eee140bdc673
  {}
  output features:
    value: 95

Feature key:
  value:  A return value

""",
        )
        tf1.close()
        tf2.close()

    def test_child_output(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("child_feat", "A feature of the child entity", FeatureType.String)]

            def execute(self, job: Job) -> dict:
                c = self._add_child(
                    "child_ID",
                    {"relationship": "extracted"},
                )
                c.add_data("content", {}, b"Child content")
                c.add_feature_values("child_feat", "child feature value")

                c = self._add_child(
                    "child2_ID",
                    {"relationship": "extracted"},
                )
                c.add_data("content", {}, b"Child content")
                c.add_feature_values("child_feat", "child feature value")
                c = self._add_child(
                    "child2_ID",
                    {"relationship": "extracted"},
                )
                c.add_data("content", {}, b"Child content")
                c.add_feature_values("child_feat", "child feature value")
                c = self._add_child(
                    "child3_ID",
                    {"relationship": "extracted"},
                )
                c.add_data("content", {}, b"Child content")
                c.add_feature_values("child_feat", "child feature value")
                c = self._add_child(
                    "child3_ID",
                    {"relationship": "extracted", "additional": "extracted twice"},
                )
                c.add_data("content", {}, b"Child content")
                c.add_feature_values("child_feat", "a different feature value")
                return

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            res = self._stdout_result(TestPlugin)
            print(res)
        self.assertEqual(
            res,
            f"""----- TestPlugin results -----
COMPLETED

events (3)

event for child_ID:None
  {{'relationship': 'extracted'}}
  child of {sha256}
  output data streams (1):
    13 bytes - EventData(hash='20ec1767a3be628df0b67c6bd9016ad50d106da0775a3fe65e49ca54121747ea', label='content')
  output features:
    child_feat: child feature value

event for child2_ID:None
  {{'relationship': 'extracted'}}
  child of {sha256}
  output data streams (1):
    13 bytes - EventData(hash='20ec1767a3be628df0b67c6bd9016ad50d106da0775a3fe65e49ca54121747ea', label='content')
  output features:
    child_feat: child feature value

event for child3_ID:None
  {{'relationship': 'extracted'}}
  child of {sha256}
  output data streams (1):
    13 bytes - EventData(hash='20ec1767a3be628df0b67c6bd9016ad50d106da0775a3fe65e49ca54121747ea', label='content')
  output features:
    child_feat: a different feature value
                child feature value

Feature key:
  child_feat:  A feature of the child entity

""",
        )

    def test_with_config(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = []

            def execute(self, job: Job) -> dict:
                return

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sys.argv = ["test_main.py", f.name, "-c", "alpha", "beta"]
            res = self._stdout_result(TestPlugin)
        print(res)
        self.assertEqual(res, "----- TestPlugin results -----\nCOMPLETED_EMPTY\n\n")

    def test_grandchild_output(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [
                Feature("child_feat", "A feature of the child entity", FeatureType.String),
                Feature("gc_feat", "A feature of the grandchild", FeatureType.String),
            ]

            def execute(self, job: Job) -> dict:
                c = self._add_child(
                    "child_ID",
                    {"relationship": "extracted"},
                )
                c.add_data("content", {}, b"Child content")
                c.add_feature_values("child_feat", "child feature value")
                gc = c._add_child("grandchild_ID", {"relationship": "decoded"})
                gc.add_data("content", {"language": "ocelot"}, b"Grandchild content")
                gc.add_feature_values("gc_feat", ["value1", "value2"])
                return

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            res = self._stdout_result(TestPlugin)
        print(res)
        self.assertEqual(
            res,
            f"""----- TestPlugin results -----
COMPLETED

events (2)

event for child_ID:None
  {{'relationship': 'extracted'}}
  child of {sha256}
  output data streams (1):
    13 bytes - EventData(hash='20ec1767a3be628df0b67c6bd9016ad50d106da0775a3fe65e49ca54121747ea', label='content')
  output features:
    child_feat: child feature value

event for grandchild_ID:None
  {{'relationship': 'decoded'}}
  child of child_ID
  output data streams (1):
    18 bytes - EventData(hash='1459c1f57adc3cc9ec8da1dddcce220eac175f048f212a1392325ad112f39422', label='content', language='ocelot')
  output features:
    gc_feat: value1
             value2

Feature key:
  child_feat:  A feature of the child entity
  gc_feat:  A feature of the grandchild

""",
        )

    def test_stream_output(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"

            def execute(self, job: Job) -> dict:
                self.add_data(
                    "text",
                    {"language": "python"},
                    b"Text output content",
                )
                return

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name]
            res = self._stdout_result(TestPlugin)
        print(res)
        self.assertEqual(
            res,
            f"""----- TestPlugin results -----
COMPLETED

events (1)

event for {sha256}:None
  {{}}
  output data streams (1):
    19 bytes - EventData(hash='1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54', label='text', language='python')

""",
        )

    def test_JSON_output(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "1.0"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.add_data(
                    "text",
                    {"language": "python"},
                    b"Text output content",
                )
                self.add_feature_values("value", 12345)

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name, "--output-json"]
            res = self._stdout_result(TestPlugin)
            res = json.loads(res)
        # Ignore nondeterministic outputs
        res.pop("date_start")
        res.pop("date_end")
        res.pop("runtime")
        print(res["events"])
        self.assertEqual(
            res,
            {
                "state": {"label": "completed"},
                "feature_types": [{"desc": "A return value", "name": "value", "type": "integer"}],
                "events": [
                    {
                        "sha256": sha256,
                        "data": [
                            {
                                "hash": "1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54",
                                "label": "text",
                                "language": "python",
                            }
                        ],
                        "features": {"value": [{"value": "12345"}]},
                    }
                ],
                "data": {
                    "1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54": "VGV4dCBvdXRwdXQgY29udGVudA==",
                },
            },
        )

    def test_output_folder(self):
        """Test that the plugin can output artifacts and run info to folder."""

        class TestPlugin(MainBaseTestPlugin):
            VERSION = "1.0"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.add_data(
                    "text",
                    {"language": "python"},
                    b"Text output content",
                )
                self.add_feature_values("value", 12345)

        with tempfile.NamedTemporaryFile() as f, tempfile.TemporaryDirectory() as tmpdir:
            f.write(b"blah")
            f.seek(0)
            sys.argv = ["test_main.py", f.name, "--output-folder", tmpdir]
            self._stdout_result(TestPlugin)
            self.assertEqual(
                os.listdir(tmpdir),
                ["1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54_text.data"],
            )
            with open(
                os.path.join(tmpdir, "1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54_text.data"),
                "rb",
            ) as f2:
                databytes = f2.read()
            self.assertEqual(databytes, b"Text output content")

    def test_multiplugin_regular(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", FeatureType.String)]

            def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
                super().__init__(config)
                self.register_multiplugin("66_dec", "version 5", self.execute_order_66)
                self.register_multiplugin("rol_67", "version 5", self.execute_order_67)

            def execute_order_66(self, entity) -> dict:
                return {"features": {"value": "for a safe and secure galaxy"}}

            def execute_order_67(self, entity) -> dict:
                return {"features": {"value": "hello there"}}

        # First, execute without input data streams
        sys.argv = ["test_main.py"]
        res = self._stdout_result(TestPlugin)
        expected = """
----- TestPlugin-66_dec results -----
COMPLETED

Output features:
  value: for a safe and secure galaxy


Feature key:
  value:  A return value

----- TestPlugin-rol_67 results -----
COMPLETED

Output features:
  value: hello there


Feature key:
  value:  A return value
        """.strip()
        self.assertEqual(res.strip(), expected)

    def test_multiplugin_regular(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", FeatureType.String)]

            def __init__(self, config: dict[str, dict[str, Any]] = None) -> None:
                super().__init__(config)
                self.register_multiplugin("66_dec", "version 5", self.execute_order_66)
                self.register_multiplugin("rol_67", "version 5", self.execute_order_67)

            def execute(self, job: Job) -> dict:
                pass

            def execute_order_66(self, job: Job) -> dict:
                self.add_feature_values("value", "for a safe and secure galaxy")

            def execute_order_67(self, job: Job) -> dict:
                self.add_feature_values("value", "hello there")

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"blah")
            f.seek(0)
            sha256 = hashlib.sha256(f.read()).hexdigest()
            f.seek(0)
            sys.argv = ["test_main.py", f.name, "--output-json"]
            results = self._stdout_result(TestPlugin)
        res1, res2, _, _ = results.split("\n")

        res1 = json.loads(res1)
        # Ignore nondeterministic outputs
        res1.pop("date_start")
        res1.pop("date_end")
        res1.pop("runtime")
        print(res1)
        self.assertEqual(
            res1,
            {
                "state": {"label": "completed"},
                "feature_types": [{"name": "value", "desc": "A return value", "type": "string"}],
                "events": [
                    {
                        "sha256": sha256,
                        "features": {
                            "value": [
                                {
                                    "value": "for a safe and secure galaxy",
                                }
                            ]
                        },
                    }
                ],
            },
        )

        res2 = json.loads(res2)
        # Ignore nondeterministic outputs
        res2.pop("date_start")
        res2.pop("date_end")
        res2.pop("runtime")
        print(res2)
        self.assertEqual(
            res2,
            {
                "state": {"label": "completed"},
                "feature_types": [{"name": "value", "desc": "A return value", "type": "string"}],
                "events": [
                    {
                        "sha256": sha256,
                        "features": {"value": [{"value": "hello there"}]},
                    }
                ],
            },
        )

    def test_folder(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"

            def execute(self, job: Job) -> dict:
                self.add_data(
                    "text",
                    {"language": "python"},
                    b"Text output content",
                )
                return

        with tempfile.TemporaryDirectory() as tmpdir:
            pf1 = os.path.join(tmpdir, "f1")
            pf2 = os.path.join(tmpdir, "f2")
            with open(pf1, "w+b") as f:
                f.write(b"data1")
                f.seek(0)
                sha2561 = hashlib.sha256(f.read()).hexdigest()

            with open(pf2, "w+b") as f:
                f.write(b"data2")
                f.seek(0)
                sha2562 = hashlib.sha256(f.read()).hexdigest()
            sys.argv = ["test_main.py", tmpdir]
            res = self._stdout_result(TestPlugin)
        print(res)
        self.assertEqual(
            res,
            f"""{pf1}
----- TestPlugin results -----
COMPLETED

events (1)

event for {sha2561}:None
  {{}}
  output data streams (1):
    19 bytes - EventData(hash='1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54', label='text', language='python')

{pf2}
----- TestPlugin results -----
COMPLETED

events (1)

event for {sha2562}:None
  {{}}
  output data streams (1):
    19 bytes - EventData(hash='1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54', label='text', language='python')

""",
        )

    def test_folder_json(self):
        class TestPlugin(MainBaseTestPlugin):
            VERSION = "1.0"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.add_data(
                    "text",
                    {"language": "python"},
                    b"Text output content",
                )
                self.add_feature_values("value", 12345)

        with tempfile.TemporaryDirectory() as tmpdir:
            pf1 = os.path.join(tmpdir, "f1")
            pf2 = os.path.join(tmpdir, "f2")
            with open(pf1, "w+b") as f:
                f.write(b"data1")
                f.seek(0)
                sha2561 = hashlib.sha256(f.read()).hexdigest()

            with open(pf2, "w+b") as f:
                f.write(b"data2")
                f.seek(0)
                sha2562 = hashlib.sha256(f.read()).hexdigest()
            sys.argv = ["test_main.py", tmpdir, "--output-json"]
            res = self._stdout_result(TestPlugin)

        # every second line is valid json
        lines = res.split("\n")
        line1 = json.loads(lines[1])
        line2 = json.loads(lines[3])
        # Ignore nondeterministic outputs
        line1.pop("date_start")
        line1.pop("date_end")
        line1.pop("runtime")
        print(line1)
        self.assertEqual(
            line1,
            {
                "state": {"label": "completed"},
                "feature_types": [{"name": "value", "desc": "A return value", "type": "integer"}],
                "events": [
                    {
                        "sha256": sha2561,
                        "data": [
                            {
                                "hash": "1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54",
                                "label": "text",
                                "language": "python",
                            }
                        ],
                        "features": {"value": [{"value": "12345"}]},
                    }
                ],
                "data": {
                    "1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54": "VGV4dCBvdXRwdXQgY29udGVudA==",
                },
            },
        )
        # Ignore nondeterministic outputs
        line2.pop("date_start")
        line2.pop("date_end")
        line2.pop("runtime")
        print(line2)
        self.assertEqual(
            line2,
            {
                "state": {"label": "completed"},
                "feature_types": [{"name": "value", "desc": "A return value", "type": "integer"}],
                "events": [
                    {
                        "sha256": sha2562,
                        "data": [
                            {
                                "hash": "1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54",
                                "label": "text",
                                "language": "python",
                            }
                        ],
                        "features": {"value": [{"value": "12345"}]},
                    }
                ],
                "data": {
                    "1c2568562238a338919f87598737756a67e22f35d5031a45796c9d312b5beb54": "VGV4dCBvdXRwdXQgY29udGVudA==",
                },
            },
        )

    def test_server_exception_handling_plugin(self):
        """Tests the server's handling of errors from coordinator."""

        class TestPlugin(MainBaseTestPlugin):
            VERSION = "none"
            FEATURES = [Feature("value", "A return value", type=FeatureType.Integer)]

            def execute(self, job: Job) -> dict:
                self.add_feature_values("value", 97)

        # Have to use shared memory because it's the only way for the mock method to communicate from the subprocess.
        # back to the test, to validate the right number of calls were made.
        with managers.SharedMemoryManager() as smm:
            share_mem = smm.ShareableList([0])

            # Normal exit
            def fake_run_loop_return(*args, **kwargs):
                share_mem[0] += 1
                return

            with mock.patch.object(monitor.Monitor, "run_loop", wraps=fake_run_loop_return):
                main.execute(TestPlugin, main.Args())
                self.assertEqual(share_mem[0], 1)

            share_mem[0] = 0

            # Exception exit
            def fake_run_loop_exception(*args, **kwargs):
                share_mem[0] += 1
                raise Exception()

            with mock.patch.object(monitor.Monitor, "run_loop", wraps=fake_run_loop_exception):
                with self.assertRaises(Exception):
                    main.execute(TestPlugin, main.Args())
                    self.assertEqual(share_mem[0], 1)

            # Re-boot plugin once and cleanup temp
            share_mem[0] = 0

            def fake_run_loop_retry(*args, **kwargs):
                share_mem[0] += 1
                if share_mem[0] == 1:
                    raise monitor.RecreateException()
                else:
                    return

            with mock.patch.object(monitor.Monitor, "run_loop", wraps=fake_run_loop_retry):
                with self.assertRaises(monitor.RecreateException):
                    main.execute(TestPlugin, main.Args())
                    self.assertEqual(share_mem[0], 2)
