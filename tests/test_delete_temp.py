"""Test cases that verify monitor's temp deletion works as expected."""

import os
import tempfile
import unittest

from azul_bedrock import models_network as azm

from azul_runner import local, monitor
from azul_runner.models import Job

from . import plugin_support as sup


class DeleteTempDirectoryTests(unittest.TestCase):
    def test_delete_tempdirectory_contents(self):
        file1 = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
        file1.write(b"abc")
        file1.close()
        file_path = file1.name
        self.assertTrue(os.path.exists(file_path))
        monitor.Monitor.delete_tempfiles()
        self.assertFalse(os.path.exists(file_path))

    def base_delete_file_with_custom_prefix(self, file_name: str, prefix: str):
        # Create a file manually and confirm it's deleted
        file_to_create = os.path.join(tempfile.gettempdir(), file_name)
        with open(file_to_create, "w") as f:
            f.write("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.assertTrue(os.path.exists(file_to_create))
        # Default delete doesn't find the file
        monitor.Monitor.delete_tempfiles()
        self.assertTrue(os.path.exists(file_to_create))
        # Providing prefix with different case still finds the file.
        monitor.Monitor.delete_tempfiles(prefix)
        self.assertFalse(os.path.exists(file_to_create))

    def test_delete_prefix_lowercase_prefix_uppercase_filename(self):
        self.base_delete_file_with_custom_prefix("customTempFile", "customtemp")

    def test_delete_prefix_uppercase_prefix_lowercase_filename(self):
        self.base_delete_file_with_custom_prefix("customtempfile", "customTemp")

    def test_delete_same_case_on_prefix_and_filename(self):
        self.base_delete_file_with_custom_prefix("customTempFile", "customTempFile")

    def test_tempfile_gets_cleared(self):
        """Tests that bad temp files created by a plugin get cleared on re-create."""

        class DP(sup.DummyPlugin):
            def execute(self, job: Job):
                with open(job.event.entity.info.get("file"), "wb") as f:
                    f.write(b"abjksdlfjalksdf aslkdfjlkasdj lasdjf laksdfjlkasdf")

        loop = monitor.Monitor(DP, {})
        first_dummy_file = os.path.join(tempfile.gettempdir(), "tmpdummyfile1")
        second_dummy_file = os.path.join(tempfile.gettempdir(), "tmpdummyfile2")
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={"file": first_dummy_file})
        loop.run_once(local.gen_event(entity))

        self.assertTrue(os.path.exists(first_dummy_file))
        entity = azm.BinaryEvent.Entity(sha256="id", datastreams=[], features=[], info={"file": second_dummy_file})
        loop.run_once(local.gen_event(entity))

        # First dummy file should have been deleted on next run when plugin was re-created.
        self.assertFalse(os.path.exists(first_dummy_file))

        self.assertTrue(os.path.exists(second_dummy_file))
        monitor.Monitor.delete_tempfiles()
        self.assertFalse(os.path.exists(second_dummy_file))
