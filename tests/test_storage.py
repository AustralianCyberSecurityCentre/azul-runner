from __future__ import annotations

import io
import time
import unittest
from typing import Any, ClassVar

import httpx
from azul_bedrock import dispatcher
from azul_bedrock import models_network as azm
from bitarray import bitarray

from azul_runner import local
from azul_runner.storage import StorageError, StorageProxyFile

from . import mock_dispatcher as md


class TestStorageProxyFile(unittest.TestCase):
    mock_server: ClassVar[md.MockDispatcher]
    server: ClassVar[str]  # Endpoint to the mock server in the form 'http://host:port'

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_server = md.MockDispatcher()
        cls.mock_server.start()
        while not cls.mock_server.is_alive():
            time.sleep(0.2)  # Wait for server to start
        cls.server = "http://%s:%s" % (cls.mock_server.host, cls.mock_server.port)
        cls.dp = dispatcher.DispatcherAPI(
            events_url=cls.server,
            data_url=cls.server,
            retry_count=2,
            timeout=2,
            author_name="TestStorageProxyFile",
            author_version="1",
            deployment_key="scorpion",
        )
        # Wait for server to be ready to respond
        tries = 0
        while True:
            time.sleep(0.2)
            tries += 1
            try:
                _ = httpx.get(cls.server + "/mock/get_var/fetch_count")
                break  # Exit loop if successful
            except (httpx.TimeoutException, httpx.ConnectError):
                if tries > 20:  # Time out after about 4 seconds
                    raise RuntimeError("Timed out waiting for mock server to be ready")

        cls.editor = md.Editor(cls.server)
        cls.editor.set_stream("500-internal-server-error", 500, b"")
        cls.editor.set_stream("test-ff-128", 206, b"\xff" * 128)
        cls.editor.set_stream("test-ff-129", 206, b"\xff" * 129)
        cls.editor.set_stream("test-ff-256", 206, b"\xff" * 256)
        cls.editor.set_stream("test-ff-192", 206, b"\xff" * 192)
        cls.editor.set_stream("test-ff-191", 206, b"\xff" * 191)
        cls.editor.set_stream("test-zeros-256", 206, b"\x00" * 256)
        cls.editor.set_stream("test-zeros-255", 206, b"\x00" * 255)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()

    # ############### #
    # #### Tests #### #
    # ############### #

    def test_instantiate(self):
        with StorageProxyFile("source", "content", "test-zeros-128", dp=self.dp) as f:
            # Size should now be None on first init, until something is read
            self.assertEqual(f._size, None)

    def test_invalid_params(self):
        with StorageProxyFile("source", "content", "test-zeros-256", dp=self.dp) as f:
            self.assertRaisesRegex(TypeError, "read length must be an int", f.read, 2.5)
            self.assertRaisesRegex(ValueError, "read length must be positive or -1", f.read, -50)
            self.assertRaisesRegex(ValueError, "start_chunk cannot be negative", f._request_chunks, -1, 20)
        with StorageProxyFile("source", "content", "INVALID URL", dp=self.dp) as f:
            self.assertRaisesRegex(StorageError, r"Got 404 requesting INVALID URL", f.read)
        with StorageProxyFile("source", "content", "nosuchfile", dp=self.dp) as f:
            self.assertRaisesRegex(StorageError, r"Got 404 requesting nosuchfile", f.read)

    def test_size_changed_assertion(self):
        # Must use a small chunk size for this one, or it will cache the whole file and not pick up the changed size
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=16, dp=self.dp) as f:
            f.read(1)
            f._hash = "test-zeros-255"
            self.assertRaisesRegex(AssertionError, "Object size mismatch", f.read)
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=256, dp=self.dp) as f:
            f.read(1)
            f._hash = "test-zeros-255"
            # No assertion error, as the entire file is already cached
            f.read()

    def test_internal_server_error(self):
        """Test remote internal server error."""
        with StorageProxyFile("source", "content", "test-zeros-256", dp=self.dp) as f:
            f._hash = "500-internal-server-error"
            self.assertRaisesRegex(StorageError, "Got 500 requesting 500-internal-server-error", f.read, 1)

    def test_chunkmap_and_cache(self):
        """Tests that the chunkmap bitarray behaves properly"""
        with StorageProxyFile("source", "content", "test-ff-128", chunk_size=32, dp=self.dp) as f:
            # Chunks should not yet be set, as nothing has been read / no requests to server
            self.assertEqual(f._chunks, bitarray())
            f.read(1)
            self.assertEqual(f._chunks, bitarray("1000"))
            f._content.seek(0)
            self.assertEqual(f._content.read(), b"\xff" * 32)
            f.seek(-1, io.SEEK_END)
            f.read(1)
            self.assertEqual(f._chunks, bitarray("1001"))
            f._content.seek(0)
            self.assertEqual(f._content.read(), b"\xff" * 32 + b"\x00" * 64 + b"\xff" * 32)
        with StorageProxyFile("source", "content", "test-ff-129", chunk_size=32, dp=self.dp) as f:
            f.MIN_LAST_CHUNK_SIZE = 0  # Disable last-chunk-merging for this test
            self.assertEqual(f._chunks, bitarray(""))
            f.seek(-1, io.SEEK_END)
            f.read(1)
            self.assertEqual(f._chunks, bitarray("00001"))
            f._content.seek(0)
            # Should actually have the last 32 bytes of real data, due to reading the final <chunk_size> bytes
            # (But only the last byte actually forms a valid chunk, since a byte in the 2nd-last chunk is missing)
            self.assertEqual(f._content.read(), b"\x00" * 97 + b"\xff" * 32)

    def test_last_chunk_merging(self):
        """Tests that auto-merging of small final chunks works as expected"""
        with StorageProxyFile("source", "content", "test-ff-192", chunk_size=128, dp=self.dp) as f:
            # Last chunk is >= the limit, so last chunk is fetched separately
            f.MIN_LAST_CHUNK_SIZE = 0.5
            self.assertEqual(f._chunks, bitarray())  # Should be zero-length since the size is unknown.
            f.read(1)
            self.assertEqual(f._chunks, bitarray("10"))
            f._content.seek(0)
            self.assertEqual(f._content.read(), b"\xff" * 128)
            f.seek(-1, io.SEEK_END)
            f.read(1)
            self.assertEqual(f._chunks, bitarray("11"))
            f._content.seek(0)
            self.assertEqual(f._content.read(), b"\xff" * 192)
        with StorageProxyFile("source", "content", "test-ff-191", chunk_size=128, dp=self.dp) as f:
            # The size is initially not known, so the extra partial chunk won't be read.
            f.MIN_LAST_CHUNK_SIZE = 0.5
            self.assertEqual(f._chunks, bitarray())
            f.read(1)
            self.assertEqual(f._chunks, bitarray("10"))
        with StorageProxyFile("source", "content", "test-ff-191", chunk_size=128, expected_size=191, dp=self.dp) as f:
            # Size known to be one byte smaller than the min size limit;  should auto-fetch the last chunk
            #  when the second-last chunk is read (in this case the first chunk)
            f.MIN_LAST_CHUNK_SIZE = 0.5
            self.assertEqual(f._chunks, bitarray("00"))
            f.read(1)
            self.assertEqual(f._chunks, bitarray("11"))
            f._content.seek(0)
            self.assertEqual(f._content.read(), b"\xff" * 191)

    def test_read_whole_file(self):
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=16, dp=self.dp) as f:
            self.assertEqual(f.read(), b"\x00" * 256)
            self.assertEqual(self.editor.get_last_request().range, [0, 255])
            self.assertEqual(f.read(), b"")
            # Further requests should not go back to the server, as the full content has been acquired.
            f.seek(128)
            f.read()
            self.assertEqual(self.editor.get_last_request().range, [0, 255])

    def test_read_partial_chunks(self):
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=16, dp=self.dp) as f:
            self.assertEqual(f.read(16), b"\x00" * 16)
            self.assertEqual(self.editor.get_last_request().range, [0, 15])

    def test_read_middle_chunk(self):
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=256, dp=self.dp) as f:
            # This should read the entire file due to large chunk size
            f.seek(128)
            self.assertEqual(f.read(64), b"\x00" * 64)

            self.assertEqual(self.editor.get_last_request().range, [0, 255])
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=16, dp=self.dp) as f:
            # This should read only what's requested
            f.seek(128)
            self.assertEqual(f.read(64), b"\x00" * 64)
            self.assertEqual(self.editor.get_last_request().range, [128, 191])
            # This should not go to the server as it's already at the EOF. (size is known)
            f.seek(256)
            self.assertEqual(f.read(), b"")
            self.assertEqual(self.editor.get_last_request().range, [128, 191])
            # This does go to the server as the file was seeked back one and that block is not cached yet:
            f.seek(-1, io.SEEK_CUR)
            self.assertEqual(f.read(), b"\x00")
            self.assertEqual(self.editor.get_last_request().range, [240, 255])

    def test_read_past_end(self):
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=32, dp=self.dp) as f:
            # Size is known so only request what is needed in this version of StorageProxyFile
            self.assertEqual(f.read(500), b"\x00" * 256)
            self.assertEqual(self.editor.get_last_request().range, [0, 255])

    def test_seek_from_end(self):
        """Tests seeking back from end on a newly-opened file"""
        with StorageProxyFile("source", "content", "test-zeros-256", dp=self.dp) as f:
            f.seek(-1, io.SEEK_END)
            self.assertEqual(f.tell(), 255)

    def test_seek_past_end(self):
        """Tests seeking past the end of a file and trying to read"""
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=32, dp=self.dp) as f:
            f.read(32)
            self.assertEqual(f._size, 256)  # Should now know this
            f.seek(512)
            self.assertEqual(f.read(), b"")
            # Should not have gone to the server for this, as it's past EOF
            self.assertEqual(self.editor.get_last_request().range, [0, 31])
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=32, dp=self.dp) as f:
            f.seek(512)
            self.assertEqual(f.read(), b"")
            # This time request from the server because the size was not known before the request (and got no content)
            self.assertIsNone(self.editor.get_last_request().range)
            # Should have been given the size
            self.assertEqual(f._size, 256)

    def test_expected_size(self):
        """Tests that there is no GET issued when an expected size is provided"""
        with StorageProxyFile("source", "content", "test-zeros-256", chunk_size=128, dp=self.dp) as f:
            # No expected size
            f.seek(-1, io.SEEK_END)
            self.assertEqual(f.tell(), 255)
            # Should have fetched the last chunk when trying to determine the size
            self.assertEqual(f._chunks, bitarray("01"))
        with StorageProxyFile(
            "source", "content", "test-zeros-256", chunk_size=128, expected_size=256, dp=self.dp
        ) as f:
            f.seek(-1, io.SEEK_END)
            self.assertEqual(f.tell(), 255)
            # Should not have fetched anything, but should have initialised _chunks
            self.assertEqual(f._chunks, bitarray("00"))

    def test_splitting(self):
        """Tests that requests are split into multiple GETs when the cached data is already in chunks."""
        with StorageProxyFile("source", "content", "test-ff-256", chunk_size=16, expected_size=256, dp=self.dp) as f:
            f.seek(128)
            f.read(16)
            self.assertEqual(f._chunks, bitarray("0000000010000000"))
            f.seek(0)
            # Make sure the whole file is fetched.
            self.assertEqual(f.read(), b"\xff" * 256)
            self.assertEqual(f._chunks, bitarray("1111111111111111"))
            # ... but the request should have been split in two, so this will be the range of the second request.
            self.assertEqual(self.editor.get_last_request().range, [144, 255])

        with StorageProxyFile("source", "content", "test-ff-256", chunk_size=16, dp=self.dp) as f:
            f.seek(32)
            f.read(16)
            f.seek(128)
            f.read(64)
            # No expected_size, but size should be collected after the first GET
            self.assertEqual(f._chunks, bitarray("0010000011110000"))
            f.seek(0)

            # Monkeypatch to check the expected three requests are made
            getlist = []

            def _getchunks(start, end):
                bsize = f._chunk_size
                getlist.append((start * bsize, end * bsize - 1))
                f._content.seek(start * bsize)
                f._content.write(b"\xff" * (end - start) * bsize)
                f._chunks[start:end] = True

            f._request_chunks = _getchunks

            self.assertEqual(f.read(193), b"\xff" * 193)
            self.assertEqual(f._chunks, bitarray("1111111111111000"))
            self.assertEqual(getlist, [(0, 31), (48, 127), (192, 207)])

    def test_init_data(self):
        """Tests for correct behaviour when input is init_data=b'...'"""
        # This also confirms that no HTTP requests are made, or an error would be raised due to invalid "url" and
        #  no f._session value set.
        with StorageProxyFile(
            "source", "content", "foo", init_data=b"This is the file content to be used", dp=self.dp
        ) as f:
            self.assertEqual(f._size, 35)
            self.assertEqual(f.read(10), b"This is th")
            self.assertEqual(f.read(), b"e file content to be used")

    def test_get_hash(self):
        """Tests for correct behaviour when input is init_data=b'...'"""
        # This also confirms that no HTTP requests are made, or an error would be raised due to invalid "url" and
        #  no f._session value set.
        content = b"This is the file content to be used"
        fi = local.gen_api_content(io.BytesIO(content), azm.DataLabel.CONTENT)
        with StorageProxyFile("source", "content", "foo", init_data=content, file_info=fi) as f:
            self.assertEqual(f.get_hash(), "a2da980d45b8fef3281dd222bb45c6725d62fc8a8b3ffda8fabea8c5211cd85f")
        with StorageProxyFile("source", "content", "foo", init_data=content, file_info=fi, dp=self.dp) as f:
            self.assertEqual(f.get_hash(), "a2da980d45b8fef3281dd222bb45c6725d62fc8a8b3ffda8fabea8c5211cd85f")

    def test_tempfile(self):
        """Tests that tempfile works."""
        with StorageProxyFile("source", "content", "test-ff-256", chunk_size=16, expected_size=256, dp=self.dp) as spf:
            f = spf.get_tempfile()
            f.seek(128)
            self.assertEqual(f.read(16), b"\xff" * 16)
            f.seek(0)
            # Make sure whole file was fetched.
            self.assertEqual(f.read(), b"\xff" * 256)

    def test_filepath(self):
        """Tests that filepath works."""
        with StorageProxyFile("source", "content", "test-ff-256", chunk_size=16, expected_size=256, dp=self.dp) as spf:
            path = spf.get_filepath()
            with open(path, "rb") as f:
                f.seek(128)
                self.assertEqual(f.read(16), b"\xff" * 16)
                f.seek(0)
                # Make sure whole file was fetched.
                self.assertEqual(f.read(), b"\xff" * 256)
