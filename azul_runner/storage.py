"""Defines a file-like interface to S3-compatible storage.

Used by plugins to access event binary data.
"""

import hashlib
import io
import logging
import re
import typing
from typing import Optional

import httpx
from azul_bedrock import dispatcher
from azul_bedrock import models_network as azm
from azul_bedrock.exceptions import DispatcherApiException
from bitarray import bitarray

from . import storage_spooled

# Matches the Content-Range: header in server responses
RANGE_RE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)$")

logger = logging.getLogger(__name__)

# Declare the hash algorithm to be used for data stream identification
DATA_HASH_NAME = "sha256"
DATA_HASH = hashlib.sha256


def calc_stream_hash(binaryio: typing.BinaryIO, hash_type=DATA_HASH) -> str:
    """Calculate the hexdigest of a binary file."""
    binaryio.seek(0)
    h = hash_type()
    while True:
        # read 1MB at a time
        buf = binaryio.read(2**20)
        if not buf:
            break
        h.update(buf)
    binaryio.seek(0)
    return h.hexdigest()


class StorageError(Exception):  # noqa: D204
    """StorageException.

    The storage system has failed in some way. This indicates a system fault such as a server timeout;
     user errors (eg seek to negative offset, read of closed file) should raise the usual exceptions.
    """

    pass


class ProxyFileNotFoundError(StorageError, FileNotFoundError):
    """Storage Proxy File backing not found."""


class StorageProxyFile(io.RawIOBase):
    """StorageProxyFile class.

    Access to storage/data proxy API via a file-like object.
    This is intended to avoid the need for fetching entire files/data streams if the plugin only needs part of it.

    Data will be fetched in multiples of chunk_size and cached for future httpx. The chunk_size should be relatively
     large in order to minimise number of GETs, especially when reading small pieces of data one after the other or
     spread out through a particular range; however, you may wish to reduce it if your network is slow and you
     know you will only need small, randomly-distributed parts of the file.
    NOTE: There is no support for resources that can change (eg if-modified-since etc), as this is intended to be used
     with immutable data objects identified by content hash.
    """

    tags: dict[str, str]  # Tags for the data stream - hashes, MIME types, whatever you want

    _url: str
    _session: httpx.Client  # a requests session allows us to re-use the same connection
    _offset: int
    _size: Optional[int]  # None if size is currently unknown.
    _chunk_size: int  # Chunk size for content caching
    _chunks: bitarray  # Bitfield indicating which chunks have data (may be zero-len)
    _content: storage_spooled.SpooledNamedTemporaryFile  # cached content data

    # If the last chunk is less than this fraction of a full chunk, fetch it and the second-last chunk together.
    MIN_LAST_CHUNK_SIZE: float = 0.66

    def __init__(
        self,
        source: str,
        label: azm.DataLabel,
        hash: str,
        *,
        dp: dispatcher.DispatcherAPI = None,
        request_timeout: int = 10,
        # retrieve in 1mb chunks
        chunk_size: int = 1_048_576,
        # cache ~20 mb in memory before moving to disk
        mem_cache_limit: int = 20_971_520,
        init_data: bytes | None = None,
        expected_size: int | None = None,
        file_info: azm.Datastream | None = None,
        allow_unbounded_read: bool = True,
    ):
        """Init.

        :param url: URL of the file to proxy
        :param request_timeout: Timeout for HTTP connections and requests
        :param chunk_size: Size of data chunks for caching; always request full chunks for every read.
                           This should be relatively large, to avoid excessive numbers of GET requests (default 256k).
        :param mem_cache_limit: size limit for memory caching before falling back to a disk file (default 64M).
        :param init_data: Initialise the instance with existing byte data. This must be the full content of the file;
                          if this is set, url is ignored and no HTTP requests are made.
        :param expected_size: Initialises our data structures for the expected size of the file. This saves us from
                              having to fetch the first chunk of file to determine the size on first access.
        """
        self._source = source
        self._label = label
        self._hash = hash

        self._timeout = request_timeout
        self._content = storage_spooled.SpooledNamedTemporaryFile(max_size=mem_cache_limit)
        self._offset = 0
        self._dispatcher = dp
        self._allow_unbounded_read = allow_unbounded_read
        self.file_info = file_info
        self._content._max_size
        if isinstance(init_data, bytes):
            logger.debug("Init StorageProxyFile with init_data (%d bytes), hash=%s" % (len(init_data), self._hash))
            # noinspection PyTypeChecker
            self._session = None  # Unset because full data is already provided.
            self._content.write(init_data)
            self._chunk_size = self._size = len(init_data)
            self._chunks = bitarray([True])
            if expected_size is not None and expected_size != len(init_data):
                raise ValueError("expected_size does not match provided init_data")
        else:
            logger.debug(
                "Init StorageProxyFile with hash=%s%s"
                % (self._hash, (", expected size %d" % expected_size) if expected_size else ", size unknown")
            )
            self._session = httpx.Client(timeout=5.0)
            self._chunk_size = chunk_size
            self._size = expected_size
            self._chunks = bitarray()
            if self._size is not None:
                self._recalc_chunks(self._size)

    def __setstate__(self, state: dict):
        """Used to set the state of the object after pickling."""
        self._source = state.get("_source", None)
        self._label = state.get("_label", None)
        self._hash = state.get("_hash", None)
        self._timeout = state.get("_timeout", None)
        self._content = storage_spooled.SpooledNamedTemporaryFile(max_size=state.get("mem_cache_limit", 20_971_520))
        self._content.write(state.get("_content", b""))
        self._offset = state.get("_offset", None)
        self._dispatcher = state.get("_dispatcher", None)
        self._allow_unbounded_read = state.get("_allow_unbounded_read", None)
        self.file_info = state.get("file_info", None)
        self._session = state.get("_session", None)
        self._chunk_size = state.get("_chunk_size", None)
        self._chunks = state.get("_chunks", None)
        self._size = state.get("_size", None)

    def __getstate__(self):
        """Used to save the state of an object prior to pickling."""
        # Ensure full content is retrieved.
        self._retrieve(-1)
        return {
            "_source": self._source,
            "_label": self._label,
            "_hash": self._hash,
            "_timeout": self._timeout,
            "mem_cache_limit": self._content._max_size,
            # Need to read content in because buffered random can't be read in.
            "_content": self._content.read(),
            "_offset": self._offset,
            "_dispatcher": self._dispatcher,
            "_allow_unbounded_read": self._allow_unbounded_read,
            "file_info": self.file_info,
            "_session": self._session,
            "_chunk_size": self._chunk_size,
            "_chunks": self._chunks,
            "_size": self._size,
        }

    def get_tempfile(self) -> storage_spooled.SpooledNamedTemporaryFile:
        """Read all data and return the temporary file.

        This is useful for third party libraries that take a file object, as it
        can reduces memory usage vs a bytearray.
        """
        # go to beginning of file
        self.seek(0)
        # read all bytes into temporary file
        self._retrieve(-1)
        return self._content

    def get_filepath(self) -> str:
        """Read all data and return a filepath.

        This is useful for third party libraries that take a file path, as it
        can reduces memory usage vs a bytearray.
        """
        # go to beginning of file
        self.seek(0)
        # read all bytes into temporary file
        self._retrieve(-1)
        # force write to disk if under size threshold
        self._content.rollover()
        return self._content.name

    def get_hash(self) -> str:
        """Return hash for identifying this data."""
        return self.file_info.sha256

    def _recalc_chunks(self, size: int):
        """(Re)initialises the chunks array for our file based on a given file size."""
        logger.debug("Recalculating chunks (size %d) for %s" % (size, self._hash))
        new_len = 1 + (size - 1) // self._chunk_size
        if len(self._chunks) > new_len:
            raise RuntimeError("Cannot shrink existing StorageProxyFile - unexpected change in size")
        b = bitarray(new_len - len(self._chunks))
        b.setall(False)
        self._chunks.extend(b)

    def close(self) -> None:
        """Close and delete the underlying file or network connection."""
        if self._session:
            self._session.close()
        self._content.close()
        del self._content
        del self._chunks
        super(StorageProxyFile, self).close()

    def _assert_size(self, size: int) -> None:
        """assert_size; asserts that the object's size hasn't changed."""
        logger.debug("Assert size: %s == %s" % (self._size, size))
        if self._size != size:
            raise AssertionError("Object size mismatch for %s" % self._hash)

    def _assert_content_len(self, resp: httpx.Response) -> None:
        """assert_content_len. Verifies that the content's length matches the Content-Length header."""
        if "Content-Length" in resp.headers and int(resp.headers["Content-Length"]) != len(resp.content):
            raise StorageError("Content length mismatch for %s" % self._hash)

    def _make_request(self, start_pos: Optional[int], end_pos: Optional[int]) -> httpx.Response:
        """make_request. Wraps the GET request and handles errors, raising the appropriate exceptions."""

        def s_fmt(v):
            return v if v is not None else ""

        try:
            logger.debug(f"Requesting bytes={s_fmt(start_pos)}-{s_fmt(end_pos)} of {s_fmt(self._hash)}")
            resp = self._dispatcher.get_binary(
                source=self._source, label=self._label, sha256=self._hash, start_pos=start_pos, end_pos=end_pos
            )
        except DispatcherApiException as e:
            if e.status_code == 404:
                raise ProxyFileNotFoundError(2, "Got 404 requesting %s" % self._hash)
            elif e.status_code == 416:  # Range not satisfiable; return this to the caller for processing
                resp = e.response
            else:
                raise StorageError("Got %s requesting %s" % (e.status_code, self._hash)) from e
        if resp.headers["Content-Type"] != "application/octet-stream":
            raise StorageError(
                "Server provided %s instead of octet-stream for %s" % (resp.headers["Content-Type"], self._hash)
            )
        return resp

    def _handle_full_content(self, resp: httpx.Response):
        """When called, takes a response containing the full file and sets self._content/self._chunks appropriately."""
        self._assert_content_len(resp)
        if self._size:
            self._assert_size(len(resp.content))
        if "Content-Range" in resp.headers:
            raise StorageError("Content-Range header unexpectedly present in 200 response for %s" % self._hash)
        # Handle the file as a single 'chunk', as no further fetches are needed.
        self._chunk_size = self._size = len(resp.content)
        self._chunks = bitarray([True])
        self._content.seek(0)
        self._content.write(resp.content)

    def _parse_partial_response(self, resp: httpx.Response) -> tuple[str, str, str]:
        """Return (request start, end, total file size) if response is valid."""
        self._assert_content_len(resp)
        if "Content-Range" not in resp.headers:
            # RFC specifies 'server MUST generate a Content-Range header'; if missing, server is broken.
            raise StorageError("Server failed to provide Content-Range header for %s" % self._hash)
        if "bytes */" in resp.headers["Content-Range"]:
            # 'unsatisfied range' response header - RFC implies this is only valid with 416 code
            raise StorageError("Server returned 206 with 'Content-Range: bytes */...' for %s" % self._hash)
        hits = RANGE_RE.match(resp.headers["Content-Range"])
        if not hits:
            raise StorageError(
                "Could not parse 'Content-Range: %s' (requesting %s)" % (resp.headers["Content-Range"], self._hash)
            )
        s, e, t = hits.groups()  # Ensures that there are exactly 3 hits to unpack
        return s, e, t

    def _request_chunks(self, start_chunk: Optional[int], end_chunk: Optional[int]) -> None:
        """Handles requests for file data.

        end_chunk is exclusive, like slice notation; fetch up to end_chunk - 1.
        This method fetches the data, updates self._chunks, and stores the data to self._content.

        Special cases:
            end_chunk = None reads all remaining data, the same as read(-1).
            start_chunk = None reads the last chunk of file (used to definitively determine size; end_chunk ignored)
        """
        # Variable naming:
        #  start_chunk, end_chunk - the chunks to be fetched
        #  req_start, req_end - the byte ranges requested from the server
        #  resp_start, resp_end, resp_total - the range the server actually sent (and total size, if provided)

        if self.closed:
            raise ValueError("_request_chunks called on closed file")
        if start_chunk is not None and start_chunk < 0:
            raise ValueError("start_chunk cannot be negative")
        logger.debug("Get chunks %s...%s for %s" % (start_chunk, end_chunk, self._hash))
        if end_chunk is not None and start_chunk is not None:
            if end_chunk < start_chunk:
                raise ValueError("end_chunk must be >= start_chunk")
            if end_chunk == start_chunk:
                return  # No-op
            if self._size is not None and start_chunk > (self._size - 1) // self._chunk_size:
                return

        if start_chunk is None:
            req_start = None
            req_end = self._chunk_size  # Note: not negative, because it will be subbed into '-%s'
        else:
            req_start = start_chunk * self._chunk_size
            if end_chunk is None:
                req_end = None
            else:
                # end_chunk is exclusive, so need to request up to the byte just before that chunk
                req_end = end_chunk * self._chunk_size - 1
                if self._size is not None and req_end >= self._size:
                    req_end = self._size - 1

        resp = self._make_request(req_start, req_end)
        if resp.status_code == 416:
            # Range not satisfiable -- must be past EOF
            if req_start is None:
                raise StorageError("Server returned 416 for a suffix range request - broken server?")
            if "Content-Range" in resp.headers:
                # server 'SHOULD' generate a Content-Range header with current length in a 416
                cr: str = resp.headers["Content-Range"]
                if not cr.startswith("bytes */"):
                    raise StorageError(
                        "Server returned unexpected Content-Range header for 416 ('%s')"
                        % resp.headers["Content-Range"]
                    )
                else:
                    if cr == "bytes */*":
                        logger.warning("Server failed to return content size in 416 Content-Range header")
                    else:
                        self._size = int(cr[8:])
            return  # Nothing further to do
        elif resp.status_code == 200:
            # Full content returned, even if it wasn't requested.
            self._handle_full_content(resp)
        elif resp.status_code == 206:
            # Partial Content
            resp_start, resp_end, resp_total = self._parse_partial_response(resp)
            if resp_total != "*":
                if self._size is None:
                    self._size = int(resp_total)
                    self._recalc_chunks(self._size)
                    if req_start is None:
                        # This was a request for the last chunk of file; adjust expected response values
                        req_start = max(self._size - self._chunk_size, 0)
                        req_end = self._size - 1
                        end_chunk = 1 + req_end // self._chunk_size  # end_chunk is exclusive
                        start_chunk = end_chunk - 1
                    elif req_end is None or req_end >= self._size:
                        # Adjust the expected end byte to avoid raising an exception below
                        req_end = self._size - 1
                else:
                    self._assert_size(int(resp_total))
            elif req_start is None:
                # Get 'the last <chunk_size> bytes of the file'; calculate length from the response
                req_end = int(resp_end)
                self._size = req_end + 1
                req_start = max(self._size - self._chunk_size, 0)
                end_chunk = 1 + req_end // self._chunk_size  # end_chunk is exclusive
                start_chunk = end_chunk - 1  # Only asked for one chunk
                self._recalc_chunks(self._size)
            elif req_end is None:
                # Get "all the rest of the file", so assume the end of the response is the total size
                req_end = int(resp_end)
                end_chunk = 1 + req_end // self._chunk_size  # end_chunk is exclusive
                if self._size is None:
                    self._size = req_end + 1
                    self._recalc_chunks(self._size)
                else:
                    self._assert_size(req_end + 1)

            if (int(resp_start), int(resp_end)) != (req_start, req_end):
                if self._size is None and int(resp_start) == req_start:
                    # Got less than the total requested size; assume it's because of EOF
                    self._size = int(resp_end) + 1
                    self._recalc_chunks(self._size)
                else:
                    raise StorageError(
                        "Server returned unrequested range: req %s != resp %s (requesting %s)"
                        % (resp.request.headers["Range"], resp.headers["Content-Range"], self._hash)
                    )
            self._assert_content_len(resp)
            # Add data to cache
            self._content.seek(int(resp_start))
            self._content.write(resp.content)
            if self._size is None and end_chunk > len(self._chunks):
                # Don't yet know the true size, but need to extend our chunks to mark the data acquired so far.
                self._recalc_chunks(int(resp_end) + 1)
            self._chunks[start_chunk:end_chunk] = True
        else:
            # If this point is reached, it's not an error but also not '200 OK' or '206 Partial Content'
            raise StorageError("Unexpected response code %s while requesting %s" % (resp.status_code, self._hash))

    def _retrieve(self, size: int):
        """Retrieve bytes from remote storage into the temporary file."""
        if not isinstance(size, int):
            raise TypeError("read length must be an int")
        if self.closed:
            raise ValueError("read of closed file")
        if size < -1:
            raise ValueError("read length must be positive or -1")
        logger.debug("Read %s@%s for %s" % (size, self._offset, self._hash))
        if self._size is not None:
            # Limit size to the remaining size of data in the file
            if size == -1:
                size = max(self._size - self._offset, 0)
            else:
                size = max(min(self._size - self._offset, size), 0)
        if size == 0:
            logger.debug("Short-circuiting zero-size read")
            return b""

        # Calculate start and end chunks. end_chunk is exclusive, like in slice notation.
        start_chunk = self._offset // self._chunk_size
        if size == -1:
            # This implies that self._size is None, else size would have been calculated above
            end_chunk = -1
        else:
            end_chunk = 1 + (self._offset + size - 1) // self._chunk_size

        if self._size is not None:
            # Special-case the last chunk together with the second-last if it's small
            if len(self._chunks) > 1 and (
                (self._size - 1) % self._chunk_size + 1 < self.MIN_LAST_CHUNK_SIZE * self._chunk_size
            ):
                if end_chunk == len(self._chunks) - 1:
                    end_chunk += 1
                elif start_chunk == len(self._chunks) - 1:
                    start_chunk -= 1

        # Check for any missing data chunks in the desired range that will need to be fetched
        while not all(self._chunks[start_chunk:end_chunk]):
            # Find first missing chunk
            next_req_start = self._chunks.index(False, start_chunk, end_chunk)
            # Find the first cached chunk after that (if any) in the requested range,
            #  so the same data isn't re-requested.
            if any(self._chunks[next_req_start:end_chunk]):
                next_req_end = self._chunks.index(True, next_req_start, end_chunk)
            else:
                next_req_end = end_chunk
            self._request_chunks(next_req_start, None if next_req_end == -1 else next_req_end)
        if self._size is None and (end_chunk == -1 or len(self._chunks) < end_chunk):
            # This should only happen if NO chunks have been acquired yet, or if some chunks have been fetched
            #  but the server hasn't told us the total file size.
            self._request_chunks(max(start_chunk, len(self._chunks)), None if end_chunk == -1 else end_chunk)

        # reset seek position in case it was changed during collect
        self._content.seek(self._offset)

    def read(self, size: int = -1) -> bytes:
        """Read the provided number of bytes or an unbounded read of the whole file."""
        if size < 0 and not self._allow_unbounded_read:
            raise ValueError("unbounded/full read() calls are disabled due to memory usage")
        self._retrieve(size)
        # Should now have all the required chunks
        rv = self._content.read(size)
        self._offset += len(rv)
        return rv

    def seekable(self) -> bool:
        """Is the file seekable - Always True."""
        return True

    def readable(self) -> bool:
        """Is the file is readable - Always True."""
        return True

    def writable(self) -> bool:
        """Is the file is writable - Always False."""
        return False

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Seek to a specified offset within the file."""
        if not isinstance(offset, int):
            raise TypeError("offset must be an int")
        if self.closed:
            raise ValueError("seek of closed file")

        if whence == io.SEEK_SET:
            self._offset = offset
        elif whence == io.SEEK_CUR:
            self._offset += offset
        elif whence == io.SEEK_END:
            if self._size is None:
                # Request the last full chunk of the file. It's the only way to be 100% sure of finding the total size.
                self._request_chunks(start_chunk=None, end_chunk=-1)
            self._offset = self._size + offset
        else:
            raise ValueError("whence value %s unsupported" % whence)
        if self._offset < 0:
            raise OSError(22, "Invalid argument")  # Duplicate behaviour of regular file
        # No errors for offsets beyond end of file; they just return nothing from read()

        return self._offset

    def tell(self) -> int:
        """Get the current offset without moving it."""
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self._offset
