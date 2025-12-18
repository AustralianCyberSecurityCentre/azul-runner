from __future__ import annotations

import time
import unittest

import httpx
from azul_bedrock import dispatcher

from . import mock_dispatcher as md


class TestMockFileServer(unittest.TestCase):
    def setUp(self) -> None:
        md.clear()
        return super().setUp()

    @classmethod
    def setUpClass(cls) -> None:
        cls.mock_server = md.MockDispatcher()
        cls.mock_server.start()
        while not cls.mock_server.is_alive():
            time.sleep(0.2)  # Wait for server to start
        cls.server = "http://%s:%s" % (cls.mock_server.host, cls.mock_server.port)
        cls.dp = dispatcher.DispatcherAPI(
            events_url=cls.server,
            data_url=cls.server + "/mock",
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.mock_server.stop()
        cls.mock_server.kill()

    def test_get_zero_file(self):
        self.editor.set_stream("test-zeros-100", 200, b"\x00" * 100)
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-zeros-100", headers={"range": "bytes=0-50"})
        self.assertEqual(resp.status_code, 206, resp.content)
        self.assertEqual(resp.content, b"\x00" * 51)
        self.assertEqual(resp.headers["content-range"], "bytes 0-50/100")
        last = self.editor.get_last_request()
        self.assertEqual(last.range_raw, "bytes=0-50")
        self.assertEqual(last.range, [0, 50])

        self.editor.set_stream("test-zeros-50", 200, b"\x00" * 50)
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-zeros-50", headers={"range": "bytes=-30"})
        self.assertEqual(resp.status_code, 206, resp.content)
        self.assertEqual(resp.content, b"\x00" * 30)
        self.assertEqual(resp.headers["content-range"], "bytes 20-49/50")
        last = self.editor.get_last_request()
        self.assertEqual(last.range_raw, "bytes=-30")
        self.assertEqual(last.range, [20, 49])

        self.editor.set_stream("test-zeros-20", 200, b"\x00" * 20)
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-zeros-20", headers={"range": "bytes=-50"})
        self.assertEqual(resp.status_code, 206, resp.content)
        self.assertEqual(resp.content, b"\x00" * 20)
        self.assertEqual(resp.headers["content-range"], "bytes 0-19/20")
        last = self.editor.get_last_request()
        self.assertEqual(last.range_raw, "bytes=-50")
        self.assertEqual(last.range, [0, 19])

        self.editor.set_stream("test-zeros-100", 200, b"\x00" * 100)
        resp = httpx.get(
            self.server + "/api/v3/stream/source/content/test-zeros-100", headers={"range": "bytes=50-500"}
        )
        self.assertEqual(resp.status_code, 206, resp.content)
        self.assertEqual(resp.content, b"\x00" * 50)
        self.assertEqual(resp.headers["content-range"], "bytes 50-99/100")
        last = self.editor.get_last_request()
        self.assertEqual(last.range_raw, "bytes=50-500")
        self.assertEqual(last.range, [50, 99])

        self.editor.set_stream("test-zeros-256", 200, b"\x00" * 256)
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-zeros-256", headers={"range": "bytes=256-"})
        self.assertEqual(resp.status_code, 416, resp.content)
        self.assertEqual(resp.content, b"")
        self.assertEqual(resp.headers["content-range"], "bytes */256")
        last = self.editor.get_last_request()
        self.assertEqual(last.range_raw, "bytes=256-")
        self.assertEqual(last.range, None)

        self.editor.set_stream("test-zeros-100", 200, b"\x00" * 100)
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-zeros-100", headers={"range": "bytes=500-"})
        self.assertEqual(resp.status_code, 416, resp.content)
        self.assertEqual(resp.content, b"")
        self.assertEqual(resp.headers["content-range"], "bytes */100")
        last = self.editor.get_last_request()
        self.assertEqual(last.range_raw, "bytes=500-")
        self.assertEqual(last.range, None)

    def test_get_ff_file(self):
        self.editor.set_stream("test-ff-200", 200, b"\xff" * 100)
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-ff-200", headers={"range": "bytes=0-50"})
        self.assertEqual(resp.status_code, 206, resp.content)
        self.assertEqual(resp.content, b"\xff" * 51, len(resp.content))
        last = self.editor.get_last_request()
        self.assertEqual(last.range_raw, "bytes=0-50")
        self.assertEqual(last.range, [0, 50])

    def test_get_mod256(self):
        self.editor.set_stream("test-mod256", 200, bytes(range(256)))
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-mod256")
        self.assertEqual(resp.status_code, 200, resp.content)
        last = self.editor.get_last_request()
        self.assertEqual(last.range, None)
        self.assertEqual(len(resp.content), 256)

        self.editor.set_stream("test-mod256", 200, bytes(range(256)))
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-mod256", headers={"range": "bytes=50-500"})
        self.assertEqual(resp.status_code, 206, resp.content)
        self.assertEqual(resp.headers["content-range"], "bytes 50-255/256")
        last = self.editor.get_last_request()
        self.assertEqual(last.range, [50, 255])
        self.assertEqual(len(resp.content), 206)

        self.editor.set_stream("test-mod256", 200, bytes(range(256)))
        resp = httpx.get(self.server + "/api/v3/stream/source/content/test-mod256", headers={"range": "bytes=-2"})
        self.assertEqual(resp.status_code, 206, resp.content)
        self.assertEqual(resp.headers["content-range"], "bytes 254-255/256")
        last = self.editor.get_last_request()
        self.assertEqual(last.range, [254, 255])
        self.assertEqual(len(resp.content), 2)
