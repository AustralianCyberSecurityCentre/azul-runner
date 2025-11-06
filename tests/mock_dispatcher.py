"""Dispatcher API Mocking with enough state-tracking to be useful for testing runner."""

import copy
import datetime
import hashlib
import json
import logging
import os
import re
from typing import Any

import urllib3
from azul_bedrock import models_api as azapi
from azul_bedrock import models_network as azm
from azul_bedrock.mock import Editor, MockDispatcher, dp
from fastapi import Request, Response

app = dp.app


class RawResponse(Response):
    media_type = "binary/octet-stream"

    def render(self, content: bytes) -> bytes:
        if not isinstance(content, bytes):
            raise Exception(f"bad response type {type(content)} {content}")
        return content


logger = logging.getLogger(__name__)


REQUEST_RANGE_RE = re.compile(r"bytes=([0-9]*)-([0-9]*)")

# Content for GET /test-mockfile/{foo>
mock_file_data: dict[str, bytes] = {}
posted_files: dict[str, bytes] = {}  # Stores files for later retrieval

vars = {}


def clear():
    global vars, mock_file_data, posted_files
    mock_file_data = {
        "mod256": bytes(range(256)),
    }
    # Stores files for later retrieval
    posted_files = {
        "b4b389c849d799d9331d5937cde7f0dfd297d76083242366cbef53b498cd6051": b"small content",
    }

    # note - not configured/updated for all endpoints
    vars = {
        "last_req_params": {},  # query parameters
        "last_request_body": b"",
        "fetch_count": 0,
        "all_requests": [],
    }


clear()

# ################
# Bottle methods


@app.get("/mock/get_var/{var}")
def mock_get_var(var: str, request: Request, response: Response) -> Any:
    """
    Retrieve a var from the mock server; used to check correct parameters were passed.
    :param var: Name of var
    :return: Value of the var
    """
    response.status_code = 200
    try:
        if var == "all_requests":
            # Format response
            resp = "["
            for val in vars[var]:
                resp += val.decode("utf-8") + ","
            resp = resp[:-1] + "]"
            # Reset requests
            vars[var] = []
            content = resp.encode()
        else:
            content = vars[var]

    except KeyError:
        response.status_code = 400
        return

    if not isinstance(content, bytes):
        content = json.dumps(content).encode()

    return RawResponse(content, status_code=response.status_code, headers=response.headers)


#
# events
#


@app.post("/{path}/api/v2/event")
async def post_event(path: str, request: Request, response: Response) -> azapi.ResponsePostEvent:
    """Called to create new output events on plugin completion"""
    body = await request.body()
    # logger.debug(f"body is {body}")
    vars["last_request_body"] = body
    vars["all_requests"] += [body]

    data = json.loads(body)

    # mock processing all events successfully
    return azapi.ResponsePostEvent(total_ok=len(data), total_failures=0, failures=[])


@app.get("/{variant}/api/v2/event/{ent_type}/active")
async def get_events_switchable(
    variant: str, ent_type: str, request: Request, response: Response
) -> dict | bytes | str:
    """Returns events based on required variant."""
    vars["fetch_count"] += 1
    # save params multi items data structure into dict of list of str
    params = {}
    for k, v in request.query_params.multi_items():
        params.setdefault(k, []).append(v)
    vars["last_req_params"] = params

    respInfo = azapi.GetEventsInfo(filtered=5, fetched=1, ready=True)
    respEvents = azapi.GetEventsBinary(
        events=[
            azm.BinaryEvent(
                model_version=azm.CURRENT_MODEL_VERSION,
                kafka_key="test-dummy",
                dequeued="test-dummy-dequeued",
                action=azm.BinaryAction.Enriched,
                timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                source=azm.Source(
                    name="source",
                    path=[],
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                ),
                author=azm.Author(name="TestServer", category="blah"),
                entity=azm.BinaryEvent.Entity(sha256="1234", datastreams=[], features=[]),
            )
        ],
    )
    if ent_type == "binary":
        if variant == "test_path":
            # returns a binary event message with path
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/binary_event_message.json")
            respEvents.events = json.loads(open(path, "rb").read())
        elif variant == "test_data":
            # returns an entity with data streams
            respEvents.events[0].entity.datastreams = [
                azm.Datastream(
                    identify_version=1,
                    label=azm.DataLabel.CONTENT,
                    size=13,
                    sha256=hashlib.sha256(b"small content").hexdigest(),
                    sha1="1",
                    md5="5",
                    sha512="512",
                    mime="mt",
                    magic="mm",
                    file_format_legacy="ft",
                    file_format="#TEST/ONLY",
                ),
                azm.Datastream(
                    identify_version=1,
                    label=azm.DataLabel.TEST,
                    size=54,
                    sha256="ed12278c588dcf2932a7db54cf917d8d4da74ce810cf8b7935112174ff2e0df6",
                    # hash of b'This is test data that should be fetched by the runner'
                    # other hashes are not currently used for identification
                    sha1="1",
                    md5="5",
                    sha512="512",
                    mime="mt",
                    magic="mm",
                    file_format_legacy="ft",
                    file_format="#TEST/ONLY",
                ),
            ]
            respEvents.events[0].source.path = [
                azm.PathNode(
                    author=azm.Author(name="foo"),
                    action=azm.BinaryAction.Extracted,
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                    sha256="1234",
                ),
            ]
        elif variant == "depth_dupe":
            # returns a dummy entity with a source history containing the same author twice
            respEvents.events[0].source.path = [
                azm.PathNode(
                    author=azm.Author(name="foo"),
                    action=azm.BinaryAction.Enriched,
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                    sha256="42",
                ),
                azm.PathNode(
                    author=azm.Author(name="foo"),
                    action=azm.BinaryAction.Sourced,
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                    sha256="1234",
                ),
            ]
            respEvents.events[0].entity.features = [azm.FeatureValue(name="tester", type="integer", value="100")]
        elif variant == "depth_2":
            # returns a dummy entity with a source history two deep from different authors
            respEvents.events[0].source.path = [
                azm.PathNode(
                    author=azm.Author(name="foo"),
                    action=azm.BinaryAction.Enriched,
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                    sha256="42",
                ),
                azm.PathNode(
                    author=azm.Author(name="bar"),
                    action=azm.BinaryAction.Sourced,
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                    sha256="1234",
                ),
            ]
            respEvents.events[0].entity.features = [azm.FeatureValue(name="tester", type="integer", value="100")]
        elif variant == "depth_1":
            # returns a dummy entity with a source history one deep
            respEvents.events[0].source.path = [
                azm.PathNode(
                    author=azm.Author(name="foo"),
                    action=azm.BinaryAction.Enriched,
                    timestamp=datetime.datetime(year=1900, month=1, day=1, tzinfo=datetime.timezone.utc),
                    sha256="1234",
                ),
            ]
            respEvents.events[0].entity.features = [azm.FeatureValue(name="tester", type="integer", value="100")]
        elif variant == "null_sleep":
            # returns a dummy entity with a test feature
            respEvents.events[0].entity.features = [azm.FeatureValue(name="tester", type="integer", value="100")]
        elif variant == "null":
            pass
        elif variant == "invalid":
            respEvents = "This is not valid JSON. This should cause an exception."
        elif variant == "too_many":
            respEvents.events.append(copy.deepcopy(respEvents.events[0]))
        else:
            response.status_code = 404
        content, response.headers["Content-Type"] = urllib3.encode_multipart_formdata(
            {
                "info": (None, respInfo.model_dump_json()),
                "events": (None, respEvents.model_dump_json()),
            }
        )
        return Response(content=content, media_type=response.headers["Content-Type"])
    else:
        response.status_code = 404
