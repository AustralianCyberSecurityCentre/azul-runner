"""Transform models to/from network event format and runner internal format."""

import copy
import datetime
import json
import logging

from azul_bedrock import models_network as azm
from pydantic import TypeAdapter

from .models import Event, EventData, EventParent, Feature, FeatureValue, JobResult
from .plugin import Multiplugin, Plugin

logger = logging.getLogger(__name__)


def get_registrations(plugin: Plugin) -> list[azm.PluginEvent]:
    """Generate registration messages for all multiplugins."""
    ret = []
    for multiplugin in plugin.get_registered_multiplugins():
        now = datetime.datetime.now(datetime.timezone.utc)
        # send all config that is not prefixed with 'secret_'
        # encode as json to get the real value to allow future access to it via the registration.
        safe_config = {x: json.dumps(y) for x, y in plugin.cfg.model_dump().items() if not x.startswith("secret_")}
        mp = plugin.get_multiplugin(multiplugin)
        description = plugin.DESCRIPTION
        if mp.description is not None and len(mp.description) > 0:
            description = mp.description

        author_details = azm.PluginEvent.Entity(
            **gen_author(plugin, mp).model_dump(exclude_defaults=True),
            description=description,
            # sort features by name
            features=TypeAdapter(list[Feature]).dump_python(sorted(plugin.FEATURES)),
            contact=plugin.CONTACT,
            # sort config by key
            config=dict(sorted(safe_config.items())),
        )
        reg_event = azm.PluginEvent(
            model_version=azm.CURRENT_MODEL_VERSION,
            kafka_key="runner-placeholder",
            entity=author_details,
            timestamp=now,
            author=author_details.summary(),
        )
        ret.append(reg_event)
    return ret


def gen_author(plugin: Plugin, mp: Multiplugin | None) -> azm.Author:
    """Return an author object for the multiplugin."""
    name = plugin.NAME
    version = plugin.VERSION
    security = plugin.SECURITY
    if mp:
        # add multiplugin name as a suffix
        if mp.name:
            name += f"-{mp.name}"
        # add multiplugin version as a suffix
        if mp.version:
            version += f"-{mp.version}"
        if mp.security:
            security = mp.security

    return azm.Author(
        category="plugin",
        name=name,
        version=version,
        security=security,
    )


def gen_processing_events(
    plugin: Plugin, posted: dict[str, azm.Datastream], author: azm.Author, src: azm.BinaryEvent, result: JobResult
) -> azm.StatusEvent:
    """Generates new output events based on the results of this plugin run."""
    # history/audit is added at every step that produces a message (including source events)
    # entity summary can be useful for retaining context about each step which would otherwise be
    # lost on a message as it decomposed into it's children
    # entity summary is expected to contain 'type', 'size' and optionally 'filename' for binaries
    # history also captures any 'relationship' information for how this message/child was derived
    # from its parent event/entity.  This will currently contain the plugin's relationship info.
    now = _get_now()
    out_events = []
    for event in result.events:
        if not event.parent:
            # event has no parent
            out_events.append(_gen_event_output(plugin, posted, author, src, event, now))
        else:
            out_events.append(_gen_event_extracted(plugin, posted, author, src, event, now))

    # do not track large fields from input event on outgoing status
    # this reduces the size of outgoing event
    min_src = copy.copy(src)  # avoid side effects
    min_src.entity = copy.copy(src.entity)  # avoid side effects
    min_src.entity.features = []
    min_src.entity.datastreams = []
    min_src.entity.info = {}

    # build outgoing status
    return azm.StatusEvent(
        model_version=azm.CURRENT_MODEL_VERSION,
        # provide a temporary id, as dispatcher must calculate this value
        kafka_key="runner-placeholder",
        timestamp=now,
        author=author,
        entity=azm.StatusEvent.Entity(
            input=min_src,
            status=result.state.label,
            error=result.state.failure_name or None,
            message=result.state.message or None,
            results=out_events,
            runtime=result.runtime,
        ),
    )


def _to_api_features(plugin: Plugin, feats_in: dict[str, set[FeatureValue]]) -> list[azm.FeatureValue]:
    """Convert a structured dict of features-by-name to a list of (name, value, <meta>) for the API.

    Any features that are not registered with the plugin are silently dropped,
    they should have been filtered during plugin execution.
    """
    ret = []
    for feature in plugin.FEATURES:
        for v in sorted(feats_in.get(feature.name, [])):
            fv = azm.FeatureValue(
                name=feature.name,
                type=feature.type,
                value=v.value_encoded(),
                label=v.label,
                offset=v.offset,
                size=v.size,
            )
            ret.append(fv)
    return ret


def _to_api_content(posted: dict[str, azm.Datastream], data: list[EventData]) -> list[azm.Datastream]:
    """Convert data from runner to the structure used by the restapi."""
    data = sorted(data)
    # update label and language for posted data
    return [
        posted[x.hash].model_copy(update=dict(label=x.label, language=x.language)) for x in data if x.hash in posted
    ]


def _gen_event_output(
    plugin: Plugin,
    posted: dict[str, azm.Datastream],
    author: azm.Author,
    src: azm.BinaryEvent,
    event: Event,
    now: datetime.datetime,
) -> azm.BinaryEvent:
    """Generate an enrichment event for existing entity."""
    # enriched events drop the entity.data of parent event
    action = azm.BinaryAction.Enriched
    if plugin._IS_USING_PUSHER:
        # Action is the same as whatever the source action is (sourced or mapped are the only two expected)
        action = src.action

    out_data = []
    extra_data = _to_api_content(posted, event.data)
    if extra_data:
        # this plugin run produced extra event data
        action = azm.BinaryAction.Augmented
        # Remove content entries other than the primary content stream and newly generated streams
        out_data += [ds for ds in (src.entity.datastreams or []) if ds.label == "content"]
        out_data += extra_data

    out_event = azm.BinaryEvent(
        model_version=azm.CURRENT_MODEL_VERSION,
        # provide a temporary id, as dispatcher must calculate this value
        kafka_key="runner-placeholder",
        action=action,
        timestamp=now,
        source=src.source.model_copy(
            update=dict(
                path=copy.copy(src.source.path)
                + [
                    azm.PathNode(
                        action=action,
                        timestamp=now,
                        author=author,
                        sha256=src.entity.sha256,
                    )
                ]
            )
        ),
        author=author,
        entity=src.entity.model_copy(
            update=dict(
                features=_to_api_features(plugin, event.features),
                datastreams=out_data,
                info=event.info,
            )
        ),
    )
    if plugin._IS_USING_PUSHER:
        # Pusher doesn't need any additional path added because the provided event is the output event..
        out_event.source.path.pop()
        src_entity_copy = src.entity.model_copy(deep=True)
        # # Merging features
        for base_feats in src_entity_copy.features:
            out_event.entity.features.append(base_feats)

        # Merging datastreams
        for base_stream in src_entity_copy.datastreams:
            is_stream_already_present = False
            for out_stream in out_event.entity.datastreams:
                if base_stream.label == out_stream.label:
                    is_stream_already_present = True
                    break
            # If the stream is already in the output don't add it again.
            if is_stream_already_present:
                continue

            out_event.entity.datastreams.append(base_stream)

        # Add base info if the plugin didn't add any.
        if src_entity_copy.info and not out_event.entity.info:
            out_event.entity.info = src_entity_copy.info
    return out_event


def _gen_event_extracted(
    plugin: Plugin,
    posted: dict[str, azm.Datastream],
    author: azm.Author,
    src: azm.BinaryEvent,
    event: Event,
    now: datetime.datetime,
) -> azm.BinaryEvent:
    """Generate an extraction event for a new entity."""
    # generate path between this event and the original source-entity submission
    path_extension = []
    c = event.as_parent()
    while c.parent:
        try:
            # For 'binary', entity.sha256 is equivalent to the hash of the 'content' stream
            pc: azm.Datastream | EventParent = posted[c.sha256]
        except KeyError:
            # try data-less submission
            pc = c

        # add to front of list
        path_extension.insert(
            0,
            azm.PathNode(
                action=azm.BinaryAction.Extracted,
                timestamp=now,
                author=author,
                relationship=c.relationship,
                sha256=c.sha256,
                file_format_legacy=pc.file_format_legacy,
                file_format=pc.file_format,
                size=pc.size,
                filename=c.filename,
                language=c.language,
            ),
        )
        # now inspect parent, unless it is top parent
        c = c.parent

    # generate output entity
    try:
        # For 'binary', entity.sha256 is equivalent to the hash of the 'content' stream
        pc = posted[event.sha256]
        out_entity: azm.BinaryEvent.Entity = pc.to_input_entity()
    except KeyError:
        # try data-less submission
        out_entity: azm.BinaryEvent.Entity = event.to_input_entity()

    out_entity.sha256 = event.sha256
    out_entity.features = out_entity.features + _to_api_features(plugin, event.features)
    out_entity.datastreams = _to_api_content(posted, event.data)
    out_entity.info = event.info

    return azm.BinaryEvent(
        model_version=azm.CURRENT_MODEL_VERSION,
        # provide a temporary id, as dispatcher must calculate this value
        kafka_key="runner-placeholder",
        action=azm.BinaryAction.Extracted,
        timestamp=now,
        source=src.source.model_copy(update=dict(path=copy.copy(src.source.path) + path_extension)),
        author=author,
        entity=out_entity,
    )


def _get_now():
    """Exists as cannot mock built-in datetime."""
    return datetime.datetime.now(datetime.timezone.utc)
