"""Subclass of Plugin designed for processing binary files."""

from azul_bedrock import models_network

from . import plugin, settings


class BinaryPlugin(plugin.Plugin):
    """Base class for plugins that want to run on new sightings of a binary.

    See readme.md for usage information.
    """

    SETTINGS = settings.add_settings(
        # plugin code generally assumes it can access data
        assume_streams_available=True,
        # event must be a new sighting of a label='content' binary
        filter_allow_event_types=[
            models_network.BinaryAction.Sourced,
            models_network.BinaryAction.Extracted,
        ],
    )
