"""Plugin template for handling plugin metadata and execution methods.

Coordination of execution and parsing of results is beyond the scope of this module.
"""

import logging
import typing
from typing import Callable, ClassVar, Iterable, Optional, Type, Union

from azul_bedrock.models_network import FeatureType
from pydantic import BaseModel

from . import settings
from .models import Event, Feature, Job, State

logger = logging.getLogger(__name__)


class Multiplugin(BaseModel):
    """Class to store data about multi-plugins."""

    name: str | None = None
    version: str | None = None
    security: str | None = None
    description: str | None = None
    callback: Callable[[Job], Optional[Union[State, State.Label]]]


class Plugin:
    """Defines the template for an Azul 3 plugin."""

    # Plaintext security string applied to all results of a plugin.
    # Examples:
    # SECURITY = "UNCLASSIFIED"
    # SECURITY = "UNCLASSIFIED TLP:GREEN"
    # SECURITY = "AMAZING TOMATO TLP:GREEN REL:ME,YOU,THATONE"
    SECURITY: ClassVar[Optional[str]] = None

    # Default settings for this plugin. This is accumulated through all descending classes.
    # Child classes will inherit unless they override in their own SETTINGS.
    SETTINGS: ClassVar[Type[settings.PluginBaseSettings]] = settings.PluginBaseSettings

    NAME: ClassVar[str]  # Plugin name, set automatically from class name.
    DESCRIPTION: ClassVar[str]  # Plugin description, set automatically from class comment.
    VERSION: ClassVar[str] = ""  # Plugin version
    CONTACT: ClassVar[Optional[str]] = None  # Contact point for issues with this plugin

    # Toggle output entity type to mapped from if no output streams are provided. (must be enriched or mapped)
    _IS_USING_PUSHER: ClassVar[bool] = False

    # List of features that the plugin produces.
    FEATURES: ClassVar[Iterable[Feature]] = {
        # Standard features set by all binary plugins.
        # Child classes that set this property will ADD to this feature set.
        Feature(name="malformed", desc="File is malformed in some way.", type=FeatureType.String),
        # these features are set from bedrock code as part of generating the binary entity
        Feature(name="file_extension", desc="File extension of the 'content' stream.", type=FeatureType.String),
        Feature(name="file_format", desc="Assemblyline file type of the 'content' stream.", type=FeatureType.String),
        Feature(name="file_format_legacy", desc="Azul file type of the 'content' stream.", type=FeatureType.String),
        Feature(name="filename", desc="Name on disk of the 'content' stream.", type=FeatureType.Filepath),
        Feature(name="magic", desc="File magic found for the 'content' stream.", type=FeatureType.String),
        Feature(name="mime", desc="Mimetype found for the 'content' stream.", type=FeatureType.String),
    }

    def __init__(self, config: settings.Settings | dict | None = None) -> None:
        if not isinstance(config, settings.Settings):
            config = settings.parse_config(type(self), config or {})
            logger.warning("plugin started using simple config mode, this is intended only for tests")
        config = self._alter_config(config)

        # parse config
        self.cfg = config
        # log settings if they are interesting
        to_print = self.cfg.model_dump(exclude_defaults=True, exclude_unset=True)
        if to_print:
            logger.info("custom startup options:")
            for k, v in self.cfg.model_dump(exclude_defaults=True, exclude_unset=True).items():
                logger.info(f"{k:20}: {v}")

        self.shutting_down = False

        # add suffix to name and version
        self.NAME = f"{type(self).__name__}"
        self.NAME = self.NAME.replace(self.cfg.name_remove_prefix, "", 1)
        if ns := self.cfg.name_suffix:
            self.NAME += f"-{ns}"
        if vs := self.cfg.version_suffix:
            self.VERSION += f"-{vs}"
        # load description from docstring
        desc = type(self).__doc__
        self.DESCRIPTION = desc if desc else ""

        # create logger for plugin creator to use
        self.logger = logging.getLogger("azul.plugin." + self.NAME.lower())

        # add the multiplugin for execute()
        self._multiplugins: dict[str | None, Multiplugin] = {
            None: Multiplugin(name=None, version=None, callback=lambda x: self.execute(x), description=None)
        }

        # travel inheritance hierarchy for all defined features
        features = {}
        for c in reversed(type(self).__mro__):
            if hasattr(c, "FEATURES"):
                for f in c.FEATURES:
                    features[f.name] = f
        self.FEATURES = sorted(features.values())

        # override security through config
        self.SECURITY = self.cfg.security_override or self.SECURITY
        if self.SECURITY:
            logger.info(f"running with security: {self.SECURITY}")
        if self.SECURITY and not isinstance(self.SECURITY, str):
            logger.error(f"self.SECURITY is {self.SECURITY}, not a string")
            exit(1)

    def _alter_config(self, config: settings.Settings) -> settings.Settings:
        """Implement this function to edit config immediately after config generation.

        This is also a function for editing other metadata of the plugin including FEATURES.
        Helpful for plugins that dynamically figure out features that are needed.
        """
        return config

    def reset(self, job: Job):
        """Clear the current result object."""
        self._job = job
        # binary data generated by plugin
        # hash -> binary, tags
        self.data: dict[str, typing.BinaryIO] = {}
        self.events: list[Event] = []
        # map sha256 -> Event
        self._event_map: dict[str | None, Event] = {}

        # main event always exists
        self._event_main = self.get_data_event(None)
        # make main event easier to manipulate
        self.add_data = self._event_main.add_data
        self.add_data_file = self._event_main.add_data_file
        self.add_info = self._event_main.add_info
        self.add_feature_values = self._event_main.add_feature_values
        self.add_many_feature_values = self._event_main.add_many_feature_values
        self._add_child = self._event_main._add_child
        self.add_child_with_data = self._event_main.add_child_with_data
        self.add_child_with_data_file = self._event_main.add_child_with_data_file
        self.add_text = self._event_main.add_text

    def get_data_event(self, data_hash: str | None):
        """Creates, or retrieve an event for manipulation.

        The data_hash provided is either None or the parent of the original job Event.
        If it's None the event is the root event and has no parent.

        If the data_hash is set it's value is the sha256 of the new event that is a parent of the current sha256.
        This is typically used to link network captures to the current job.
        """
        if data_hash not in self._event_map:
            ret = self._event_map.setdefault(
                data_hash,
                Event(
                    sha256=self._job.event.entity.sha256,
                    parent_sha256=data_hash,
                ),
            )
            ret.set_stores(self.data, self.events)
            self.events.append(ret)
        return self._event_map[data_hash]

    def register_multiplugin(
        self,
        name: str,
        version: str,
        callback: Callable[[Job], Optional[Union[State, State.Label]]],
        *,
        description: str | None = None,
        security: str | None = None,
    ):
        """Register a multiplugin callback for execution as part of the plugin run."""
        self._multiplugins[name] = Multiplugin(
            name=name, version=version, security=security, description=description, callback=callback
        )

    def get_multiplugin(self, name: str | None):
        """Get a multiplugin by it's name."""
        return self._multiplugins.get(name)

    def get_registered_multiplugins(self):
        """Return a list of the available multiplugins."""
        return self._multiplugins.keys()

    def is_ready(self) -> bool:
        """Return whether the plugin is ready to accept jobs.

        Note: It is called before each job fetch attempt which,
        depending on plugin, may have performance consequences.
        """
        return True

    def execute(self, job: Job) -> Optional[Union[State, State.Label]]:
        """Default entrypoint for plugin execution."""
        raise NotImplementedError

    def is_malformed(self, message: str):
        """Plugins should return this when a file has been processed and is invalid compared to expected format.

        Many plugins receive files that look correct for them but turn out to be corrupted in some known way.
        This is an attempt to standardise the way malformed files are handled.
        """
        self.add_feature_values("malformed", message)
        return State(label=State.Label.COMPLETED_WITH_ERRORS, message=message)
