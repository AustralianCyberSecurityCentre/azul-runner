"""Plugin settings, using pydantic environment parsing."""

from enum import Enum
from inspect import isclass
from typing import Any

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict


class PluginError(Exception):
    """Generic error with plugin."""

    pass


class SetupError(PluginError):
    """Error relating to configuration/initialisation of plugin."""

    pass


_prefix = "plugin_"


class WatchTypeEnum(str, Enum):
    """Different file watch types."""

    PLAIN = ""
    GIT = "git"


class RunnerSettings(BaseSettings):
    """Runner specific environment variables that must not be altered by plugin implementations.

    This is used before a plugin is created, so cannot be shared with regular plugin settings.
    """

    model_config = SettingsConfigDict(env_prefix="runner_", extra="allow")

    # log level of runner + plugin loggers
    log_level: str = "WARNING"


class PluginBaseSettings(BaseSettings):
    """Base settings with prefix configuration already included."""

    model_config = SettingsConfigDict(env_prefix=_prefix, extra="ignore")


class Settings(BaseSettings):
    """Plugin specific environment variables parsed into settings object."""

    model_config = SettingsConfigDict(env_prefix=_prefix, extra="allow")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.heartbeat_interval < 1:
            raise SetupError(f"{self.heartbeat_interval=} too small")

        # Check that heartbeat and timeout values are sane
        if self.run_timeout and self.heartbeat_interval > self.run_timeout:
            raise SetupError(f"{self.heartbeat_interval=} greater than {self.run_timeout=}")

        if self.plugin_depth_limit <= 0:
            raise SetupError(f"{self.plugin_depth_limit=} must be positive")

        if self.server:
            if self.events_url or self.data_url:
                raise SetupError("cannot combine server with events_url or data_url")
            self.data_url = self.events_url = self.server

    # Seconds between heartbeat status messages.
    heartbeat_interval: int = 30
    # Seconds a plugin can run on a single sample before being aborted, 0 to never time out.
    # Set to 10 minutes by default so rare long running binaries are more likely to be processed.
    run_timeout: int = 600
    # Exit run loop after this many timeouts.
    # Useful in case a plugin doesn't clean up all resources when forcefully terminated.
    max_timeouts_before_exit: int = 100

    # Base URL to the dispatcher retrieve events and post results. eg "https://some.server/path".
    server: str = ""
    # dispatcher to use for event interaction
    events_url: str = ""
    # dispatcher to use for file data interaction
    data_url: str = ""
    # Seconds to wait before timing out API requests.
    request_timeout: int = 15
    # Times to retry API requests before worker dies.
    request_retry_count: int = 3
    # Content metadata entries to carry forward between jobs.
    # 0 = clear cache after every run.
    content_meta_cache_limit: int = 0
    # Hard limit of values for a single feature.
    max_values_per_feature: int = 1000
    # Hard limit of feature value length (opensearch limit is 32766)
    # Plugins should use text streams instead if values are sufficiently large
    max_value_length: int = 4000
    # Max event processing depth before auto opting out, -1 to disable.
    plugin_depth_limit: int = 10
    # Seconds to sleep if plugin was not ready to receive events.
    not_ready_backoff: int = 5
    # Prefix to be removed from Plugin Name at startup
    name_remove_prefix: str = "AzulPlugin"
    # Suffix to append to name on startup.
    name_suffix: str = ""
    # Suffix to append to version on startup.
    version_suffix: str = ""
    # Override plugin SECURITY with this on startup.
    security_override: str = ""
    # true if plugin code assumes that all streams can be pulled from s3.
    # false if plugin will handle 'FileNotFound' exceptions when performing .get_all_data() or similar.
    assume_streams_available: bool = False
    # a unique key for all instances of this plugin, typically matching the
    # the name of the parent deployment.
    deployment_key: str = ""

    # File watch, for running plugin with up-to-date files
    # if files in this path are modified, the plugin will reload
    watch_path: str = ""
    # if watch_type==git, then on plugin restart, will set last commit hash of watch_path as version_suffix
    watch_type: WatchTypeEnum = WatchTypeEnum.PLAIN
    # wait x seconds after first change event before restarting plugin
    watch_wait: int = 10

    # Memory limits
    # Enable the memory limiting functionality (disable by default because only works in Kubernetes).
    enable_mem_limits: bool = False
    # Fractional amount of max memory usage when a warning should be raise (e.g 0.8 == 80%)
    used_mem_warning_frac: float = 0.8
    # Fractional amount of max memory usage when plugin should be shutdown and the current job listed as an error.
    used_mem_force_exit_frac: float = 0.9
    # Max memory file location. (shouldn't need changing but just in case.)
    max_mem_file_path: str = "/sys/fs/cgroup/memory.max"
    # Current memory file location. (shouldn't need changing but just in case.)
    cur_mem_file_path: str = "/sys/fs/cgroup/memory.current"
    # How often to check if the process is out of memory (milliseconds).
    mem_poll_frequency_milliseconds: int = 1000

    #
    # these options affect what events your plugin will receive to process
    #

    # process user submitted binaries at a high priority
    require_expedite: bool = True
    # process events as they arrive (i.e. only run on new binaries)
    require_live: bool = True
    # process events from the past (i.e. if version changes and you want to rerun plugin)
    require_historic: bool = True
    # only keep events that have 'content' stream below this size (must be greater than filter_min_content_size)
    # '0' indicates no restriction.
    # default to 200MiB (assume plugins will load and process whole file in memory)
    filter_max_content_size: pydantic.ByteSize = "200MiB"
    # only keep events that have 'content' stream above this size (must be less than max)
    # '0' indicates no restriction.
    filter_min_content_size: pydantic.ByteSize = 0
    # allow only specified event types
    filter_allow_event_types: list[str] = []
    # filter out events published by this plugin
    filter_self: bool = False
    # Require at least one data stream with label matching each key.
    # Outer key is the stream label. Event must have a stream with every label mentioned.
    # Inner list is all accepted file types. Event must have at least one matching file type.
    # Usually you will only want to filter with the content stream label and a limited list of file types.
    # See identify.py for valid file types.
    # Examples:
    # requires a content stream of any file type.
    # {'content': []}
    # requires a content stream either win32 or dos executable.
    # {'content': ["executable/windows/pe32", "executable/windows/dos"]}
    # requires any stream either win32 or dos executable.
    # {'*': ["executable/windows/pe32", "executable/windows/dos"]}
    # requires any stream with a dos exe and a blob stream as either gzip or bzip.
    # {'*': ['executable/windows/dos'], 'blob': ['archive/gzip', 'archive/bzip2']}
    filter_data_types: dict[str, list[str]] = {}

    # Number of concurrent instances, useful for plugins that wait a long time for an external (e.g Cape)
    # Launches this many subprocesses, each running their own instance of the plugin.
    concurrent_plugin_instances: int = 1

    # Useful for testing in cases where forking allows users to preserve mocks between multiprocessors.
    # It can also be much faster than using forkserver.
    # fork is not the default because it is deprecated and in Python 3.14 it forkserver will be the default.
    # This is because having threads in a parent can cause deadlocks which includes some libraries
    # azul plugins use, as well as our Queue based log handlers.
    use_multiprocessing_fork: bool = False


def add_settings(**field_definitions: Any) -> type[PluginBaseSettings]:
    """Create a pydantic settings class useable for plugins SETTINGS field.

    If overriding settings from runner, only the value needs to be supplied, not the type.

    Otherwise kwargs provided to this method are same as field_definitions from the pydantic docs:
        https://docs.pydantic.dev/2.6/api/base_model/#pydantic.create_model

    which are in the form:
        <name>=(<type>, <default>)
    e.g:
        timeout=(int, 300),
        default_passwords=(list[str], ["password1", "infected", "password"]),
        useful=(bool, True)
    """
    # Handle case where a plugin overrides existing config defined in Settings
    filled_fields = {}
    for k, v in field_definitions.items():
        if not isinstance(v, tuple):
            if k in Settings.model_fields:
                filled_fields[k] = (Settings.model_fields[k].annotation, v)
            else:
                raise Exception(f"A custom setting did not provide type information: '{k}'")
        else:
            filled_fields[k] = v

    return pydantic.create_model("PluginSettings", __base__=PluginBaseSettings, **filled_fields)


def parse_config(cls, in_cfg: dict) -> Settings:
    """Validate and transform/merge the plugin configs into unified config.

    cls is the class for the plugin
    """
    inheritance = cls.__mro__
    s = Settings()
    c = s.model_dump()

    # update with any configs passed in to __init__
    c.update(in_cfg)

    # plugins can override standard options and add custom config options
    # Note that it only looks at the plugin class and parent classes,
    #  so SETTINGS set on an *instance* will be ignored. (You shouldn't be doing that anyway.)
    for cls in reversed(inheritance):
        # check for config
        config = getattr(cls, "SETTINGS", None)
        if config is None:
            continue
        error_message = (
            f"Plugin misconfigured: SETTINGS is a type '{type(config)}' and value '{config}' but must use the"
            + " method 'azul_runner.add_settings()' to add config."
        )
        if not isclass(config) or not issubclass(config, PluginBaseSettings):
            raise SetupError(error_message)

        # Get custom configuration options for the plugin from the commandline, environment variables and defaults.
        # NOTE: ignores in_cfg found in Settings that aren't in the PluginSettings
        config_model = config(**in_cfg)
        if config_model.model_config.get("extra", None) != "ignore":
            raise SetupError("Bad Extra's parameter - " + error_message)

        # Dump the plugin configuration over top of current config
        c.update(config_model.model_dump())

    # sanity check / normalise
    return Settings(**c)
