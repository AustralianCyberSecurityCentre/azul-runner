# Azul Runner

Core framework for writing Python plugins for Azul.

It handles filtering and fetching events, setting up data streams, and
parsing and validating plugin results before posting output events
back to the dispatcher.

Additionally, it allows plugins to be run
locally from the commandline, with local files or folders as input data.

## Usage

To run a plugin (after installing azul-runner and the plugin as Python packages):

```bash
# against local samples
$ azul-plugin-(name) local_file.bin
$ azul-plugin-(name) local_folder/

# against remote dispatcher
$ azul-plugin-(name) --server http://server.address/path

# with custom config
$ azul-plugin-(name) -c KEY VALUE -c KEY2 VALUE2
```

Plugins may define multiple entrypoints, check the plugins setup.py for details.

Plugin configuration may be passed in as environment variables.

Check ./azul_runner/settings.py for specific options

- PLUGIN_RUN_TIMEOUT=600
- PLUGIN_SERVER=http://widgets.co.internal

## Plugin Development

Use the project `azure-generator` to generate boilerplate code for a new plugin.

For most common use cases, inherit from `azul_runner.BinaryPlugin` and implement the `execute` method.
It is recommended to look at existing plugins in order to implement your plugin.
You can also check binary_plugin.py and plugin.py for more properties that can be configured.

Advanced use cases may inherit from `azul_runner.Plugin` instead.

For more documentation on development process, see [here](./docs/index.md)

For more documentation on specific api usage and plugin code, see [here](./docs/coding/index.md)

See [structure](./docs/structure.md) for more information about the structure of azul-runner.

See [migration guide](./docs/migration.md) to update your plugin in line with api changes in azul-runner.

## Example Plugin

```python
from azul_runner import BinaryPlugin, Feature, Job, cmdline_run, State, FeatureType


class LookForThings(BinaryPlugin):
    """Look for things."""

    VERSION = "1.0"
    SETTINGS = add_settings(
      # You can filter assemblyline file types as below.
      # These are the files that your plugin will process.
      # Note: you can filter by prefix (document/).
      # Check azul-bedrock/identify.yaml for valid file types.
        filter_data_types={
            "content": [
                "text/plain",
                "document/",
            ]
        }
    )
    FEATURES = [
        Feature("tag", "Custom tag", FeatureType.String),
    ]

    def execute(self, job: Job):
        """Find peanuts."""
        data = job.get_data()
        # 'data' is a file-like object that supports seeking and being read from
        # (The content may be retrieved in parts if the file is large and non-local)
        header: bytes = data.read(7)
        if header == b"PEANUT:":
            # create a tag
            self.add_feature_values("tag", "may contain nuts")
            # add the next 24 bytes as a child
            c = self.add_child_with_data(
                relationship={"label": "peanut"},
                data=data.read(24),
            )
            c.add_feature_values("tag", "may be hard to crack")
        else:
            return State.Label.OPT_OUT

if __name__ == '__main__':
    cmdline_run(plugin=LookForThings)
```

## Execution

As the example plugin was not constructed in `azure-generator`, running them requires directly executing
the script they are contained in. A copy can be found in `tests/example_plugins.py`.

```bash
azul-runner$ python tests/example_plugins.py tests/data/peanut.txt
----- LookForThings results -----
COMPLETED

events (2)

event for cmdline_entity:None
  {}
  output features:
    tag: may contain nuts

event for 7c4cd5274277dde41aa3f5e06cfca8c6cc703951a642b6c268cb43a2b345780a:None
  {'label': 'peanut'}
  child of cmdline_entity
  output data streams (1):
    24 bytes - EventData(hash='7c4cd5274277dde41aa3f5e06cfca8c6cc703951a642b6c268cb43a2b345780a', label='content')
  output features:
    tag: may be hard to crack

Feature key:
  tag:  Custom tag

```
