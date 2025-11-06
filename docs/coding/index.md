## Topics

Our documentation is designed to show the fundamentals of plugin development.

If you are having trouble we recommend investigating:

- plugins that already exist
- source code of azul-runner

If this documentation is out of date or contains errors, please submit a PR or raise an issue.

[Running Plugins and Passing Configuration](runtime.md)

[Logging](logging.md)

[Writing Plugin Tests](tests.md)

[Reading Binary Data](streams.md)

[Defining and Adding Features](features.md)

[Adding deobfuscated/extracted children](children.md)

[Adding Flexible Data Structures](info.md)

[Plugin Completion, Opting Out and Raising Exceptions](status.md)

[Defining Plugin Security](security.md)

[Default configuration and environments](config.md)

## Basic Plugin

The easiest way to get started is to use the `azure-generator` project to create an initial plugin.

Here is a simple example of a plugin:

```python
from azul_runner import BinaryPlugin, Feature, Job, cmdline_run, State, add_settings


class LookForThings(BinaryPlugin):
    """Look for things."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Custom tag", str),
    ]
    SETTINGS = add_settings(
        filter_data_types={"content": ["plain/text"]},
    )


    def execute(self, job: Job):
        """Find peaches."""
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

## Non-Binary Plugin Type

If you want to process entity types other than 'binary', you will need to use the `Plugin` template directly.

It is nearly the same as BinaryPlugin, however you need to specify the INPUT_TYPE, etc.
You will need to read BinaryPlugin and Plugin to figure out what you want.

```python
class MyPlugin(Plugin):
    VERSION = '1.0'

    def execute(self, job: Job):
        return
```

### MZ Filter

This plugin will set a tag feature if the incoming file is identified as MZ.

```python
from azul_runner import BinaryPlugin, Feature, Job, cmdline_run, State


class LookForMZ1(BinaryPlugin):
    """Look for MZ type 1."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Tags files that might be .EXEs", str),
    ]

    def execute(self, job: Job):
        """Find MZ."""
        data = job.get_data()
        # 'data' is a file-like object that supports seeking and being read from
        # (The content may be retrieved in parts if the file is large and non-local)
        header: bytes = data.read(2)
        if header == b"MZ":
            self.add_feature_values("tag", "might be EXE")
        else:
            return State.Label.OPT_OUT

if __name__ == '__main__':
    cmdline_run(plugin=LookForMZ1)
```

### Filtering Jobs

Files in Azul are already pre-inspected to identify many common file types.

MZ filtering is better accomplished by using an `INPUT_DATA` filtering setting.
This is many times more efficient on network, cpu and memory than the plugin checking for itself.

The `INPUT_DATA` filter will only send jobs to your plugin if they match the filtered data type.

For a list of recognised `INPUT_DATA` type names, check the identify.py of this package.
This listing is duplicated logic from dispatcher source code (which is the official source).

```python
class LookForMZ2(BinaryPlugin):
    """Look for MZ type 2."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Tags files that might be .EXEs", str),
    ]
    SETTINGS = add_settings(
        filter_data_types={"content": ["executable/windows/pe32", "executable/windows/dos"]},
    )

    def execute(self, job: Job):
        """Find MZ."""
        # This plugin will only run over files that have been identified
        #  by azul-runner/dispatcher as EXE files.
        self.add_feature_values("tag", "It's an EXE!")
```

### Add child

In a real plugin you may also extract binary artifacts that should be analysed using other plugins.

```python
class LookForMZ3(BinaryPlugin):
    """Look for MZ type 3."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Tags files that might be .EXEs", str),
    ]
    SETTINGS = add_settings(
        filter_data_types={"content": ["executable/windows/pe32", "executable/windows/dos"]},
    )

    def execute(self, job: Job):
        """Find MZ."""
        buffer = job.get_data().read(64)
        # add child binary
        c = self.add_child_with_data(
            relationship={"label": "First 64 bytes"},
            data=buffer,
        )
        # add feature to the child binary
        c.add_feature_values("tag", "the extracted child")
        # add feature to the original incoming binary
        self.add_feature_values("tag", ["Might be an exe", "Extracted header"])

```

This demonstrates adding a child, as well as returning multiple values for a features. Features can also be set on the extracted children; the should be provided in the same way as returned features for the parent binary.
