The runner framework provides a means of passing options to plugins from the commandline or from
environment variables. These options require an expected type and value.

Plugins can access `self.cfg.<setting_name>` to retrieve the values passed to them.
Custom options will not be picked up by your IDE but will still work.

```python
class MyPlugin(BinaryPlugin):
    SETTINGS = add_settings(
        subprocess_timeout_seconds=(int, 0)
    )

    def execute(job: Job):
        print(self.cfg.subprocess_timeout_seconds)
```

Default config is accumulated across the inheritance chain.

The Plugin and BinaryPlugin classes set many default options that affect how your plugin runs.

Please see `settings.py` in the source code for more information about these options.

## Limiting File Size

The binary template can also filter input jobs based on a maximum content size limit.

This is implemented as a config option to make it more easily adjustable at runtime.

```python
class MyPlugin(BinaryPlugin):
    SETTINGS = add_settings(
        filter_max_file_size=5 * 2 ** 20 # 5 MB limit
    )
```

## Filtering file types

You can filter assemblyline file types. These are the files that your plugin will process.

You can filter by prefix (document/). Check azul-bedrock/identify.yaml for valid file types.

```python
class MyPlugin(BinaryPlugin):
    SETTINGS = add_settings(
        filter_data_types={
            "content": [
                "text/plain",
                "document/",
            ]
        }
    )

```

## Protecting Secret Config values

Config values are sent to the dispatcher on plugin registration, and recorded in the Azul event queues.

Plugins may have secret config values that should not be visible to others outside the plugin.
If this is the case, the config's name should be prefixed with `secret_`, and it will be filtered out
before sending configs to the server.

Example:

- `secret_myservice_api_key` - this will **not** be sent to the server.
- `super_secret_password` - This **will** be sent to the server - names must _start_ with `secret_` to be filtered.
