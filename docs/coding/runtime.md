A plugin can be executed on the command line like this:

```bash
# run against single file
(venv) user@host:~/demoplugin$ python myplugin.py sample.bin
# run against folder
(venv) user@host:~/demoplugin$ python myplugin.py samples/
```

## Options

There are a variety of flags and parameters that can be supplied to a plugin.

```bash
# view possible flags and parameters
(venv) user@host:~/demoplugin$ python myplugin.py --help
```

In particular, if you provide `--server` then the plugin will monitor the supplied address
for events to process instead of targeting the local file system.

## Config via Command Line

To pass config to your plugin from the command line, use `-c <key> <value>`.

Values supplied this way take precedence over anything else set via `DEFAULT_CONFIG`.
