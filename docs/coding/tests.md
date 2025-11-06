There can be a lot of boilerplate present when creating tests for plugins.

In an effort to alleviate some of your pain, there is a convenience function that will
generate the correct response to make your test pass when a binary is supplied.

This does not mean that your plugin logic is correct, and you will need to read each line of the produced
output to verify it meets your requirements. If you don't do this, your plugin will be garbage.

Some alteration of the output may be needed to remove timestamps, specific exception messages, etc.

```python
from azul_runner import DATA_HASH, FV, Event, EventData, EventParent, Filepath, JobResult, State, Uri
from azul_runner import TestPlugin

from azul_myplugin import MyPlugin


class TestExecute(TestPlugin):
    PLUGIN_TO_TEST = MyPlugin

    def test_execute(self):
        """Test an expected normal run"""
        data = self.load_test_file_bytes("u_testfile.cart")
        result = self.do_execution(data_in=[("content", data)])
        # this check will fail, and will print the expected value instead of 'None'
        # to the console, which you can paste into your test and then format with 'black .'
        # to make readable.
        self.assertJobResult(result, None)
```

Here is a test with the correct expected JobResult. The output should be fairly self explanatory.

Something to note is that if your plugin produces children, the first Event() is always the parent,
followed by the order in which you added children and grandchildren.

You will need to always add the children and grandchildren in the same order for the tests to always pass.
For this reason, if you are loading in a bunch of child files, you should sort them alphabetically first.

```python
from azul_runner import DATA_HASH, FV, Event, EventData, EventParent, Filepath, JobResult, State, Uri
from azul_runner import TestPlugin

from azul_myplugin import MyPlugin


class TestExecute(TestPlugin):
    PLUGIN_TO_TEST = MyPlugin

    def test_execute(self):
        """Test an expected normal run"""
        data = self.load_test_file_bytes("u_testfile.cart")
        result = self.do_execution(data_in=[("content", data)])
        self.assertJobResult(
            result,
            JobResult(
                state=State(State.Label.COMPLETED),
                events=[
                    Event(
                        sha256="7bb6f9f7a47a63e684925af3608c059edcc371eb81188c48c9714896fb1091fd",
                        features={
                            "compiletime": [FV(datetime.datetime(2021, 3, 17, 23, 10, 5))],
                            "embedded_url": [
                                FV("https://another.uri:8081"),
                                FV("https://example.uri:8080/feature"),
                            ],
                            "filepath": [FV("/path/to/a/bad/file.exe")],
                            "port": [FV(8080, label="optional info", offset=65536, size=4)],
                            "tag": [FV("This is an example of an ordinary string feature")],
                        },
                    )
                ],
            ),
        )
```

Plugin config for tests is the same as `SETTINGS` with the exception of:

- `"request_retry_count"` set to zero (so that a failed request raises an exception)
- `"run_timeout"` set to zero when the test is run with a debugger attached (so that plugin timeout doesn't interrupt live debugging).

Default config can be changed per-test by passing a config dict to `do_execution()`:

```python
from azul_runner import DATA_HASH, FV, Event, EventData, EventParent, Filepath, JobResult, State, Uri
from azul_runner import TestPlugin

from azul_myplugin import MyPlugin


class TestExecute(TestPlugin):
    PLUGIN_TO_TEST = MyPlugin

    def test_execute(self):
        """Test an expected normal run"""
        config = {
            "run_timeout": 60,
            "plugin_specific_config": "someconfig",
        }
        result = self.do_execution(config=config)
```
