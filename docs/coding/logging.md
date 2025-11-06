A standard python logging object is available for use during your plugin execution.

It is important to avoid message spam during execution.

```python
class MyPlugin(BinaryPlugin):

    def execute(job: Job):
        iterations=45
        self.logger.debug(f"{iterations=}")
        if True:
            self.logger.warning("something unknown happened")
```
