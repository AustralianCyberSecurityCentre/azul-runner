In order to return a result other than 'COMPLETED', the plugin must either

- return nothing
- return a `State` type
- return a `State.Label` type
- raise an error

If your code, or code beneath yours raises an exception in production then
it will be visible in the Azul UI for debugging.

Completion Examples:

```python
def execute(job: Job):
    t = "apple"
    if t == 1:
        # opt out with no details
        return State.Label.OPT_OUT
    elif t == 2:
        # opt out with details
        return State(
            State.Label.OPT_OUT,
            "OOBang no match",
            "This file did not match on OOBang",
        )
    elif t == 3:
        # return an error
        return State(
            State.Label.ERROR_INPUT,
            "Bad",
            "Bad but handled",
    )
    elif t == 4:
        return State(
            State.Label.COMPLETED_WITH_ERRORS,
            "Completed with some errors.",
            "An error occurred that could be recovered from while processing the plugin results data."
        )
    elif t == 5:
        # finished but no features or augmented streams were added to the result. (unnecessary) is inferred.
        return State.Label.COMPLETED_EMPTY
    elif t == 6:
        # finished execute with COMPLETED status (unnecessary)
        return State.Label.COMPLETED
    elif t == 7:
        # finished execute with COMPLETED status (unnecessary)
        return
    # finished execute with COMPLETED status (implied)
    t = 7
```

Valid labels are defined in azul-bedrock:

```python
class StatusEventEnum(str, Enum):
    # Successfully completed
    COMPLETED = "completed"
    # Successfully completed but no features or augmented streams were produced
    COMPLETED_EMPTY = "completed-empty"
    # Successfully completed but errors occurred which means the plugin might not have gotten all data.
    COMPLETED_WITH_ERRORS = "completed-with-errors"
    # Entity not suitable for this plugin (eg wrong size, type, ...)
    OPT_OUT = "opt-out"
    # Plugin heartbeat
    HEARTBEAT = "heartbeat"

    # Errors
    # Plugin-specific code raised an unhandled exception
    # This is dedicated to errors that can only be resolved by the plugin author
    ERROR_EXCEPTION = "error-exception"
    # Plugin could not communicate with some required service (including dispatcher)
    ERROR_NETWORK = "error-network"
    # Generic error in plugin harness
    ERROR_RUNNER = "error-runner"
    # Error processing input entity (eg incorrect format, corrupted) - legacy "entity error"
    ERROR_INPUT = "error-input"
    # Plugin returned something that couldn't be understood by the runner
    ERROR_OUTPUT = "error-output"
    # Plugin exceeded its maximum execution time on a sample
    ERROR_TIMEOUT = "error-timeout"
    # Plugin execution was cancelled due to being out of memory.
    ERROR_OOM = "error-out-of-memory"
```
