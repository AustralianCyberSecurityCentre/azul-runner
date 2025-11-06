Azul-runner is split into modules based on related functionality.

All names necessary for plugins are imported into the package
namespace, so users can simply write `from azul_runner import thing`.

## Class Diagram

```mermaid
classDiagram
    Plugin <|-- BinaryPlugin
    Plugin *-- Coordinator
    Network *-- Coordinator
    Coordinator *-- Monitor
    Dispatcher *-- Network
    Plugin *-- Network
    Multiplugin "*" *-- "1" Plugin

    Job <-- Plugin

    class Dispatcher{
        +submit_events()
        +submit_binary()
        +get_binary_events()
        +get_binary()
    }
    class Network{
        +post_registration()
        +fetch_job(): BinaryEvent
        +ack_job()
    }
    class Coordinator{
        +run_once()
        +run_loop()
    }
    class Plugin{
        +is_ready()
        +execute(Job)
    }
    class Multiplugin{

    }
    class BinaryPlugin{
    }

    BinaryEvent *-- Job
    class BinaryEvent{
        ...
    }
    class Job{
        +BinaryEvent event
        +load_streams()
        +get_data()
        +get_all_data()
    }
    class Monitor{
        +run_once()
        +run_loop()
    }
```

## Sequence Diagram

Simple case for looping plugin execution.

```mermaid
sequenceDiagram
    main->>coordinator: run_loop()
    coordinator->>network: post_registrations()
    network->>dispatcher: submit_events()
    coordinator->>plugin: is_ready()
    loop forever
        coordinator->>network: fetch_job()
        network->>dispatcher: get_binary_events()
        coordinator->>job: load_streams()
        loop each multiplugin
            coordinator->>executor: run_plugin_with_job(job)
            executor->>plugin: execute(job)
            coordinator->>network: ack_job()
            loop each new stream
                network->>dispatcher: submit_binary()
            end
            network->>dispatcher: submit_events()
        end
    end
```

## binary_plugin.py

Subclass of Plugin designed for processing binary files.

## coordinator.py

Handle full Plugin execution loop.

## dispatcher.py

Handle interaction with dispatcher restapi.


## identify.py

Python implementation of the Dispatcher identify code.

It is intended for use during plugin tests only.

## main.py

CLI based execution of plugins for local testing or remote deployment.

## models.py

Data structures used by plugins to record job results, such as features, info, data.

## monitor.py

Uses multiprocessing to monitor the process of a plugin running in coordinator and track memory
and create heartbeat events.

## network_transform.py

Transform models to/from network event format and runner internal format.

## network.py

Higher level interactions with Azul Dispatcher vs dispatcher.py.

## plugin_executor.py

Execute Plugin with Job and return JobResult instance.

## plugin.py

Plugin template for handling plugin metadata and execution methods.

## settings.py

Plugin settings, using pydantic environment parsing.

## storage_spooled.py

Guarantee a valid path to a SpooledTemporaryFile after rollover has occurred.

## storage.py

Defines a file-like interface to S3-compatible storage.

Used by plugins to access event binary data.

## test_template.py

Contains a template test case class to simplify testing of Azul plugins.
