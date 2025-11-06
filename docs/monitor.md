# Logic for Coordinator and Monitor

This documentation goes through the logic of how monitor and coordinator interact.

## Monitor logic flow
Monitor operates by starting the child process which is coordinator.
It is continually checking if the coordinator child process has exited or not
and then getting a new job from coordinator if there are any.

If the job acquired from the queue is None it means coordinator is waiting for a job.
Monitor will then continue to verify the subprocess is still alive until it sees a job.

Once there is a job being processed by the plugin monitor will continually check to see
if the plugin has run out of memory, run out of time or needs to send a heartbeat.
It will send a message to dispatcher in any of those cases.

Finally if the child process does exit monitor will check the exit code and depending on the 
value either restart the child process or exit the application.

```mermaid
flowchart TD
    start -- start process --> check[Check Process Ended]
    queue{Coordinator Queue} --> get
    check[Check Process Ended] -- alive --> get[Get Current Job]
    get[Get Current Job] -- No job --> start
    get[Get Current Job] -- Job found --> checkoom[Check out of Memory]
    checkoom[Check out of Memory] -- out of memory --> restart
    checkoom[Check out of Memory] -- fine --> checktime[Check Timeout]
    checktime[Check Timeout] -- timed out --> restart
    checktime[Check Timeout] -- fine --> start
    restart[Restart Child Process] --> start
    check -- dead fatal --> Exit
    check -- dead recover --> restart

```

## Coordinator logic flow

Coordinator sets it's queue to None initially and then sends a request to dispatcher to get a job.
This is done so monitor won't timeout as it has a None job to indicate coordinator is looking for work.

Once Coordinator gets a job from dispatcher it will then put that job into the communication queue
back to monitor.

It will then download the job's streams from dispatcher, and run the job in the plugin.
Coordinator will then run any multiplugin's ensuring to add the job with the multiplugin that is running
to the queue for monitor to ensure if the plugin times out which multiplugin failed can be determined.

```mermaid
flowchart TD
    start -- initalise plugin --> checkWD[Check watchdog]
    checkWD[Check watchdog] -- Git Changes --> Exit
    checkWD[Check watchdog] -- No git changes --> lfj[Look for Job]
    lfj -. set job to None .-o queue{Coordinator Queue}
    lfj -- get job from dispatcher --> fetch[Job found]
    fetch -. set job with plugin .-o queue{Coordinator Queue}
    fetch -- get streams --> getStream[Get Streams from dispatcher]
    getStream -- run Plugin --> pfj[Plugin Finished]
    pfj -- run multiplugin --> mpf[mp finished]
    pfj -. set job with multiplugin .-o queue
    mpf -- next multiplugin --> pfj
    mpf -- done --> checkWD
```
