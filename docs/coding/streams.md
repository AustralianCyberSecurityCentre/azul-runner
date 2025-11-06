## Binary Data

The 'content' stream returned by `job.get_data()` supports methods such as `read()`, `seek()`, `tell()`.

```python
# Check this file should be processed the file
# this prevents all files travelling over the network (which is terrible if you are processing 200/s)
if b"12345" != job.get_data().read(5):
    return status.OPT_OUT
# reset pointer (important!)
job.get_data().seek(0)
# read all bytes
main = job.get_data().read(0)
```

### Tags

Data streams have a `tags` property, which is a dict of key:value pairs describing the data stream.

Currently, the tags will contain at least 'sha1', 'sha256', 'md5', 'mime', 'magic' (`file` output),
and 'file_format_legacy' (categorised type of the file such as 'RAR' or 'Win32 DLL').

### Non-Content Streams

There may be some situations where you want to process non-'content' streams, or specific behaviour for
different file types.

The Azul framework identifies the file type of data streams and allows you to filter on them.

The 'file_format_legacy' is intended to match with the types from virustotal. Types are listed in identify.py in azul-bedrock.

```python
class MyPlugin(BinaryPlugin):
    SETTINGS = add_settings(
        filter_data_types={"content": ["executable/windows/dos"],'blob': ['archive/gzip', 'archive/bzip2']},
    )

    def execute(self, job: Job):
        # get_data returns None if no stream has matching label
        single = job.get_data(label="invalid")
        # get_data cannot match on specific file_format_legacy
        single = job.get_data(label=azm.DataLabel.CONTENT)
        # get_data raises Exception if multiple streams have matching label
        blobby = job.get_data(label=azm.DataLabel.TEXT)
        # get_all_data returns a list of streams with matching label and/or file_format_legacy
        files = job.get_all_data(file_format="executable/windows/dos")
        blobs = job.get_all_data(label=azm.DataLabel.TEXT)
        gzips = job.get_all_data(file_format="archive/gzip")
        bzips = job.get_all_data(file_format="archive/bzip2", label=azm.DataLabel.TEXT)
```

```python
class FilteredLookForMZ(BinaryPlugin):
    SETTINGS = add_settings(
        filter_data_types={azm.DataLabel.CONTENT: ['executable/windows/pe32', 'executable/windows/dos', 'executable/windows/dll32']},
    )

    def execute(self, job: Job):
        mz_thing = job.get_data()
```

### Text reports

It is common to generate a text report or tool output log.

Content added this way will be shown in the Azul UI.

```python
def execute(job: Job):
    raw_tool_result = run_tool_subprocess(
        data,
    )
    try:
        tool_result = raw_tool_result.decode('cp1252')
    except UnicodeDecodeError:
        tool_result = 'decoding error: ...'
        # Do some error handling
    self.add_text(tool_result)
```

You can also set the 'language' parameter if the output is source code of some kind.
This should be a language name supported by prism.js.

Examples 'html', 'js'/'javascript', 'bash'/'shell',
'c', 'dotnet', 'php', 'go', 'powershell', 'python', 'regex', 'vb'/'

```python
def execute(job: Job):
    self.add_text("eval(var5);", language="js")
```

### Data

You may wish to add some binary data as part of your output.

This could be something like a pcap or jpg.

```python
def execute(job: Job):
    pcap: bytes = run_and_get_me_a_pcap(job.get_data().read())
    self.add_data(label=azm.DataLabel.PCAP, tags={}, data=pcap)
```
