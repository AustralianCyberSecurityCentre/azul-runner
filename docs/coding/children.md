It is common to decode additional binary files derived from the current file.

If you want Azul to recursively analyse these files, you need to add them as children.

```python
def execute(job: Job):
    data = job.get_data()
    # do some processing to generate a child file
    # open a temporary file in read/write binary mode
    with tempfile.TemporaryFile("r+b") as tf:
        tf.write(b"\xff\x01\x01\xff")
        data.seek(200, 1)
        tf.write(data.read(20))
        tf.write(b"\xff\x01\x01\xff")
        data.seek(20, 1)
        tf.write(data.read(20))
        # not required but is best practice
        tf.seek(0)

        # add a child binary
        c = self.add_child_with_data_file(
            relationship={
                # 'action' key is preferred, and you can add any other keys you like
                'action': 'deobfuscated',
                'obfus_type': "tangerine",
            },
            data_file=tf,
        )
    # add a feature to the child binary
    c.add_feature_values('filename', "my_child_file.ex5")
    # add a grandchild using a binary string rather than a binary file
    gc = c.add_child_with_data(
        relationship={
            # 'action' key is preferred, and you can add any other keys you like
            'action': 'deobfuscated',
            'obfus_type': "tangerine",
        },
        data=b"986754893",
    )
    # add a feature to the grandchild binary
    gc.add_feature_values('filename', "my_grandchild_file.ex5")
    gc.add_info({"refund": "this tangerine is supposed to be seedless but it actually has seeds for some reason"})
    # you can add great-grandchildren, etc. At some point it gets a bit silly though.
    # remember to carefully track the returned values from x.add_child_with_data
```
