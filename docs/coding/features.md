## Defining Features

Your plugin will raise errors if you try to return features you have not documented.

```python
class MyPlugin(BinaryPlugin):
    FEATURES = [
        Feature("embedded_url", desc="URL found embedded in content", type=FeatureType.Uri),
        Feature("tag", desc="Any informational label about the sample", type=FeatureType.String),
        Feature("port", desc="Port used to communicate", type=FeatureType.Integer),
        Feature("compiletime", desc="DateTime the entity was compiled", type=FeatureType.Datetime),
        Feature("filepath", desc="Filepath on disk", type=FeatureType.Filepath),
    ]
```

The `Feature` class has a `name`, a `desc`, and a `type`. Valid types are `int`, `float`, `datetime.datetime`,
`str`, `bytes`, `Filepath` and `Uri`.

You should reuse feature definitions from other plugins where it makes sense.

Plugins accumulate features defined by their parents, and they can override these if necessary.

There are several features that all Plugin's share:

```python
FEATURES = {
    # Standard features set by all binary plugins.
    # Child classes that set this property will ADD to this feature set.
    Feature(name="file_extension", desc="File extension of the 'content' stream.", type=FeatureType.String),
    Feature(name="file_format", desc="Assemblyline file type of the 'content' stream.", type=FeatureType.String),
    Feature(name="file_format_legacy", desc="Azul file type of the 'content' stream.", type=FeatureType.String),
    Feature(name="filename", desc="Name on disk of the 'content' stream.", type=FeatureType.Filepath),
    Feature(name="magic", desc="File magic found for the 'content' stream.", type=FeatureType.String),
    Feature(name="malformed", desc="File is malformed in some way.", type=FeatureType.String),
    Feature(name="mime", desc="Mimetype found for the 'content' stream.", type=FeatureType.String),
}
```

## Creating Feature Values

```python
def execute(job: Job):
    # add one value to one feature
    self.add_feature_values("tag", "might be EXE")
    # add many values to one feature
    self.add_feature_values("tag", ["might be EXE", "has_flag"])
    # add many values to many features
    self.add_all_feature_values({
        # A feature with a single value
        'badness_level': 9001,
        # A feature with multiple values. Note that it can be any iterable, not just a list.
        'detection_AVs': {'Sophos', 'Kaspersky'},
        # A list (or other iterable) with a single value is also allowed
        'malware_categories': ['ransomware'],
        # Empty lists are acceptable; it will be ignored in final output
        'detected_known_malware': [],
    })
```

## Advanced output features

Feature values can be enriched with additional metadata if relevant.

The plugin will need to return `FeatureValue` objects instead of simple data types.

`FeatureValue` objects must have a `value` and can also support a `size` and `offset`
and a `label`.

- offset - position in file where the feature was extracted from.
- size - number of bytes in file used to extract feature.
- label - additional contextual information about the feature value.

These will be used for feature grouping and improved visualisation in the Azul UI.

Example:

```python
class DeobSomething(BinaryPlugin):
    FEATURES = [
        Feature('decoded', desc='A decoded string.', type=FeatureType.String)
    ]

    def execute(self, job: Job):
        deob_list = foo_lib.get_deob_strings(data)
        if deob_list:
            self.add_feature_value('decoded', [
                FeatureValue(txt, offset=offs, label='Deob key %s' % key)
                for offs, txt, key in deob_list
            ])
```
