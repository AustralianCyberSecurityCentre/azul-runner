# Migrate plugin between azul-runner versions

Note that version was reset to conform to azul-app chart versions

## 7.0

- Moved `test_template.py` to a folder `test_utils` and made it an extra import so now `azul_runner[test_utils]` will need to be added to the `requirements_test.txt` file, and imported from the `azul_runner` or `azul_runner.test_utils` package.
- Removed `TestLoad` should be replaced with `FileManager` which is exported by `azul_runner.test_utils` and `azul_runner`
- `load_cart` has been removed from the plugin test and need to be replaced with `load_test_file_bytes` or `load_test_file_path`
- `load_raw` has been renamed to `load_local_raw`

## 6.0

- Remove support for config `filter_gjson`
- Removed `filter_data_types_legacy`
- A number of properties on Event() classes (used in tests) were renamed and 'entity_type' was removed.
  - For backwards compatibility, legacy names are renamed dynamically with a deprecation warning.
- Removed plugin property ENTITY_TYPE, all plugins implicitly operate on 'binary' events.
- Remove several obsolete property checks from legacy releases (all plugins have been updated)

## 5.0

No breaking changes

## Legacy releases (before version realignment)

### Legacy 6.0

Changes azul-runner to using multiprocessing ('forkserver') by default from previous threading model.

- `assertFormatted` has been replaced with `assertJobResultsDict` and `assertReprEqual`
- `do_execution` loop replaced with `provided_coordinator` also `no_multiprocessing` was added

- New setting
  - `use_multiprocessing_fork` to change multiprocessing over to 'fork' from 'forkserver'.

### Legacy 5.0

- config changes:
  - `filter_require_data` removed
    - use `filter_allow_event_types=[]` to select specific event types to process
    - use `assume_streams_available=true` to catch errors accessing underlying streams
  - `filter_deny_event_types` removed
    - use `filter_allow_event_types=[]`

### Legacy 4.0

- config & metadata changes:
  - overrides for existing config options defined in Settings() do not require type information
  - `add_custom_config_settings()` -> `add_settings()`
  - `REQUIRES_DATA = x` -> `add_settings(filter_require_data: x)`
  - `INPUT_DATA = x` -> `add_settings(filter_data_types_legacy: x)`
  - `ADDITIONAL_INPUT_CRITERIA = x` -> `add_settings(filter_gjson: x)`
  - `add_settings(max_file_size=x)` -> `add_settings(filter_max_content_size: x)`
  - `DEFAULT_CONFIGS` -> `SETTINGS`
  - `OUTPUT_FEATURES` -> `FEATURES`
  - add more dispatcher filters to runner settings (`filter_` settings)
  - export `TestPlugin` from `test_template`
  - remove legacy plugin `self.config` dictionary

### Legacy 3.45

Not a breaking change, but some options deprecated

- `Filepath()` should be removed, use a string instance instead
- `Uri()` should be removed, use a string instance instead
- `Feature()` `type=` argument now should use FeatureType.String, FeatureType.Filepath, FeatureType.Binary, etc.

### Legacy 3.1 to 3.2

- Plugin metadata now requires security to be defined as a string instead of a dict
  - security is still optional, if unset, security is assumed to be 'OFFICIAL'
  - Incoming events your plugin processes (which you have no control over) may
    have security as either a dict or str, don't worry about it
- Replace config option `security_exclusive` with `security_override` which accepts a security string
- `SECURITY` property replace security dict/list/str with only security string
- The `Security` class has been removed
  - If you were inspecting security objects at runtime for some reason,
    you will need to handle both dict and str cases.
- See [here](./coding/security.md) for more information

### Legacy 3.0 to 3.1

#### Plugin code

- Replace import `Entity` with `Job`
- Replace import `BinaryTemplate` with `BinaryPlugin`
- `REQUIRES_DATA` is set as true in `BinaryPlugin` and can be removed
- Base class should now be `BinaryPlugin` not `BinaryTemplate`
- Rename `INPUT_CONTENT` plugin property to `INPUT_DATA`
- Rename `RunResult` to `State`
- Rename `RunResultEnum` to `State.Label`
- `FEATURES` with unspecified types now default to `str`
- Change `def execute_binary(self, entity: Entity, data: Optional[StorageProxyFile]) -> dict:` to
  `def execute(self, job: Job):`
- Read `data` via `data = job.get_data()`
  - Read non-content data via `job.get_data('text')` or `job.get_all_data('pcap')`
  - `job.get_data('text')` will error if more than one stream is available with the `text` label (replaces `get_stream()`)
  - `job.get_all_data('pcap')` to return streams of a certain type, or all streams with no parameter (replaces `get_streams()`)
- Feature values, info, children are managed via Event instances.
  - The main Event has helpers on BinaryTemplate
    - `self.add_feature_values(feature, values)` to set values for a specific feature
    - `self.add_many_feature_values(feature_values)` to set many features with a dict
      - Prefer usage of `self.add_feature_values()` for readability
    - `self.add_info(info)` to set info
  - Adding children returns a new Event instance
    - `self.add_binary(b"data", {"relation":"ship"})` is replaced with `c = self.add_child_with_data({"relation":"ship"}, b"data")`
    - `c.add_feature_values()`
    - `c.add_many_feature_values()`
    - `c.add_info()`
    - `gc = c.add_child_with_data()` returns new Event instance and supports grandchildren and further
  - Adding features, info for particular existing data streams
    - `ds = self.get_data_event(hash)` returns Event object (replaces `per_stream`)
    - Event functions used to add data, as described for children

#### Test code

- Add imports
  ```
  from azul_runner import Event, FV, JobResult, State, test_template
  ```
- self.do_execution returns a `JobResult()` instance instead of a dictionary
- Use `self.assertJobResult(run_result, JobResult(...))` instead of `self.assertEqual` to make error checking render nicely
- When developing tests, use `self.assertJobResult(run_result, None)` to print the received output of run_result that you can copy+paste to replace `None`.
  - After copy+pasting, use `black .` to transform the long JobResult(...) into a nicely formatted multiline output.
  - Make sure to check this output is as you expected.
