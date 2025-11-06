"""Execute Plugin with Job and return JobResult instance."""

import datetime
import json
import logging
import traceback

from azul_bedrock import models_network as azm

from . import models
from . import plugin as mplugin
from .models import FeatureValue, JobResult, State
from .storage import ProxyFileNotFoundError, StorageError

logger = logging.getLogger(__name__)


# true feature limit is 10_000, but reserve some for internal use
MAX_FEATURE_VALUES = 9_500


class ResultError(AssertionError):
    """Used to signal an invalid plugin output in the 'results' dict."""

    pass


def run_plugin_with_job(plugin: mplugin.Plugin, job: models.Job, multiplugin: str) -> JobResult:
    """Execute plugin and handle bad output."""
    run_start = datetime.datetime.now(datetime.timezone.utc)
    # cleanup from last run
    plugin.reset(job)

    # execute and validate
    mp = plugin.get_multiplugin(multiplugin)
    try:
        # filter job if plugin requires data
        if plugin.cfg.assume_streams_available and not job.get_all_data():
            state = State(State.Label.OPT_OUT, message="plugin requires data but none exists")
        else:
            logger.debug(f"Running job with '{multiplugin=}'")
            # Execute the plugin with the provided job data.
            response = mp.callback(job)
            state = response
            # normalise status
            if not state:
                state = State(State.Label.COMPLETED)
            elif isinstance(state, State.Label):
                state = State(state)
            elif not isinstance(state, State):
                raise ResultError(f"Returned data cannot be processed as a State instance: {state}")

        # Validate output format
        result = JobResult(state=state)
        # Catch anything that goes wrong while processing output, and report it as a failed run
        if result.state.label in azm.StatusEnumSuccess:
            input_sids = {s.file_info.sha256 for s in job.get_all_data() or []}
            new_hashes = {y.hash for x in plugin.events for y in x.data}
            all_hashes = input_sids.union(new_hashes).union({None})
            feat_types = set()
            warnings = []
            for event in plugin.events:
                if event.parent_sha256 not in all_hashes:
                    raise ResultError(f"results for non-existent stream: {event.parent_sha256}")
                warning_strings = _process_features(plugin, event.features)
                warnings.extend(warning_strings)

                feat_types.update([x for x in plugin.FEATURES if x.name in event.features])
                if event.info:
                    try:
                        json.dumps(event.info)
                    except TypeError as e:
                        raise ResultError(f"info was not JSON-serialisable: {event.info}") from e

                # sort event contents (including feature values)
                event.sort()

            result.feature_types = sorted(feat_types)
            result.events = [x for x in plugin.events if x.keep()]
            result.data = plugin.data

            if len(warnings) > 0:
                result.state = State(
                    label=State.Label.COMPLETED_WITH_ERRORS,
                    message="Partial completion occurred with the following errors: " + "\n".join(warnings),
                )
            # Automatically set the label to COMPLETED_EMPTY if there are no features or data.
            elif (
                result.state.label == State.Label.COMPLETED
                and len(feat_types) == 0
                and len(result.data) == 0
                and len(event.info) == 0
            ):
                result.state.label = State.Label.COMPLETED_EMPTY

    except ProxyFileNotFoundError:
        if plugin.cfg.assume_streams_available:
            result = JobResult(
                state=State(
                    label=State.Label.OPT_OUT,
                    message="Plugin requires binary content but none exists",
                )
            )
        else:
            result = JobResult(
                state=State(
                    label=State.Label.ERROR_EXCEPTION,
                    failure_name="Plugin failed to handle FileNotFoundError",
                    message=traceback.format_exc(),
                )
            )
    except StorageError:
        # Handled differently from regular exception, as it indicates a problem in the system
        #  rather than an error in the plugin
        # FUTURE Consider handling the case where one stream fails but other streams succeed
        logger.error("Encountered StorageError for entity %s:\n%s" % (job.id, traceback.format_exc()))
        result = JobResult(
            state=State(
                label=State.Label.ERROR_NETWORK,
                failure_name="Storage Failure",
                message=traceback.format_exc(),
            )
        )

    except (ResultError, models.ModelError):
        result = JobResult(
            state=State(
                label=State.Label.ERROR_OUTPUT,
                failure_name="Invalid Plugin Output",
                message=traceback.format_exc(),
            )
        )

    except Exception as e:
        logger.error("%s for %s(%s)\n%s" % (e.__class__.__name__, plugin.NAME, job.id, str(e)))
        result = JobResult(
            state=State(
                label=State.Label.ERROR_EXCEPTION,
                failure_name=e.__class__.__name__,
                message=traceback.format_exc(),
            )
        )

    # reset position of reads for data
    for store in job.get_all_data():
        # will fail on closed files
        store.seek(0)

    # set timing information
    result.date_start = run_start
    now = datetime.datetime.now(datetime.timezone.utc)
    result.date_end = now
    result.runtime = int((now - run_start).total_seconds())

    return result


def _process_features(plugin: mplugin.Plugin, feature_values: dict[str, list[FeatureValue]]) -> list[str]:
    """Verify and normalise feature values produced by a plugin.

    Returns, recoverable warning messages in a list.
    """
    warnings = []
    # verify only registered features were set
    bad_features = set(feature_values.keys()).difference(set(x.name for x in plugin.FEATURES))
    if bad_features:
        raise ResultError(f"Plugin tried to set undeclared features: {sorted(bad_features)}")

    total_values = 0  # total feature values
    feature_counts = []  # list of (feature name, feature value count)
    for feature in plugin.FEATURES:
        if feature.name not in feature_values:
            # feature was not produced
            continue

        # remove duplicate feature values
        feature_values[feature.name] = sorted(set(feature_values[feature.name]))
        num_values = len(feature_values[feature.name])

        # check not too many values for a single feature
        if num_values > plugin.cfg.max_values_per_feature:
            feature_values[feature.name] = feature_values[feature.name][: plugin.cfg.max_values_per_feature]
            warnings.append(
                f"too many values for feature {feature.name} ({num_values}) capping returned values to "
                + f"values at {len(feature_values[feature.name])}"
            )
            num_values = len(feature_values[feature.name])

        # check all values have valid type and valid length
        for v in feature_values[feature.name]:
            if not isinstance(v.value, feature.typeref):
                raise ResultError(
                    "Plugin returned a value with incorrect type "
                    f"({feature.name} should be {feature.typeref}, not {type(v.value)})"
                )
            if isinstance(v.value, (str, bytes)) and len(v.value) > plugin.cfg.max_value_length:
                raise ResultError(f"feature {feature.name} has value that is too long ({v.value[:100]}...)")

        # track problematic numbers of feature values
        total_values += num_values
        feature_counts.append((feature.name, num_values))

    # Remove feature values until enough have been removed to put the total number of features below the limit.
    if total_values > MAX_FEATURE_VALUES:
        num_values_breaking = total_values
        # sort feature counts highest to lowest
        feature_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
        warnings.append(
            f"too many values for plugin ({num_values_breaking}) only returning first ({MAX_FEATURE_VALUES} values)"
        )
        # blindly drop values from the most common features until below the feature limit.
        for feature_name, num_values in feature_counts:
            num_values_drop = min(total_values - MAX_FEATURE_VALUES, num_values)
            # fix total values count
            total_values -= num_values_drop
            warnings.append(f"dropping {num_values_drop}/{num_values} values from {feature_name}")
            # cull values
            feature_values[feature_name] = feature_values[feature_name][: num_values - num_values_drop]
            if total_values <= MAX_FEATURE_VALUES:
                # reached the limit so no further culling required
                break

    return warnings
