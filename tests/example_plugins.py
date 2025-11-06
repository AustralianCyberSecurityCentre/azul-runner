"""Example plugins used in readme."""

from azul_runner import (
    BinaryPlugin,
    Feature,
    FeatureType,
    Job,
    State,
    add_settings,
    cmdline_run,
)


class LookForThings(BinaryPlugin):
    """Look for things."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Custom tag", FeatureType.String),
    ]
    SETTINGS = add_settings(
        filter_data_types={"content": ["text/plain"]},
    )

    def execute(self, job: Job):
        """Find peanuts."""
        data = job.get_data()
        # 'data' is a file-like object that supports seeking and being read from
        # (The content may be retrieved in parts if the file is large and non-local)
        header: bytes = data.read(7)
        if header == b"PEANUT:":
            # create a tag
            self.add_feature_values("tag", "may contain nuts")
            # add the next 24 bytes as a child
            c = self.add_child_with_data(
                relationship={"label": "peanut"},
                data=data.read(24),
            )
            c.add_feature_values("tag", "may be hard to crack")
        else:
            return State.Label.OPT_OUT


if __name__ == "__main__":
    cmdline_run(plugin=LookForThings)


class LookForMZ1(BinaryPlugin):
    """Look for MZ type 1."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Tags files that might be .EXEs", FeatureType.String),
    ]

    def execute(self, job: Job):
        """Find MZ."""
        data = job.get_data()
        # 'data' is a file-like object that supports seeking and being read from
        # (The content may be retrieved in parts if the file is large and non-local)
        header: bytes = data.read(2)
        if header == b"MZ":
            self.add_feature_values("tag", "might be EXE")
        else:
            return State.Label.OPT_OUT


class LookForMZ2(BinaryPlugin):
    """Look for MZ type 2."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Tags files that might be .EXEs", FeatureType.String),
    ]
    SETTINGS = add_settings(
        filter_data_types={"content": ["executable/windows/pe32", "executable/windows/dos"]},
    )

    def execute(self, job: Job):
        """Find MZ."""
        # This plugin will only run over files that have been identified
        #  by azul-runner/dispatcher as EXE files.
        self.add_feature_values("tag", "It's an EXE!")


class LookForMZ3(BinaryPlugin):
    """Look for MZ type 3."""

    VERSION = "1.0"
    FEATURES = [
        Feature("tag", "Tags files that might be .EXEs", FeatureType.String),
    ]
    SETTINGS = add_settings(
        filter_data_types={"content": ["executable/windows/pe32", "executable/windows/dos"]},
    )

    def execute(self, job: Job):
        """Find MZ."""
        buffer = job.get_data().read(64)
        # add child binary
        c = self.add_child_with_data(
            relationship={"label": "First 64 bytes"},
            data=buffer,
        )
        # add feature to the child binary
        c.add_feature_values("tag", "the extracted child")
        # add feature to the original incoming binary
        self.add_feature_values("tag", ["Might be an exe", "Extracted header"])


if __name__ == "__main__":
    cmdline_run(plugin=LookForThings)
