#!/usr/bin/env python3
"""Setup script."""
import os

from setuptools import setup


def open_file(fname):
    """Open and return a file-like object for the relative filename."""
    return open(os.path.join(os.path.dirname(__file__), fname))


setup(
    name="azul-runner",
    description="Core framework for writing Python plugins for Azul.",
    author="Azul",
    author_email="azul@asd.gov.au",
    url="https://www.asd.gov.au/",
    packages=["azul_runner"],
    include_package_data=True,
    python_requires=">=3.12",
    classifiers=[],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=[r.strip() for r in open_file("requirements.txt") if not r.startswith("#")],
    extras_require={
        # test_utils is only required for plugin tests and has additional requirements.
        "test_utils": [r.strip() for r in open_file("requirements_tests.txt") if not r.startswith("#")],
    },
)
