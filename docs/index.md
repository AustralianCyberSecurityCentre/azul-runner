# Overview

This guide will outline the process of creating and deploying a plugin for Azul;

* Decide if you should create a separate library for your plugin
* Create the basic source code repository for your plugin
* Understand how a basic plugin works
* Adapt the basic plugin to function with your automated extraction
* Choose good feature names for your data
* Decide if you need more advanced feature value properties
    * labels
    * offsets
    * data types (url, filepath, integer)
* Create regression tests + unit tests
* Ensure the docker build of your plugin succeeds
* Getting a code review done on your plugin ( and responding to feedback )
* Check your plugin functions correctly in a test instance of Azul
* Check your plugin functions correctly in a prod instance of Azul

[The coding guide is found here.](./coding/index.md)

## Before Starting

Recommendations before starting plugin development

* installed and configured your system as required
    * Refer to system specific documentation for Azul (if required)
    * install docker, access to python repositories, etc.
* some knowledge about what you want to achieve with your plugin
    * decode malware configuration including c2 data
    * decompress/decode a unique file type
    * something else.
* some knowledge about Azul and how it presents information, including
    * what feature values are
    * what parent/child relationships are
    * what text reports are
* a basic process or library or program that demonstrates the information can be extracted automatically


## Decide if you should create a separate library for your plugin

Separate libraries allow for common code to be reused between different projects. They would be 
stored in a separate source code repository from the main plugin.

Avoid using a separate library where possible.

If your plugin is the only thing using your other code, do not make a separate library.

If you want to run your code outside of the plugin harness, put it in a separate script in your plugin repository.

If you have a few utility functions that you believe would be useful for many plugins, they should be integrated into the azul-runner project so that everyone can benefit.

Using a separate library will not allow you to make changes to plugins without a PR, as the master branch will need to be marked read-only as well.

Using a separate library will require another set of test cases in that library to prevent regressions.

Using a separate library requires double the amount of automated build jobs that Azul devs need to maintain.

Finally, please consider that someone else will have to pick up your project, and the simpler the 
layout of your project is, the easier it will be for them to understand. 

## Update LibMagic

libmagic has changed over time, and the default ubuntu/debian install is usually out of date.

The current expected version is 5.45

Check your current version via `file --version`

Update libmagic via

```
git clone --depth 1 --branch FILE5_46 https://github.com/file/file
cd file/ && \
    autoreconf -f -i && \
    ./configure --disable-silent-rules && \
    make -j4 && \
    sudo make install && \
    cp ./magic/magic.mgc /etc/magic
ldconfig -v && file --version
```

## Create the basic source code repository for your plugin

1. Create a new source code repository where your organisation stores code.
    Named like `azul-plugin-mypluginname`.
1. Ensure your team has write/admin permissions on the repository.
1. Generate the basic code for your plugin via the `azure-generator`.
1. Commit this master/main branch to version control.
1. Create a branch where you will perform your plugin development.

## Understand how a basic plugin works

Look through the  files that are generated. The generated code defines a basic plugin with some trivial test cases.

Read the comments and check the links below to understand what does what. Also ask the Azul team for more information.

### .devcontainer/

Contains configuration for VisualStudio Code to enable development inside of a docker container.
 This enables development and local testing/debugging in a consistent environment no matter what operating system the developer is using. You should not need to edit this.

### tests/

Contains example of test for plugin

https://docs.python.org/3/library/unittest.html

### mypluginname/

Contains python code for your plugin

### .gitignore

Controls what files in the directory are never added to git version control

https://git-scm.com/docs/gitignore

### .dockerignore

Controls what files are available from the directory during a docker build operation

https://docs.docker.com/engine/reference/builder/#dockerignore-file

### .pre-commit-config.yaml

Controls what checks are performed before allowing a `git commit` operation (if activated)

https://pre-commit.com/

### Dockerfile

Specifies build steps for this repository during the docker build process.

Avoid getting too fancy/complex with build steps here.

https://docs.docker.com/engine/reference/builder/

### Makefile

Helper for local testing of docker build process. You should avoid editing this.

https://en.wikipedia.org/wiki/Make_(software)

### README.md

Information about your plugin.

https://www.makeareadme.com/

### debian.txt

Specify debian specific packages that must be installed for your plugin to function.

This file is used in the Dockerfile and the CI job to install system packages for your plugin.

It is important that any system requirements (apt install) are specified here and not in the dockerfile.

### pyproject.toml

Defines important metadata about your python package.

https://www.python.org/dev/peps/pep-0621/

### requirements.txt

Defines python packages required for this plugin to work.

https://pip.pypa.io/en/stable/user_guide/#requirements-files

### requirements_tests.txt

Defines python packages required for the tests for this plugin to work.

https://pip.pypa.io/en/stable/user_guide/#requirements-files

### setup.py

Defines important metadata about your python package.

https://docs.python.org/3/distutils/setupscript.html

### tox.ini

Defines test execution and style checking for the repository.

https://tox.wiki/en/latest/index.html

## Adapt the basic plugin to function with your automated extraction

This varies based on what you are trying to achieve.

1. figure out how files are supplied to your plugin
1. define file types that are good for your plugin to process
1. import your script/library in the main plugin module
1. use your library to get some output from the file

At this point, you are just trying to get some basic dictionary or raw data from your library accessible from
 your plugin.
You are generally not trying to adapt the data to fit in with feature values just yet unless you are sure 
what the feature values should be (next step).

Please see items in the coding subsection for more information about developing with the azul-runner

## Choose good feature names for your data

You might have some ideas already what feature names you want to use for all your information.

It is preferred to use existing feature names and reuse them where possible, as this makes pivoting between
features different values easier event when different plugins have produced the feature values.

Example; 
* plugin a has a feature name and value ('config_callback_domain', 'httpx://evil.bad.com') for a binary 
* plugin b has a feature name and value ('config_domain_callback', 'httpx://evil.bad.com') for a different binary

Azul and you will not be able to recognise this relationship because these two plugins produced the same 
value for two DIFFERENT features.

You want to be using the same feature name in this case.

Please reference [features.md](./features.md) for much more comprehensive information about features


## Decide if you need more advanced feature value properties

* labels 
* offsets
* data types (uri, filepath, etc.)

Please see items in the coding subsection for more information about developing with the azul-runner


## Create regression tests + unit tests

Tests are run frequently in CI to check that the next deployment of a plugin won't produce bad results.
 Even if your code doesn't change, bad things can happen. 

For example, a system library might update which breaks libmagic and therefore breaks your plugin. 
If you have a test that checks your plugin thoroughly, it should then fail, which will keep the old
 version of your plugin running in Azul until the problem can be resolved.

Tests that include running a plugin against a known sample of malware and checking the feature values
 are output correctly would be considered regression tests.

Tests that run a particular function in relative isolation for specific inputs and outputs would be
 considered unit tests.

Azul maintainers like to see both in a plugin, but for us the most critical tests are the regression tests.

Please note that if you are using a separate library (in a separate repository) you have written, that
 library MUST have unit tests, as it is not a complete plugin, but it still needs to be tested to ensure you
  don't break your plugin.


## Run code quality checks

Code quality checks help ensure that common python errors or gotchas are avoided and make code easier to read.

They also make it easier for people to review pull requests, as reviewers can focus more on the overall logic of the code and less on the specific python operations that are being performed :)

Code quality checks are run in CI and will cause builds to fail if they are not successful.

### black

https://github.com/psf/black

Black is an automated styling tool, it will automatically format your code to make lines a standard length and
change spacing and other aspects of code to have a uniform layout.

Black is configured through the `pyproject.toml` file which should not need to be changed from the defaults.

`pip install black`

`black .`

Remember to look at the changes black has made and commit them to your branch.

### isort

isort imports and separate into groups of first-party, second-party and third-party modules.

`pip install isort`

`isort .`


### flake8

https://flake8.pycqa.org/en/latest/

Flake8 checks for code quality in a different way, it will alert you when some part of the python code looks like it might potentially have an error in it.

Flake8 is run within the tox styling checks it has been configured to print the files and line numbers where it has found an issue.

`tox -e style`

The errors should usually be descriptive enough to be fixable relatively easily. If you have a good reason for doing something, you can add an flake ignore comment to make a specific check pass. Use the code for the check you want to disable.

`suspect.call()  # noqa: B008`


## Ensure the docker build of your plugin succeeds

A successful docker build represents a platform and configuration independent build and test of your plugin.
It is important, because the CI systems used for Azul don't have the same software installed as your desktop does.

There are two parts that make this complex.

### Ensure your environment is setup correctly

If building straight from the internet with no proxy, you should be fine.

You will need to set your `PIP_INDEX_URL` and `PIP_TRUSTED_HOST` variables to
point to devpi or another pypi mirror with Azul packages available.

Otherwise you must have environment variables `http_proxy`, `https_proxy` and `no_proxy` configured correctly.

You must be able to pull any base images from either docker hub or your private registry as required. 
Certain environments require custom images to be used instead of the raw dockerhub images.

You must be able to use docker commands without putting 'sudo' at the front.

docker instead of podman is what is typically used.

Please to refer to environment specific setup as required.

### Make changes to get docker build working

#### Makefile

Don't edit this, not used by CI.

#### Dockerfile

Generally, you shouldn't need to edit this unless you need to perform a custom and separate compilation 
of some weird python or system libraries. Hopefully you don't have to do this as it is usually painful.

Avoid connecting to the internet during the build, as it is likely to prevent your plugin being 
integrated with CI tools in other environments.

#### debian.txt

This is where any required system dependencies should go, like 'libmagic-dev'.

They need to go here instead of directly in the dockerfile as there is a special build step in CI
 which compiles and publishes your wheel after installing things from debian.txt.

#### requirements.txt

python packages your plugin needs during normal operation.
e.g. requests

#### requirements_test.txt

python packages your plugin needs during test execution.
e.g. malpz & pytest


### Try to build the dockerfile

If your CI build system (not local machine) is fully internet connected, `make`.

Otherwise refer to system specific instructions, which will need to override something like

`BASE_IMAGE=artifactory.something/python BASE_TAG=yep BUILD_IMAGE=artifactory.something/python BUILD_TAG=cool make`

## Getting a code review done on your plugin ( and responding to feedback )

At this point you will have a 'master' branch with the basic template for your plugin, and 
another branch where you have been doing your actual development.

Submit a pull request from your development branch into the master branch and add responsible 
people to the list of reviewers.

They will then make comments and ask questions about the plugin to ensure it looks stable enough to deploy.

You will need to respond appropriately, and then your code will be merged into the master branch.

## Check your plugin functions correctly in a test instance of Azul

The Azul admins will need to configure your plugin to build and deploy into the test environment of Azul.

You will then be able to upload files to the test instance and see if the plugin works as you wanted it to.

If it does not, you will need to create a new branch in the plugin and make your changes, submit 
PR and respond to feedback.

## Check your plugin functions correctly in a prod instance of Azul

If your plugin is functioning correctly, it will be added to the next production deployment of Azul.

## Celebrate that your plugin is finding the bad

Good job!
