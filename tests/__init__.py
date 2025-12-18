"""Init for all tests."""

import logging
import multiprocessing

# FUTURE, shouldn't rely on fork and should switch to forkserver or spawn (defaults for python 3.14)
# Ensure start method is set to fork but ignore if the start method has already been set.
try:
    multiprocessing.set_start_method("fork")
except Exception:
    pass
# Ensure logging is enabled during tests.
logging.basicConfig(force=True)
