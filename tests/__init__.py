"""Init for all tests."""

import logging
import multiprocessing

# Preload azul_runner, this dramatically improves test speed when running in forkserver.
multiprocessing.set_forkserver_preload(["azul_runner"])

# Ensures forkserver which is used which is how all future code is expected to run (python 3.14+)
try:
    multiprocessing.set_start_method("forkserver")
except Exception:
    pass
# Ensure logging is enabled during tests.
logging.basicConfig(force=True)
