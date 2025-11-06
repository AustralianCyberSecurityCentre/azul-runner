"""Common logging utilities for azul-runner."""

import logging
import multiprocessing
import sys
from logging import handlers
from typing import Optional, TypeAlias

AZUL_RUNNER_LOGGER_NAME = "azul_runner"
LogLevel: TypeAlias = int | str


def setup_logger(log_level: LogLevel, queue: Optional[multiprocessing.Queue] = None):
    """Setup the loggers for azul runner."""
    # use log level settings
    rootlogger = logging.getLogger()
    # Set all loggers to error/warning
    rootlogger.setLevel(log_level)

    # Clear existing handlers
    for old_handle in rootlogger.handlers:
        rootlogger.removeHandler(old_handle)

    # Set the handler and formatter for azul_runner logggers.
    if queue:
        # Logging Queue is in use, send all logs to the parent process via the queue and don't log to stderr.
        h3 = handlers.QueueHandler(queue)
        h3.setLevel(log_level)
        rootlogger.addHandler(h3)
    else:
        # Root process is configured to log to stderr.
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(log_level)
        h.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)-30s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S%z"
            )
        )
        rootlogger.addHandler(h)

    # Set HTTPX logging to debug to avoid logging spam in production
    logging.getLogger("httpx").setLevel("DEBUG")
    logging.getLogger("httpcore").setLevel("DEBUG")


class AddLoggingQueueListener:
    """Handle adding and removing a logging queue handler."""

    def __init__(self, queue: multiprocessing.Queue, logger: logging.Logger):
        """Handle adding and removing a logging queue handler."""
        self.queue = queue
        self.logger = logger

    def __enter__(self):
        """Open the queue listener."""
        self.q_listen = handlers.QueueListener(self.queue, *self.logger.handlers)
        self.q_listen.start()

    def __exit__(self, *args, **kwargs):
        """Close the Queue listener to stop log capture."""
        self.q_listen.stop()
