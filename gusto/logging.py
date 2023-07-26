import logging
import sys
import os

from datetime import datetime
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa: F401

from pyop2.mpi import COMM_WORLD

__all__ = ["logger", "set_log_handler"]

logging.captureWarnings(True)
logger = logging.getLogger("gusto")

# Set the log level based on environment variables
log_level = os.environ.get("GUSTO_LOG_LEVEL", WARNING)
logfile_level = os.environ.get("GUSTO_FILE_LOG_LEVEL", DEBUG)
logconsole_level = os.environ.get("GUSTO_CONSOLE_LOG_LEVEL", INFO)
log_level_list = [log_level, logfile_level, logconsole_level]
log_levels = [
    logging.getLevelNamesMapping().get(x) if isinstance(x, str)
    else x
    for x in log_level_list
]
logger.setLevel(min(log_levels))

# Setup parallel logging based on environment variables
parallel_log = os.environ.get("GUSTO_PARALLEL_LOG", None)
options = ["CONSOLE", "FILE", "BOTH"]
if parallel_log is not None:
    parallel_log = parallel_log.upper()
    if parallel_log.upper() not in options:
        parallel_log = None


def create_logfile_handler(path):
    ''' Handler for logfiles

    Args:
        path: path to log file

    '''
    logfile = logging.FileHandler(filename=path, mode="w")
    logfile.setLevel(logfile_level)
    logfile_formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s'
    )
    logfile.setFormatter(logfile_formatter)
    return logfile


def create_console_handler(fmt):
    ''' Handler for console logging

    Args:
        fmt: format string for log output
    '''
    console = logging.StreamHandler()
    console.setLevel(logconsole_level)
    console_formatter = logging.Formatter(fmt)
    console.setFormatter(console_formatter)
    return console


def set_log_handler(comm=COMM_WORLD):
    """
    Set all handlers for logging.

    Args:
        comm (:class:`MPI.Comm`): MPI communicator.
    """
    # Set up logging
    timestamp = datetime.now()
    logfile_name = f"gusto-{timestamp.strftime('%Y-%m-%dT%H%M%S')}"
    if parallel_log in ["FILE", "BOTH"]:
        logfile_name += f"_{comm.rank}"
    logfile_name += ".log"
    if comm.rank == 0:
        os.makedirs("results", exist_ok=True)
    logfile_path = os.path.join("results", logfile_name)

    console_format_str = ""
    if parallel_log in ["CONSOLE", "BOTH"]:
        console_format_str += f"[{comm.rank}] "
    console_format_str += '%(levelname)-8s %(message)s'

    if comm.rank == 0:
        # Always log on rank 0
        logger.addHandler(create_logfile_handler(logfile_path))
        logger.addHandler(create_console_handler(console_format_str))
    else:
        # Only log on other ranks if enabled
        if parallel_log in ["FILE", "BOTH"]:
            logger.addHandler(create_logfile_handler(logfile_path))
        if parallel_log in ["CONSOLE", "BOTH"]:
            logger.addHandler(create_console_handler(console_format_str))
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())

    logger.info("Running %s" % " ".join(sys.argv))

