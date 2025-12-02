"""
Logging configuration for biofilm project

Usage:
    from src.logger import setup_logger
    logger = setup_logger()
    logger.info("Starting calibration...")
"""
import logging
import sys
from pathlib import Path


def setup_logger(name="biofilm", level=logging.INFO, log_file=None):
    """
    Configure project-wide logger

    Parameters
    ----------
    name : str, default="biofilm"
        Logger name
    level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str or Path, optional
        If provided, also log to this file

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logger()
    >>> logger.info("Starting simulation")

    >>> # Debug mode with file output
    >>> logger = setup_logger(level=logging.DEBUG, log_file="biofilm.log")
    >>> logger.debug("Detailed debug info")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name="biofilm"):
    """
    Get existing logger instance

    Parameters
    ----------
    name : str, default="biofilm"
        Logger name

    Returns
    -------
    logging.Logger
        Existing logger instance
    """
    return logging.getLogger(name)
