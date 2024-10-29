import logging
import argparse
from typing import Optional


def get_debug_flag() -> bool:
    """Get debug flag from command line arguments."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--debug", "-v", action="store_true", help="Enable debug logging"
    )
    args, _ = parser.parse_known_args()
    return args.debug


def setup_logging(
    name: Optional[str] = None, debug: Optional[bool] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name (str, optional): Logger name. Defaults to __name__ of caller.
        debug (bool, optional): Override debug flag. Defaults to command line arg.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Use module name if none provided
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)

    # Only configure root logger once
    if not logging.getLogger().handlers:
        debug_enabled = debug if debug is not None else get_debug_flag()
        log_level = logging.DEBUG if debug_enabled else logging.WARNING

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add debug status to log
        logger.debug("Debug logging enabled")

    return logger
