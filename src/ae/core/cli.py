"""Core CLI helper utility to run standardized CLI commands and use cases."""

import argparse
import logging
import signal
import sys
import threading
from typing import Callable, Optional, TypeVar

from ae.core.config.settings import Settings
from ae import setup_logging
from ae.core.exceptions import AEException

TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")

logger = logging.getLogger(__name__)


def run_cli_use_case(
    argv: Optional[list[str]],
    parser: argparse.ArgumentParser,
    build_request_fn: Callable[[argparse.Namespace, Settings, threading.Event], TRequest],
    execute_use_case_fn: Callable[[TRequest], TResponse],
    format_response_fn: Callable[[TResponse], int],
    setup_llms_fn: Optional[Callable[[Settings], None]] = None,
) -> int:
    """Standardized runner for AE command line interfaces.

    This helper handles configuration loading, logging setup, LLM configuration,
    graceful signal handling, use case execution, formatting response, and exit code handling.
    """
    args = parser.parse_args(argv)

    # 1. Load Settings
    try:
        settings = Settings.load(config_path=args.config)
        if hasattr(args, "task") and args.task:
            settings.task.name = args.task
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
        return 1

    # 2. Setup Logging
    setup_logging(settings)

    # 3. Setup LLMs (if applicable)
    if setup_llms_fn:
        try:
            setup_llms_fn(settings)
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return 1

    # 4. Handle Signals
    cancel_event = threading.Event()
    def signal_handler(signum, frame):
        logger.warning(f"Interrupt signal ({signum}) received. Initiating graceful shutdown...")
        cancel_event.set()

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass  # If not in main thread

    # 5. Build and execute UseCase
    try:
        request = build_request_fn(args, settings, cancel_event)
        response = execute_use_case_fn(request)
        return format_response_fn(response)

    except KeyboardInterrupt:
        logger.warning("Operation aborted by user.")
        print("\n\n⚠ Operation aborted by user.", file=sys.stderr)
        return 130
    except AEException as e:
        logger.error(f"Execution failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        return 1
