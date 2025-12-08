import os
import sys
from pathlib import Path

from src.Dispatcher import Dispatcher
from src.Display import print_results
from src.logging_utils import get_logger
from src.Metrics import (AvailabilityMetric, BusFactorMetric, CodeQuality,
                         DatasetQuality, LicenseMetric,
                         PerformanceClaimsMetric, RampUpTime, SizeMetric)
from src.Parser import Parser

logger = get_logger(__name__)


def _validate_env_or_exit() -> None:
    """Validate required environment variables and exit(1) if invalid.

    Requirements (per spec provided by user):
    - GITHUB_TOKEN must be set, non-empty, and look like a GitHub token
      (simple heuristic: length >= 8, contains at least one underscore or
      starts with 'ghp_' / 'github_' prefix).
    - LOG_FILE must be a writable path; its parent directory must exist
      or be creatable; we attempt to open it in append mode.
    """
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    log_path_raw = os.environ.get("LOG_FILE", "").strip()

    def fail(msg: str) -> None:
        logger.error("ENV VALIDATION FAILED: %s", msg)
        sys.exit(1)

    # Validate GitHub token
    if not gh_token:
        fail("GITHUB_TOKEN is empty or unset")
    # Heuristic token format check
    if not (
        len(gh_token) >= 8
        and (
            gh_token.startswith("ghp_")
            or gh_token.startswith("github_")
            or "_" in gh_token
        )
    ):
        fail("GITHUB_TOKEN format appears invalid")

    # Validate log file path
    if not log_path_raw or not log_path_raw.endswith(".log"):
        fail("LOG_FILE is empty or unset")
    log_path = Path(log_path_raw)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8"):
            pass
    except Exception as e:
        fail(f"LOG_FILE is not writable: {e}")

    logger.debug("Environment validation passed (log: %s)", log_path)


if __name__ == "__main__":
    _validate_env_or_exit()
    if len(sys.argv) < 2:
        logger.error("No input file path provided")
        sys.exit(1)
    input_path = sys.argv[1]
    logger.info("Starting CLI processing for %s", input_path)
    parse = Parser(input_path)
    url_groups = parse.getGroups()

    dispatcher = Dispatcher([LicenseMetric(),
                             SizeMetric(),
                             RampUpTime(),
                             AvailabilityMetric(),
                             DatasetQuality(),
                             CodeQuality(),
                             PerformanceClaimsMetric(),
                             BusFactorMetric()])
    for group in url_groups:
        logger.debug("Dispatching metrics for group %s", group)
        results = dispatcher.dispatch(group)
        logger.debug("Metrics complete for group %s", group)
        print_results(group, results)
    logger.info("Finished processing %d group(s)", len(url_groups))
