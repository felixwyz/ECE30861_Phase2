# src/utils.py
# Store any helper functions here

import re
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup

from src.Client import HFClient
from src.logging_utils import get_logger

logger = get_logger(__name__)


def browse_hf_repo(
    client: HFClient,
    repo_id: str,
    repo_type: str = "model",
    revision: str = "main",
    recursive: bool = True,
) -> List[Tuple[str, int]]:
    """
    Browse files of a Hugging Face repo using HFClient.

    Parameters
    ----------
    client : HFClient
        An instance of your HFClient (already configured with token).
    repo_id : str
        Repo identifier, e.g. "facebook/opt-125m".
    repo_type : str
        One of "model", "dataset", or "space".
    revision : str
        Branch, tag, or commit SHA (default: "main").
    recursive : bool
        Whether to traverse all subfolders.

    Returns
    -------
    List[Tuple[str, int]]
        A list of (file_path, size_in_bytes). Directories are skipped.
    """
    plural = {"model": "models",
              "dataset": "datasets",
              "space": "spaces"}[repo_type]
    path = f"/api/{plural}/{repo_id}/tree/{revision}"
    params = {"recursive": 1} if recursive else {}

    logger.info("Browsing HF repo %s (%s) at %s", repo_id, repo_type, revision)
    data = client.request("GET", path, params=params)

    # Handle response format
    if isinstance(data, dict) and "tree" in data:
        entries = data["tree"]
    elif isinstance(data, list):
        entries = data
    else:
        return []

    files = [
        (e["path"], e.get("size", -1))
        for e in entries
        if e.get("type") != "directory"
    ]
    logger.debug("Retrieved %d file(s) from %s", len(files), repo_id)
    return files


_HF_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def injectHFBrowser(
    model: str,
    headless: bool = True,
    timeout: float = 20.0,
) -> str:
    """
    Retrieve the Hugging Face model page using an HTTP request.

    Parameters
    ----------
    model : str
        Fully-qualified URL for the target Hugging Face model repository.
    headless : bool
        Retained for compatibility with the previous Selenium-based API.
    timeout : float
        Maximum time (seconds) to wait for the HTTP response.

    Returns
    -------
    str
        Visible text contained within the page `<main>` element (or `<body>`).

    Raises
    ------
    RuntimeError
        If the Hugging Face page cannot be downloaded.
    """
    _ = headless  # maintained for call-sites; no longer used.

    logger.info("Fetching Hugging Face page %s", model)
    try:
        response = requests.get(
            model,
            headers=_HF_DEFAULT_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover
        logger.info(
            "Failed to fetch Hugging Face page %s: %s",
            model,
            exc,
        )
        raise RuntimeError(
            f"Failed to fetch Hugging Face page: {model}"
        ) from exc

    soup = BeautifulSoup(response.text, "html.parser")
    container = soup.find("main") or soup.find("body")
    if container is None:
        logger.debug("No <main> or <body> element found for %s", model)
        return ""

    text = container.get_text(separator="\n")
    cleaned = _normalize_hf_text(text)
    logger.debug("Fetched page text length %d for %s", len(cleaned), model)
    return cleaned


def _normalize_hf_text(raw_text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""

    # Replace Windows-style line endings and collapse multiple blank lines.
    text = raw_text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing spaces on each line for consistency.
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()
