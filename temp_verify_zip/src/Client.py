# src/Client.py
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, List, Optional

import requests
# from dotenv import load_dotenv  # Not needed in Lambda, uses env vars directly

from src.logging_utils import get_logger

# load_dotenv()  # Not needed in Lambda


logger = get_logger(__name__)


class Client(ABC):
    """
    Abstract client that enforces a rate-limit check before sending
    requests.

    Subclasses must implement:
      - ``can_send() -> bool``
      - ``_send(...): Any``
    """

    @abstractmethod
    def can_send(self) -> bool:
        """
        Return whether a request is currently allowed.

        Returns
        -------
        bool
            True if allowed (rate limit respected), False otherwise.
        """
        ...

    @abstractmethod
    def _send(self, *args: Any, **kwargs: Any) -> Any:
        """
        Do the actual request (HTTP, API call, etc.).

        Returns
        -------
        Any
            Response payload (implementation-defined by subclass).
        """
        ...

    def request(self, *args: Any, **kwargs: Any) -> Any:
        """
        Public entrypoint: honor the rate limit before delegating to
        ``_send()``. When the rate limit is reached, the call blocks until a
        new slot becomes available instead of raising immediately.

        Returns
        -------
        Any
            Whatever ``_send(...)`` returns.
        """
        while True:
            if self.can_send():
                logger.debug("Rate limit ok for %s", self.__class__.__name__)
                break

            delay = max(
                self._rate_limit_wait_time(),
                self._rate_limit_min_delay(),
            )
            logger.info(
                "Rate limit hit for %s; sleeping %.3fs",
                self.__class__.__name__,
                delay,
            )
            self._rate_limit_sleep(delay)

        return self._send(*args, **kwargs)

    @staticmethod
    def _rate_limit_min_delay() -> float:
        """Return the minimum backoff to avoid busy-waiting."""
        return 0.05

    def _rate_limit_wait_time(self) -> float:
        """
        Compute how long to sleep before retrying after a rate limit hit.

        Subclasses backed by a shared sliding window should override this to
        return a value based on their internal state (e.g., time until the
        oldest timestamp leaves the window).
        """
        return 1.0

    @staticmethod
    def _rate_limit_sleep(delay: float) -> None:
        time.sleep(delay)


class PurdueClient(Client):
    """
    Client for the Purdue GenAI API that implements a per-class (process-local)
    rate limit shared across all instances.

    Notes
    -----
    - The limiter is global to the class: all instances share the same
      window and request counter via ``request_history``.
    - A request is counted when it is allowed (during ``can_send``), even
      if the subsequent network call fails.
    """
    # Shared, process-local state
    _lock = threading.Lock()
    request_history: Deque[float] = deque()   # shared across all instances

    # ^ Will keep track of requests done accross all instances of this object

    def __init__(self,
                 max_requests: int,
                 token: Optional[str] = None,
                 base_url: str = "https://genai.rcac.purdue.edu/api",
                 window_seconds: float = 60.0) -> None:
        super().__init__()
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Prefer explicit token; otherwise fall back to env var
        if token is None:
            env_token = os.getenv("GEN_AI_STUDIO_API_KEY")
            if not env_token:
                # Don't raise error - allow client to work without token (degraded mode)
                logger.warning("GEN_AI_STUDIO_API_KEY not set - PurdueClient will operate in degraded mode")
                self.token = None
            else:
                self.token = env_token
        else:
            self.token = token
        self.base_url = base_url

    def can_send(self) -> bool:
        """
        Determine whether the request limit has been reached.

        Returns
        -------
        bool
            True if a request is allowed; False otherwise.
        """
        # Get current time for the window
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Use the lock to avoid accessing same memory during multiprocessing
        with PurdueClient._lock:
            # Remove any requests from before the window
            hist = PurdueClient.request_history
            while hist and hist[0] <= cutoff:
                hist.popleft()

            # If we can still make the request, we keep going
            if len(PurdueClient.request_history) < self.max_requests:
                PurdueClient.request_history.append(now)
                return True
        return False

    def _rate_limit_wait_time(self) -> float:
        with PurdueClient._lock:
            history = PurdueClient.request_history
            if not history:
                return self.window_seconds / max(self.max_requests, 1)
            oldest = history[0]

        now = time.monotonic()
        wait = oldest + self.window_seconds - now
        if wait <= 0:
            return self.window_seconds / max(self.max_requests, 1)
        return wait

    def _send(self, method: str, path: str, **kwargs: Any) -> Any:
        """
        Perform an HTTP request to Purdue's API.

        Parameters
        ----------
        method : str
            HTTP method (e.g., "GET", "POST").
        path : str
            API path starting with "/".
        **kwargs : Any
            Additional keyword arguments passed to ``requests.request``.

        Returns
        -------
        Any
            Parsed JSON when possible, otherwise response text.

        Raises
        ------
        RuntimeError
            If the request fails or the response is not OK.

        Examples
        --------
        >>> client.request("GET", "/models", params={"limit": 5})
        """
        if self.token is None:
            raise RuntimeError("PurdueClient cannot make requests without API token")
        
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.token}",
                   "Content-Type": "application/json"}
        try:
            logger.debug("%s request to %s", method, url)
            resp = requests.request(method=method,
                                    url=url,
                                    headers=headers,
                                    timeout=15,
                                    **kwargs)
        except requests.RequestException as e:
            logger.info("Purdue API request failed: %s", e)
            raise RuntimeError(f"Purdue API request failed: {e}") from e

        if not resp.ok:
            msg = f"Purdue API error {resp.status_code}: {resp.text}"
            logger.info("Purdue API error %s for %s", resp.status_code, url)
            raise RuntimeError(msg)

        # Try to parse JSON, else return text
        try:
            logger.debug("Purdue API returned JSON for %s", url)
            return resp.json()
        except ValueError:
            logger.debug("Purdue API returned text for %s", url)
            return resp.text

    def llm(self, message: str) -> str:
        """
        Run the ``llama-3.1-8b`` model on the given input and
        return the assistant's response text.

        Parameters
        ----------
        message : str
            User message for the chat completion API.

        Returns
        -------
        str
            Assistant message content from the Groq API response.
        """
        completion = self.request(
            "POST",
            "/chat/completions",
            json={
                "model": "llama3.2:latest",
                "messages": [{"role": "user", "content": message}],
                "stream": False,
                "temperature": 0.0,
            },
        )

        # Extract from OpenAI-compatible schema: choices[0].message.content
        try:
            return completion["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover - defensive parsing
            logger.info("Unexpected Purdue API response format: %s", e)
            raise RuntimeError("Unexpected Purdue API response format") from e


class HFClient(Client):
    """
    Client for the Huggingface API that implements a per-class (process-local)
    rate limit shared across all instances.

    Notes
    -----
    - The limiter is global to the class: all instances share the same
      window and request counter via ``request_history``.
    - A request is counted when it is allowed (during ``can_send``), even
      if the subsequent network call fails.
    """
    # Shared, process-local state
    _lock = threading.Lock()
    request_history: Deque[float] = deque()   # shared across all instances

    # ^ Will keep track of requests done accross all instances of this object

    def __init__(self,
                 max_requests: int,
                 token: Optional[str] = None,
                 base_url: str = "https://huggingface.co",
                 window_seconds: float = 60.0) -> None:
        super().__init__()
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.base_url = base_url
        # Prefer explicit token; otherwise fall back to env var
        self.token = token

    def can_send(self) -> bool:
        """
        Determine whether the request limit has been reached.

        Returns
        -------
        bool
            True if a request is allowed; False otherwise.
        """
        # Get current time for the window
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Use the lock to avoid accessing same memory during multiprocessing
        with HFClient._lock:
            # Remove any requests from before the window
            hist = HFClient.request_history
            while hist and hist[0] <= cutoff:
                hist.popleft()

            # If we can still make the request, we keep going
            if len(HFClient.request_history) < self.max_requests:
                HFClient.request_history.append(now)
                return True
        return False

    def _rate_limit_wait_time(self) -> float:
        with HFClient._lock:
            history = HFClient.request_history
            if not history:
                return self.window_seconds / max(self.max_requests, 1)
            oldest = history[0]

        now = time.monotonic()
        wait = oldest + self.window_seconds - now
        if wait <= 0:
            return self.window_seconds / max(self.max_requests, 1)
        return wait

    def _send(self, method: str, path: str, **kwargs: Any) -> Any:
        """
        Perform an HTTP request to HF's API.

        Parameters
        ----------
        method : str
            HTTP method (e.g., "GET", "POST").
        path : str
            API path starting with "/".
        **kwargs : Any
            Additional keyword arguments passed to ``requests.request``.

        Returns
        -------
        Any
            Parsed JSON when possible, otherwise response text.

        Raises
        ------
        RuntimeError
            If the request fails or the response is not OK.

        Examples
        --------
        >>> client.request("GET", "/api/spaces", params={"limit": 5})
        >>> client.request("GET", f"/api/models/{repo_id}")['cardData']
        """
        url = f"{self.base_url}{path}"
        try:
            logger.debug("%s request to %s", method, url)
            resp = requests.request(method=method,
                                    url=url,
                                    headers={},
                                    timeout=15,
                                    **kwargs)
        except requests.RequestException as e:
            logger.info("HF API request failed: %s", e)
            raise RuntimeError(f"HF API request failed: {e}") from e

        if not resp.ok:
            msg = f"HF API error {resp.status_code}: {resp.text}"
            logger.info("HF API error %s for %s", resp.status_code, url)
            raise RuntimeError(msg)

        # Try to parse JSON, else return text
        try:
            logger.debug("HF API returned JSON for %s", url)
            return resp.json()
        except ValueError:
            logger.debug("HF API returned text for %s", url)
            return resp.text


class GitClient(Client):
    """
    Client for interacting with a local Git repository using the `git`
    command-line. Implements a per-class (process-local) rate limit shared
    across all instances, matching the style of other clients.

    Notes
    -----
    - The limiter is global to the class: all instances share the same
      window and request counter via ``request_history``.
    - A request is counted when it is allowed (during ``can_send``), even
      if the subsequent command fails.
    """
    # Shared, process-local state
    _lock = threading.Lock()
    request_history: Deque[float] = deque()  # shared across all instances

    # ^ Will keep track of requests done accross all instances of this object

    def __init__(self,
                 max_requests: int,
                 repo_path: str = ".",
                 window_seconds: float = 30.0) -> None:
        super().__init__()
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Normalize path but do not validate at construction time
        # to ease testing
        self.repo_path = os.fspath(repo_path)

    def can_send(self) -> bool:
        """
        Determine whether the request limit has been reached.

        Returns
        -------
        bool
            True if a request is allowed; False otherwise.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with GitClient._lock:
            hist = GitClient.request_history
            while hist and hist[0] <= cutoff:
                hist.popleft()

            if len(GitClient.request_history) < self.max_requests:
                GitClient.request_history.append(now)
                return True
        return False

    def _rate_limit_wait_time(self) -> float:
        with GitClient._lock:
            history = GitClient.request_history
            if not history:
                return self.window_seconds / max(self.max_requests, 1)
            oldest = history[0]

        now = time.monotonic()
        wait = oldest + self.window_seconds - now
        if wait <= 0:
            return self.window_seconds / max(self.max_requests, 1)
        return wait

    def _send(self, *git_args: str) -> str:
        """
        Execute a git command in the configured repository path.

        Parameters
        ----------
        *git_args : str
            Arguments to pass to the `git` binary,
            e.g., ("status", "--porcelain").

        Returns
        -------
        str
            Standard output from the command (stripped of trailing whitespace).

        Raises
        ------
        RuntimeError
            If the command fails (non-zero return code) or cannot be executed.

        Examples
        --------
        >>> client = GitClient(max_requests=3, repo_path=".")
        >>> client.request("status", "--porcelain")
        """
        cmd = ["git", *git_args]
        logger.debug("Running git command: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.info("Git command failed: %s", e)
            raise RuntimeError(f"Git command failed to run: {e}") from e

        if proc.returncode != 0:
            err = proc.stderr.strip() or "unknown git error"
            logger.info("Git command error %s: %s", proc.returncode, err)
            raise RuntimeError(f"Git command error {proc.returncode}: {err}")

        logger.debug("Git command succeeded with %d bytes of output",
                     len(proc.stdout or ""))
        return (proc.stdout or "").rstrip("\n")

    def list_files(self) -> List[str]:
        """
        List tracked files under the repository using `git ls-files`.

        Returns
        -------
        list[str]
            File paths relative to repository root.

        Examples
        --------
        >>> client = GitClient(max_requests=3, repo_path=".")
        >>> client.list_files()  # doctest: +SKIP
        ["src/main.py", "README.md", ...]
        """
        out = self.request("ls-files")
        if not out:
            return []
        return out.splitlines()


if __name__ == "__main__":
    pass
