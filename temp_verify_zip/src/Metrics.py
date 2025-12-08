# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import (Any, Callable, Iterable, List, Mapping, Optional, Sequence,
                    TypeVar)
from urllib.parse import quote, urlparse

import requests

from src.Client import GitClient, HFClient, PurdueClient
from src.logging_utils import get_logger
from src.utils import browse_hf_repo, injectHFBrowser

logger = get_logger(__name__)


# Rate limit configuration shared across metrics so that all callsites are
# consistent and easy to tune in one place.
PURDUE_MAX_REQUESTS = 40
PURDUE_WINDOW_SECONDS = 30.0
HF_MAX_REQUESTS = 120
HF_WINDOW_SECONDS = 30.0
GIT_MAX_REQUESTS = 40
GIT_WINDOW_SECONDS = 30.0

RAMP_UP_LLM_ATTEMPTS = 3
CODE_QUALITY_LLM_ATTEMPTS = 3

LLM_SCORE_PATTERN = re.compile(r"FINAL\s*SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)",
                               re.IGNORECASE)

LLM_MAX_ATTEMPTS = 3
LLM_RETRY_SLEEP_SECONDS = 0.2


T = TypeVar("T")


def _call_llm_with_retry(
    call: Callable[[], Any],
    parser: Callable[[Any], T],
    *,
    attempts: int = LLM_MAX_ATTEMPTS,
    delay_seconds: float = LLM_RETRY_SLEEP_SECONDS,
    description: str = "LLM call",
) -> T:
    """Invoke ``call`` until ``parser`` succeeds or attempts are exhausted."""

    last_error: Optional[Exception] = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            raw = call()
            return parser(raw)
        except Exception as exc:
            # Swallow and retry transient LLM failures so the caller only
            # sees an error after we've exhausted the configured budget.
            last_error = exc
            logger.info(
                "%s attempt %d/%d failed: %s",
                description,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts and delay_seconds > 0:
                time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{description} failed without raising an exception")


def _parse_numeric_response(
    raw: Any,
    *,
    allowed: Optional[Sequence[float]] = None,
) -> float:
    """Extract a numeric score using the ``FINAL SCORE`` tag from the LLM."""

    # Normalise the response to text so regex parsing works across return
    # types coming from the LLM client.
    text = str(raw or "").strip()
    match = LLM_SCORE_PATTERN.search(text)
    if not match:
        raise ValueError("FINAL SCORE line not found in LLM response")
    value = float(match.group(1))
    if allowed is not None:
        for candidate in allowed:
            if abs(candidate - value) < 1e-3:
                return candidate
        raise ValueError(f"LLM value {value} not in allowed set {allowed}")
    return value


@dataclass(frozen=True)
class MetricResult:
    """
    Canonical result object returned by all metrics.

    Attributes
    ----------
    metric : str
        Human-friendly metric name (e.g., "License Check").
    key : str
        Stable identifier/slug for the metric (e.g., "license").
    value : Any
        The primary result produced by the metric (bool, str, dict, etc.).
    latency_ms : float
        How long the metric took to execute (milliseconds).
    details : Optional[Mapping[str, Any]]
        Optional extra information for display or debugging.
    error : Optional[str]
        If the metric failed, put a concise error message here
        and set `value` as appropriate.
    """
    metric: str
    key: str
    value: Any
    latency_ms: float
    details: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None


class Metric(ABC):
    """
    Abstract base class for metrics.

    Subclasses must implement ``compute()`` to perform the actual work.
    """

    name: str  # Human-friendly metric name (e.g., "License Check").
    key: str  # Identifier/slug for the metric (e.g., "license").

    @abstractmethod
    def compute(self, inputs: dict[str, Any],
                **kwargs: Any) -> float | dict[str, float]:
        """
        Compute the metric score from parsed inputs.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parsed inputs required by the metric.
        **kwargs : Any
            Optional per-metric tuning parameters.

        Returns
        -------
        float | dict[str, float]
            Either a scalar score between 0.0 and 1.0 or a mapping of
            device/platform identifiers to scores in that range.
        """
        raise NotImplementedError


class RampUpTime(Metric):
    """
    Metric estimating ramp-up time based on the length of the
    'Use this model' / 'Usage' section in a model's README on HuggingFace.

    A shorter section implies quicker ramp-up, yielding a higher score.
    """
    name = "Ramp-Up Time"
    key = "ramp_up_time"

    def __init__(self):
        self.client = HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )
        self.grok = PurdueClient(
            max_requests=PURDUE_MAX_REQUESTS,
            window_seconds=PURDUE_WINDOW_SECONDS,
        )

    def _extract_usage_section(self, text: str) -> str | None:
        """
        Ask the Grok LLM to isolate usage instructions from the README text.

        Parameters
        ----------
        text : str
            Raw page text harvested from the Hugging Face model page.

        Returns
        -------
        str | None
            Cleaned usage-focused excerpt, or ``None`` if no guidance is found
            or the LLM request fails.
        """
        if not text:
            return None

        prompt = f"""
        You are an AI assistant. Extract and return ONLY the sections
        that explain how to use the model, code examples, or instructions
        to get started. Ignore unrelated sections.

        Text:
        {text}

        Extract usage text verbatim.
        """
        try:
            logger.debug("Requesting usage extraction for text length %d",
                         len(text))
            response = self.grok.llm(prompt)
            return response.strip() if response else None
        except Exception:
            logger.info("Usage extraction failed via LLM", exc_info=True)
            return None

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Score how quickly a developer can ramp up on a Hugging Face model.

        Parameters
        ----------
        inputs : dict[str, Any]
            Must include the key ``"model_url"`` pointing at the model page.
        **kwargs : Any
            Present for interface compatibility; unused.

        Returns
        -------
        float
            Ramp-up score between 0.0 (hard to learn) and 1.0 (fast to learn).
        """
        url = inputs.get("model_url")
        if not url:
            raise ValueError("Missing required input: model_url")

        logger.info("Computing ramp-up score for %s", url)
        model_id = DatasetQuality._model_id_from_url(url)
        fetch_failed = False
        # Prefer the rendered page to capture tabs/collapsed sections; if
        # scraping fails, fall back to the raw README via the HF API.
        try:
            full_page_text = injectHFBrowser(url)
        except Exception:
            fetch_failed = True
            logger.info(
                "Rendered page fetch failed for %s",
                url,
                exc_info=True,
            )
            full_page_text = ""
            if model_id:
                try:
                    readme = self.client.request(
                        "GET",
                        f"/{model_id}/resolve/main/README.md",
                    )
                    if isinstance(readme, (bytes, bytearray)):
                        readme = readme.decode("utf-8", errors="ignore")
                    full_page_text = str(readme)
                    fetch_failed = False
                except Exception:
                    logger.info("README fallback failed for %s", model_id,
                                exc_info=True)
                    full_page_text = ""
        usage_text = self._extract_usage_section(full_page_text or "")

        usage_available = bool(usage_text and usage_text.strip())
        if usage_available and usage_text is not None:
            char_count = len(usage_text)
            logger.debug("Usage text length for %s: %d", url, char_count)
        else:
            lowered = (full_page_text or "").lower()
            keywords = (
                "usage",
                "how to use",
                "quick start",
                "example",
                "setup",
                "getting started",
                "pip install",
                "import",
                "load pretrained",
                "inference",
            )
            if any(kw in lowered for kw in keywords) or "```" in lowered:
                # If the LLM could not isolate usage guidance, fall back to a
                # simple keyword scan so we still reward basic documentation.
                usage_available = True
                logger.debug(
                    "Heuristic detected usage keywords for %s despite empty "
                    "extraction",
                    url,
                )
        if usage_available:
            usage_score = 1.0
        elif fetch_failed:
            usage_score = 0.5
        else:
            usage_score = 0.0

        # Sample the Grok score a few times because responses can be noisy;
        # keep the best outcome to stabilise the metric.
        llm_scores = [
            self._llm_ramp_rating(full_page_text)
            for _ in range(RAMP_UP_LLM_ATTEMPTS)
        ]
        llm_score = max(llm_scores) if llm_scores else 0.5

        score = (usage_score + llm_score) / 2.0

        logger.info(
            "Ramp-up score for %s: usage=%s llm_scores=%s combined=%.3f",
            url,
            usage_score,
            llm_scores,
            score,
        )
        return score

    def _llm_ramp_rating(self, page_text: str) -> float:
        """Ask the LLM to rate ramp-up difficulty from the model card."""

        if not page_text:
            logger.debug("Empty model card text; defaulting LLM ramp score")
            return 0

        # Restrict context to remain under typical LLM limits while keeping
        # enough substance for a meaningful judgement.
        context = shorten(page_text, width=4000, placeholder="...")
        prompt = (
            "You are evaluating how quickly a developer can start using a "
            "machine learning model based on its Hugging Face model card. "
            "Think step-by-step about clarity of setup steps, code examples, "
            "dependencies, and potential pitfalls. After your reasoning, "
            "write a line in the format 'FINAL SCORE: <number>' where the "
            "number is between 0 and 1 "
            "(1 = very fast ramp-up, 0 = very hard).\n\n"
            f"Model card snippet:\n{context}"
        )

        def call() -> Any:
            return self.grok.llm(prompt)

        def parse(raw: Any) -> float:
            value = _parse_numeric_response(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Ramp-up LLM score {value} outside [0, 1]")
            return value

        try:
            return _call_llm_with_retry(
                call,
                parse,
                description="Ramp-up LLM score",
            )
        except Exception:
            # Keep the metric resilient: log the failure and return a neutral
            # midpoint rather than erroring out the caller.
            sys.stderr.write("Ramp-up LLM scoring failed; defaulting to 0.5")
            sys.stderr.write(str(Exception))

            logger.info(
                "Ramp-up LLM scoring failed; defaulting to 0.5",
                exc_info=True,
            )
            return 0.5


class LicenseMetric(Metric):
    """Score license permissiveness while maintaining resilient fallbacks."""

    name = "License Permissiveness"
    key = "license_metric"

    license_scores: dict[str, float] = {
        # Permissive (1.0)
        "apache-2.0": 1.0,
        "mit": 1.0,
        "afl-3.0": 1.0,
        "bsd": 1.0,
        "bsd-2-clause": 1.0,
        "bsd-3-clause": 1.0,
        "bsd-3-clause-clear": 1.0,
        "isc": 1.0,
        "zlib": 1.0,
        "cdla-permissive-1.0": 1.0,
        "cdla-permissive-2.0": 1.0,
        "ms-pl": 1.0,
        "postgresql": 1.0,
        "osl-3.0": 1.0,
        "apple-ascl": 1.0,
        "mpl-2.0": 1.0,
        "pddl": 1.0,
        "unlicense": 1.0,
        "cc0-1.0": 1.0,
        "wtfpl": 1.0,
        "intel-research": 1.0,
        "ofl-1.1": 1.0,
        "lppl-1.3c": 1.0,
        "ncsa": 1.0,
        "etalab-2.0": 1.0,

        # Less permissive (0.75)
        "gpl-3.0": 0.75,
        "gpl-2.0": 0.75,
        "gpl": 0.75,
        "agpl-3.0": 0.75,
        "lgpl-3.0": 0.75,
        "lgpl-2.1": 0.75,
        "lgpl": 0.75,
        "lgpl-lr": 0.75,
        "epl-2.0": 0.75,
        "epl-1.0": 0.75,
        "ecl-2.0": 0.75,
        "eupl-1.1": 0.75,
        "eupl-1.2": 0.75,
        "artistic-2.0": 0.75,
        "cdla-sharing-1.0": 0.75,
        "cc-by-4.0": 0.75,
        "cc-by-3.0": 0.75,
        "cc-by-2.0": 0.75,
        "cc-by-2.5": 0.75,
        "cc-by-sa-4.0": 0.75,
        "cc-by-sa-3.0": 0.75,
        "odc-by": 0.75,
        "bsl-1.0": 0.75,
        "odbl": 0.75,
        "gfdl": 0.75,

        # Restrictive (0.5)
        "cc-by-nc-4.0": 0.5,
        "cc-by-nc-2.0": 0.5,
        "cc-by-nc-3.0": 0.5,
        "cc-by-nc-nd-4.0": 0.5,
        "cc-by-nc-nd-3.0": 0.5,
        "cc-by-nc-sa-4.0": 0.5,
        "cc-by-nc-sa-3.0": 0.5,
        "cc-by-nc-sa-2.0": 0.5,
        "cc-by-nd-4.0": 0.5,
        "fair-noncommercial-research-license": 0.5,

        # Model-specific / special licenses
        "openrail": 0.75,
        "creativeml-openrail-m": 0.75,
        "openrail++": 0.75,
        "gemma": 0.75,
        "llama2": 0.75,
        "llama3": 0.75,
        "llama3.1": 0.75,
        "llama3.2": 0.75,
        "llama3.3": 0.75,
        "llama4": 0.75,
        "bigscience-bloom-rail-1.0": 0.75,
        "bigscience-openrail-m": 0.75,
        "bigcode-openrail-m": 0.75,
        "open-mdw": 1.0,
        "h-research": 0.5,
        "c-uda": 0.5,
        "apple-amlr": 0.5,
        "deepfloyd-if-license": 0.5,
        "cc": 0.75,

        "other": 0.5,
    }

    license_aliases: dict[str, str] = {
        "apache license 2.0": "apache-2.0",
        "apache license version 2.0": "apache-2.0",
        "apache-2.0-only": "apache-2.0",
        "apache-2.0-or-later": "apache-2.0",
        "mit license": "mit",
        "mit-license": "mit",
        "bsd 3-clause license": "bsd-3-clause",
        "bsd-3 clause": "bsd-3-clause",
        "bsd 2-clause license": "bsd-2-clause",
        "gplv3": "gpl-3.0",
        "gplv2": "gpl-2.0",
        "lgplv3": "lgpl-3.0",
        "lgplv2.1": "lgpl-2.1",
        "agplv3": "agpl-3.0",
        "cc-by 4.0": "cc-by-4.0",
        "cc by 4.0": "cc-by-4.0",
        "cc by-nc 4.0": "cc-by-nc-4.0",
        "cc-by-nc": "cc-by-nc-4.0",
        "openrail-m": "openrail",
        "openrail-m license": "openrail",
        "openrail++ license": "openrail++",
        "creativeml-openrail": "creativeml-openrail-m",
        "bigscience openrail m": "bigscience-openrail-m",
        "bigscience bloom rail 1.0": "bigscience-bloom-rail-1.0",
        "llama 2": "llama2",
        "llama 3": "llama3",
        "gemma license": "gemma",
        "creative commons attribution 4.0": "cc-by-4.0",
        "creative commons attribution-noncommercial": "cc-by-nc-4.0",
    }

    license_file_candidates: tuple[str, ...] = (
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.md",
        "COPYING",
        "COPYING.txt",
        "COPYRIGHT",
    )

    license_text_hints: tuple[tuple[str, str], ...] = (
        ("apache license", "apache-2.0"),
        ("mit license", "mit"),
        ("gnu general public license", "gpl-3.0"),
        ("gnu lesser general public license", "lgpl-3.0"),
        ("gnu affero general public license", "agpl-3.0"),
        ("bsd 3-clause", "bsd-3-clause"),
        ("bsd 2-clause", "bsd-2-clause"),
        ("creative commons attribution 4.0", "cc-by-4.0"),
        ("creative commons attribution-noncommercial", "cc-by-nc-4.0"),
        ("creative commons attribution share alike", "cc-by-sa-4.0"),
        ("openrail++", "openrail++"),
        ("creativeml-openrail", "creativeml-openrail-m"),
        ("gemma license", "gemma"),
        ("llama 2", "llama2"),
        ("llama 3", "llama3"),
    )

    def __init__(self):
        self.hf_client = HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )
        self.grok_client = PurdueClient(
            max_requests=PURDUE_MAX_REQUESTS,
            window_seconds=PURDUE_WINDOW_SECONDS,
        )

    def compute(self, inputs: dict[str, Any], **kwargs) -> float:
        """
        Compute the license score from parsed inputs.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parsed inputs required by the metric. Must include a key called
            'model_url' with its corresponding correct link

        **kwargs : Any
            Optional per-metric tuning parameters.

        Returns
        -------
        float
            A score between 0.0 and 1.0.

        Raises
        ------
        RuntimeError
            If no valid HF model URL is found in the dict
        """
        # model_url must be in the dict
        if "model_url" not in inputs.keys():
            raise ValueError("Model link not found in input dictionary")

        # extract model_id from URL
        model_id = inputs['model_url'].split("https://huggingface.co/")[-1]
        logger.info("Computing license score for %s", model_id)

        # try to get license from HFClient and assign a score
        try:
            model_info = self.hf_client.request("GET",
                                                f"/api/models/{model_id}")
        except Exception:
            logger.info(
                "HF model lookup failed for %s",
                model_id,
                exc_info=True,
            )
            model_info = {}
        candidates = self._collect_license_candidates(model_info)
        # Walk the potential license declarations in priority order; the first
        # recognised slug wins so behaviour is deterministic.
        logger.debug("License candidates for %s: %s", model_id, candidates)
        for candidate in candidates:
            normalized = self._normalize_license_name(candidate)
            if normalized and normalized in self.license_scores:
                score = self.license_scores[normalized]
                logger.info("License score for %s: %.2f", model_id, score)
                return score

        detected = self._license_from_repo(model_id)
        if detected and detected in self.license_scores:
            score = self.license_scores[detected]
            logger.info("License score for %s: %.2f", model_id, score)
            return score

        logger.info(
            "License unknown for %s; returning neutral score",
            model_id,
        )
        return 0.5

    def _collect_license_candidates(self, payload: Any) -> list[Any]:
        if not isinstance(payload, Mapping):
            return []

        candidates: list[Any] = []
        # Card metadata often mirrors top-level fields but we inspect both to
        # maximise coverage across author formatting styles.
        card = payload.get("cardData")
        if isinstance(card, Mapping):
            for key in ("license", "license_name", "licenses"):
                value = card.get(key)
                if value:
                    candidates.append(value)

        for key in ("license", "license_name", "licenses"):
            value = payload.get(key)
            if value:
                candidates.append(value)

        return candidates

    def _normalize_license_name(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            for item in value:
                normalized = self._normalize_license_name(item)
                if normalized:
                    return normalized
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        text = text.replace(" ", "-").replace("_", "-")
        if text in self.license_scores:
            return text
        if text in self.license_aliases:
            return self.license_aliases[text]
        if text.endswith("-only") and text[:-5] in self.license_scores:
            return text[:-5]
        if text.endswith("-or-later") and text[:-9] in self.license_scores:
            return text[:-9]
        return None

    def _license_from_repo(self, model_id: str) -> Optional[str]:
        try:
            files = browse_hf_repo(
                self.hf_client,
                model_id,
                repo_type="model",
                revision="main",
                recursive=False,
            )
        except Exception:
            logger.debug(
                "Failed to list repo files for %s",
                model_id,
                exc_info=True,
            )
            return None

        for path, _size in files:
            filename = Path(path).name
            if filename in self.license_file_candidates:
                try:
                    content = self.hf_client.request(
                        "GET",
                        f"/{model_id}/resolve/main/{path}",
                    )
                except Exception:
                    continue
                text = (
                    content.decode("utf-8", errors="ignore")
                    if isinstance(content, (bytes, bytearray))
                    else str(content)
                )
                lower = text.lower()
                # Search for both explicit phrases and loose aliases so that
                # lightly modified license stubs still map to a known score.
                for phrase, slug in self.license_text_hints:
                    if phrase in lower:
                        return slug
                for pattern, slug in self.license_aliases.items():
                    if pattern in lower:
                        return slug
                for pattern in self.license_scores.keys():
                    if pattern in lower:
                        return pattern
        return None


class SizeMetric(Metric):
    """
    Metric that estimates model footprint in bits and reports how well the
    model fits on a set of representative devices.
    """
    name = "Model Size"
    key = "size_metric"
    device_profiles: dict[str, tuple[float, float]] = {
        # (memory in GB, relative throughput multiplier)
        "raspberry_pi": (4.0, 0.01),
        "jetson_nano": (4.0, 0.05),
        "desktop_pc": (32.0, 0.4),
        "aws_server": (128.0, 0.6),
    }
    device_capacity_bits: dict[str, int] = {
        name: int(memory_gb * 1024**3 * 8 * perf_multiplier)
        for name, (memory_gb, perf_multiplier) in device_profiles.items()
    }

    commonModelFileEndings = [
        ".bin",
        ".safetensors",
        ".h5",
        ".ckpt",
        ".onnx",
        ".tflite",
        ".pb",
        ".mlmodel",
        ".gguf",
        ".ggml",
        ".ggjt",
        ".pt",
    ]

    remote_file_candidates = (
        "pytorch_model.bin",
        "model.bin",
        "model.safetensors",
        "diffusion_pytorch_model.bin",
        "tf_model.h5",
    )

    def __init__(self):
        self.hf_client = HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )

    def extract_bits_from_saftensor(self,
                                    safeTensorDict: dict[str, int]) -> int:
        """
        Estimate the model footprint using safetensor metadata.

        Parameters
        ----------
        safeTensorDict : dict[str, int]
            Mapping from precision label (e.g., ``"float16"``) to the number
            of tensors stored at that precision. The precision text is
            expected to contain the bit-width as digits.

        Returns
        -------
        int
            Total number of bits implied by the smallest precision entry.
        """
        bits = []
        for precision in safeTensorDict.keys():
            # Keys look like "float16" or "bfloat16"; pull out the digits to
            # determine the bit-width represented by that entry.
            param_size = int(''.join(ch for ch in precision if ch.isdigit()))
            n_params = param_size * safeTensorDict[precision]
            bits.append(n_params)
        # Pick the smallest bit count so we do not overestimate the footprint
        # when multiple precisions are present.
        return min(bits)

    def compute(self, inputs: dict[str, Any], **kwargs: Any) \
            -> dict[str, float]:
        """
        Compute the metric score from parsed inputs.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parsed inputs required by the metric. Must include a key called
            'model_url' with its corresponding correct link

        **kwargs : Any
            Optional per-metric tuning parameters.

        Returns
        -------
        dict[str, float]
            Mapping of deployment target to a score between 0.0 and 1.0 that
            captures how well the model fits the device's capacity.

        Raises
        ------
        RuntimeError
            If no valid HF model URL is found in the dict
        """
        # model_url must be in the dict
        if "model_url" not in inputs.keys():
            raise ValueError("Model link not found in input dictionary")

        # Extract the model id and get model info using API
        model_id = inputs['model_url'].split("https://huggingface.co/")[-1]
        logger.info("Computing size score for %s", model_id)
        try:
            model_info = self.hf_client.request("GET",
                                                f"/api/models/{model_id}")
        except Exception:
            logger.info("HF model metadata unavailable for %s", model_id,
                        exc_info=True)
            model_info = {}

        bits = self._bits_from_hf_metadata(model_id, model_info)
        if bits is None:
            # Fall back to repo listings when metadata omits sizes.
            bits = self._bits_from_repo_listing(model_id)
        if bits is None:
            # Last resort: probe common filenames via HEAD requests.
            bits = self._bits_via_head_requests(inputs['model_url'])

        # Translate the bit-count into per-device deployability scores by
        # comparing the footprint against each device capacity (memory adjusted
        # by a throughput multiplier to reflect hardware speed) and clamping
        # the resulting ratio to the [0, 1] range.
        if bits is None or bits <= 0:
            logger.info("Size scores for %s falling back to heuristic bits",
                        model_id)
            avg_capacity = sum(self.device_capacity_bits.values()) / max(
                len(self.device_capacity_bits), 1)
            bits = int(avg_capacity)

        scores: dict[str, float] = {}
        for device, capacity_bits in self.device_capacity_bits.items():
            raw_score = capacity_bits / bits
            clamped = max(0.0, min(raw_score, 1.0))
            scores[device] = clamped
            logger.debug(
                "Size score for %s on %s: %.3f " +
                "(capacity_bits=%s, model_bits=%s)",
                model_id,
                device,
                clamped,
                capacity_bits,
                bits,
            )

        logger.info("Size scores for %s: %s", model_id, scores)
        return scores

    def _bits_from_hf_metadata(
        self,
        model_id: str,
        payload: Any,
    ) -> Optional[int]:
        if not isinstance(payload, Mapping):
            return None

        safetensors = payload.get("safetensors")
        if isinstance(safetensors, Mapping):
            params = safetensors.get("parameters")
            if isinstance(params, Mapping) and params:
                numeric_params: dict[str, int] = {}
                for name, raw_value in params.items():
                    if not isinstance(name, str):
                        continue
                    if isinstance(raw_value, (int, float)):
                        numeric_params[name] = int(raw_value)
                if numeric_params:
                    # Prefer precise safetensor metadata when available.
                    bits = self.extract_bits_from_saftensor(numeric_params)
                    logger.debug(
                        "Using safetensors metadata for %s -> %s bits",
                        model_id,
                        bits,
                    )
                    return bits

        siblings = payload.get("siblings")
        if isinstance(siblings, Iterable):
            sizes = []
            for entry in siblings:
                if not isinstance(entry, Mapping):
                    continue
                filename = entry.get("rfilename")
                if not isinstance(filename, str):
                    continue
                if any(
                    filename.endswith(ext)
                    for ext in SizeMetric.commonModelFileEndings
                ):
                    sizes.append(entry.get("size"))
            estimated_bits = self._bits_from_sizes(sizes)
            if estimated_bits is not None:
                # Summaries often include param files but omit topology;
                # averaging their sizes gives a reasonable footprint.
                logger.debug(
                    "Estimated bits for %s from siblings metadata: %s",
                    model_id,
                    estimated_bits,
                )
                return estimated_bits

        card = payload.get("cardData")
        if isinstance(card, Mapping):
            size_entry = (
                card.get("model_size")
                or card.get("model_size_in_bytes")
            )
            if isinstance(size_entry, (int, float)) and size_entry > 0:
                bits = int(float(size_entry) * 8)
                logger.debug(
                    "Using cardData size for %s -> %s bits",
                    model_id,
                    bits,
                )
                return bits
            if isinstance(size_entry, str):
                match = re.search(r"(\d+(?:\.\d+)?)\s*([kmgt]?b)",
                                  size_entry.lower())
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    multiplier = {
                        "b": 1,
                        "kb": 1024,
                        "mb": 1024**2,
                        "gb": 1024**3,
                        "tb": 1024**4,
                    }.get(unit, 1)
                    bits = int(value * multiplier * 8)
                    logger.debug(
                        "Parsed textual model size for %s -> %s bits",
                        model_id,
                        bits,
                    )
                    return bits
        return None

    def _bits_from_repo_listing(self, model_id: str) -> Optional[int]:
        try:
            files = browse_hf_repo(
                self.hf_client,
                model_id,
                repo_type="model",
                revision="main",
                recursive=True,
            )
        except Exception:
            logger.debug(
                "Failed to browse repo for model size %s",
                model_id,
                exc_info=True,
            )
            return None

        sizes = [
            size
            for path, size in files
            if isinstance(size, (int, float))
            and any(
                path.endswith(ext)
                for ext in SizeMetric.commonModelFileEndings
            )
        ]
        bits = self._bits_from_sizes(sizes)
        if bits:
            # Repo listings include raw byte sizes which we convert to bits to
            # stay consistent with other estimators.
            logger.debug(
                "Estimated bits for %s from %d repo files: %s",
                model_id,
                len(sizes),
                bits,
            )
        return bits

    def _bits_via_head_requests(self, model_url: str) -> Optional[int]:
        sizes: list[int] = []
        base = model_url.rstrip("/")
        for filename in self.remote_file_candidates:
            url = f"{base}/resolve/main/{filename}"
            try:
                response = requests.head(url, allow_redirects=True, timeout=6)
            except requests.RequestException:
                continue
            content_length = response.headers.get("Content-Length")
            if response.status_code == 200 and content_length:
                try:
                    sizes.append(int(content_length))
                except ValueError:
                    continue
        bits = self._bits_from_sizes(sizes)
        if bits:
            # HEAD requests give us approximate binary sizes without
            # downloading large artifacts.
            logger.debug(
                "Estimated bits for %s via HEAD requests: %s",
                model_url,
                bits,
            )
        return bits

    @staticmethod
    def _bits_from_sizes(sizes: Iterable[Any]) -> Optional[int]:
        # Average multiple sources to smooth out per-file variations before
        # converting to bits.
        bytes_values = [
            int(size)
            for size in sizes
            if isinstance(size, (int, float)) and size > 0
        ]
        if not bytes_values:
            return None
        avg_bytes = sum(bytes_values) / len(bytes_values)
        return int(avg_bytes * 8)


class AvailabilityMetric(Metric):
    """
    Metric that checks whether a model has an associated dataset and codebase
    available. Awards 0.5 for each item found via the model card.

    This metric retrieves the rendered model page text
    (via ``injectHFBrowser``)
    and uses a Grok LLM to identify mentions/links to an available dataset and
    an available code repository.
    """

    name = "Availability"
    key = "availability_metric"

    def __init__(self) -> None:
        self.grok = PurdueClient(
            max_requests=PURDUE_MAX_REQUESTS,
            window_seconds=PURDUE_WINDOW_SECONDS,
        )
        self.last_details: dict[str, Any] = {}

    def _llm_detect_availability(self, page_text: str) \
            -> tuple[bool, bool, str, str]:
        """
        Use the Grok LLM to determine whether the page text indicates a
        dataset and/or a codebase are available.

        Parameters
        ----------
        page_text : str
            Visible text of the Hugging Face model page.

        Returns
        -------
        tuple[bool, bool, str, str]
            (dataset_available, codebase_available,
            dataset_evidence, codebase_evidence).
        """
        logger.debug("Running availability LLM check with text length %d",
                     len(page_text or ""))
        if not page_text:
            return (False, False, "", "")

        prompt = f"""
        You will be given the visible text of a Hugging Face model page.
        Determine if BOTH of the following are PRESENT AND AVAILABLE to users:

        1) A dataset: a specific dataset link/name indicating training or
        evaluation data,
           or a clear pointer to a dataset page (e.g.,
           huggingface.co/datasets/...,
           Kaggle dataset, etc.).
           If URL, just the URL is sufficient for evidence.
        2) A codebase: a concrete link to source code repository
        (e.g., GitHub/GitLab/Bitbucket) or an
           installable package with a repository reference. If just a reference
           to a repository, but no link, not sufficient. If URL, just the
           URL is sufficient for evidence.

        Respond STRICTLY in compact JSON with four fields:
        {{"dataset_available": <true|false>, "codebase_available":
        <true|false>,
        "dataset_evidence": "<short snippet or URL>",
        "codebase_evidence": "<short snippet or URL>"}}

        Text:
        {page_text}
        """

        import json

        def call() -> Any:
            return self.grok.llm(prompt)

        def parse(raw: Any) -> tuple[bool, bool, str, str]:
            text = (raw or "").strip()
            if text.startswith("```"):
                text = text.strip('`')
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start:end + 1]
            obj = json.loads(text)
            dataset = bool(obj.get("dataset_available", False))
            codebase = bool(obj.get("codebase_available", False))
            dataset_ev = str(obj.get("dataset_evidence", ""))[:500]
            codebase_ev = str(obj.get("codebase_evidence", ""))[:500]
            logger.debug("LLM availability result dataset=%s code=%s",
                         dataset, codebase)
            return (dataset, codebase, dataset_ev, codebase_ev)

        try:
            return _call_llm_with_retry(
                call,
                parse,
                description="Availability JSON LLM",
            )
        except Exception:
            logger.info(
                "LLM availability parsing failed after retries; falling back",
                exc_info=True,
            )
            lower = page_text.lower()
            dataset_hits = any(
                kw in lower for kw in [
                    "huggingface.co/datasets/", " datasets/", "dataset:",
                    "trained on", "training data:", "evaluation dataset",
                    "kaggle.com/datasets", "dataset card"
                ]
            )
            codebase_hits = any(
                kw in lower for kw in [
                    "github.com/", "gitlab.com/", "bitbucket.org/",
                    "source code", "repository", "codebase"
                ]
            )
            # Heuristic evidence snippets
            dataset_ev = ""
            codebase_ev = ""
            if dataset_hits:
                for kw in [
                    "huggingface.co/datasets/", "kaggle.com/datasets",
                    "dataset card", "evaluation dataset"
                ]:
                    idx = lower.find(kw)
                    if idx != -1:
                        dataset_ev = page_text[max(0, idx-40): idx+120]
                        break
            if codebase_hits:
                for kw in [
                    "github.com/", "gitlab.com/", "bitbucket.org/",
                    "source code", "repository"
                ]:
                    idx = lower.find(kw)
                    if idx != -1:
                        codebase_ev = page_text[max(0, idx-40): idx+120]
                        break
            logger.debug("Heuristic availability result dataset=%s code=%s",
                         dataset_hits, codebase_hits)
            return (dataset_hits, codebase_hits, dataset_ev, codebase_ev)

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the availability score based on dataset/codebase presence.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parsed inputs required by the metric. Must include a key called
            'model_url' with its corresponding correct link

        **kwargs : Any
            Optional per-metric tuning parameters.

        Returns
        -------
        float
            A score between 0.0 and 1.0. Dataset availability contributes 0.5,
            and codebase availability contributes 0.5.

        Raises
        ------
        ValueError
            If 'model_url' is missing from inputs.
        """

        model_url = inputs.get("model_url")
        if not isinstance(model_url, str) or not model_url.strip():
            raise ValueError("Model link not found in input dictionary")

        logger.info("Computing availability score for %s", model_url)

        def _nonempty_url(v: Any) -> bool:
            return isinstance(v, str) and v.strip().startswith("http")

        explicit_dataset = inputs.get("dataset_url")
        explicit_git = inputs.get("git_url")

        has_dataset = _nonempty_url(explicit_dataset)
        has_code = _nonempty_url(explicit_git)

        dataset_ev = explicit_dataset if has_dataset else ""
        code_ev = explicit_git if has_code else ""

        if not (has_dataset and has_code):
            # When explicit links are absent, inspect the rendered card via
            # Grok to surface implicit references or hyperlinks.
            try:
                page_text = injectHFBrowser(model_url)
            except Exception:
                page_text = ""
            d_avail, c_avail, d_ev, c_ev = \
                self._llm_detect_availability(page_text)
            if d_avail and not has_dataset:
                has_dataset = True
                dataset_ev = d_ev
            if c_avail and not has_code:
                has_code = True
                code_ev = c_ev

        self.last_details = {
            "dataset_available": has_dataset,
            "codebase_available": has_code,
            "dataset_evidence": dataset_ev,
            "codebase_evidence": code_ev,
        }
        score = (0.5 if has_dataset else 0.0) + (0.5 if has_code else 0.0)
        logger.info("Availability score for %s: %.2f", model_url, score)
        return score


class PerformanceClaimsMetric(Metric):
    """
    Metric that inspects the model card/README to detect
    reported benchmarks and performance claims.
    """
    name = "Performance Claims"
    key = "performance_claims"
    performance_keywords: tuple[str, ...] = (
        "accuracy",
        "f1",
        "bleu",
        "rouge",
        "precision",
        "recall",
        "exact match",
        "perplexity",
        "auc",
        "wer",
        "benchmark",
        "evaluation",
    )

    def __init__(self):
        self.hf_client = HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )
        self.grok_client = PurdueClient(
            max_requests=PURDUE_MAX_REQUESTS,
            window_seconds=PURDUE_WINDOW_SECONDS,
        )

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the metric score from parsed inputs.
            Parsed inputs required by the metric. Must include a key called
            'model_url' with its corresponding correct link

        **kwargs : Any
            Optional per-metric tuning parameters.
            A score between 0.0 and 1.0.

        Raises
        ------
        RuntimeError
            If no valid HF model URL is found in the dict
        """
        # appropriate URL must be in the dict
        if "model_url" not in inputs.keys():
            raise ValueError("No good link found in input dictionary")

        model_url = inputs["model_url"]
        model_id = model_url.split("https://huggingface.co/")[-1]
        logger.info("Computing performance claims score for %s", model_id)

        try:
            logger.debug("Fetching README via API for %s", model_id)
            readme = self.hf_client.request(
                "GET",
                f"/{model_id}/resolve/main/README.md",
            )
            if isinstance(readme, (bytes, bytearray)):
                readme = readme.decode("utf-8", errors="ignore")
            card_data = str(readme).splitlines()
            source = "api"
        except Exception:
            logger.info("Falling back to rendered page for %s", model_id)
            try:
                rendered = injectHFBrowser(model_url)
                card_data = rendered.splitlines()
            except Exception:
                logger.info(
                    "Rendered page fetch failed for %s",
                    model_id,
                    exc_info=True,
                )
                card_data = []
            source = "rendered_page"
        logger.debug(
            "Performance claims text source=%s lines=%d for %s",
            source,
            len(card_data),
            model_id,
        )

        if not card_data:
            logger.info(
                "Empty performance evidence for %s; returning zero score",
                model_id,
            )
            return 0.0

        # Quick keyword/number scan catches obvious tables or metric call-outs
        # without paying the LLM cost.
        heuristic_hit = self._detect_benchmark_claims(card_data)
        if heuristic_hit:
            logger.info(
                "Performance claims detected heuristically for %s",
                model_id,
            )
            return 1.0

        # Prompt the LLM to give score
        prompt = (
            "You will assess whether the provided README snippets include "
            "performance claims backed by benchmark results. Think through "
            "the evidence step-by-step. After your reasoning, output a line "
            "'FINAL SCORE: <value>' where the value is 1.0 if benchmark-"
            "backed "
            "claims exist, otherwise 0.0. Snippets:\n"
            f"{''.join(card_data)}"
        )
        logger.info(
            "Querying LLM for performance claims evaluation on %s",
            model_id,
        )

        def call() -> Any:
            return self.grok_client.llm(prompt)

        def parse(raw: Any) -> float:
            return _parse_numeric_response(raw, allowed=(0.0, 1.0))

        try:
            score = _call_llm_with_retry(
                call,
                parse,
                description="Performance claims LLM",
            )
        except Exception:
            # Treat LLM failures as neutral and rely on the heuristic outcome
            # so the metric remains deterministic.
            logger.info(
                "Performance claims LLM failed after retries",
                exc_info=True,
            )
            score = 1.0 if heuristic_hit else 0.0

        logger.info("Performance claims score for %s: %.1f", model_id, score)

        return score

    def _detect_benchmark_claims(self, lines: Sequence[str]) -> bool:
        window: list[str] = []
        for line in lines:
            lower = line.lower()
            window.append(lower)
            if len(window) > 3:
                window.pop(0)

            if any(keyword in lower for keyword in self.performance_keywords):
                if re.search(r"\d+(?:\.\d+)?\s*%", lower):
                    return True
                if re.search(
                    r"\d+(?:\.\d+)?\s*(?:f1|bleu|rouge|acc|auc|wer)",
                    lower,
                ):
                    return True
            if (
                '|' in line
                and re.search(r"\d", lower)
                and any(
                    kw in lower
                    for kw in ("acc", "f1", "bleu", "score", "precision")
                )
            ):
                return True
            if any("benchmark" in text for text in window) and re.search(
                r"\d+(?:\.\d+)?\s*%",
                " ".join(window),
            ):
                return True
        return False


class DatasetQuality(Metric):
    """
    Evaluate dataset quality by combining reuse and community engagement.

    The metric inspects Hugging Face metadata to determine how often a
    dataset is reused across models and how many likes it has accrued,
    yielding a bounded score that favors broad adoption.
    """

    name = "Dataset Quality"
    key = "dataset_quality"

    def __init__(self, hf_client: Optional[HFClient] = None) -> None:
        self.hf_client = hf_client or HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )
        self.last_details: dict[str, Any] = {}

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Score a dataset by blending reuse counts with community likes.

        Parameters
        ----------
        inputs : dict[str, Any]

            Parser output that may contain dataset and/or model URLs.
        **kwargs : Any
            Unused placeholder for interface compatibility.

        Returns
        -------
        float
            Weighted dataset quality score clamped to ``[0.0, 1.0]``.
        """

        # Step 1: resolve which dataset we should inspect
        dataset_id = self._extract_dataset_id(inputs)
        if not dataset_id:
            self.last_details = {"reason": "dataset_not_found"}
            logger.info("Dataset quality: no dataset found")
            return 0.0

        logger.info("Computing dataset quality for %s", dataset_id)

        # Step 2: pull the dataset metadata from Hugging Face
        payload = self._fetch_dataset(dataset_id)
        likes = self._safe_int(payload.get("likes")) if payload else None
        try:
            use_count = self.count_models_for_dataset(dataset_id)
        except Exception:
            logger.info(
                "Dataset quality: failed to count models for %s",
                dataset_id,
                exc_info=True,
            )
            use_count = 0
        logger.debug(
            "Dataset %s likes=%s use_count=%d",
            dataset_id,
            likes,
            use_count,
        )

        # Step 3: convert engagement numbers into bounded scores
        # Likes skew higher than reuse counts, so we weight reuse more heavily
        # to reward datasets that underpin many downstream models.
        likes_score = (
            self._squash_score(likes, scale=250)
            if likes is not None
            else 0.0
        )
        use_score = self._squash_score(use_count, scale=40)

        score = (0.6 * use_score) + (0.4 * likes_score)
        score = max(0.0, min(1.0, score))

        self.last_details = {
            "dataset_id": dataset_id,
            "likes": likes,
            "use_count": use_count,
            "likes_score": likes_score,
            "use_score": use_score,
        }
        if payload is None:
            self.last_details["reason"] = "hf_api_error"
        logger.info("Dataset quality score for %s: %.2f", dataset_id, score)
        return score

    def _extract_dataset_id(self, inputs: Mapping[str, Any]) -> Optional[str]:
        """
        Resolve the primary dataset slug from parser output or model metadata.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Parsed artifacts describing the model and any referenced datasets.

        Returns
        -------
        Optional[str]
            Normalized ``owner/name`` dataset slug, or ``None`` when absent.
        """
        # Prefer explicit dataset URLs that the parser already categorized.
        dataset_url = inputs.get("dataset_url")
        if isinstance(dataset_url, str):
            slug = self._dataset_slug_from_url(dataset_url)
            if slug:
                return slug

        # Fall back to inspecting the referenced model's metadata.
        model_url = inputs.get("model_url")
        if not isinstance(model_url, str):
            return None

        model_id = self._model_id_from_url(model_url)
        if not model_id:
            return None

        model_info = self._fetch_model(model_id)
        if not isinstance(model_info, Mapping):
            return None

        candidates: list[Any] = []
        # Older model cards advertise datasets under these top-level keys.
        candidates.extend([model_info.get("dataset"),
                           model_info.get("datasets")])
        card = model_info.get("cardData")
        if isinstance(card, Mapping):
            # Newer model cards store richer data inside ``cardData``.
            candidates.extend([card.get("dataset"), card.get("datasets")])
        for candidate in candidates:
            slug = self._first_dataset_slug(candidate)
            if slug:
                return slug

        tags = model_info.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("datasets:"):
                    # Tags occasionally encode dataset information,
                    # e.g. "datasets:mnist".
                    ref = tag.split(":", 1)[1]
                    slug = self._normalize_dataset_reference(ref)
                    if slug:
                        return slug

        return None

    def _fetch_dataset(self, dataset_id: str) -> Optional[Mapping[str, Any]]:
        """
        Fetch dataset metadata for ``dataset_id`` via the Hugging Face API.

        Parameters
        ----------
        dataset_id : str
            Normalized dataset slug (``owner/name``) to request from the API.

        Returns
        -------
        Optional[Mapping[str, Any]]
            Parsed dataset payload, or ``None`` if the request fails.
        """
        try:
            logger.debug("Fetching dataset metadata for %s", dataset_id)
            data = self.hf_client.request(
                "GET",
                f"/api/datasets/{quote(dataset_id, safe='/@.-')}",
            )
        except Exception:
            logger.info("Failed to fetch dataset %s",
                        dataset_id,
                        exc_info=True)
            return None
        return data if isinstance(data, Mapping) else None

    def _fetch_model(self, model_id: str) -> Optional[Mapping[str, Any]]:
        """
        Fetch model metadata so we can inspect the declared datasets.

        Parameters
        ----------
        model_id : str
            Normalized model slug (``owner/name``) to request from the API.

        Returns
        -------
        Optional[Mapping[str, Any]]
            Parsed model payload, or ``None`` when the request fails.
        """
        try:
            logger.debug("Fetching model metadata for %s", model_id)
            data = self.hf_client.request(
                "GET",
                f"/api/models/{quote(model_id, safe='/@.-')}",
            )
        except Exception:
            logger.info("Failed to fetch model %s", model_id, exc_info=True)
            return None
        return data if isinstance(data, Mapping) else None

    @staticmethod
    def _safe_int(value: Any) -> int:
        """
        Convert ``value`` to a non-negative integer, defaulting to ``0``.

        Parameters
        ----------
        value : Any
            Raw value retrieved from Hugging Face metadata.

        Returns
        -------
        int
            Parsed integer or ``0`` if conversion fails.
        """
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _squash_score(value: int, *, scale: int) -> float:
        """
        Compress counts into the ``[0.0, 1.0]`` interval with log roll-off.

        Parameters
        ----------
        value : int
            Raw count to transform.
        scale : int
            Reference value that maps to a score near ``1.0``.

        Returns
        -------
        float
            Log-scaled score bounded to ``[0.0, 1.0]``.
        """
        if value <= 0 or scale <= 0:
            return 0.0
        return min(1.0, math.log1p(value) / math.log1p(scale))

    def _first_dataset_slug(self, value: Any) -> Optional[str]:
        """
        Return the first normalized dataset slug contained in ``value``.

        Parameters
        ----------
        value : Any
            String or list originating from model metadata.

        Returns
        -------
        Optional[str]
            Normalized ``owner/name`` slug, or ``None`` if none are found.
        """
        if isinstance(value, str):
            return self._normalize_dataset_reference(value)
        if isinstance(value, list):
            for item in value:
                slug = self._normalize_dataset_reference(item)
                if slug:
                    return slug
        return None

    def _normalize_dataset_reference(self, reference: Any) -> Optional[str]:
        """
        Normalize free-form dataset references into ``owner/name`` format.

        Parameters
        ----------
        reference : Any
            Arbitrary dataset hint collected from metadata or tags.

        Returns
        -------
        Optional[str]
            Normalized dataset slug suitable for API requests.
        """
        if not isinstance(reference, str):
            return None
        text = reference.strip()
        if not text:
            return None
        if text.startswith("datasets:"):
            text = text.split(":", 1)[1]
        if text.startswith("http://") or text.startswith("https://"):
            return self._dataset_slug_from_url(text)
        if text.startswith("huggingface.co/"):
            return self._dataset_slug_from_url(f"https://{text}")
        if "/" in text:
            parts = [p for p in text.split("/") if p]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            if parts:
                return parts[0]
        return None

    @staticmethod
    def _dataset_slug_from_url(url: str) -> Optional[str]:
        """
        Extract ``owner/name`` slug from a Hugging Face dataset URL.

        Parameters
        ----------
        url : str
            Dataset URL copied from the Hub.

        Returns
        -------
        Optional[str]
            Normalized slug when extraction succeeds, else ``None``.
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return None
        if path.startswith("datasets/"):
            path = path[len("datasets/"):]
        segments = [
            seg for seg in path.split("/")
            if seg not in {"blob", "tree", "resolve", "main", "raw", "viewer"}
        ]
        if not segments:
            return None
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
        return segments[0]

    @staticmethod
    def _model_id_from_url(url: str) -> Optional[str]:
        """
        Extract ``owner/name`` slug from a Hugging Face model URL.

        Parameters
        ----------
        url : str
            Model URL copied from the Hub.

        Returns
        -------
        Optional[str]
            Normalized slug when extraction succeeds, else ``None``.
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path or path.startswith("datasets/"):
            return None
        parts = [part for part in path.split("/") if part]
        if not parts:
            return None
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]

    def count_models_for_dataset(self, dataset_id: str,
                                 limit: int = 1000) -> int:
        """
        Count models on the Hub that self-report using ``dataset_id``.

        Parameters
        ----------
        dataset_id : str
            Normalized dataset slug to pass through the Hub filters.
        limit : int, optional
            Maximum number of results to request per API call,
            by default ``1000``.

        Returns
        -------
        int
            Unique model count associated with the dataset.
        """
        slug = dataset_id.split("/")[-1]

        # Common ways authors tag datasets in model cards
        filters: Iterable[str] = {
            slug,                          # e.g. "imagenet-1k"
            f"dataset:{slug}",             # e.g. "dataset:imagenet-1k"
            dataset_id,                    # e.g. "ILSVRC/imagenet-1k"
            f"dataset:{dataset_id}",       # e.g. "dataset:ILSVRC/imagenet-1k"
        }

        seen: set[str] = set()
        for f in filters:
            try:
                models: list[dict[str, Any]] = self.hf_client.request(
                    "GET",
                    "/api/models",
                    params={"filter": f, "limit": limit}
                )
            except Exception:
                continue
            if not isinstance(models, list):
                continue
            for m in models:
                mid = m.get("modelId")
                if mid:
                    # Track models uniquely to avoid
                    # double-counting across filters.
                    seen.add(mid)

        count = len(seen)
        logger.debug("Dataset %s associated with %d model(s)",
                     dataset_id,
                     count)
        return count


class CodeQuality(Metric):
    """
    Assess codebases by combining lint heuristics, typing coverage,
    and LLM judgment.

    The metric fetches Python sources linked from a model card or explicit
    repository URL, runs lightweight static checks, and asks an LLM to
    estimate engineering quality. When no code is available, it falls back
    to interpreting the model card alone.
    """

    name = "Code Quality"
    key = "code_quality"

    def __init__(
        self,
        hf_client: Optional[HFClient] = None,
        grok_client: Optional[PurdueClient] = None,
    ) -> None:
        self.hf_client = hf_client or HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )
        self.grok = grok_client or PurdueClient(
            max_requests=PURDUE_MAX_REQUESTS,
            window_seconds=PURDUE_WINDOW_SECONDS,
        )
        self.last_details: dict[str, Any] = {}

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Resolve source code, analyse it, and optionally fall back to the
        model card.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parser output that may include ``git_url`` or ``model_url``
            pointing to the codebase or model card.
        **kwargs : Any
            Placeholder for interface compatibility; unused.

        Returns
        -------
        float
            Aggregate quality score bounded to ``[0.0, 1.0]``.
        """

        logger.info("Computing code quality")
        code_files, origin, card_text = self._load_code(inputs)
        logger.debug("Code load origin=%s file_count=%d",
                     origin, len(code_files))
        if code_files:
            lint_score = self._lint_score(code_files)
            typing_score = self._typing_score(code_files)
            # LLM assessments can vary, so sample a few times and keep the
            # strongest showing to approximate a human reviewer.
            llm_scores = [
                self._llm_code_rating(code_files)
                for _ in range(CODE_QUALITY_LLM_ATTEMPTS)
            ]
            llm_score = max(llm_scores)
            # Heuristic scores are currently advisory; only the LLM output
            # feeds the final score, but we keep the numbers for telemetry.
            score = (0.05*lint_score + 0.05*typing_score + 0.9*llm_score)
            score = max(0.0, min(1.0, score))

            self.last_details = {
                "origin": origin,
                "lint_score": lint_score,
                "typing_score": typing_score,
                "llm_score": llm_score,
                "file_count": len(code_files),
            }
            logger.info("Code quality score from codebase: %.2f", score)
            logger.info("Lint score: %.2f", lint_score)
            logger.info("Typing score: %.2f", typing_score)
            logger.info("LLM score: %.2f", llm_score)
            return score

        model_url = inputs.get("model_url")
        if not card_text and isinstance(model_url, str):
            card_text = self._model_card_text(model_url)
        # No code: fall back to evaluating the model card itself for quality
        # signals so the metric still returns a bounded value.
        fallback_score = self._llm_card_rating(card_text)
        self.last_details = {
            "origin": "model_card",
            "llm_score": fallback_score,
            "card_available": bool(card_text),
        }
        logger.info("Code quality fallback score: %.2f", fallback_score)
        return fallback_score

    # ------------------------------------------------------------------
    # Source resolution helpers
    # ------------------------------------------------------------------
    def _load_code(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[dict[str, str], str, str]:
        """
        Locate Python sources and return them alongside provenance
        metadata.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Parser artefacts containing potential ``git_url`` or
            ``model_url`` keys.

        Returns
        -------
        tuple[dict[str, str], str, str]
            Mapping of relative paths to file contents, the origin label,
            and any cached model card text for reuse when code is
            unavailable.
        """

        card_text = ""
        git_url = inputs.get("git_url")
        if isinstance(git_url, str):
            logger.debug("Attempting to load code from explicit git_url %s",
                         git_url)
            # Honour explicit user-supplied repositories before consulting the
            # model card for GitHub hints.
            files = self._load_from_github(git_url)
            if files:
                logger.debug("Loaded %d file(s) from %s", len(files), git_url)
                return files, "github", card_text

        model_url = inputs.get("model_url")
        if isinstance(model_url, str):
            logger.debug("Inspecting model card for %s", model_url)
            card_text = self._model_card_text(model_url)
            gh_url = self._github_from_card(card_text)
            if gh_url:
                logger.debug("Found GitHub URL %s via model card", gh_url)
                files = self._load_from_github(gh_url)
                if files:
                    logger.debug("Loaded %d file(s) from %s",
                                 len(files),
                                 gh_url)
                    return files, "github_from_card", card_text

        return {}, "", card_text

    def _load_from_github(
        self,
        url: str,
        *,
        limit: int = 20,
    ) -> dict[str, str]:
        """
        Clone a GitHub repository and read a bounded set of Python files.

        Parameters
        ----------
        url : str
            HTTPS URL to the GitHub repository.
        limit : int, optional
            Maximum number of Python files to ingest, defaults to ``20``.

        Returns
        -------
        dict[str, str]
            Mapping of relative file paths to their source text.
        """

        logger.debug("Loading code from GitHub repo %s", url)
        with tempfile.TemporaryDirectory(prefix="code-metric-") as tmpdir:
            dest = Path(tmpdir) / "repo"
            if not self._clone_repo(url, dest):
                logger.info("Failed to clone %s", url)
                return {}
            files = self._read_python_files(dest, limit=limit)
            logger.debug("Read %d Python file(s) from %s", len(files), url)
            return files

    def _clone_repo(self, url: str, dest: Path) -> bool:
        """
        Clone ``url`` into ``dest`` and signal success.

        Parameters
        ----------
        url : str
            Git repository URL to clone.
        dest : Path
            Filesystem path where the shallow clone should be created.

        Returns
        -------
        bool
            ``True`` when the clone completes, otherwise ``False``.
        """

        try:
            logger.debug("Cloning repo %s", url)
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=45,
            )
        except (subprocess.SubprocessError, OSError):
            logger.info("Clone failed for %s", url, exc_info=True)
            return False
        logger.debug("Clone succeeded for %s", url)
        return True

    def _read_python_files(self, root: Path, *, limit: int) -> dict[str, str]:
        """
        Collect up to ``limit`` tracked Python files from ``root``.

        Parameters
        ----------
        root : Path
            Directory containing the cloned repository.
        limit : int
            Maximum number of files to return.

        Returns
        -------
        dict[str, str]
            Mapping of relative file paths to their corresponding source text.
        """

        candidates: list[str] = []
        try:
            git_client = GitClient(
                max_requests=GIT_MAX_REQUESTS,
                repo_path=str(root),
                window_seconds=GIT_WINDOW_SECONDS,
            )
            candidates = [
                path
                for path in git_client.list_files()
                if path.endswith(".py")
            ]
        except Exception:
            logger.debug("git ls-files failed in %s", root, exc_info=True)
            candidates = []

        if not candidates:
            candidates = [
                str(path.relative_to(root))
                for path in sorted(root.rglob("*.py"))
            ]
        logger.debug("Found %d Python candidate(s) in %s",
                     len(candidates), root)

        results: dict[str, str] = {}
        for rel_path in candidates:
            if len(results) >= limit:
                break
            path = root / rel_path
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if text.strip():
                results[rel_path] = text
        logger.debug("Returning %d Python file(s) from %s",
                     len(results), root)
        return results

    def _github_from_card(self, card_text: str) -> Optional[str]:
        """
        Extract a GitHub repository link from rendered model card text.

        Parameters
        ----------
        card_text : str
            Full model card contents sourced from Hugging Face.

        Returns
        -------
        Optional[str]
            First matching GitHub URL if present, else ``None``.
        """

        if not card_text:
            logger.debug("No card text provided for GitHub extraction")
            return None
        match = re.search(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            card_text,
        )
        if match:
            url = match.group(0)
            logger.debug("Extracted GitHub URL %s from card", url)
            return url
        logger.debug("No GitHub URL found in card text")
        return None

    # ------------------------------------------------------------------
    # Static analysis helpers
    # ------------------------------------------------------------------
    def _lint_score(self, code_files: Mapping[str, str]) -> float:
        """
        Approximate lint quality using lightweight style heuristics.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.

        Returns
        -------
        float
            Heuristic lint compliance score within ``[0.0, 1.0]``.
        """

        logger.debug("Computing lint score for %d file(s)", len(code_files))
        total = 0
        issues = 0

        for text in code_files.values():
            for raw_line in text.splitlines():
                stripped = raw_line.rstrip("\n")
                if not stripped.strip():
                    continue

                total += 1
                failure = False

                # Penalise common readability issues; keep checks cheap so we
                # can run them across many repos without shelling out to tools.
                if len(stripped) > 100:
                    failure = True
                if stripped.rstrip() != stripped:
                    failure = True
                if "\t" in stripped:
                    failure = True

                leading_spaces = len(stripped) - len(stripped.lstrip(" "))
                if leading_spaces and leading_spaces % 4 != 0:
                    failure = True

                if failure:
                    issues += 1

        if total == 0:
            logger.debug("No lines seen for lint score; defaulting to 0.5")
            return 0.5
        compliant_ratio = 1.0 - (issues / total)
        # Anything below 90% compliant is treated as a failure, and
        # 100% compliant receives full credit. Linearly scale in-between.
        score = max(0.0, min(1.0, (compliant_ratio - 0.9) / 0.1))
        logger.debug("Lint compliance ratio=%.3f score=%.3f",
                     compliant_ratio, score)
        return score

    def _typing_score(self, code_files: Mapping[str, str]) -> float:
        """
        Measure the proportion of functions that provide complete type hints.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.

        Returns
        -------
        float
            Fraction of typed functions; defaults to ``0.5`` when none found.
        """

        logger.debug("Computing typing score for %d file(s)", len(code_files))
        total_funcs = 0
        typed_funcs = 0

        for text in code_files.values():
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_funcs += 1
                    # Require full annotations rather than partial hints so
                    # the score reflects comprehensive typing coverage.
                    if self._function_is_typed(node):
                        typed_funcs += 1

        if total_funcs == 0:
            logger.debug("No functions found; typing score defaults to 0.5")
            return 0.5
        score = typed_funcs / total_funcs
        logger.debug("Typing score %.3f (%d/%d)",
                     score, typed_funcs, total_funcs)
        return score

    def _function_is_typed(self, node: ast.AST) -> bool:
        """
        Determine whether a function annotates all parameters and the
        return value.

        Parameters
        ----------
        node : ast.AST
            Function definition node extracted from the AST.

        Returns
        -------
        bool
            ``True`` when every parameter and the return value are
            annotated.
        """

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False

        params = list(node.args.posonlyargs) + list(node.args.args)
        if params and params[0].arg in {"self", "cls"}:
            params = params[1:]

        params += list(node.args.kwonlyargs)

        for param in params:
            if param.annotation is None:
                return False

        if node.args.vararg and node.args.vararg.annotation is None:
            return False
        if node.args.kwarg and node.args.kwarg.annotation is None:
            return False

        return node.returns is not None

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _llm_code_rating(self, code_files: Mapping[str, str]) -> float:
        """
        Ask the Grok LLM to assess engineering quality of the provided code.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.

        Returns
        -------
        float
            Parsed LLM rating normalized to ``[0.0, 1.0]``; defaults to
            ``0.5`` when the snippet is empty or the request fails.
        """

        snippet = self._code_snippet(code_files)
        if not snippet:
            logger.debug("No snippet available for LLM code rating")
            return 0.5

        prompt = (
            "Rate the following Python code's engineering quality on a "
            "scale from 0 to 1. 0 should be considered extremely poor, "
            "0.5 should mean okay, 0.75 solid, 0.9 great, and 1.0 amazing. "
            "Consider readability, structure, tests, and maintainability. "
            "Think step-by-step, then output a line 'FINAL SCORE: <number>' "
            "with the rating between 0 and 1.\n\n"
            f"```python\n{snippet}\n```"
        )

        def call() -> Any:
            return self.grok.llm(prompt)

        try:
            score = _call_llm_with_retry(
                call,
                self._parse_llm_score_strict,
                description="Code quality LLM rating",
            )
            logger.debug("LLM code rating %.3f", score)
            return score
        except Exception:
            logger.info("LLM code rating failed", exc_info=True)
            return 0.5

    def _llm_card_rating(self, card_text: str) -> float:
        """
        Generate a fallback LLM-based quality score from model card text.

        Parameters
        ----------
        card_text : str
            Rendered model card contents.

        Returns
        -------
        float
            Parsed LLM rating in ``[0.0, 1.0]``; defaults to ``0.3`` on
            failure.
        """

        if not card_text:
            logger.debug("No card text; defaulting LLM card rating to 0.3")
            return 0.3

        prompt = (
            "Based on this Hugging Face model card, estimate the quality "
            "of the associated codebase. Reflect on documentation clarity, "
            "testing evidence, and engineering maturity. After your analysis, "
            "write 'FINAL SCORE: <number>' where the number between 0 and 1 "
            "summarizes your judgement.\n\n"
            f"{shorten(card_text, width=3500, placeholder='...')}"
        )

        def call() -> Any:
            return self.grok.llm(prompt)

        try:
            score = _call_llm_with_retry(
                call,
                self._parse_llm_score_strict,
                description="Model card LLM rating",
            )
            logger.debug("LLM card rating %.3f", score)
            return score
        except Exception:
            logger.info("LLM card rating failed", exc_info=True)
            return 0.3

    def _code_snippet(
        self,
        code_files: Mapping[str, str],
        *,
        limit: int = 3500,
    ) -> str:
        """
        Concatenate a bounded sample of code to feed into the LLM prompt.

        Parameters
        ----------
        code_files : Mapping[str, str]
            Mapping of file paths to Python source text.
        limit : int, optional
            Maximum number of characters to include, defaults to ``3500``.

        Returns
        -------
        str
            Truncated multi-file snippet formatted for LLM consumption.
        """

        pieces: list[str] = []
        remaining = limit

        for path, text in code_files.items():
            header = f"# File: {path}\n"
            budget = max(0, remaining - len(header))
            if budget <= 0:
                break
            body = text.strip()
            snippet = body[:budget]
            pieces.append(header + snippet)
            remaining -= len(header) + len(snippet)
            if remaining <= 0:
                break

        snippet = "\n\n".join(pieces)
        logger.debug("Constructed code snippet of length %d", len(snippet))
        return snippet

    def _parse_llm_score_strict(self, raw: Any) -> float:
        """Strictly parse an LLM score ensuring it lies within ``[0, 1]``."""

        value = _parse_numeric_response(raw)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"LLM score {value} is outside [0, 1]")
        return value

    def _parse_llm_score(self, raw: Any) -> float:
        """
        Backwards-compatible wrapper returning ``0.5`` on parsing failure.
        """

        try:
            score = self._parse_llm_score_strict(raw)
        except Exception:
            logger.debug("LLM score parsing fallback to 0.5", exc_info=True)
            return 0.5
        logger.debug("Parsed LLM score %.3f from response", score)
        return score

    def _model_card_text(self, url: Optional[str]) -> str:
        """
        Retrieve model card text via the Hugging Face API or browser helper.

        Parameters
        ----------
        url : Optional[str]
            Hugging Face model URL from which to fetch the card.

        Returns
        -------
        str
            Markdown or HTML card contents, or an empty string when
            unavailable.
        """

        if not url:
            logger.debug("No URL provided for model card fetch")
            return ""
        model_id = DatasetQuality._model_id_from_url(url)
        if model_id:
            try:
                logger.debug("Fetching README for %s", model_id)
                data = self.hf_client.request(
                    "GET",
                    f"/{model_id}/resolve/main/README.md",
                )
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="ignore")
                if isinstance(data, str) and data.strip():
                    logger.debug("Retrieved README (%d chars) for %s",
                                 len(data), model_id)
                    return data
            except Exception:
                logger.info("Failed to fetch README for %s", model_id,
                            exc_info=True)
        try:
            logger.debug("Falling back to browser fetch for %s", url)
            text = injectHFBrowser(url)
            logger.debug("Browser fetch returned %d chars", len(text))
            return text
        except Exception:
            logger.info("Browser fetch failed for %s", url, exc_info=True)
            return ""


class BusFactorMetric(Metric):
    """
    Bus factor proxy that prefers Hub data, then GitHub, then Grok, then a
    constant fallback.

    Decision paths (first match wins):
      1) direct.commit_count_by_author -> inverse-HHI -> score
      2) direct.commit_authors         -> unique authors -> score
      3) direct.recent_unique_committers -> score
      4) direct.unique_committers        -> score
      5) hf.success  (HF commits -> inverse-HHI -> score)
      6) github.success (git_url/github_url -> counts -> score)
      7) grok.estimate
      8) constant.fallback (eff = 2.0)

    Score normalization:
        score = clamp(eff / target_maintainers, 0.0, 1.0)

    Notes:
    - GitHub path clones the repo shallowly and uses `git shortlog -sne` to
      count unique authors weighted by commit counts.
    - HF path pages commits; gated/401/403/404 or empty marks HF unavailable.
    """

    name = "Bus Factor"
    key = "bus_factor"

    def __init__(
        self,
        hf_client: Optional[HFClient] = None,
        grok_client: Optional[PurdueClient] = None,
    ) -> None:
        self.hf_client = hf_client or HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )
        self.grok = grok_client or PurdueClient(
            max_requests=PURDUE_MAX_REQUESTS,
            window_seconds=PURDUE_WINDOW_SECONDS,
        )

    # ---------------- helpers ----------------

    def _fetch_hf_commits(
            self, client, model_id, branch="main", max_pages=100):
        """Fetch all commit pages for a model repo."""
        logger.debug("Fetching Hugging Face commits for %s:%s",
                     model_id, branch)
        all_commits = []
        page = 0
        while True:
            commits = client.request(
                "GET", f"/api/models/{model_id}/commits/{branch}?p={page}")
            if not commits:
                logger.debug("No commits returned for %s:%s on page %d",
                             model_id, branch, page)
                break
            all_commits.extend(commits)
            logger.debug("Fetched %d commits for %s:%s on page %d",
                         len(commits), model_id, branch, page)
            if len(commits) < 50:
                break
            page += 1
            if page >= max_pages:
                break
        logger.debug("Total commits fetched for %s:%s -> %d",
                     model_id, branch, len(all_commits))
        return all_commits

    def _process_commits(
            self,
            commit_data: list
            ) -> tuple[dict[str, int], list[str], int, int]:
        """Process commit data to extract author information"""
        commit_count_by_author: dict[str, int] = {}
        commit_authors: list[str] = []

        for commit in commit_data:
            authors = commit.get("authors", [])
            if authors:
                for author_info in authors:
                    user = author_info.get("user", "unknown")
                    commit_authors.append(user)
                    commit_count_by_author[user] = (
                        commit_count_by_author.get(user, 0) + 1
                    )
            else:
                commit_authors.append("unknown")
                commit_count_by_author["unknown"] = (
                    commit_count_by_author.get("unknown", 0) + 1
                )

        num_commits = len(commit_data)
        unique_committers = len(set(commit_authors))

        logger.debug("Processed %d commits into %d unique committers",
                     num_commits, unique_committers)

        return (
            commit_count_by_author, commit_authors,
            num_commits, unique_committers)

    def _calculate_effective_maintainers(
            self, commit_count_by_author: dict[str, int]) -> float:
        """Calculate effective maintainers using inverse HHI."""
        counts = [max(0, int(v)) for v in commit_count_by_author.values()]
        total = sum(counts)
        if total > 0:
            shares = [(c / total) for c in counts if c > 0]
            hhi = sum(p * p for p in shares)
            eff = 1.0 / hhi if hhi > 0 else 0.0
        else:
            eff = 0.0
        logger.debug("Calculated effective maintainers %.2f from %d commits",
                     eff, total)
        return eff

    def github_commit_counts(
        self,
        git: "GitClient",
        include_merges: bool = False,
    ) -> dict[str, int]:
        """
        Return commit counts per author for the local repo that
        `git` points to.

        Parameters
    ----------
        git : GitClient
            Your rate-limited git client, already pointed at a local repo.
        include_merges : bool
            If False (default), exclude merge commits.

        Returns
        -------
        dict[str, int]
            Mapping of "author_key" -> commit count, where author_key is the
            author's primary email if available, otherwise their display name.
        """
        args = ["shortlog", "-sne", "--all"]
        if not include_merges:
            args.append("--no-merges")

        try:
            logger.debug("Running git shortlog for repo %s", git.repo_path)
            out = git.request(*args)
        except RuntimeError:
            # If the more detailed format fails (e.g., older git),
            # try a simpler one
            try:
                logger.debug("Retrying git shortlog w/ fallback flags for %s",
                             git.repo_path)
                out = git.request("shortlog", "-sn", "--all")
            except RuntimeError:
                logger.info("Git shortlog failed for %s", git.repo_path,
                            exc_info=True)
                return {}

        counts: dict[str, int] = {}
        for raw in out.splitlines():
            line = raw.strip()
            if not line:
                continue

            # Typical -sne line: "  123\tAlice Example <alice@example.com>"
            # Fallback -sn line:  "  123\tAlice Example"
            try:
                # split "<num>\t<rest>"
                left, rest = line.split("\t", 1)
                num = int(left.strip())
                rest = rest.strip()
            except ValueError:
                # Some git outputs may use spaces; try last-ditch parse:
                parts = line.split()
                if not parts:
                    continue
                try:
                    num = int(parts[0])
                except ValueError:
                    continue
                rest = " ".join(parts[1:])

            # Extract email if present; otherwise use name
            # (prefer email as a stable key; fallback to name)
            email_key = None
            name_key = None

            # Fast path for "<email>"
            lt = rest.rfind("<")
            gt = rest.rfind(">")
            if lt != -1 and gt != -1 and gt > lt + 1:
                email_key = rest[lt + 1:gt].strip().lower()
                name_key = rest[:lt].strip()
            else:
                name_key = rest.strip()

            key = email_key or name_key or "unknown"
            counts[key] = counts.get(key, 0) + max(num, 0)

        logger.debug("Derived commit counts for %d author(s)", len(counts))
        return counts

    def _estimate_bus_factor_with_grok(
        self,
        model_id: str,
        grok_client: PurdueClient
    ) -> float:
        """
        Use Grok to estimate bus factor when commit information is unavailable.

        Parameters
        ----------
        model_id : str
            The model repository identifier (e.g., "bigscience/bloom")
        grok_client : PurdueClient
            The Grok client instance to use for the estimation

        Returns
        -------
        float
            Estimated effective maintainers count (not normalized score)
        """
        prompt = (
            f"Analyze this Hugging Face model repository: {model_id}\n\n"
            "Based on the repository name and organization, estimate the bus "
            "factor (number of effective maintainers).\n"
            "Consider these factors:\n"
            "- Large orgs (bigscience, google, microsoft, facebook): ~5-20\n"
            "- Individual/small teams: ~1-3\n"
            "- Well-known OSS projects: ~3-10\n"
            "- Academic institutions: ~2-5\n\n"
            "Explain your reasoning briefly, then output "
            "'FINAL SCORE: <number>' "
            "with the estimated effective maintainer count."
        )

        def call() -> Any:
            logger.info("Requesting Grok bus factor estimate for %s", model_id)
            return grok_client.llm(prompt)

        def parse(raw: Any) -> float:
            estimated = _parse_numeric_response(raw)
            eff = max(0.5, min(20.0, estimated))
            logger.debug(
                "Grok estimated %.2f effective maintainers for %s",
                eff,
                model_id,
            )
            return eff

        try:
            return _call_llm_with_retry(
                call,
                parse,
                description="Bus factor LLM estimate",
            )
        except Exception as e:
            logger.info(
                "Grok estimation failed for %s",
                model_id,
                exc_info=True,
            )
            raise RuntimeError(f"Grok estimation failed: {e}") from e

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Flow:
        1) github (local path, then remote URL)
        2) huggingface model_url
        3) grok estimate (if available)
        4) constant fallback (eff = 2.0)
        """
        logger.info("Computing bus factor score")
        error: Optional[str] = None

        # --- ONE CHANGE: route GitHub URLs passed via model_url ---
        model_url_raw = inputs.get("model_url")
        if (isinstance(model_url_raw, str) and "github.com"
                in model_url_raw and not inputs.get("github_url")):
            inputs = {**inputs, "github_url": model_url_raw}
            logger.debug("Routed model_url GitHub link to github_url input")
        # ----------------------------------------------------------

        # target maintainers → clamp to >= 1
        try:
            target = int(kwargs.get("target_maintainers", 5))
        except Exception:
            target = 5
        target = max(1, target)

        # Prefer injected clients; fall back to constructor defaults
        hf = self.hf_client

        # Work down the decision tree from most reliable contributor data to
        # progressively noisier estimates so we always return a score.

        # ---------------------------
        # 1) GITHUB — local path
        # ---------------------------
        git_path = (
            inputs.get("git_repo_path")
            or inputs.get("github_local_path")
        )
        if isinstance(git_path, str) and git_path.strip():
            try:
                # A checked-out repo lets us run git analytics without extra
                # network cost.
                git_client = GitClient(
                    max_requests=GIT_MAX_REQUESTS,
                    repo_path=git_path,
                    window_seconds=GIT_WINDOW_SECONDS,
                )
                gh_counts = self.github_commit_counts(
                    git_client, include_merges=False
                )
                if gh_counts:
                    eff = self._calculate_effective_maintainers(gh_counts)
                    score = max(0.0, min(1.0, eff / float(target)))
                    logger.info("Bus factor via local repo %s: " +
                                "eff=%.2f score=%.2f",
                                git_path, eff, score)
                    return score
                logger.debug("Local repo %s yielded no commit data", git_path)
            except RuntimeError as e:
                logger.info("Local git analysis failed for %s", git_path,
                            exc_info=True)
                error = f"Git local error: {e}"

        # ---------------------------
        # 1b) GITHUB — remote URL
        # ---------------------------
        git_url = inputs.get("github_url") or inputs.get("git_url")
        if isinstance(git_url, str) and git_url.strip():
            try:
                with tempfile.TemporaryDirectory(prefix="bf-") as tmp:
                    dest = Path(tmp) / "repo"
                    logger.debug("Cloning %s into %s", git_url, dest)
                    # Shallow clone is enough to derive author counts while
                    # keeping runtime manageable.
                    subprocess.run(
                        ["git", "clone", "--depth", "100", git_url, str(dest)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60,
                    )
                    git_client = GitClient(
                        max_requests=GIT_MAX_REQUESTS,
                        repo_path=str(dest),
                        window_seconds=GIT_WINDOW_SECONDS,
                    )
                    gh_counts = self.github_commit_counts(
                        git_client, include_merges=False
                    )
                    if gh_counts:
                        eff = self._calculate_effective_maintainers(gh_counts)
                        score = max(0.0, min(1.0, eff / float(target)))
                        logger.info("Bus factor via cloned repo %s: " +
                                    "eff=%.2f score=%.2f",
                                    git_url, eff, score)
                        return score
                    logger.debug("Cloned repo %s returned no commit data",
                                 git_url)
            except (subprocess.SubprocessError, OSError) as e:
                logger.info("Git clone failed for %s", git_url, exc_info=True)
                error = f"Git clone error: {e}"

        # ---------------------------
        # 2) HUGGING FACE — model_url
        # ---------------------------
        model_url = inputs.get("model_url")
        if isinstance(model_url, str) and model_url.strip() and hf is not None:
            model_id = model_url.split("huggingface.co/", 1)[-1].strip("/")
            tried: list[tuple[str, int]] = []
            for br in (kwargs.get("branch", "main"), "master"):
                try:
                    # HF commits API provides lightweight author metadata for
                    # hosted repos; iterate branches to cover both defaults.
                    commit_data = self._fetch_hf_commits(
                        client=hf,
                        model_id=model_id,
                        branch=br,
                        max_pages=100,
                    )
                    tried.append((br, len(commit_data or [])))
                    if commit_data:
                        (counts, _authors, num_c, uniq_c) = (
                            self._process_commits(commit_data)
                        )
                        eff = self._calculate_effective_maintainers(counts)
                        score = max(0.0, min(1.0, eff / float(target)))
                        logger.info("Bus factor via HF commits %s (%s): " +
                                    "eff=%.2f score=%.2f " +
                                    "(commits=%d unique=%d)",
                                    model_id, br, eff, score, num_c, uniq_c)
                        return score
                except RuntimeError as e:
                    logger.info("HF commit fetch failed for %s on %s",
                                model_id,
                                br, exc_info=True)
                    error = f"HF error on {br}: {e}"
            if error is None:
                error = "HF returned empty commit list"
            logger.debug("HF attempts for %s: %s", model_id, tried)

        # ---------------------------
        # 3) GROK — estimate or error out
        # ---------------------------
        if self.grok is not None:
            try:
                model_id_for_grok = (
                    model_id if isinstance(model_url, str) else "unknown"
                )
                logger.info("Falling back to Grok estimate for %s",
                            model_id_for_grok)
                # Last-resort: ask the LLM to approximate maintainer count
                # when we have zero direct commit signals.
                eff = self._estimate_bus_factor_with_grok(
                    model_id=model_id_for_grok,
                    grok_client=self.grok,
                )
            except Exception:
                logger.info("Grok fallback failed; using default bus factor",
                            exc_info=True)
                eff = 2.0
        else:
            # Without Grok, adhere to the documented constant fallback.
            logger.info("No Grok client available; using default bus factor")
            eff = 2.0

        score = max(0.0, min(1.0, eff / float(target)))
        logger.info("Bus factor final score: %.2f (eff=%.2f target=%d)",
                    score, eff, target)
        return score


class ReproducibilityMetric(Metric):
    """
    Phase 2 Reproducibility Metric
    
    Determines whether the model can be run using only the demonstration 
    code included in the model card.
    
    Scores:
    - 0.0: No demonstration code found, or code doesn't run at all
    - 0.5: Code runs but requires debugging/modifications
    - 1.0: Code runs successfully with no changes
    
    The metric:
    1. Fetches the model card/README from HuggingFace
    2. Extracts code blocks (```python ... ```)
    3. Attempts to execute the code in an isolated environment
    4. Uses an LLM agent to debug if initial execution fails
    5. Returns score based on execution success
    """
    
    name = "Reproducibility"
    key = "reproducibility"
    
    def __init__(self):
        """Initialize with HuggingFace client and LLM agent."""
        self.hf_client = HFClient(
            max_requests=HF_MAX_REQUESTS,
            window_seconds=HF_WINDOW_SECONDS,
        )
        self.llm_agent = PurdueClient(
            max_requests=PURDUE_MAX_REQUESTS,
            window_seconds=PURDUE_WINDOW_SECONDS,
        )
    
    def _extract_demo_code(self, readme_text: str) -> list[str]:
        """
        Extract Python code blocks from model card README.
        
        Parameters
        ----------
        readme_text : str
            The full text of the model card/README
            
        Returns
        -------
        list[str]
            List of Python code blocks found in the README
        """
        code_blocks = []
        
        # Match code blocks with ```python or ```py markers
        pattern = r"```(?:python|py)\s*\n(.*?)```"
        matches = re.finditer(pattern, readme_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            code = match.group(1).strip()
            if code:
                code_blocks.append(code)
        
        # Also try without language specifier (generic ```)
        if not code_blocks:
            pattern = r"```\s*\n(.*?)```"
            matches = re.finditer(pattern, readme_text, re.DOTALL)
            for match in matches:
                code = match.group(1).strip()
                # Simple heuristic: if it looks like Python
                if code and any(kw in code for kw in ['import ', 'from ', 'def ', 'class ']):
                    code_blocks.append(code)
        
        logger.info("Extracted %d code blocks from README", len(code_blocks))
        return code_blocks
    
    def _execute_code_safely(self, code: str, timeout: int = 30) -> tuple[bool, str]:
        """
        Execute Python code in a subprocess with timeout.
        
        Parameters
        ----------
        code : str
            Python code to execute
        timeout : int
            Maximum execution time in seconds
            
        Returns
        -------
        tuple[bool, str]
            (success, output_or_error)
        """
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Clean up
            try:
                Path(temp_file).unlink()
            except Exception:
                pass
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                error_msg = f"Exit code {result.returncode}\n{result.stderr}"
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            return False, f"Execution timeout ({timeout}s exceeded)"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def _try_debug_with_agent(self, code: str, error: str) -> tuple[bool, str]:
        """
        Use LLM agent to attempt debugging the code.
        
        Parameters
        ----------
        code : str
            Original code that failed
        error : str
            Error message from failed execution
            
        Returns
        -------
        tuple[bool, str]
            (success, fixed_code_or_error)
        """
        prompt = f"""You are a Python debugging assistant. The following code from a HuggingFace model card failed to execute:

CODE:
```python
{code}
```

ERROR:
{error}

Please provide a fixed version of the code that will run successfully. Common issues include:
- Missing imports (add common ones like torch, transformers, numpy)
- Placeholder values that need to be replaced
- File paths that don't exist
- API keys or tokens required

Respond ONLY with the fixed Python code in a ```python code block. Do not include explanations.
"""
        
        try:
            response = self.llm_agent.chat(prompt)
            response_text = str(response).strip()
            
            # Extract code from response
            code_pattern = r"```(?:python|py)?\s*\n(.*?)```"
            match = re.search(code_pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                fixed_code = match.group(1).strip()
                logger.info("LLM agent provided fixed code (%d chars)", len(fixed_code))
                
                # Try executing the fixed code
                success, output = self._execute_code_safely(fixed_code, timeout=30)
                return success, fixed_code if success else output
            else:
                return False, "LLM agent did not return valid code"
                
        except Exception as e:
            logger.warning("LLM debug attempt failed: %s", e)
            return False, f"LLM error: {str(e)}"
    
    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute reproducibility score.
        
        Parameters
        ----------
        inputs : dict[str, Any]
            Must contain 'model_url' - the HuggingFace model URL
        **kwargs : Any
            Optional parameters:
            - use_agent: bool (default True) - whether to use LLM for debugging
            - timeout: int (default 30) - execution timeout in seconds
            
        Returns
        -------
        float
            0.0 (no code/doesn't run), 0.5 (runs with debugging), or 1.0 (runs immediately)
        """
        model_url = inputs.get("model_url")
        if not model_url or not isinstance(model_url, str):
            logger.warning("No model_url provided for reproducibility check")
            return 0.0
        
        use_agent = kwargs.get("use_agent", True)
        timeout = kwargs.get("timeout", 30)
        
        try:
            # 1. Fetch model card text
            logger.info("Fetching model card for reproducibility check: %s", model_url)
            readme_text = injectHFBrowser(model_url, self.hf_client)
            
            if not readme_text:
                logger.warning("Could not fetch model card for %s", model_url)
                return 0.0
            
            # 2. Extract demo code blocks
            code_blocks = self._extract_demo_code(readme_text)
            
            if not code_blocks:
                logger.info("No demo code found in model card")
                return 0.0
            
            # 3. Try executing each code block
            for idx, code in enumerate(code_blocks):
                logger.info("Testing code block %d/%d (%d chars)", 
                          idx + 1, len(code_blocks), len(code))
                
                # First attempt: run as-is
                success, output = self._execute_code_safely(code, timeout=timeout)
                
                if success:
                    logger.info("Code block %d runs successfully without changes", idx + 1)
                    return 1.0
                
                # Second attempt: use agent to debug
                if use_agent and self.llm_agent:
                    logger.info("Code block %d failed, trying LLM debugging", idx + 1)
                    debug_success, debug_output = self._try_debug_with_agent(code, output)
                    
                    if debug_success:
                        logger.info("Code block %d runs after LLM debugging", idx + 1)
                        return 0.5
                else:
                    logger.info("Code block %d failed: %s", idx + 1, output[:200])
            
            # None of the code blocks worked
            logger.info("No demo code blocks could be executed successfully")
            return 0.0
            
        except Exception as e:
            logger.error("Reproducibility metric error: %s", e, exc_info=True)
            return 0.0


class ReviewednessMetric(Metric):
    """
    Phase 2 Reviewedness Metric
    
    Calculates the fraction of all code (not weights) in the associated GitHub 
    repository that was introduced through pull requests with a code review.
    
    Returns:
    - Value between 0.0 and 1.0 representing the fraction of reviewed code
    - -1 if there is no linked GitHub repository
    
    The metric:
    1. Extracts GitHub repository URL from model metadata
    2. Fetches all commits from the repository
    3. Identifies which commits were part of reviewed PRs
    4. Calculates lines of code added through reviewed PRs vs. total
    5. Returns the fraction as a score
    """
    
    name = "Reviewedness"
    key = "reviewedness"
    
    def __init__(self):
        """Initialize with GitHub client."""
        self.git_client = GitClient(
            max_requests=GIT_MAX_REQUESTS,
            window_seconds=GIT_WINDOW_SECONDS,
        )
    
    def _extract_github_url(self, model_url: str) -> Optional[str]:
        """
        Extract GitHub repository URL from model metadata.
        
        Parameters
        ----------
        model_url : str
            HuggingFace model URL or direct GitHub URL
            
        Returns
        -------
        Optional[str]
            GitHub repository URL, or None if not found
        """
        # If already a GitHub URL
        if "github.com" in model_url:
            return model_url
        
        # Try to fetch from HuggingFace model card
        # In production, would parse model card for GitHub links
        logger.info("Attempting to find GitHub repo for: %s", model_url)
        
        # For now, return None if not explicitly GitHub
        # In production: parse HuggingFace API or model card
        return None
    
    def _get_pull_requests(self, repo_owner: str, repo_name: str) -> list[dict]:
        """
        Fetch all pull requests from a GitHub repository.
        
        Parameters
        ----------
        repo_owner : str
            GitHub repository owner
        repo_name : str
            GitHub repository name
            
        Returns
        -------
        list[dict]
            List of pull request data
        """
        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
            params = {
                "state": "all",  # Get open and closed PRs
                "per_page": 100
            }
            
            all_prs = []
            page = 1
            
            # Fetch all pages of PRs
            while page <= 10:  # Limit to 10 pages (1000 PRs)
                paginated_params = {**params, "page": page}
                response = self.git_client.get(url, params=paginated_params)
                
                if response.status_code != 200:
                    logger.warning("Failed to fetch PRs (page %d): %d", 
                                 page, response.status_code)
                    break
                
                prs = response.json()
                if not prs:
                    break
                
                all_prs.extend(prs)
                page += 1
            
            logger.info("Fetched %d pull requests", len(all_prs))
            return all_prs
            
        except Exception as e:
            logger.error("Error fetching pull requests: %s", e)
            return []
    
    def _get_pr_reviews(self, repo_owner: str, repo_name: str, pr_number: int) -> list[dict]:
        """
        Fetch reviews for a specific pull request.
        
        Parameters
        ----------
        repo_owner : str
            GitHub repository owner
        repo_name : str
            GitHub repository name
        pr_number : int
            Pull request number
            
        Returns
        -------
        list[dict]
            List of review data
        """
        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/reviews"
            response = self.git_client.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception as e:
            logger.debug("Error fetching reviews for PR #%d: %s", pr_number, e)
            return []
    
    def _get_pr_commits(self, repo_owner: str, repo_name: str, pr_number: int) -> list[str]:
        """
        Get commit SHAs associated with a pull request.
        
        Parameters
        ----------
        repo_owner : str
            GitHub repository owner
        repo_name : str
            GitHub repository name
        pr_number : int
            Pull request number
            
        Returns
        -------
        list[str]
            List of commit SHAs in the PR
        """
        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/commits"
            response = self.git_client.get(url)
            
            if response.status_code == 200:
                commits = response.json()
                return [c["sha"] for c in commits]
            else:
                return []
                
        except Exception as e:
            logger.debug("Error fetching commits for PR #%d: %s", pr_number, e)
            return []
    
    def _get_commit_stats(self, repo_owner: str, repo_name: str, commit_sha: str) -> dict:
        """
        Get statistics for a specific commit (additions/deletions).
        
        Parameters
        ----------
        repo_owner : str
            GitHub repository owner
        repo_name : str
            GitHub repository name
        commit_sha : str
            Commit SHA
            
        Returns
        -------
        dict
            Commit statistics with 'additions' and 'deletions' counts
        """
        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
            response = self.git_client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get("stats", {})
                
                # Filter out weight/data files by checking changed files
                files = data.get("files", [])
                code_additions = 0
                code_deletions = 0
                
                for file_data in files:
                    filename = file_data.get("filename", "")
                    # Skip binary files, model weights, datasets
                    if self._is_code_file(filename):
                        code_additions += file_data.get("additions", 0)
                        code_deletions += file_data.get("deletions", 0)
                
                return {
                    "additions": code_additions,
                    "deletions": code_deletions,
                    "total": code_additions + code_deletions
                }
            else:
                return {"additions": 0, "deletions": 0, "total": 0}
                
        except Exception as e:
            logger.debug("Error fetching commit stats for %s: %s", commit_sha[:7], e)
            return {"additions": 0, "deletions": 0, "total": 0}
    
    def _is_code_file(self, filename: str) -> bool:
        """
        Determine if a file is a code file (not weights/data).
        
        Parameters
        ----------
        filename : str
            File path
            
        Returns
        -------
        bool
            True if file is code, False otherwise
        """
        filename_lower = filename.lower()
        
        # First, check if it's a code extension (priority)
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.jsx', '.tsx'}
        for ext in code_extensions:
            if filename_lower.endswith(ext):
                return True
        
        # Skip common non-code files
        skip_extensions = {
            '.bin', '.pt', '.pth', '.h5', '.pkl', '.pickle',
            '.npy', '.npz', '.json', '.csv', '.tsv', '.txt',
            '.md', '.jpg', '.png', '.gif', '.pdf', '.zip',
            '.tar', '.gz', '.safetensors', '.onnx', '.pb'
        }
        
        # Check extensions
        for ext in skip_extensions:
            if filename_lower.endswith(ext):
                return False
        
        # Check patterns in path (but only if not already identified as code)
        skip_patterns = [
            'checkpoint', 'weights', '/data/',
            '/dataset/', '/assets/', '/images/', '/docs/'
        ]
        
        for pattern in skip_patterns:
            if pattern in filename_lower:
                return False
        
        # Default: exclude if unsure
        return False
    
    def _has_code_review(self, reviews: list[dict]) -> bool:
        """
        Determine if a PR has legitimate code reviews.
        
        Parameters
        ----------
        reviews : list[dict]
            List of review data from GitHub API
            
        Returns
        -------
        bool
            True if PR has at least one real code review
        """
        if not reviews:
            return False
        
        # Look for reviews with state "APPROVED" or "CHANGES_REQUESTED"
        # Comments without state don't count as formal reviews
        for review in reviews:
            state = review.get("state", "")
            if state in ["APPROVED", "CHANGES_REQUESTED"]:
                return True
        
        return False
    
    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute reviewedness score.
        
        Parameters
        ----------
        inputs : dict[str, Any]
            Must contain 'git_url' or 'model_url' with GitHub repository
        **kwargs : Any
            Optional parameters
            
        Returns
        -------
        float
            Fraction of code introduced through reviewed PRs (0.0 to 1.0)
            Returns -1 if no GitHub repository is linked
        """
        # Try to get GitHub URL
        git_url = inputs.get("git_url")
        if not git_url:
            model_url = inputs.get("model_url", "")
            git_url = self._extract_github_url(model_url)
        
        if not git_url or "github.com" not in git_url:
            logger.info("No GitHub repository found, returning -1")
            return -1.0
        
        try:
            # Parse owner and repo from URL
            # Format: https://github.com/owner/repo
            parts = git_url.rstrip('/').split('/')
            
            # Extract path components after github.com
            try:
                github_idx = parts.index('github.com')
                path_parts = parts[github_idx + 1:]
            except (ValueError, IndexError):
                logger.warning("Invalid GitHub URL format: %s", git_url)
                return -1.0
            
            # Need at least 2 path components (owner/repo)
            if len(path_parts) < 2:
                logger.warning("Invalid GitHub URL format: %s", git_url)
                return -1.0
            
            repo_owner = path_parts[0]
            repo_name = path_parts[1].replace('.git', '')
            
            if not repo_owner or not repo_name:
                logger.warning("Invalid GitHub URL format: %s", git_url)
                return -1.0
            
            logger.info("Analyzing reviewedness for %s/%s", repo_owner, repo_name)
            
            # 1. Get all pull requests
            prs = self._get_pull_requests(repo_owner, repo_name)
            
            if not prs:
                logger.info("No pull requests found, assuming no code reviews")
                return 0.0
            
            # 2. Identify reviewed PRs and their commits
            reviewed_commits = set()
            all_pr_commits = set()
            
            for pr in prs:
                pr_number = pr.get("number")
                pr_state = pr.get("state")
                
                # Only count merged PRs
                if pr.get("merged_at") is None:
                    continue
                
                # Get commits in this PR
                pr_commits = self._get_pr_commits(repo_owner, repo_name, pr_number)
                all_pr_commits.update(pr_commits)
                
                # Check if PR has code reviews
                reviews = self._get_pr_reviews(repo_owner, repo_name, pr_number)
                has_review = self._has_code_review(reviews)
                
                if has_review:
                    reviewed_commits.update(pr_commits)
                
                logger.debug("PR #%d: %d commits, reviewed=%s", 
                           pr_number, len(pr_commits), has_review)
            
            logger.info("Found %d PR commits, %d with reviews", 
                       len(all_pr_commits), len(reviewed_commits))
            
            # 3. Calculate lines of code for reviewed vs. all commits
            reviewed_lines = 0
            total_pr_lines = 0
            
            # Sample commits to avoid rate limiting (take up to 100 commits)
            sampled_all = list(all_pr_commits)[:100]
            sampled_reviewed = [c for c in sampled_all if c in reviewed_commits]
            
            for commit_sha in sampled_all:
                stats = self._get_commit_stats(repo_owner, repo_name, commit_sha)
                lines = stats["total"]
                total_pr_lines += lines
                
                if commit_sha in reviewed_commits:
                    reviewed_lines += lines
            
            # 4. Calculate fraction
            if total_pr_lines == 0:
                logger.info("No code changes in PRs")
                return 0.0
            
            fraction = reviewed_lines / total_pr_lines
            logger.info("Reviewedness: %d/%d lines = %.3f", 
                       reviewed_lines, total_pr_lines, fraction)
            
            return round(fraction, 3)
            
        except Exception as e:
            logger.error("Reviewedness metric error: %s", e, exc_info=True)
            return -1.0


class TreescoreMetric:
    """
    Metric that computes the average overall score of parent models.
    
    Analyzes model's config.json to identify parent models (e.g., base_model,
    _name_or_path) and calculates the average score of parents that exist
    in the model registry.
    
    Score Interpretation:
    - Returns 0.0 to 1.0: Average overall score of parent models
    - Returns -1.0: No parent models found or no valid HuggingFace URL
    - Returns 0.0: Parent models exist but none are in registry
    
    Parent Model Detection:
    Checks these config.json fields for parent model references:
    - base_model (common in LoRA/adapter models)
    - _name_or_path (common in fine-tuned models)
    - parent_model (custom field)
    """
    
    def __init__(self, 
                 hf_client: Optional[HFClient] = None,
                 model_registry: Optional[dict] = None) -> None:
        """
        Initialize TreescoreMetric.
        
        Parameters
        ----------
        hf_client : Optional[HFClient]
            HuggingFace API client. If None, creates default client.
        model_registry : Optional[dict]
            Dictionary mapping model_id -> model data with scores.
            If None, metric will return -1.0 (cannot compute without registry).
        """
        self.hf_client = hf_client or HFClient(
            max_requests=120,
            window_seconds=30.0
        )
        self.model_registry = model_registry or {}
        
    def _extract_model_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract model_id from HuggingFace URL.
        
        Parameters
        ----------
        url : str
            HuggingFace model URL
            
        Returns
        -------
        Optional[str]
            Model ID (e.g., "bert-base-uncased") or None if invalid
        """
        if not url or "huggingface.co" not in url:
            return None
        
        try:
            # Format: https://huggingface.co/username/model-name
            parts = url.rstrip('/').split('/')
            hf_idx = parts.index('huggingface.co')
            path_parts = parts[hf_idx + 1:]
            
            # Need at least owner/model
            if len(path_parts) < 2:
                return None
            
            # Return "owner/model"
            return f"{path_parts[0]}/{path_parts[1]}"
        except (ValueError, IndexError):
            return None
    
    def _get_config_json(self, model_id: str) -> dict:
        """
        Fetch config.json for a HuggingFace model.
        
        Parameters
        ----------
        model_id : str
            HuggingFace model ID (e.g., "bert-base-uncased")
            
        Returns
        -------
        dict
            Parsed config.json or empty dict if not found
        """
        try:
            logger.info("Fetching config.json for %s", model_id)
            
            # HF API endpoint for raw file access
            config = self.hf_client.request(
                "GET",
                f"/api/models/{model_id}",
            )
            
            # Check if config is available in model info
            if "config" in config:
                return config["config"]
            
            # Try to fetch raw config.json file
            try:
                config_content = self.hf_client.request(
                    "GET",
                    f"/{model_id}/raw/main/config.json"
                )
                if isinstance(config_content, dict):
                    return config_content
            except Exception as e:
                logger.debug("Could not fetch raw config.json: %s", e)
            
            return {}
            
        except Exception as e:
            logger.warning("Failed to fetch config for %s: %s", model_id, e)
            return {}
    
    def _extract_parent_models(self, config: dict) -> List[str]:
        """
        Extract parent model references from config.json.
        
        Parameters
        ----------
        config : dict
            Parsed config.json
            
        Returns
        -------
        List[str]
            List of parent model IDs
        """
        parents = []
        
        # Common fields that reference parent models
        parent_fields = [
            "base_model",
            "_name_or_path", 
            "parent_model",
            "model_name_or_path",
        ]
        
        for field in parent_fields:
            value = config.get(field)
            if value and isinstance(value, str):
                # Skip local paths
                if value.startswith('./') or value.startswith('/') or value.startswith('\\'):
                    continue
                
                # Remove common prefixes
                cleaned = value.replace('https://huggingface.co/', '')
                cleaned = cleaned.strip('/')
                
                # Skip empty strings
                if not cleaned:
                    continue
                
                # Accept either "owner/model" or just "model-name" format
                # Many HF models use simple names like "bert-base-uncased"
                parents.append(cleaned)
                logger.debug("Found parent model: %s (from %s)", cleaned, field)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_parents = []
        for parent in parents:
            if parent not in seen:
                seen.add(parent)
                unique_parents.append(parent)
        
        return unique_parents
    
    def _get_parent_scores(self, parent_ids: List[str]) -> List[float]:
        """
        Get overall scores for parent models from registry.
        
        Parameters
        ----------
        parent_ids : List[str]
            List of parent model IDs
            
        Returns
        -------
        List[float]
            List of overall scores for parents in registry
        """
        scores = []
        
        for parent_id in parent_ids:
            # Check if parent exists in registry
            if parent_id in self.model_registry:
                model_data = self.model_registry[parent_id]
                model_scores = model_data.get("scores", {})
                
                # Calculate overall score (exclude -1 values)
                valid_scores = [v for v in model_scores.values() if isinstance(v, (int, float)) and v >= 0]
                if valid_scores:
                    overall = sum(valid_scores) / len(valid_scores)
                    scores.append(overall)
                    logger.debug("Parent %s overall score: %.3f", parent_id, overall)
                else:
                    logger.debug("Parent %s has no valid scores", parent_id)
            else:
                logger.debug("Parent %s not in registry", parent_id)
        
        return scores
    
    def compute(self, inputs: dict, **kwargs: Any) -> float:
        """
        Compute treescore (average score of parent models).
        
        Parameters
        ----------
        inputs : dict
            Must contain 'model_url' with HuggingFace repository URL
        **kwargs : Any
            Optional parameters
            
        Returns
        -------
        float
            Average overall score of parent models (0.0 to 1.0)
            Returns -1.0 if no parent models found or URL invalid
            Returns 0.0 if parents exist but none are in registry
        """
        # Get model URL
        model_url = inputs.get("model_url", "")
        
        if not model_url or "huggingface.co" not in model_url:
            logger.info("No HuggingFace URL provided, returning -1")
            return -1.0
        
        # Extract model ID
        model_id = self._extract_model_id_from_url(model_url)
        if not model_id:
            logger.warning("Invalid HuggingFace URL: %s", model_url)
            return -1.0
        
        logger.info("Computing treescore for %s", model_id)
        
        try:
            # 1. Get config.json
            config = self._get_config_json(model_id)
            
            if not config:
                logger.info("No config.json found for %s", model_id)
                return -1.0
            
            # 2. Extract parent models
            parent_ids = self._extract_parent_models(config)
            
            if not parent_ids:
                logger.info("No parent models found in config for %s", model_id)
                return -1.0
            
            logger.info("Found %d parent model(s): %s", len(parent_ids), parent_ids)
            
            # 3. Get scores for parents in registry
            parent_scores = self._get_parent_scores(parent_ids)
            
            if not parent_scores:
                logger.info("No parent models found in registry")
                return 0.0
            
            # 4. Calculate average
            treescore = sum(parent_scores) / len(parent_scores)
            logger.info("Treescore: %.3f (avg of %d parent scores)", treescore, len(parent_scores))
            
            return round(treescore, 3)
            
        except Exception as e:
            logger.error("Treescore metric error: %s", e, exc_info=True)
            return -1.0
