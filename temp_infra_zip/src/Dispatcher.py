# src/Dispatcher.py

"""Concurrent dispatcher for Metric computations."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.logging_utils import get_logger
from src.Metrics import Metric, MetricResult

logger = get_logger(__name__)


class Dispatcher:
    """Run a collection of metrics in parallel and capture their results."""

    def __init__(
        self,
        metrics: Optional[Iterable[Metric]] = None,
        *,
        max_workers: Optional[int] = None,
    ) -> None:
        self._metrics: List[Metric] = (
            list(metrics) if metrics is not None else []
        )
        self._max_workers = max_workers

    def add_metric(self, metric: Metric) -> None:
        """Register an additional metric to be dispatched."""

        self._metrics.append(metric)

    def clear_metrics(self) -> None:
        """Remove all registered metrics."""

        self._metrics.clear()

    @property
    def metrics(self) -> List[Metric]:
        """Return a copy of the registered metrics."""

        return list(self._metrics)

    def dispatch(self, inputs: Dict[str, Any]) -> List[MetricResult]:
        """Execute all registered metrics concurrently with shared inputs."""

        if not self._metrics:
            logger.info("No metrics registered; returning empty results")
            return []

        worker_count = self._resolve_worker_count()
        logger.info(
            "Dispatching %d metric(s) using %d worker(s)",
            len(self._metrics),
            worker_count,
        )
        logger.debug("Dispatch inputs: %s", inputs)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._execute_metric, metric, inputs)
                for metric in self._metrics
            ]
            return [future.result() for future in futures]

    def _resolve_worker_count(self) -> int:
        if self._max_workers is not None and self._max_workers > 0:
            return self._max_workers
        worker_cap = (os.cpu_count() or 1) + 4
        return max(1, min(len(self._metrics), worker_cap))

    def _execute_metric(
        self,
        metric: Metric,
        inputs: Dict[str, Any],
    ) -> MetricResult:
        logger.debug("Starting metric %s", metric.key)
        value, latency_ms, error = self._run_with_timing(
            metric.compute,
            inputs,
        )
        result = MetricResult(
            metric=metric.name,
            key=metric.key,
            value=value,
            latency_ms=latency_ms,
            error=error,
        )
        logger.debug("Finished metric %s with result %s", metric.key, result)
        return result

    def _run_with_timing(
        self,
        func: Callable[..., float | dict[str, float]],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[float | dict[str, float], int, Optional[str]]:
        """Run ``func`` with timing, capturing the result or error."""

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            error: Optional[str] = None
        except Exception as exc:
            logger.debug("Metric execution raised %s", exc, exc_info=True)
            result = float("nan")
            error = f"{exc.__class__.__name__}: {exc}"
        latency_ms = int((time.perf_counter() - start) * 1000.0)
        return result, latency_ms, error
