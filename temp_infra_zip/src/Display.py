# src/Display.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import urlparse

from src.logging_utils import get_logger
from src.Metrics import MetricResult

logger = get_logger(__name__)


def _extract_model_name(model_url: str) -> str:
    parsed = urlparse(model_url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return ""
    # Typical HF model path: /org/model[/tree/main]
    # Prefer the second segment when present, else last.
    if len(parts) >= 2:
        return parts[1]
    return parts[-1]


def _results_by_key(results: Iterable[MetricResult]) \
                    -> Dict[str, MetricResult]:
    return {r.key: r for r in results}


def _get_value_latency(
    res_map: Dict[str, MetricResult],
    key: str,
) -> tuple[float | dict[str, float], int]:
    res = res_map.get(key)
    if res is None or res.value is None:
        return 0.0, 0

    def _clamp_score(raw: Any) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if value != value:
            return 0.0
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    value = res.value
    if isinstance(value, Mapping):
        normalized: dict[str, float] = {}
        for device, raw_score in value.items():
            normalized[str(device)] = _clamp_score(raw_score)
        metric_value: float | dict[str, float] = normalized
    else:
        metric_value = _clamp_score(value)

    try:
        latency = int(res.latency_ms)
    except (TypeError, ValueError):
        latency = 0

    return metric_value, latency


def build_output_object(group: Dict[str, str], results: List[MetricResult]) \
                        -> Dict[str, Any]:
    logger.debug("Building output object for group %s", group)
    res_map = _results_by_key(results)

    ramp_val, ramp_lat = _get_value_latency(res_map, "ramp_up_time")
    lic_val, lic_lat = _get_value_latency(res_map, "license_metric")
    size_val, size_lat = _get_value_latency(res_map, "size_metric")
    avail_val, avail_lat = _get_value_latency(res_map, "availability_metric")
    dquality_val, dquality_lat = _get_value_latency(res_map, "dataset_quality")
    cquality_val, cquality_lat = _get_value_latency(res_map, "code_quality")
    pclaim_val, pclaim_lat = _get_value_latency(res_map, "performance_claims")
    bfact_val, bfact_lat = _get_value_latency(res_map, "bus_factor")

    def _score_mean(value: float | dict[str, float]) -> float:
        if isinstance(value, Mapping):
            values = list(value.values())
            if not values:
                return 0.0
            return sum(values) / len(values)
        return float(value)

    component_values = [
        _score_mean(ramp_val),
        _score_mean(lic_val),
        _score_mean(size_val),
        _score_mean(avail_val),
        _score_mean(dquality_val),
        _score_mean(cquality_val),
        _score_mean(pclaim_val),
        _score_mean(bfact_val),
    ]
    net_val = sum(component_values) / len(component_values)
    net_lat = (
        ramp_lat
        + lic_lat
        + avail_lat
        + size_lat
        + dquality_lat
        + cquality_lat
        + pclaim_lat
        + bfact_lat
    )

    # Size score as multi-device object to match expected shape
    size_keys = ["raspberry_pi", "jetson_nano", "desktop_pc", "aws_server"]
    if isinstance(size_val, dict):
        size_obj = {}
        for device in size_keys:
            raw_score = size_val.get(device, 0.0)
            try:
                size_obj[device] = float(raw_score)
            except Exception:
                size_obj[device] = 0.0
    else:
        try:
            size_scalar = float(size_val)
        except Exception:
            size_scalar = 0.0
        size_obj = {device: size_scalar for device in size_keys}

    model_url = group.get("model_url", "")
    name = _extract_model_name(model_url) if isinstance(model_url, str) else ""

    # Build output dict in the expected key order
    out: Dict[str, Any] = {}
    out["name"] = name
    out["category"] = "MODEL"
    out["net_score"] = net_val
    out["net_score_latency"] = net_lat
    out["ramp_up_time"] = ramp_val
    out["ramp_up_time_latency"] = ramp_lat
    # Optional/unknown metrics in our pipeline: set to 0.0
    out["bus_factor"] = bfact_val
    out["bus_factor_latency"] = bfact_lat
    out["performance_claims"] = pclaim_val
    out["performance_claims_latency"] = pclaim_lat
    out["license"] = lic_val
    out["license_latency"] = lic_lat
    out["size_score"] = size_obj
    out["size_score_latency"] = size_lat
    out["dataset_and_code_score"] = avail_val
    out["dataset_and_code_score_latency"] = avail_lat
    out["dataset_quality"] = dquality_val
    out["dataset_quality_latency"] = dquality_lat
    out["code_quality"] = cquality_val
    out["code_quality_latency"] = cquality_lat
    logger.debug("Output object ready: %s", out)
    return out


def print_results(group: Dict[str, str], results: List[MetricResult]) -> None:
    obj = build_output_object(group, results)
    logger.info("Printing results for %s", group.get("model_url", "unknown"))
    print(json.dumps(obj, separators=(",", ":")))
