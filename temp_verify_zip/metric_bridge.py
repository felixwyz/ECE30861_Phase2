"""
Metrics Bridge - Connects OpenAPI routes to Phase 1 & 2 metrics system.
"""
import time
import sys
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.Metrics import (
    BusFactorMetric, RampUpTime, LicenseMetric,
    AvailabilityMetric, CodeQuality, DatasetQuality,
    PerformanceClaimsMetric, SizeMetric,
    ReproducibilityMetric, ReviewednessMetric, TreescoreMetric
)
from src.Client import HFClient, GitClient, PurdueClient
from src.logging_utils import get_logger
from src.utils import browse_hf_repo

logger = get_logger(__name__)

# Rate limits from Metrics.py
PURDUE_MAX_REQUESTS = 40
PURDUE_WINDOW_SECONDS = 30.0
HF_MAX_REQUESTS = 120
HF_WINDOW_SECONDS = 30.0
GIT_MAX_REQUESTS = 40
GIT_WINDOW_SECONDS = 30.0


class MetricsBridge:
    """Bridge between API layer and metrics computation system."""
    
    def __init__(self, model_registry: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics bridge.
        
        Parameters
        ----------
        model_registry : Optional[Dict[str, Any]]
            Registry of existing models for treescore computation
        """
        self.model_registry = model_registry or {}
        
        # Initialize clients with error handling
        try:
            self.hf_client = HFClient(
                max_requests=HF_MAX_REQUESTS,
                window_seconds=HF_WINDOW_SECONDS
            )
        except Exception as e:
            logger.warning(f"HFClient initialization failed: {e}")
            self.hf_client = None
            
        try:
            self.git_client = GitClient(
                max_requests=GIT_MAX_REQUESTS,
                window_seconds=GIT_WINDOW_SECONDS
            )
        except Exception as e:
            logger.warning(f"GitClient initialization failed: {e}")
            self.git_client = None
            
        try:
            self.purdue_client = PurdueClient(
                max_requests=PURDUE_MAX_REQUESTS,
                window_seconds=PURDUE_WINDOW_SECONDS
            )
        except Exception as e:
            logger.warning(f"PurdueClient initialization failed (likely missing API key): {e}")
            self.purdue_client = None
        
    def compute_all_metrics(
        self,
        artifact_url: str,
        artifact_type: str = "model",
        artifact_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute all metrics for an artifact.
        
        Parameters
        ----------
        artifact_url : str
            URL to the artifact (HuggingFace or GitHub)
        artifact_type : str
            Type of artifact: "model", "dataset", or "code"
        artifact_name : Optional[str]
            Name to use for the artifact (extracted from URL if not provided)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all metric scores and latencies
        """
        logger.info(f"Computing metrics for {artifact_type}: {artifact_url}")
        
        # Extract name from URL if not provided
        if not artifact_name:
            artifact_name = self._extract_name_from_url(artifact_url)
        
        # Build inputs dict
        inputs = {
            "model_url": artifact_url,
            "artifact_url": artifact_url,
            "artifact_type": artifact_type,
            "name": artifact_name
        }
        
        # Try to get README content for metrics that need it
        try:
            if "huggingface.co" in artifact_url:
                readme_content = self._get_hf_readme(artifact_url)
                inputs["readme"] = readme_content
        except Exception as e:
            logger.warning(f"Could not fetch README: {e}")
            inputs["readme"] = ""
        
        # Initialize all metrics
        metrics_to_compute = {
            'bus_factor': BusFactorMetric(
                hf_client=self.hf_client,
                grok_client=self.purdue_client
            ),
            'ramp_up_time': RampUpTimeMetric(),
            'license': LicenseMetric(),
            'availability': AvailabilityMetric(),
            'code_quality': CodeQualityMetric(
                hf_client=self.hf_client,
                grok_client=self.purdue_client
            ),
            'dataset_quality': DatasetQualityMetric(hf_client=self.hf_client),
            'performance_claims': PerformanceClaimsMetric(),
            'size': SizeMetric(),
            'reproducibility': ReproducibilityMetric(),
            'reviewedness': ReviewednessMetric(),
            'tree_score': TreescoreMetric(
                hf_client=self.hf_client,
                model_registry=self.model_registry
            )
        }
        
        results = {}
        
        # Compute each metric with timing
        for metric_name, metric_obj in metrics_to_compute.items():
            start_time = time.time()
            try:
                score = metric_obj.compute(inputs)
                latency = time.time() - start_time
                
                # Handle size metric which returns dict
                if metric_name == 'size' and isinstance(score, dict):
                    results['size_score'] = score
                    results['size_score_latency'] = latency
                else:
                    results[metric_name] = score if score >= 0 else 0.0
                    results[f"{metric_name}_latency"] = latency
                    
                logger.info(f"✓ {metric_name}: {score:.3f} ({latency:.2f}s)")
                
            except Exception as e:
                logger.error(f"✗ {metric_name} failed: {e}", exc_info=True)
                # Return -1 for failed metrics
                if metric_name == 'size':
                    results['size_score'] = {
                        'raspberry_pi': 0.0,
                        'jetson_nano': 0.0,
                        'desktop_pc': 0.0,
                        'aws_server': 0.0
                    }
                    results['size_score_latency'] = time.time() - start_time
                else:
                    results[metric_name] = -1.0
                    results[f"{metric_name}_latency"] = time.time() - start_time
        
        # Compute dataset_and_code_score as average
        dataset_score = results.get('dataset_quality', 0.0)
        code_score = results.get('code_quality', 0.0)
        results['dataset_and_code_score'] = (dataset_score + code_score) / 2.0
        results['dataset_and_code_score_latency'] = (
            results.get('dataset_quality_latency', 0.0) +
            results.get('code_quality_latency', 0.0)
        ) / 2.0
        
        # Calculate net score (average of all valid non-latency metrics)
        valid_scores = []
        for key, value in results.items():
            if not key.endswith('_latency') and key != 'size_score':
                if isinstance(value, (int, float)) and value >= 0:
                    valid_scores.append(value)
        
        results['net_score'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        results['net_score_latency'] = sum(
            v for k, v in results.items() if k.endswith('_latency')
        )
        
        return results
    
    def _extract_name_from_url(self, url: str) -> str:
        """Extract artifact name from URL."""
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split('/') if p]
        if len(parts) >= 2 and 'huggingface.co' in url:
            # Format: huggingface.co/owner/model-name
            return parts[-1]
        elif parts:
            return parts[-1]
        return "unknown"
    
    def _get_hf_readme(self, url: str) -> str:
        """Get README content from HuggingFace model."""
        try:
            # Extract model ID from URL
            parsed = urlparse(url)
            parts = [p for p in parsed.path.split('/') if p]
            if len(parts) >= 2:
                model_id = f"{parts[-2]}/{parts[-1]}"
                # Use browse_hf_repo to get README
                content = browse_hf_repo(model_id, self.hf_client)
                return content
        except Exception as e:
            logger.warning(f"Failed to fetch README: {e}")
        return ""


def compute_artifact_metrics(
    artifact_url: str,
    artifact_type: str = "model",
    artifact_name: Optional[str] = None,
    model_registry: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to compute metrics for an artifact.
    
    Parameters
    ----------
    artifact_url : str
        URL to the artifact
    artifact_type : str
        Type: "model", "dataset", or "code"
    artifact_name : Optional[str]
        Artifact name (extracted if not provided)
    model_registry : Optional[Dict[str, Any]]
        Registry for treescore computation
        
    Returns
    -------
    Dict[str, Any]
        All metric scores and latencies
    """
    bridge = MetricsBridge(model_registry=model_registry)
    return bridge.compute_all_metrics(artifact_url, artifact_type, artifact_name)
