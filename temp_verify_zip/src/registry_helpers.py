"""
Helper functions for registry operations: lineage, license checking, cost calculation.
"""
import re
import requests
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


# License compatibility matrix for fine-tuning + inference/generation
# Based on ModelGo paper principles: permissive licenses are generally compatible
LICENSE_COMPATIBILITY = {
    # GitHub license -> Compatible model licenses for fine-tune + inference
    "mit": {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", 
            "openrail", "creativeml-openrail-m", "llama2", "llama3", "llama3.1", 
            "llama3.2", "llama3.3", "gemma", "bigscience-openrail-m"},
    "apache-2.0": {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0",
                   "openrail", "creativeml-openrail-m", "llama2", "llama3", "llama3.1",
                   "llama3.2", "llama3.3", "gemma", "bigscience-openrail-m"},
    "bsd-2-clause": {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0"},
    "bsd-3-clause": {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0"},
    "gpl-3.0": {"gpl-3.0", "agpl-3.0"},  # GPL is restrictive
    "gpl-2.0": {"gpl-2.0", "gpl-3.0"},
    "lgpl-3.0": {"lgpl-3.0", "gpl-3.0", "mit", "apache-2.0"},  # LGPL allows linking
    "agpl-3.0": {"agpl-3.0"},  # AGPL most restrictive
    "cc-by-4.0": {"cc-by-4.0", "cc-by-sa-4.0", "mit", "apache-2.0"},
    "cc-by-sa-4.0": {"cc-by-sa-4.0", "gpl-3.0"},  # Share-alike requires same
    # If license not in matrix, be conservative
}

# Normalize license names
LICENSE_ALIASES = {
    "apache license 2.0": "apache-2.0",
    "apache 2.0": "apache-2.0",
    "apache-2": "apache-2.0",
    "mit license": "mit",
    "bsd": "bsd-3-clause",
    "bsd license": "bsd-3-clause",
    "gpl": "gpl-3.0",
    "gpl-3": "gpl-3.0",
    "gplv3": "gpl-3.0",
    "lgpl": "lgpl-3.0",
    "lgpl-3": "lgpl-3.0",
    "agpl": "agpl-3.0",
    "cc-by": "cc-by-4.0",
    "cc by 4.0": "cc-by-4.0",
    "cc-by-sa": "cc-by-sa-4.0",
    "openrail-m": "openrail",
    "llama 2": "llama2",
    "llama 3": "llama3",
    "llama 3.1": "llama3.1",
    "llama 3.2": "llama3.2",
}


def normalize_license_name(license_str: str) -> str:
    """Normalize license name to standard form."""
    if not license_str:
        return "unknown"
    
    license_lower = license_str.strip().lower()
    license_lower = license_lower.replace("_", "-").replace(" ", "-")
    
    # Direct match
    if license_lower in LICENSE_COMPATIBILITY:
        return license_lower
    
    # Check aliases
    if license_lower in LICENSE_ALIASES:
        return LICENSE_ALIASES[license_lower]
    
    # Check if it contains a known license
    for known_license in LICENSE_COMPATIBILITY.keys():
        if known_license in license_lower:
            return known_license
    
    return license_lower


def get_github_license(github_url: str, git_token: Optional[str] = None) -> Optional[str]:
    """
    Fetch license from GitHub repository.
    
    Parameters
    ----------
    github_url : str
        GitHub repository URL
    git_token : Optional[str]
        GitHub token for authenticated requests
        
    Returns
    -------
    Optional[str]
        Normalized license key or None if not found
    """
    try:
        # Extract owner/repo from URL
        parsed = urlparse(github_url)
        parts = [p for p in parsed.path.split('/') if p]
        
        if len(parts) < 2:
            logger.warning(f"Invalid GitHub URL format: {github_url}")
            return None
        
        owner, repo = parts[0], parts[1]
        
        # GitHub API endpoint
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {}
        if git_token:
            headers['Authorization'] = f"token {git_token}"
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            repo_data = response.json()
            license_info = repo_data.get("license")
            
            if license_info and isinstance(license_info, dict):
                license_key = license_info.get("key", "").lower()
                if license_key:
                    normalized = normalize_license_name(license_key)
                    logger.info(f"GitHub license for {owner}/{repo}: {normalized}")
                    return normalized
        
        elif response.status_code == 404:
            logger.warning(f"GitHub repository not found: {github_url}")
            return None
        else:
            logger.warning(f"GitHub API error {response.status_code} for {github_url}")
            return None
    
    except Exception as e:
        logger.error(f"Error fetching GitHub license: {e}")
        return None


def get_huggingface_license(artifact_url: str, hf_token: Optional[str] = None) -> Optional[str]:
    """
    Fetch license from HuggingFace model.
    
    Parameters
    ----------
    artifact_url : str
        HuggingFace model URL
    hf_token : Optional[str]
        HuggingFace token for authenticated requests
        
    Returns
    -------
    Optional[str]
        Normalized license key or None if not found
    """
    try:
        if "huggingface.co" not in artifact_url:
            return None
        
        # Extract model ID from URL
        parsed = urlparse(artifact_url)
        parts = [p for p in parsed.path.split('/') if p]
        
        if len(parts) < 2:
            return None
        
        model_id = f"{parts[-2]}/{parts[-1]}"
        
        # HuggingFace API endpoint
        api_url = f"https://huggingface.co/api/models/{model_id}"
        headers = {}
        if hf_token:
            headers['Authorization'] = f"Bearer {hf_token}"
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            model_data = response.json()
            
            # Try multiple sources for license info
            # 1. Direct license field
            if "license" in model_data and model_data["license"]:
                license_str = model_data["license"]
                if isinstance(license_str, str):
                    normalized = normalize_license_name(license_str)
                    logger.info(f"HF license for {model_id}: {normalized}")
                    return normalized
            
            # 2. Card data
            card_data = model_data.get("cardData", {})
            if isinstance(card_data, dict) and "license" in card_data:
                license_str = card_data["license"]
                if isinstance(license_str, str):
                    normalized = normalize_license_name(license_str)
                    logger.info(f"HF license (from card) for {model_id}: {normalized}")
                    return normalized
            
            # 3. Tags (e.g., "license:apache-2.0")
            tags = model_data.get("tags", [])
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("license:"):
                    license_str = tag.split("license:")[1]
                    normalized = normalize_license_name(license_str)
                    logger.info(f"HF license (from tags) for {model_id}: {normalized}")
                    return normalized
        
        elif response.status_code == 404:
            logger.warning(f"HuggingFace model not found: {artifact_url}")
            return None
        
        logger.info(f"No license found for HuggingFace model: {model_id}")
        return None
    
    except Exception as e:
        logger.error(f"Error fetching HuggingFace license: {e}")
        return None


def check_license_compatibility(github_url: str, artifact_url: str, 
                                git_token: Optional[str] = None,
                                hf_token: Optional[str] = None) -> bool:
    """
    Check if GitHub license is compatible with model license for fine-tuning + inference.
    
    Parameters
    ----------
    github_url : str
        GitHub repository URL
    artifact_url : str
        HuggingFace model URL
    git_token : Optional[str]
        GitHub API token
    hf_token : Optional[str]
        HuggingFace API token
        
    Returns
    -------
    bool
        True if licenses are compatible, False otherwise
    """
    try:
        github_license = get_github_license(github_url, git_token)
        model_license = get_huggingface_license(artifact_url, hf_token)
        
        if not github_license or not model_license:
            logger.warning("Could not determine one or both licenses")
            # Conservative: return False if we can't determine
            return False
        
        # Check compatibility matrix
        compatible_licenses = LICENSE_COMPATIBILITY.get(github_license, set())
        
        is_compatible = model_license in compatible_licenses
        
        logger.info(f"License compatibility check: GitHub({github_license}) + Model({model_license}) = {is_compatible}")
        
        return is_compatible
    
    except Exception as e:
        logger.error(f"License compatibility check failed: {e}")
        return False


def extract_lineage_graph(artifact_url: str, artifact_id: str, 
                          hf_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract lineage graph from model config.json.
    
    Parameters
    ----------
    artifact_url : str
        HuggingFace model URL
    artifact_id : str
        Current artifact ID
    hf_token : Optional[str]
        HuggingFace API token
        
    Returns
    -------
    Dict[str, Any]
        Lineage graph with nodes and edges
    """
    nodes = []
    edges = []
    
    # Add current model as a node
    try:
        parsed = urlparse(artifact_url)
        parts = [p for p in parsed.path.split('/') if p]
        if len(parts) >= 2:
            model_name = parts[-1]
        else:
            model_name = "unknown"
    except Exception:
        model_name = "unknown"
    
    nodes.append({
        "artifact_id": artifact_id,
        "name": model_name,
        "source": "registry"
    })
    
    if "huggingface.co" not in artifact_url:
        # Not a HuggingFace model, return minimal graph
        return {"nodes": nodes, "edges": edges}
    
    try:
        # Extract model ID from URL
        parsed = urlparse(artifact_url)
        parts = [p for p in parsed.path.split('/') if p]
        
        if len(parts) < 2:
            return {"nodes": nodes, "edges": edges}
        
        model_id = f"{parts[-2]}/{parts[-1]}"
        
        # Fetch config.json
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        headers = {}
        if hf_token:
            headers['Authorization'] = f"Bearer {hf_token}"
        
        response = requests.get(config_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.info(f"No config.json found for {model_id}")
            return {"nodes": nodes, "edges": edges}
        
        config = response.json()
        
        # Extract parent model references from config
        parent_fields = {
            "base_model": "base_model",
            "_name_or_path": "name_or_path",
            "model_name_or_path": "model_name_or_path",
            "parent_model": "parent_model"
        }
        
        seen_parents = set()
        
        for field, relationship in parent_fields.items():
            if field in config and config[field]:
                parent_value = str(config[field]).strip()
                
                # Skip local paths
                if parent_value.startswith('./') or parent_value.startswith('/') or parent_value.startswith('\\'):
                    continue
                
                # Skip empty or very short values
                if len(parent_value) < 2:
                    continue
                
                # Clean up HuggingFace URLs
                parent_value = parent_value.replace('https://huggingface.co/', '')
                parent_value = parent_value.strip('/')
                
                # Avoid duplicates
                if parent_value in seen_parents:
                    continue
                
                seen_parents.add(parent_value)
                
                # Create parent node
                # Use consistent ID generation
                parent_id = f"parent_{abs(hash(parent_value)) % 1000000000}"
                
                nodes.append({
                    "artifact_id": parent_id,
                    "name": parent_value,
                    "source": "config_json"
                })
                
                # Create edge from parent to current model
                edges.append({
                    "from_node_artifact_id": parent_id,
                    "to_node_artifact_id": artifact_id,
                    "relationship": relationship
                })
                
                logger.info(f"Found parent model: {parent_value} (via {field})")
        
        logger.info(f"Extracted lineage: {len(nodes)} nodes, {len(edges)} edges")
        
    except Exception as e:
        logger.error(f"Error extracting lineage graph: {e}")
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def calculate_artifact_cost_with_dependencies(artifact_id: str, 
                                              artifacts_store: Dict[str, Dict],
                                              include_dependencies: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Calculate artifact download cost, optionally including dependencies.
    
    Parameters
    ----------
    artifact_id : str
        ID of artifact to calculate cost for
    artifacts_store : Dict[str, Dict]
        Store of all artifacts with their data
    include_dependencies : bool
        Whether to include dependency costs
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Cost information per artifact
    """
    result = {}
    
    artifact = artifacts_store.get(artifact_id)
    if not artifact:
        return result
    
    # Calculate standalone cost for this artifact
    standalone_cost = _calculate_single_artifact_cost(artifact)
    
    if not include_dependencies:
        # Simple case: just return total_cost
        result[artifact_id] = {
            "total_cost": standalone_cost
        }
        return result
    
    # Complex case: include dependencies
    result[artifact_id] = {
        "standalone_cost": standalone_cost,
        "total_cost": standalone_cost  # Will be updated
    }
    
    # Get dependencies from lineage (if stored)
    visited = {artifact_id}
    to_visit = []
    
    # Extract parent artifacts from stored lineage or config
    artifact_url = artifact.get("url", "")
    if "huggingface.co" in artifact_url:
        try:
            # Use extract_lineage_graph to find parents
            lineage = extract_lineage_graph(artifact_url, artifact_id)
            
            # Find parent artifacts in registry
            for edge in lineage.get("edges", []):
                parent_id = edge.get("from_node_artifact_id")
                if parent_id and parent_id != artifact_id:
                    # Try to match parent by name in registry
                    for aid, adata in artifacts_store.items():
                        if aid == artifact_id or aid in visited:
                            continue
                        # Simple name matching (in real system, use proper ID resolution)
                        if adata.get("name") in [n.get("name") for n in lineage.get("nodes", [])]:
                            to_visit.append(aid)
                            break
        except Exception as e:
            logger.debug(f"Could not extract dependencies: {e}")
    
    # Calculate costs for dependencies
    total_cost = standalone_cost
    
    for dep_id in to_visit:
        if dep_id in visited:
            continue
        visited.add(dep_id)
        
        dep_artifact = artifacts_store.get(dep_id)
        if dep_artifact:
            dep_cost = _calculate_single_artifact_cost(dep_artifact)
            
            result[dep_id] = {
                "standalone_cost": dep_cost,
                "total_cost": dep_cost
            }
            
            total_cost += dep_cost
    
    # Update total cost for main artifact
    result[artifact_id]["total_cost"] = total_cost
    
    return result


def _calculate_single_artifact_cost(artifact: Dict) -> float:
    """
    Calculate cost (size in MB) for a single artifact.
    
    Parameters
    ----------
    artifact : Dict
        Artifact data
        
    Returns
    -------
    float
        Size in MB
    """
    # Try multiple sources for size information
    
    # 1. Check if size is stored directly
    if "size_mb" in artifact:
        return float(artifact["size_mb"])
    
    # 2. Check if size_score exists (from metrics)
    metric_results = artifact.get("metric_results", {})
    size_score = metric_results.get("size_score", {})
    
    # Size score from metrics is actually the model size, not a score
    # We need to estimate from the model
    
    # 3. Check S3 object size if available
    s3_key = artifact.get("s3_key")
    if s3_key:
        try:
            import boto3
            import os
            s3_client = boto3.client('s3', region_name='us-east-1')
            bucket = os.getenv('S3_BUCKET', 'tmr-dev-models')
            
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            size_bytes = response.get('ContentLength', 0)
            size_mb = size_bytes / (1024 * 1024)
            return round(size_mb, 2)
        except Exception as e:
            logger.debug(f"Could not get S3 object size: {e}")
    
    # 4. Estimate from URL if it's HuggingFace
    url = artifact.get("url", "")
    if "huggingface.co" in url:
        try:
            # This is a placeholder - in production, query HF API for model size
            # For now, return a reasonable default
            return 500.0  # Default 500 MB
        except Exception:
            pass
    
    # 5. Default fallback
    return 100.0  # Default 100 MB if we can't determine
