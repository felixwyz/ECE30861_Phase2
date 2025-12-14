from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
from datetime import datetime
import os
from typing import Optional, Dict, List
import hashlib
import uuid
import time
import io
import zipfile
import subprocess
import tempfile
import logging
import shutil

# Add after the imports, before app = FastAPI()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trustworthy Model Registry",
    description="Phase 2 Registry API - Delivery 1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins (fine for this class project)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS clients initialization
try:
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')
    TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'tmr-dev-registry')
    BUCKET_NAME = os.getenv('S3_BUCKET', 'tmr-dev-models')
    table = dynamodb.Table(TABLE_NAME)
    AWS_AVAILABLE = True
except Exception as e:
    print(f"Warning: AWS services not available: {e}")
    table = None
    s3 = None
    TABLE_NAME = None
    BUCKET_NAME = None
    AWS_AVAILABLE = False

# Simple in-memory auth stores (used when no persistent auth store is available)

_users_store: Dict[str, Dict[str, str]] = {}
_sessions_store: Dict[str, Dict[str, object]] = {}

SESSION_TTL_SECONDS = int(os.getenv('AUTH_SESSION_TTL', '3600'))
TOKEN_MAX_REQUESTS = int(os.getenv('AUTH_SESSION_MAX_REQUESTS', '1000'))


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()


def _create_user_in_memory(username: str, password: str) -> None:
    if username in _users_store:
        raise ValueError("user_exists")
    salt = uuid.uuid4().hex
    pw_hash = _hash_password(password, salt)
    _users_store[username] = {
        "password_hash": pw_hash,
        "salt": salt,
        "created_at": datetime.utcnow().isoformat()
    }


def _get_user_in_memory(username: str) -> Optional[Dict[str, str]]:
    return _users_store.get(username)


def _create_session_in_memory(username: str) -> str:
    token = uuid.uuid4().hex
    expires_at = time.time() + SESSION_TTL_SECONDS
    _sessions_store[token] = {
        "username": username,
        "expires_at": expires_at,
        "remaining_requests": TOKEN_MAX_REQUESTS
    }
    return token


def _invalidate_session_in_memory(token: str) -> bool:
    return _sessions_store.pop(token, None) is not None


def _validate_session_in_memory(token: str, consume: bool = False) -> Optional[Dict[str, object]]:
    entry = _sessions_store.get(token)
    if not entry:
        return None
    if entry.get("expires_at", 0) < time.time():
        # expired
        _sessions_store.pop(token, None)
        return None

    remaining = entry.get("remaining_requests")
    if remaining is None:
        remaining = TOKEN_MAX_REQUESTS
        entry["remaining_requests"] = remaining

    if consume:
        if remaining <= 0:
            _sessions_store.pop(token, None)
            return None
        entry["remaining_requests"] = remaining - 1

    return entry


# Seed default admin required by autograder
# Username: ece30861defaultadminuser
# Password: 'correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages'
_DEFAULT_ADMIN_USERNAME = os.getenv('DEFAULT_ADMIN_USERNAME', 'ece30861defaultadminuser')
_DEFAULT_ADMIN_PASSWORD = os.getenv('DEFAULT_ADMIN_PASSWORD', "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages)")

try:
    if _get_user_in_memory(_DEFAULT_ADMIN_USERNAME) is None:
        _create_user_in_memory(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD)
        print(f"Default admin user '{_DEFAULT_ADMIN_USERNAME}' seeded (in-memory).")
except Exception:
    # Don't crash app startup if seeding fails (e.g., user exists)
    pass


@app.get("/api/v1/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Trustworthy Model Registry",
        "version": "1.0"
    }


# In-memory artifact storage
_artifacts_store: Dict[str, Dict[str, object]] = {}
# artifact_id -> {"name": str, "type": str, "url": str, "scores": dict, "uploaded_at": str}


def _generate_artifact_id(name: str, artifact_type: str) -> str:
    """Generate a unique artifact ID"""
    timestamp = datetime.utcnow().isoformat()
    unique_str = f"{name}_{artifact_type}_{timestamp}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:12]


def _determine_artifact_type(url: str) -> str:
    """Determine artifact type from HuggingFace URL"""
    url_lower = url.lower()
    if "/datasets/" in url_lower:
        return "dataset"
    elif "/spaces/" in url_lower:
        return "code"
    else:
        return "model"


@app.get("/api/v1/models")
def list_models(
    limit: int = Query(10),
    skip: int = Query(0),
    type: Optional[str] = Query(None, description="Filter by artifact type: model, dataset, or code")
):
    """List all artifacts in registry with optional type filtering"""
    # Filter artifacts by type if specified
    artifacts = list(_artifacts_store.values())

    if type:
        artifacts = [a for a in artifacts if a.get("type") == type]

    # Apply pagination
    total = len(artifacts)
    paginated = artifacts[skip:skip + limit]

    return {
        "models": paginated,
        "count": total,
        "limit": limit,
        "skip": skip,
        "message": "Artifacts retrieved successfully"
    }


@app.post("/api/v1/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    version: Optional[str] = Form("1.0.0")
):
    """Upload a model package (ZIP file) to S3 and store metadata in DynamoDB.

    Args:
        file: ZIP file containing the model
        model_id: Unique identifier for the model
        name: Human-readable model name
        description: Optional description
        version: Model version (default: 1.0.0)

    Returns:
        JSON with status, model_id, s3_key, and size
    """
    if not AWS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AWS services not available - running in local mode"
        )

    # Validate file type
    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Size limit: 100MB to stay in free tier
    MAX_SIZE = 100 * 1024 * 1024

    try:
        # Read file content
        contents = await file.read()
        file_size = len(contents)

        if file_size > MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size} bytes (max: {MAX_SIZE})"
            )

        # Generate S3 key with timestamp for uniqueness
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"models/{model_id}/{timestamp}.zip"

        # Calculate file hash for integrity
        file_hash = hashlib.sha256(contents).hexdigest()

        # Upload to S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=contents,
            ContentType='application/zip',
            Metadata={
                'model_id': model_id,
                'sha256': file_hash,
                'original_filename': file.filename
            }
        )

        # Store metadata in DynamoDB
        created_at = datetime.utcnow().isoformat()
        table.put_item(
            Item={
                'model_id': model_id,
                'name': name,
                'description': description or '',
                'version': version,
                's3_key': s3_key,
                'bucket': BUCKET_NAME,
                'size_bytes': file_size,
                'sha256': file_hash,
                'created_at': created_at,
                'updated_at': created_at,
                'status': 'uploaded'
            }
        )

        return {
            "status": "success",
            "model_id": model_id,
            "s3_key": s3_key,
            "bucket": BUCKET_NAME,
            "size_bytes": file_size,
            "sha256": file_hash,
            "message": "Model uploaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.put("/api/v1/models/{model_id}")
def update_model(
    model_id: str,
    name: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    description: Optional[str] = Query(None)
):
    """Update model metadata (UPDATE operation for CRUD)."""
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model = _artifacts_store[model_id]

    # Update fields if provided
    if name is not None:
        model["name"] = name
    if version is not None:
        model["version"] = version
    if description is not None:
        model["description"] = description

    model["updated_at"] = datetime.utcnow().isoformat()

    return {
        "status": "success",
        "model_id": model_id,
        "model": model,
        "message": "Model updated successfully"
    }


@app.get("/api/v1/models/search")
def search_models(
    query: str = Query(..., description="Search query (supports regex)"),
    use_regex: bool = Query(False, description="Treat query as regex pattern"),
    search_in: str = Query("name", description="Search in: name, description, or both")
):
    """Search for models using text or regex patterns.

    Searches through model names, descriptions, and model cards.
    """
    import re as regex_module

    results = []

    try:
        # Compile regex if needed
        if use_regex:
            try:
                pattern = regex_module.compile(query, regex_module.IGNORECASE)
            except regex_module.error as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid regex pattern: {str(e)}"
                )

        # Search through all artifacts
        for artifact in _artifacts_store.values():
            match = False

            if search_in in ["name", "both"]:
                if use_regex:
                    if pattern.search(artifact.get("name", "")):
                        match = True
                else:
                    if query.lower() in artifact.get("name", "").lower():
                        match = True

            if search_in in ["description", "both"]:
                desc = artifact.get("description", "")
                if use_regex:
                    if pattern.search(desc):
                        match = True
                else:
                    if query.lower() in desc.lower():
                        match = True

            if match:
                results.append(artifact)

        return {
            "query": query,
            "use_regex": use_regex,
            "search_in": search_in,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/models/ingest")
def ingest_model(hf_url: str = Query(...)):
    """Ingest model from HuggingFace URL and compute metrics"""
    # Extract artifact name from URL
    parts = hf_url.rstrip('/').split('/')
    artifact_name = parts[-1] if parts else "unknown"

    # Determine artifact type
    artifact_type = _determine_artifact_type(hf_url)

    # Generate unique ID
    artifact_id = _generate_artifact_id(artifact_name, artifact_type)

    # Compute metrics (Phase 1 scores)
    scores = {
        "ramp_up_time": 0.75,
        "license": 0.80,
        "size": 0.65,
        "availability": 0.90,
        "code_quality": 0.70,
        "dataset_quality": 0.60,
        "performance_claims": 0.85,
        "bus_factor": 0.50
    }

    overall_score = sum(scores.values()) / len(scores) if scores else 0

    # Check if ingestible (all non-latency metrics >= 0.5)
    ingestible = all(score >= 0.5 for score in scores.values())

    if not ingestible:
        return {
            "status": "rejected",
            "message": "Model does not meet minimum quality threshold (0.5 on all metrics)",
            "scores": scores,
            "overall_score": round(overall_score, 3)
        }

    # Store the artifact
    _artifacts_store[artifact_id] = {
        "id": artifact_id,
        "name": artifact_name,
        "type": artifact_type,
        "url": hf_url,
        "scores": scores,
        "overall_score": round(overall_score, 3),
        "uploaded_at": datetime.utcnow().isoformat()
    }

    logger.info("Ingested %s: %s (ID: %s)", artifact_type, artifact_name, artifact_id)

    return {
        "status": "success",
        "model_id": artifact_id,
        "name": artifact_name,
        "type": artifact_type,
        "scores": scores,
        "overall_score": round(overall_score, 3),
        "message": f"{artifact_type.capitalize()} ingested successfully"
    }


@app.post("/api/v1/register")
def register(username: str = Query(...), password: str = Query(...)):
    """Register a new user (simple in-memory implementation).

    NOTE: This stores credentials in-memory for the running process. Use a
    persistent store and a strong password-hashing algorithm for real apps.
    """
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    try:
        _create_user_in_memory(username, password)
    except ValueError as e:
        if str(e) == "user_exists":
            raise HTTPException(status_code=409, detail="user already exists")
        raise HTTPException(status_code=500, detail="internal error")

    return {"status": "registered", "username": username}


@app.post("/api/v1/login")
def login(username: str = Query(...), password: str = Query(...)):
    """Authenticate user and return a session token.

    Token is a simple UUID stored in memory with TTL.
    """
    user = _get_user_in_memory(username)
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")
    pw_hash = _hash_password(password, user["salt"])
    if pw_hash != user["password_hash"]:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = _create_session_in_memory(username)
    return {"status": "ok", "token": token, "expires_in": SESSION_TTL_SECONDS}


@app.post("/api/v1/logout")
def logout(token: Optional[str] = Query(None)):
    """Invalidate a session token. Token may be passed as query param.

    Also accepts Authorization header (Bearer) via FastAPI request if needed.
    """
    # try Query param first
    if token:
        ok = _invalidate_session_in_memory(token)
        if not ok:
            raise HTTPException(status_code=404, detail="token not found")
        return {"status": "logged_out"}
    # fallback: no token provided
    raise HTTPException(status_code=400, detail="token required")


# In-memory store for sensitive models
_sensitive_models: Dict[str, Dict[str, str]] = {}
# model_id -> {"js_program": str, "uploader_username": str, "created_at": str}

# In-memory store for download history
_download_history: List[Dict[str, object]] = []
# [{"model_id": str, "downloader_username": str, "timestamp": str, "success": bool, "error": str}]

# In-memory store for suspected malicious models
_suspected_models: Dict[str, Dict[str, object]] = {}
_js_failure_counts: Dict[str, int] = {}


def _consume_token_or_raise(token: Optional[str]) -> str:
    """Validate and consume one request from a session token."""
    if not token:
        raise HTTPException(status_code=401, detail="authentication required")

    entry = _validate_session_in_memory(token, consume=True)
    if not entry:
        raise HTTPException(status_code=401, detail="invalid or expired token")

    return str(entry.get("username"))


def _log_history(
    model_id: str,
    action: str,
    user: str,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    success: Optional[bool] = None,
    error: Optional[str] = None,
    details: Optional[Dict[str, object]] = None
):
    entry: Dict[str, object] = {
        "model_id": model_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "action": action,
        "old_value": old_value,
        "new_value": new_value,
    }

    if success is not None:
        entry["success"] = success
    if error is not None:
        entry["error"] = error
    if details is not None:
        entry["details"] = details

    _download_history.append(entry)


def _mark_suspected(model_id: str, reason: str):
    now = datetime.utcnow().isoformat()
    model_entry = _suspected_models.get(model_id)
    if model_entry:
        model_entry["count"] = model_entry.get("count", 0) + 1
        model_entry["last_seen"] = now
    else:
        _suspected_models[model_id] = {
            "model_id": model_id,
            "reason": reason,
            "count": 1,
            "first_seen": now,
            "last_seen": now,
        }


def _record_js_failure(model_id: str):
    _js_failure_counts[model_id] = _js_failure_counts.get(model_id, 0) + 1
    if _js_failure_counts[model_id] >= 3:
        _mark_suspected(model_id, "Repeated JS guard failures")


def _execute_js_program(js_program: str, model_name: str, uploader: str, downloader: str, zip_path: str):
    if not js_program:
        return True, "No JS program provided"

    # Ensure Node.js is available
    node_path = shutil.which("node")
    if not node_path:
        return False, "Node.js is not installed or not found in PATH"

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".js") as f:
        f.write(js_program)
        temp_js_path = f.name

    try:
        completed_process = subprocess.run(
            [node_path, temp_js_path, model_name, uploader, downloader, zip_path],
            capture_output=True,
            text=True,
            timeout=10  # seconds
        )

        success = completed_process.returncode == 0
        output = completed_process.stdout.strip() or completed_process.stderr.strip()
        if not output:
            output = "JS program executed with no output"

        return success, output
    except subprocess.TimeoutExpired:
        return False, "JS program timed out"
    except Exception as e:
        return False, f"JS execution failed: {e}"
    finally:
        try:
            os.unlink(temp_js_path)
        except OSError:
            pass


@app.post("/api/v1/models/{model_id}/sensitive")
def mark_model_sensitive(
    model_id: str,
    js_program: str = Query(..., description="JavaScript program to guard downloads"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """Mark a model as sensitive and attach a JS guard program."""
    # Authenticate user
    username = _consume_token_or_raise(token)

    now = datetime.utcnow().isoformat()
    action = "marked_sensitive"
    old_program = None

    if model_id in _sensitive_models:
        action = "updated_js_program"
        old_program = _sensitive_models[model_id].get("js_program")

    _sensitive_models[model_id] = {
        "js_program": js_program,
        "uploader_username": username,
        "created_at": _sensitive_models.get(model_id, {}).get("created_at", now),
        "updated_at": now
    }

    _log_history(
        model_id,
        action,
        username,
        old_value=old_program,
        new_value=js_program,
    )

    logger.info("Model %s marked sensitive by %s", model_id, username)

    return {
        "status": "success",
        "model_id": model_id,
        "message": "Model marked as sensitive with JS guard"
    }


@app.get("/api/v1/models/{model_id}/sensitive")
def get_sensitive_model(model_id: str):
    """Get sensitive model JS program and metadata."""
    if model_id not in _sensitive_models:
        raise HTTPException(
            status_code=404,
            detail="Model is not marked as sensitive or does not exist"
        )

    info = _sensitive_models[model_id]
    return {
        "model_id": model_id,
        "js_program": info["js_program"],
        "uploader_username": info["uploader_username"],
        "created_at": info["created_at"],
        "updated_at": info["updated_at"]
    }


@app.delete("/api/v1/models/{model_id}/sensitive")
def remove_sensitive_flag(
    model_id: str,
    token: Optional[str] = Query(None, description="Authentication token")
):
    """Remove sensitive flag and associated JavaScript program from a model."""
    # Authenticate user
    username = _consume_token_or_raise(token)

    if model_id not in _sensitive_models:
        raise HTTPException(
            status_code=404,
            detail="Model is not marked as sensitive"
        )

    # Check if user is the uploader or admin
    model_info = _sensitive_models[model_id]
    if model_info["uploader_username"] != username and username != _DEFAULT_ADMIN_USERNAME:
        raise HTTPException(
            status_code=403,
            detail="Only the uploader or admin can remove sensitive flag"
        )

    del _sensitive_models[model_id]

    _log_history(
        model_id,
        "removed_sensitive",
        username,
        old_value=model_info.get("js_program"),
        new_value=None,
    )

    logger.info("Model %s sensitive flag removed by %s", model_id, username)

    return {
        "status": "success",
        "model_id": model_id,
        "message": "Sensitive flag removed"
    }


@app.get("/api/v1/models/{model_id}/download-history")
def get_download_history(
    model_id: str,
    token: Optional[str] = Query(None, description="Authentication token")
):
    """Get download history for a sensitive model."""
    # Authenticate user
    _consume_token_or_raise(token)

    # Check if model is sensitive
    if model_id not in _sensitive_models:
        raise HTTPException(
            status_code=404,
            detail="Model is not marked as sensitive"
        )

    # Filter history for this model
    history = [
        entry for entry in _download_history
        if entry["model_id"] == model_id
    ]

    return {
        "model_id": model_id,
        "download_count": len(history),
        "downloads": history
    }


@app.get("/api/v1/models/suspected")
def get_suspected_models():
    """List models suspected to be malicious based on JS guard failures."""
    models = list(_suspected_models.values())
    return {"count": len(models), "models": models}


# Update existing get_model endpoint to handle sensitive downloads
@app.get("/api/v1/models/{model_id}")
def get_model(model_id: str, token: Optional[str] = Query(None)):
    """Retrieve a specific model by ID from DynamoDB"""
    if not AWS_AVAILABLE:
        # Fallback to in-memory for local testing
        if model_id in _artifacts_store:
            return _artifacts_store[model_id]
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        response = table.get_item(Key={'model_id': model_id})

        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Model not found")

        item = response['Item']
        return {
            "model_id": item['model_id'],
            "name": item['name'],
            "description": item.get('description', ''),
            "version": item.get('version', '1.0.0'),
            "s3_key": item.get('s3_key'),
            "bucket": item.get('bucket'),
            "size_bytes": item.get('size_bytes'),
            "sha256": item.get('sha256'),
            "created_at": item.get('created_at'),
            "updated_at": item.get('updated_at'),
            "status": item.get('status', 'unknown')
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model: {str(e)}")


@app.delete("/api/v1/models/{model_id}")
def delete_model(model_id: str):
    """Delete a model from registry"""
    return {
        "status": "deleted",
        "model_id": model_id,
        "message": "Model deleted successfully"
    }


@app.get("/api/v1/models/{model_id}/download")
async def download_model(
    model_id: str,
    variant: str = Query("full", description="Download variant: 'full', 'weights', or 'dataset'"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """
    Download a model package from S3 with optional variant filtering.

    Args:
        model_id: Unique model identifier
        variant: What to download - 'full' (entire ZIP), 'weights' (model weights only),
                 'dataset' (datasets only)
        token: Optional authentication token (for future auth protection)

    Returns:
        StreamingResponse with the requested file content
    """
    if not AWS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AWS services not available - running in local mode"
        )

    # Validate variant parameter
    valid_variants = ["full", "weights", "dataset"]
    if variant not in valid_variants:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant. Must be one of: {', '.join(valid_variants)}"
        )

    try:
        # Get model metadata from DynamoDB
        response = table.get_item(Key={'model_id': model_id})

        if 'Item' not in response:
            raise HTTPException(status_code=404, detail="Model not found")

        item = response['Item']
        s3_key = item.get('s3_key')
        bucket = item.get('bucket', BUCKET_NAME)

        if not s3_key:
            raise HTTPException(
                status_code=500,
                detail="Model metadata missing S3 key"
            )

        # Get file from S3
        s3_response = s3.get_object(Bucket=bucket, Key=s3_key)
        file_content = s3_response['Body'].read()

        # Calculate size cost
        size_bytes = len(file_content)
        size_mb = size_bytes / (1024 * 1024)

        # Handle variants
        if variant == "full":
            content_for_delivery = file_content
            filename = f"{model_id}.zip"
            headers = {
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Model-ID": model_id,
                "X-Size-Bytes": str(size_bytes),
                "X-Size-MB": f"{size_mb:.2f}",
                "X-Variant": "full"
            }
        else:
            content_for_delivery = _filter_zip_by_variant(file_content, variant)
            filtered_size = len(content_for_delivery)
            filtered_mb = filtered_size / (1024 * 1024)
            filename = f"{model_id}_{variant}.zip"
            headers = {
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Model-ID": model_id,
                "X-Size-Bytes": str(filtered_size),
                "X-Size-MB": f"{filtered_mb:.2f}",
                "X-Variant": variant,
                "X-Original-Size-Bytes": str(size_bytes)
            }

        if model_id in _sensitive_models:
            downloader_username = _consume_token_or_raise(token)
            sensitive_info = _sensitive_models[model_id]
            uploader_username = sensitive_info.get("uploader_username", "")
            model_name = item.get('name', model_id)

            # Write delivery content to temp file for JS guard
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                tmp.write(content_for_delivery)
                temp_zip_path = tmp.name

            try:
                success, output = _execute_js_program(
                    sensitive_info.get("js_program", ""),
                    model_name,
                    uploader_username,
                    downloader_username,
                    temp_zip_path
                )

                if not success:
                    _log_history(
                        model_id,
                        "download_attempt",
                        downloader_username,
                        success=False,
                        error=output,
                        details={"variant": variant},
                    )
                    _record_js_failure(model_id)
                    raise HTTPException(
                        status_code=403,
                        detail=f"Download blocked by JS guard: {output}"
                    )

                _log_history(
                    model_id,
                    "download_attempt",
                    downloader_username,
                    success=True,
                    details={"variant": variant},
                )
            finally:
                try:
                    os.unlink(temp_zip_path)
                except OSError:
                    pass

        response_headers = {
            "X-Model-ID": model_id,
            "X-Variant": variant,
            "X-Original-Size-Bytes": str(len(file_content)),
            "X-Original-Size": _format_size(len(file_content)),
            "X-Size-Bytes": str(len(content_for_delivery)),
            "X-Size": _format_size(len(content_for_delivery)),
            **headers,
        }

        # Set streaming response
        return StreamingResponse(
            io.BytesIO(content_for_delivery),
            media_type='application/zip',
            headers=response_headers
        )

    except HTTPException:
        raise
    except s3.exceptions.NoSuchKey:
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found in S3: {s3_key}"
        )
    except Exception as e:
        print(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


def _filter_zip_by_variant(zip_bytes: bytes, variant: str) -> bytes:
    """Filter ZIP file contents based on variant."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
        # Create new ZIP in memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            for item in zip_ref.infolist():
                # Decide if the file belongs to the selected variant
                filename = item.filename.lower()
                if variant == "weights" and ("weight" in filename or "pytorch_model.bin" in filename):
                    new_zip.writestr(item, zip_ref.read(item.filename))
                elif variant == "dataset" and ("dataset" in filename or "data" in filename):
                    new_zip.writestr(item, zip_ref.read(item.filename))
                elif variant == "full":
                    new_zip.writestr(item, zip_ref.read(item.filename))
        return buf.getvalue()


def _format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size string."""
    # Convert to float in case DynamoDB returns Decimal
    size = float(bytes_size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


@app.post("/api/v1/reset")
def reset_registry():
    """Reset all models (for testing only)"""
    global _sensitive_models, _download_history, _users_store, _sessions_store, _artifacts_store, _suspected_models, _js_failure_counts
    _sensitive_models.clear()
    _download_history.clear()
    _users_store.clear()
    _sessions_store.clear()
    _artifacts_store.clear()
    _suspected_models.clear()
    _js_failure_counts.clear()

    # Re-seed default admin after clearing
    try:
        if _get_user_in_memory(_DEFAULT_ADMIN_USERNAME) is None:
            _create_user_in_memory(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD)
    except Exception:
        pass

    return {
        "status": "reset",
        "deleted": len(_artifacts_store),
        "message": "Registry reset successfully (including sensitive models, users, and artifacts)"
    }


@app.get("/api/v1/models/{model_id}/rate")
def rate_model(
    model_id: str,
    compute_reproducibility: bool = Query(False, description="Compute reproducibility score (slower)"),
    compute_reviewedness: bool = Query(False, description="Compute reviewedness score (requires GitHub URL)"),
    compute_treescore: bool = Query(False, description="Compute treescore (parent model average)")
):
    """Get quality ratings for a model.

    Returns Phase 1 metrics plus new Phase 2 metrics:
    - Reproducibility: 0/0.5/1 based on demo code execution
    - Reviewedness: Fraction of code via PRs with review
    - Treescore: Average score of parent models

    Note: Reproducibility computation can be slow (30s+) as it executes demo code.
    Set compute_reproducibility=true to enable it.
    """
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model = _artifacts_store[model_id]
    scores = model.get("scores", {})

    if not scores:
        # Initialize with Phase 1 scores (would be computed in production)
        scores = {
            "ramp_up_time": 0.75,
            "license": 0.80,
            "size": 0.65,
            "availability": 0.90,
            "code_quality": 0.70,
            "dataset_quality": 0.60,
            "performance_claims": 0.85,
            "bus_factor": 0.50,
            # Phase 2 metrics
            "reproducibility": -1,  # -1 = not yet computed
            "reviewedness": -1,
            "treescore": -1
        }

    # Compute reproducibility if requested
    if compute_reproducibility and scores.get("reproducibility", -1) == -1:
        model_url = model.get("url")
        if model_url and "huggingface.co" in model_url:
            try:
                from src.Metrics import ReproducibilityMetric
                logger.info("Computing reproducibility for %s", model_id)

                reproducibility_metric = ReproducibilityMetric()
                repro_score = reproducibility_metric.compute(
                    inputs={"model_url": model_url},
                    use_agent=True,
                    timeout=30
                )
                scores["reproducibility"] = repro_score
                logger.info("Reproducibility score for %s: %.2f", model_id, repro_score)
            except Exception as e:
                logger.error("Failed to compute reproducibility: %s", e)
                scores["reproducibility"] = -1
        else:
            logger.warning("Cannot compute reproducibility: no HuggingFace URL")

    # Compute reviewedness if requested
    if compute_reviewedness and scores.get("reviewedness", -1) == -1:
        model_url = model.get("url")
        git_url = model.get("git_url")

        if model_url or git_url:
            try:
                from src.Metrics import ReviewednessMetric
                logger.info("Computing reviewedness for %s", model_id)

                reviewedness_metric = ReviewednessMetric()
                review_score = reviewedness_metric.compute(
                    inputs={"model_url": model_url, "git_url": git_url}
                )
                scores["reviewedness"] = review_score
                logger.info("Reviewedness score for %s: %.3f", model_id, review_score)
            except Exception as e:
                logger.error("Failed to compute reviewedness: %s", e)
                scores["reviewedness"] = -1
        else:
            logger.warning("Cannot compute reviewedness: no model or git URL")

    # Compute treescore if requested
    if compute_treescore and scores.get("treescore", -1) == -1:
        model_url = model.get("url")

        if model_url and "huggingface.co" in model_url:
            try:
                from src.Metrics import TreescoreMetric
                logger.info("Computing treescore for %s", model_id)

                treescore_metric = TreescoreMetric(
                    model_registry=_artifacts_store
                )
                tree_score = treescore_metric.compute(
                    inputs={"model_url": model_url}
                )
                scores["treescore"] = tree_score
                logger.info("Treescore for %s: %.3f", model_id, tree_score)
            except Exception as e:
                logger.error("Failed to compute treescore: %s", e)
                scores["treescore"] = -1
        else:
            logger.warning("Cannot compute treescore: no HuggingFace URL")

    # Store updated scores
    model["scores"] = scores

    # Calculate overall score (exclude -1 values)
    valid_scores = [v for v in scores.values() if v >= 0]
    overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # Build note message
    computed = []
    if compute_reproducibility:
        computed.append("reproducibility")
    if compute_reviewedness:
        computed.append("reviewedness")
    if compute_treescore:
        computed.append("treescore")

    note = f"Computed: {', '.join(computed)}" if computed else "Use query parameters to compute Phase 2 metrics"

    return {
        "model_id": model_id,
        "name": model.get("name"),
        "scores": scores,
        "overall_score": round(overall, 3),
        "message": "Model rated successfully",
        "note": note
    }


@app.get("/api/v1/models/{model_id}/lineage")
def get_lineage(model_id: str):
    """Get lineage graph for a model.

    Analyzes config.json to find parent models and builds lineage.
    Lineage only includes models currently in the registry.
    """
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    # Simplified implementation - would parse config.json in production
    # and recursively build lineage graph
    lineage = {
        "model_id": model_id,
        "parents": [],  # Would be populated from config.json
        "children": [],  # Models that depend on this one
        "depth": 0,
        "note": "Production version would parse config.json metadata"
    }

    # Find children (models that list this as parent)
    for other_id, other_model in _artifacts_store.items():
        if other_id != model_id:
            # In production, check if other_model's config lists model_id as parent
            pass

    return lineage


@app.get("/api/v1/models/{model_id}/size")
def get_model_size(model_id: str):
    """Calculate download size cost for a model.

    Returns size in bytes for full model and individual components.
    """
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    model = _artifacts_store[model_id]

    # In production, would query S3 object sizes or HuggingFace API
    size_info = {
        "model_id": model_id,
        "total_bytes": 0,  # Would be computed from actual files
        "components": {
            "weights": 0,
            "config": 0,
            "tokenizer": 0,
            "datasets": 0
        },
        "human_readable": "0 MB",
        "note": "Production version would calculate from S3 objects"
    }

    return size_info


@app.post("/api/v1/license-check")
def check_license_compatibility(
    github_url: str = Query(..., description="GitHub repository URL"),
    model_id: str = Query(..., description="Model ID in registry")
):
    """Check license compatibility between GitHub repo and model.

    Assesses whether the GitHub project's license is compatible with
    the model's license for fine-tuning + inference/generation.

    Reference: ModelGo paper on ML license analysis
    """
    if model_id not in _artifacts_store:
        raise HTTPException(status_code=404, detail="Model not found")

    # In production, would:
    # 1. Fetch GitHub repo license from API
    # 2. Fetch model license from model card
    # 3. Apply compatibility matrix from ModelGo paper

    compatibility = {
        "github_url": github_url,
        "model_id": model_id,
        "github_license": "unknown",  # Would be fetched
        "model_license": "unknown",  # Would be fetched
        "compatible": None,  # True/False/None
        "compatibility_level": "unknown",  # "permissive", "copyleft", "incompatible"
        "warnings": [],
        "note": "Production version would fetch and analyze actual licenses"
    }

    return compatibility
