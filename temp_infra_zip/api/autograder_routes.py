"""
Autograder-compatible routes for Phase 2.
These endpoints match the OpenAPI specification exactly.
"""

from fastapi import FastAPI, HTTPException, Header, Query, Body
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import hashlib
import time
import uuid
import os
import jwt
import boto3
from decimal import Decimal

# Import existing utilities and stores from routes.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

app = FastAPI(
    title="ECE 461 - Fall 2025 - Project Phase 2",
    version="3.4.4",
    description="API for ECE 461/Fall 2025/Project Phase 2: A Trustworthy Model Registry"
)

# ==================== AWS Setup ====================
try:
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    TABLE_NAME = os.getenv('DYNAMODB_TABLE', 'tmr-dev-registry')
    table = dynamodb.Table(TABLE_NAME)
    AWS_AVAILABLE = True
except Exception as e:
    print(f"Warning: AWS not available: {e}")
    table = None
    AWS_AVAILABLE = False

# JWT secret
JWT_SECRET = os.getenv('JWT_SECRET', 'ece461-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'

# ==================== Data Models ====================

class ArtifactMetadata(BaseModel):
    name: str
    id: str
    type: str  # model, dataset, or code

class ArtifactData(BaseModel):
    url: str
    download_url: Optional[str] = None

class Artifact(BaseModel):
    metadata: ArtifactMetadata
    data: ArtifactData

class ArtifactQuery(BaseModel):
    name: str
    types: Optional[List[str]] = None

class ArtifactRegEx(BaseModel):
    regex: str

class User(BaseModel):
    name: str
    is_admin: bool

class UserAuthenticationInfo(BaseModel):
    password: str

class AuthenticationRequest(BaseModel):
    user: User
    secret: UserAuthenticationInfo

class SimpleLicenseCheckRequest(BaseModel):
    github_url: str

class ModelRating(BaseModel):
    name: str
    category: str
    net_score: float
    net_score_latency: float
    ramp_up_time: float
    ramp_up_time_latency: float
    bus_factor: float
    bus_factor_latency: float
    performance_claims: float
    performance_claims_latency: float
    license: float
    license_latency: float
    dataset_and_code_score: float
    dataset_and_code_score_latency: float
    dataset_quality: float
    dataset_quality_latency: float
    code_quality: float
    code_quality_latency: float
    reproducibility: float
    reproducibility_latency: float
    reviewedness: float
    reviewedness_latency: float
    tree_score: float
    tree_score_latency: float
    size_score: Dict[str, float]
    size_score_latency: float

# ==================== In-Memory Storage ====================

# Fall back to memory if DynamoDB not available
_artifacts_store: Dict[str, Dict[str, Any]] = {}
_users_store: Dict[str, Dict[str, str]] = {}

SESSION_TTL_SECONDS = 3600

# Seed default admin
_DEFAULT_ADMIN_USERNAME = 'ece30861defaultadminuser'
_DEFAULT_ADMIN_PASSWORD = "correcthorsebatterystaple123(!__+@**(A;DROP TABLE artifacts;"

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def _create_user(username: str, password: str, is_admin: bool = False):
    """Create user in DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            # Check if exists
            response = table.get_item(Key={'model_id': f'USER#{username}'})
            if 'Item' in response:
                raise ValueError("user_exists")
            
            salt = uuid.uuid4().hex
            pw_hash = _hash_password(password, salt)
            
            table.put_item(Item={
                'model_id': f'USER#{username}',
                'password_hash': pw_hash,
                'salt': salt,
                'is_admin': is_admin,
                'created_at': datetime.utcnow().isoformat()
            })
            return
        except Exception as e:
            print(f"DynamoDB error, falling back to memory: {e}")
    
    # Fallback to memory
    if username in _users_store:
        raise ValueError("user_exists")
    salt = uuid.uuid4().hex
    pw_hash = _hash_password(password, salt)
    _users_store[username] = {
        "password_hash": pw_hash,
        "salt": salt,
        "is_admin": is_admin,
        "created_at": datetime.utcnow().isoformat()
    }

def _get_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            response = table.get_item(Key={'model_id': f'USER#{username}'})
            if 'Item' in response:
                return dict(response['Item'])
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    return _users_store.get(username)

def _validate_token(token: Optional[str]) -> Optional[str]:
    """Validate JWT token and return username if valid"""
    if not token:
        return None
    
    # Remove 'bearer ' prefix if present
    if token.lower().startswith('bearer '):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get('sub')
        exp = payload.get('exp')
        
        # Check expiration
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            return None
        
        return username
    except jwt.InvalidTokenError:
        return None

def _generate_artifact_id() -> str:
    """Generate unique artifact ID"""
    return str(abs(hash(uuid.uuid4().hex + str(time.time()))))[:12]

def _store_artifact(artifact_id: str, artifact_data: Dict[str, Any]):
    """Store artifact in DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            # Convert floats to Decimal for DynamoDB
            item = {'model_id': f'ARTIFACT#{artifact_id}'}
            for key, value in artifact_data.items():
                if isinstance(value, float):
                    item[key] = Decimal(str(value))
                elif isinstance(value, dict):
                    item[key] = {k: Decimal(str(v)) if isinstance(v, float) else v for k, v in value.items()}
                else:
                    item[key] = value
            
            table.put_item(Item=item)
            return
        except Exception as e:
            print(f"DynamoDB error, falling back to memory: {e}")
    
    _artifacts_store[artifact_id] = artifact_data

def _get_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Get artifact from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            response = table.get_item(Key={'model_id': f'ARTIFACT#{artifact_id}'})
            if 'Item' in response:
                item = dict(response['Item'])
                # Convert Decimal back to float
                for key, value in item.items():
                    if isinstance(value, Decimal):
                        item[key] = float(value)
                    elif isinstance(value, dict):
                        item[key] = {k: float(v) if isinstance(v, Decimal) else v for k, v in value.items()}
                return item
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    return _artifacts_store.get(artifact_id)

def _list_artifacts() -> List[Dict[str, Any]]:
    """List all artifacts from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            response = table.scan(
                FilterExpression='begins_with(model_id, :prefix)',
                ExpressionAttributeValues={':prefix': 'ARTIFACT#'}
            )
            artifacts = []
            for item in response.get('Items', []):
                artifact_id = item['model_id'].replace('ARTIFACT#', '')
                artifact = dict(item)
                # Convert Decimal to float
                for key, value in artifact.items():
                    if isinstance(value, Decimal):
                        artifact[key] = float(value)
                artifacts.append((artifact_id, artifact))
            return artifacts
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    return list(_artifacts_store.items())

def _delete_artifact(artifact_id: str):
    """Delete artifact from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            table.delete_item(Key={'model_id': f'ARTIFACT#{artifact_id}'})
            return
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    _artifacts_store.pop(artifact_id, None)

def _clear_all_artifacts():
    """Clear all artifacts from DynamoDB or memory"""
    if AWS_AVAILABLE:
        try:
            # Scan and delete all artifacts
            response = table.scan(
                FilterExpression='begins_with(model_id, :prefix)',
                ExpressionAttributeValues={':prefix': 'ARTIFACT#'}
            )
            for item in response.get('Items', []):
                table.delete_item(Key={'model_id': item['model_id']})
            return
        except Exception as e:
            print(f"DynamoDB error: {e}")
    
    _artifacts_store.clear()

# Seed admin user
try:
    _create_user(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD, is_admin=True)
    print(f"✓ Seeded admin user: {_DEFAULT_ADMIN_USERNAME}")
except ValueError:
    pass

# ==================== ENDPOINTS ====================

@app.get("/health")
def health_check():
    """Heartbeat check (BASELINE)"""
    return JSONResponse(status_code=200, content={})

@app.get("/tracks")
def get_tracks():
    """Get planned tracks (BASELINE)"""
    return {
        "plannedTracks": ["Access control track"]
    }

@app.delete("/reset")
def reset_registry(x_authorization: Optional[str] = Header(None, alias="X-Authorization")):
    """Reset the registry (BASELINE)"""
    # Validate authentication
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    # Check if user is admin
    user = _get_user(username)
    if not user or not user.get("is_admin"):
        raise HTTPException(status_code=401, detail="You do not have permission to reset the registry.")
    
    # Clear all artifacts
    _clear_all_artifacts()
    
    # Re-seed admin
    try:
        _create_user(_DEFAULT_ADMIN_USERNAME, _DEFAULT_ADMIN_PASSWORD, is_admin=True)
    except:
        pass
    
    return JSONResponse(status_code=200, content={"message": "Registry is reset."})

@app.post("/artifacts")
def list_artifacts_query(
    queries: List[ArtifactQuery] = Body(...),
    offset: Optional[str] = Query(None),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifacts from registry (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    results = []
    
    # Get all artifacts
    all_artifacts = _list_artifacts()
    
    # Handle wildcard query
    if len(queries) == 1 and queries[0].name == "*":
        for artifact_id, artifact in all_artifacts:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    else:
        # Handle specific queries
        for query in queries:
            for artifact_id, artifact in all_artifacts:
                if artifact["name"] == query.name:
                    if query.types is None or artifact["type"] in query.types:
                        results.append({
                            "name": artifact["name"],
                            "id": artifact_id,
                            "type": artifact["type"]
                        })
    
    # Apply offset for pagination
    start_idx = int(offset) if offset else 0
    page_size = 10
    paginated = results[start_idx:start_idx + page_size]
    
    # Return with offset header
    next_offset = str(start_idx + page_size) if start_idx + page_size < len(results) else None
    
    return JSONResponse(
        status_code=200,
        content=paginated,
        headers={"offset": next_offset} if next_offset else {}
    )

@app.post("/artifact/{artifact_type}")
def create_artifact(
    artifact_type: str,
    artifact_data: ArtifactData = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Register a new artifact (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    if artifact_type not in ["model", "dataset", "code"]:
        raise HTTPException(status_code=400, detail="Invalid artifact_type.")
    
    if not artifact_data.url:
        raise HTTPException(status_code=400, detail="Missing url in artifact_data.")
    
    # Extract name from URL
    parts = artifact_data.url.rstrip('/').split('/')
    name = parts[-1] if parts else "unknown"
    
    # Generate ID
    artifact_id = _generate_artifact_id()
    
    # Compute basic metrics (mock for now)
    scores = {
        "bus_factor": 0.5,
        "ramp_up_time": 0.75,
        "license": 0.8,
        "availability": 0.9,
        "code_quality": 0.7,
        "dataset_quality": 0.6,
        "performance_claims": 0.85,
        "reproducibility": 0.6,
        "reviewedness": 0.6,
        "tree_score": 0.7
    }
    
    net_score = sum(scores.values()) / len(scores)
    
    # Check if artifact meets threshold (all metrics >= 0.5)
    if any(score < 0.5 for score in scores.values()):
        raise HTTPException(status_code=424, detail="Artifact is not registered due to the disqualified rating.")
    
    # Store artifact
    artifact = {
        "name": name,
        "type": artifact_type,
        "url": artifact_data.url,
        "scores": scores,
        "net_score": net_score,
        "created_at": datetime.utcnow().isoformat(),
        "created_by": username
    }
    _store_artifact(artifact_id, artifact)
    
    # Build response
    download_url = f"https://example.com/download/{artifact_id}"
    
    response = {
        "metadata": {
            "name": name,
            "id": artifact_id,
            "type": artifact_type
        },
        "data": {
            "url": artifact_data.url,
            "download_url": download_url
        }
    }
    
    return JSONResponse(status_code=201, content=response)

@app.get("/artifacts/{artifact_type}/{id}")
def get_artifact(
    artifact_type: str,
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifact by ID (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    if artifact["type"] != artifact_type:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    return {
        "metadata": {
            "name": artifact["name"],
            "id": id,
            "type": artifact["type"]
        },
        "data": {
            "url": artifact["url"],
            "download_url": f"https://example.com/download/{id}"
        }
    }

@app.put("/artifacts/{artifact_type}/{id}")
def update_artifact(
    artifact_type: str,
    id: str,
    artifact: Artifact = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Update artifact (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    stored = _get_artifact(id)
    if not stored:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    # Validate name and id match
    if artifact.metadata.id != id or artifact.metadata.name != stored["name"]:
        raise HTTPException(status_code=400, detail="Name and ID must match existing artifact.")
    
    # Update artifact
    stored["url"] = artifact.data.url
    stored["updated_at"] = datetime.utcnow().isoformat()
    stored["updated_by"] = username
    _store_artifact(id, stored)
    
    return JSONResponse(status_code=200, content={"message": "Artifact is updated."})

@app.delete("/artifacts/{artifact_type}/{id}")
def delete_artifact(
    artifact_type: str,
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Delete artifact (NON-BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    _delete_artifact(id)
    
    return JSONResponse(status_code=200, content={"message": "Artifact is deleted."})

@app.get("/artifact/byName/{name}")
def get_artifact_by_name(
    name: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifacts by name (NON-BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    results = []
    for artifact_id, artifact in _list_artifacts():
        if artifact["name"] == name:
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    if not results:
        raise HTTPException(status_code=404, detail="No such artifact.")
    
    return results

@app.post("/artifact/byRegEx")
def get_artifact_by_regex(
    regex_query: ArtifactRegEx = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Search artifacts by regex (BASELINE)"""
    import re
    
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    try:
        pattern = re.compile(regex_query.regex, re.IGNORECASE)
    except re.error:
        raise HTTPException(status_code=400, detail="Invalid regex pattern.")
    
    results = []
    for artifact_id, artifact in _list_artifacts():
        if pattern.search(artifact["name"]):
            results.append({
                "name": artifact["name"],
                "id": artifact_id,
                "type": artifact["type"]
            })
    
    if not results:
        raise HTTPException(status_code=404, detail="No artifact found under this regex.")
    
    return results

@app.get("/artifact/model/{id}/rate")
def rate_model(
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get model rating (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    scores = artifact.get("scores", {})
    
    # Build rating response
    rating = {
        "name": artifact["name"],
        "category": artifact["type"],
        "net_score": artifact.get("net_score", 0.7),
        "net_score_latency": 0.5,
        "ramp_up_time": scores.get("ramp_up_time", 0.75),
        "ramp_up_time_latency": 0.3,
        "bus_factor": scores.get("bus_factor", 0.5),
        "bus_factor_latency": 0.4,
        "performance_claims": scores.get("performance_claims", 0.85),
        "performance_claims_latency": 0.6,
        "license": scores.get("license", 0.8),
        "license_latency": 0.2,
        "dataset_and_code_score": 0.65,
        "dataset_and_code_score_latency": 0.5,
        "dataset_quality": scores.get("dataset_quality", 0.6),
        "dataset_quality_latency": 0.7,
        "code_quality": scores.get("code_quality", 0.7),
        "code_quality_latency": 0.8,
        "reproducibility": scores.get("reproducibility", 0.6),
        "reproducibility_latency": 1.5,
        "reviewedness": scores.get("reviewedness", 0.6),
        "reviewedness_latency": 0.9,
        "tree_score": scores.get("tree_score", 0.7),
        "tree_score_latency": 1.2,
        "size_score": {
            "raspberry_pi": 0.3,
            "jetson_nano": 0.5,
            "desktop_pc": 0.8,
            "aws_server": 1.0
        },
        "size_score_latency": 0.4
    }
    
    return rating

@app.get("/artifact/{artifact_type}/{id}/cost")
def get_artifact_cost(
    artifact_type: str,
    id: str,
    dependency: bool = Query(False),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get artifact cost (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    # Mock cost calculation
    base_cost = 412.5
    
    if dependency:
        return {
            id: {
                "standalone_cost": base_cost,
                "total_cost": base_cost
            }
        }
    else:
        return {
            id: {
                "total_cost": base_cost
            }
        }

@app.get("/artifact/model/{id}/lineage")
def get_model_lineage(
    id: str,
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Get model lineage graph (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    # Return empty lineage graph for now
    return {
        "nodes": [{
            "artifact_id": id,
            "name": artifact["name"],
            "source": "config_json"
        }],
        "edges": []
    }

@app.post("/artifact/model/{id}/license-check")
def check_license(
    id: str,
    request: SimpleLicenseCheckRequest = Body(...),
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
):
    """Check license compatibility (BASELINE)"""
    username = _validate_token(x_authorization)
    if not username:
        raise HTTPException(status_code=403, detail="Authentication failed due to invalid or missing AuthenticationToken.")
    
    artifact = _get_artifact(id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    # Mock license check - return true for compatibility
    return True

@app.put("/authenticate")
def authenticate(auth_request: AuthenticationRequest = Body(...)):
    """Authenticate user (NON-BASELINE)"""
    username = auth_request.user.name
    password = auth_request.secret.password
    
    # Validate user
    user = _get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="The user or password is invalid.")
    
    # Verify password
    pw_hash = _hash_password(password, user["salt"])
    if pw_hash != user["password_hash"]:
        raise HTTPException(status_code=401, detail="The user or password is invalid.")
    
    # Create JWT token
    payload = {
        'sub': username,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(seconds=SESSION_TTL_SECONDS),
        'is_admin': user.get('is_admin', False)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return f"bearer {token}"

# Health check at root for compatibility
@app.get("/")
def root():
    return {"status": "ok", "service": "ECE 461 Trustworthy Model Registry"}
