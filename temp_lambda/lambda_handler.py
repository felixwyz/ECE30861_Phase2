import json
from decimal import Decimal
from mangum import Mangum
from api.autograder_routes import app, _validate_token, _list_artifacts

# Create the Mangum handler for most requests
mangum_handler = Mangum(app)

def handler(event, context):
    """
    Lambda handler that bypasses Mangum for byName requests.
    
    The issue: Mangum/API Gateway v2 doesn't correctly handle path parameters
    with slashes (e.g., /artifact/byName/google-bert/bert-base-uncased).
    
    Solution: Intercept byName requests before they reach Mangum and handle
    them directly using the same artifact storage.
    """
    # Get request info from API Gateway v2 event format
    raw_path = event.get("rawPath", "")
    http_method = event.get("requestContext", {}).get("http", {}).get("method", "")
    
    # Check if this is a byName GET request
    if http_method == "GET" and "/artifact/byName/" in raw_path:
        return handle_byname_request(event)
    
    # All other requests go through Mangum/FastAPI
    return mangum_handler(event, context)


def handle_byname_request(event):
    """
    Handle GET /artifact/byName/{name} requests directly.
    Bypasses Mangum routing which has issues with path parameters containing slashes.
    """
    try:
        raw_path = event.get("rawPath", "")
        headers = event.get("headers", {})
        
        # Extract the artifact name from the path
        # /artifact/byName/google-bert/bert-base-uncased -> google-bert/bert-base-uncased
        name = raw_path.split("/artifact/byName/", 1)[1] if "/artifact/byName/" in raw_path else ""
        
        # URL decode the name
        from urllib.parse import unquote
        name = unquote(name)
        
        # Validate authentication
        auth_token = headers.get("x-authorization") or headers.get("X-Authorization")
        username = _validate_token(auth_token)
        
        if not username:
            return {
                "statusCode": 403,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"detail": "Authentication failed due to invalid or missing AuthenticationToken."})
            }
        
        # Search for artifacts with this name
        all_artifacts = list(_list_artifacts())
        results = []
        
        for artifact_id, artifact in all_artifacts:
            artifact_name = artifact.get("name", "")
            if artifact_name == name:
                results.append({
                    "name": artifact_name,
                    "id": artifact_id,
                    "type": artifact.get("type", "model")
                })
        
        if not results:
            return {
                "statusCode": 404,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"detail": "No such artifact."})
            }
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,X-Authorization"
            },
            "body": json.dumps(results, default=str)
        }
        
    except Exception as e:
        print(f"Error in handle_byname_request: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
