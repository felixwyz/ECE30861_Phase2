from mangum import Mangum
from api.autograder_routes import app

_mangum = Mangum(app)

def _method(event: dict) -> str:
    if isinstance(event, dict):
        if "httpMethod" in event:
            return (event.get("httpMethod") or "").upper()
        rc = event.get("requestContext") or {}
        http = rc.get("http") or {}
        if "method" in http:
            return (http.get("method") or "").upper()
    return ""

def _origin(event: dict) -> str:
    headers = (event.get("headers") or {}) if isinstance(event, dict) else {}
    return headers.get("origin") or headers.get("Origin") or "*"

def handler(event, context):
    if _method(event) == "OPTIONS":
        origin = _origin(event)
        return {
            "statusCode": 204,
            "headers": {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
                "Access-Control-Allow-Headers": "content-type,x-authorization,authorization",
                "Access-Control-Max-Age": "300",
            },
            "body": "",
        }

    return _mangum(event, context)
