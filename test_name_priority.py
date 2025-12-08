"""
Test that we use provided name (not extracted from URL)
"""
import requests
import json

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# 1. Authenticate as admin
print("Step 1: Authenticating...")
auth_response = requests.put(
    f"{BASE_URL}/authenticate",
    headers={"Content-Type": "application/json"},
    json={
        "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
        "Secret": {
            "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
        }
    }
)
print(f"Auth status: {auth_response.status_code}")

if auth_response.status_code != 200:
    print(f"Auth failed: {auth_response.text}")
    exit(1)

# Token is in plain text response with "bearer " prefix
token = auth_response.text.strip()
if token.startswith("bearer "):
    token = token.split("bearer ")[1]
print(f"Got token: {token[:20]}...")

headers = {
    "Content-Type": "application/json",
    "X-Authorization": token
}

# 2. Upload with BOTH name and URL - name should take priority
print("\nStep 2: Uploading artifact...")
print("URL: https://huggingface.co/google-bert/bert-base-uncased")
print("Provided name: bert-base-uncased")
print("Expected: Should store as 'bert-base-uncased' (not 'google-bert/bert-base-uncased')")

upload_response = requests.post(
    f"{BASE_URL}/artifact/model",
    headers=headers,
    json={
        "name": "bert-base-uncased",  # AUTOGRADER PROVIDES THIS
        "url": "https://huggingface.co/google-bert/bert-base-uncased"
    }
)
print(f"Upload status: {upload_response.status_code}")
print(f"Upload response: {upload_response.text}")

# 3. Search by name (what autograder does)
print("\nStep 3: Searching by name...")
print("Searching for: bert-base-uncased")

search_response = requests.get(
    f"{BASE_URL}/artifact/byName/bert-base-uncased",
    headers=headers
)
print(f"Search status: {search_response.status_code}")

if search_response.status_code == 200:
    result = search_response.json()
    # byName returns a list of matching artifacts
    if isinstance(result, list) and len(result) > 0:
        print(f"✅ FOUND! Count: {len(result)}")
        for item in result:
            print(f"  - Name: {item.get('metadata', {}).get('name', item.get('name'))}")
            print(f"    ID: {item.get('metadata', {}).get('id', item.get('id'))}")
    else:
        print(f"Full response: {json.dumps(result, indent=2)}")
else:
    print(f"❌ NOT FOUND: {search_response.text}")

# 4. Also try with slash (should NOT find it)
print("\nStep 4: Searching for org/repo format...")
print("Searching for: google-bert/bert-base-uncased")

search2_response = requests.get(
    f"{BASE_URL}/artifact/byName/google-bert/bert-base-uncased",
    headers=headers
)
print(f"Search status: {search2_response.status_code}")

if search2_response.status_code == 404:
    print("✅ Correctly NOT found (we stored as 'bert-base-uncased')")
elif search2_response.status_code == 200:
    print(f"❌ PROBLEM: Found with org/repo format: {search2_response.json()}")
else:
    print(f"Status: {search2_response.status_code}, Response: {search2_response.text}")
