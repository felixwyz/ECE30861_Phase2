"""Test byName endpoint to diagnose autograder failures"""
import requests
import json

BASE_URL = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

# Get auth token
auth_response = requests.put(
    f"{BASE_URL}/authenticate",
    json={
        "User": {"name": "ece30861defaultadminuser", "isAdmin": True},
        "Secret": {"password": 'correcthorsebatterystaple123(!__+@**(A\'"`;DROP TABLE packages;'}
    }
)
token = auth_response.text.strip()
headers = {"X-Authorization": token}

# Reset and upload some test artifacts
print("=== Resetting system ===")
requests.delete(f"{BASE_URL}/reset", headers=headers)

print("\n=== Uploading test artifacts ===")

# Upload model with CORRECT format (url at root, not in data)
model_response = requests.post(
    f"{BASE_URL}/artifact/model",
    headers=headers,
    json={
        "url": "https://huggingface.co/google-bert/bert-base-uncased"
    }
)
print(f"Model upload: {model_response.status_code}")
if model_response.status_code == 201:
    model_data = model_response.json()
    model_id = model_data["metadata"]["id"]
    model_name = model_data["metadata"]["name"]
    print(f"  Stored with name: '{model_name}'")
    print(f"  ID: {model_id}")
    
    # Try to get by name
    print(f"\n=== Testing byName with '{model_name}' ===")
    byname_response = requests.get(
        f"{BASE_URL}/artifact/byName/{model_name}",
        headers=headers
    )
    print(f"  Status: {byname_response.status_code}")
    if byname_response.status_code == 200:
        print(f"  Response: {json.dumps(byname_response.json(), indent=2)}")
    else:
        print(f"  Error: {byname_response.text}")
    
    # Try to get by ID
    print(f"\n=== Testing byID with {model_id} ===")
    byid_response = requests.get(
        f"{BASE_URL}/artifacts/model/{model_id}",
        headers=headers
    )
    print(f"  Status: {byid_response.status_code}")
    if byid_response.status_code == 200:
        result = byid_response.json()
        print(f"  Returned name: '{result['metadata']['name']}'")
    else:
        print(f"  Error: {byid_response.text}")
    
    # List all artifacts to see what names are stored
    print(f"\n=== Listing all artifacts ===")
    list_response = requests.post(
        f"{BASE_URL}/artifacts",
        headers=headers,
        json=[{"name": "*"}]
    )
    if list_response.status_code == 200:
        artifacts = list_response.json()
        for art in artifacts:
            print(f"  {art['type']}: '{art['name']}' (id={art['id']})")
else:
    print(f"  Error: {model_response.text}")
