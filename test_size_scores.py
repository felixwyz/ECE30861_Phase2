"""Test size score calculation for different model sizes"""
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

# Reset system
print("Resetting system...")
requests.delete(f"{BASE_URL}/reset", headers=headers)

# Test 1: Small model (distilbert)
print("\n=== Test 1: Small model (distilbert) ===")
response = requests.post(
    f"{BASE_URL}/artifact/model",
    headers=headers,
    json={
        "data": {
            "url": "https://huggingface.co/distilbert/distilbert-base-uncased"
        }
    }
)
print(f"Upload: {response.status_code}")
if response.status_code == 201:
    artifact = response.json()
    model_id = artifact["metadata"]["id"]
    
    # Get rating
    rating_response = requests.get(
        f"{BASE_URL}/artifact/model/{model_id}/rate",
        headers=headers
    )
    if rating_response.status_code == 200:
        rating = rating_response.json()
        print(f"Model: {rating['name']}")
        print(f"Size scores: {rating['size_score']}")
        print(f"Expected: rpi > 0.9, jetson > 0.8 (small model)")

# Test 2: Medium model (base)
print("\n=== Test 2: Medium model (bert-base) ===")
response = requests.post(
    f"{BASE_URL}/artifact/model",
    headers=headers,
    json={
        "data": {
            "url": "https://huggingface.co/bert-base-uncased"
        }
    }
)
print(f"Upload: {response.status_code}")
if response.status_code == 201:
    artifact = response.json()
    model_id = artifact["metadata"]["id"]
    
    # Get rating
    rating_response = requests.get(
        f"{BASE_URL}/artifact/model/{model_id}/rate",
        headers=headers
    )
    if rating_response.status_code == 200:
        rating = rating_response.json()
        print(f"Model: {rating['name']}")
        print(f"Size scores: {rating['size_score']}")
        print(f"Expected: rpi ~0.8, jetson ~0.8 (medium model)")

# Test 3: Large model
print("\n=== Test 3: Large model (bert-large) ===")
response = requests.post(
    f"{BASE_URL}/artifact/model",
    headers=headers,
    json={
        "data": {
            "url": "https://huggingface.co/bert-large-uncased"
        }
    }
)
print(f"Upload: {response.status_code}")
if response.status_code == 201:
    artifact = response.json()
    model_id = artifact["metadata"]["id"]
    
    # Get rating
    rating_response = requests.get(
        f"{BASE_URL}/artifact/model/{model_id}/rate",
        headers=headers
    )
    if rating_response.status_code == 200:
        rating = rating_response.json()
        print(f"Model: {rating['name']}")
        print(f"Size scores: {rating['size_score']}")
        print(f"Expected: rpi ~0.2, jetson ~0.3 (large model)")
