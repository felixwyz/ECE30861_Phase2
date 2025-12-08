#!/usr/bin/env python3
"""
Test the byName endpoint with the new Lambda bypass logic
"""

import requests
import json

API_BASE = "https://2047fz40z1.execute-api.us-east-1.amazonaws.com"

def get_auth_token():
    """Get authentication token"""
    auth_url = f"{API_BASE}/authenticate"
    password = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
    payload = {
        "User": {
            "name": "ece30861defaultadminuser",
            "isAdmin": True
        },
        "Secret": {
            "password": password
        }
    }
    
    response = requests.put(auth_url, json=payload)
    print(f"Auth response: {response.status_code}")
    if response.status_code == 200:
        return response.text.strip('"')
    else:
        print(f"Auth failed: {response.text}")
        return None

def test_byname(token, artifact_name):
    """Test byName endpoint"""
    byname_url = f"{API_BASE}/artifact/byName/{artifact_name}"
    headers = {
        "X-Authorization": token
    }
    
    print(f"\n{'='*60}")
    print(f"Testing byName: {artifact_name}")
    print(f"URL: {byname_url}")
    print(f"{'='*60}")
    
    response = requests.get(byname_url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json() if response.status_code != 500 else response.text, indent=2)}")
    
    return response.status_code == 200

def main():
    # Get auth token
    print("Authenticating...")
    token = get_auth_token()
    if not token:
        print("Failed to authenticate")
        return
    
    print(f"Token obtained: {token[:20]}...")
    
    # Test cases with different artifact name formats
    test_cases = [
        "bert-base-uncased",
        "google-bert/bert-base-uncased",
        "microsoft/DialoGPT-medium",
        "test-artifact"
    ]
    
    results = []
    for name in test_cases:
        success = test_byname(token, name)
        results.append((name, success))
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")

if __name__ == "__main__":
    main()
