#!/usr/bin/env python3
"""
Test creating an artifact and then searching for it by name
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
    if response.status_code == 200:
        return response.text.strip('"')
    else:
        print(f"Auth failed: {response.text}")
        return None

def create_artifact(token, name, url):
    """Create a test artifact"""
    create_url = f"{API_BASE}/artifact/model"
    headers = {
        "X-Authorization": token,
        "Content-Type": "application/json"
    }
    payload = {
        "name": name,
        "url": url,
        "debloat": False
    }
    
    print(f"\nCreating artifact:")
    print(f"  Name: {name}")
    print(f"  URL: {url}")
    
    response = requests.post(create_url, json=payload, headers=headers)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 201:
        print(f"  ✓ Created successfully")
        return True
    else:
        print(f"  ✗ Failed: {response.text}")
        return False

def search_by_name(token, name):
    """Search for artifact by name"""
    search_url = f"{API_BASE}/artifact/byName/{name}"
    headers = {
        "X-Authorization": token
    }
    
    print(f"\nSearching for: {name}")
    response = requests.get(search_url, headers=headers)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"  ✓ Found {len(results)} result(s)")
        for r in results:
            print(f"    - {r['name']} (id={r['id']}, type={r['type']})")
        return True
    else:
        print(f"  ✗ Not found: {response.text}")
        return False

def main():
    # Get auth token
    print("=== Authentication ===")
    token = get_auth_token()
    if not token:
        print("Failed to authenticate")
        return
    print(f"✓ Token obtained: {token[:20]}...")
    
    # Test case 1: Simple name (no slashes)
    print("\n" + "="*60)
    print("TEST 1: Simple name (no slashes)")
    print("="*60)
    name1 = "test-bypass-artifact"
    url1 = "https://huggingface.co/bert-base-uncased"
    create_artifact(token, name1, url1)
    search_by_name(token, name1)
    
    # Test case 2: Name with slashes (org/repo format)
    print("\n" + "="*60)
    print("TEST 2: Name with slashes (org/repo format)")
    print("="*60)
    name2 = "google-bert/bert-base-uncased"
    url2 = "https://huggingface.co/google-bert/bert-base-uncased"
    create_artifact(token, name2, url2)
    search_by_name(token, name2)
    
    # Test case 3: What the autograder does - upload with short name, search with full path
    print("\n" + "="*60)
    print("TEST 3: Autograder scenario")
    print("="*60)
    upload_name = "bert-base-uncased"
    search_name = "google-bert/bert-base-uncased"
    upload_url = "https://huggingface.co/google-bert/bert-base-uncased"
    
    print(f"\nScenario: Upload as '{upload_name}', search for '{search_name}'")
    create_artifact(token, upload_name, upload_url)
    print(f"\n  Search 1: Looking for '{upload_name}'")
    search_by_name(token, upload_name)
    print(f"\n  Search 2: Looking for '{search_name}'")
    search_by_name(token, search_name)

if __name__ == "__main__":
    main()
