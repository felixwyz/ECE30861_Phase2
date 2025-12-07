"""
Test scenarios based on autograder failures
"""
import requests
import json
import os

# Get API endpoint
API_URL = os.getenv('API_GATEWAY_URL', 'https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com')

# Admin credentials
ADMIN_USER = "ece30861defaultadminuser"
ADMIN_PASSWORD = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''

def get_auth_token():
    """Authenticate and get token"""
    response = requests.put(
        f"{API_URL}/authenticate",
        json={
            "User": {"name": ADMIN_USER, "isAdmin": True},
            "Secret": {"password": ADMIN_PASSWORD}
        }
    )
    if response.status_code == 200:
        return response.text.strip()
    else:
        print(f"Auth failed: {response.status_code} - {response.text}")
        return None

def test_regex_endpoint(token):
    """Test regex endpoint - FAILING (0/7)"""
    print("\n=== Testing Regex Endpoint ===")
    
    # Test 1: Exact match
    response = requests.post(
        f"{API_URL}/artifact/byRegEx",
        headers={"X-Authorization": token},
        json={"RegEx": "bert-base-uncased"}
    )
    print(f"Exact match test: {response.status_code}")
    if response.status_code == 200:
        print(f"  Results: {response.json()}")
    else:
        print(f"  Error: {response.text}")
    
    # Test 2: Pattern with extra chars
    response = requests.post(
        f"{API_URL}/artifact/byRegEx",
        headers={"X-Authorization": token},
        json={"RegEx": "bert.*uncased"}
    )
    print(f"Pattern test: {response.status_code}")
    if response.status_code == 200:
        print(f"  Results: {response.json()}")
    else:
        print(f"  Error: {response.text}")

def test_by_name_endpoint(token):
    """Test byName endpoint - FAILING (many tests)"""
    print("\n=== Testing ByName Endpoint ===")
    
    # First get all artifacts to see their actual names
    response = requests.post(
        f"{API_URL}/artifacts",
        headers={"X-Authorization": token},
        json=[{"name": "*"}]
    )
    
    if response.status_code == 200:
        artifacts = response.json()
        print(f"\n=== Actual artifact names in registry ===")
        for i, art in enumerate(artifacts[:10]):
            print(f"{i}. {art['name']} (type={art['type']}, id={art['id']})")
    
    # Test names that autograder might be using
    test_names = [
        "bert-base-uncased",
        "google-bert/bert-base-uncased",
        "bookcorpus",
        "dataset/bookcorpus"
    ]
    
    for name in test_names:
        # Try to get by this name
        response = requests.get(
            f"{API_URL}/artifact/byName/{name}",
            headers={"X-Authorization": token}
        )
        print(f"\nGet '{name}': {response.status_code}")
        if response.status_code == 200:
            print(f"  Results: {response.json()}")
        elif response.status_code == 404:
            print(f"  Not found - need to check how we store names")
        else:
            print(f"  Error: {response.text}")

def test_rating_endpoint(token):
    """Test rating endpoint - FAILING (5/14)"""
    print("\n=== Testing Rating Endpoint ===")
    
    # Get all artifacts first
    response = requests.post(
        f"{API_URL}/artifacts",
        headers={"X-Authorization": token},
        json=[{"name": "*"}]
    )
    
    if response.status_code == 200:
        artifacts = response.json()
        print(f"Found {len(artifacts)} artifacts")
        
        # Test rating for first model
        for artifact in artifacts[:3]:
            if artifact['type'] == 'model':
                print(f"\nTesting rating for {artifact['name']} (id={artifact['id']})")
                response = requests.get(
                    f"{API_URL}/artifact/model/{artifact['id']}/rate",
                    headers={"X-Authorization": token}
                )
                if response.status_code == 200:
                    rating = response.json()
                    print(f"  ramp_up_time: {rating.get('ramp_up_time')}")
                    print(f"  performance_claims: {rating.get('performance_claims')}")
                    print(f"  dataset_and_code_score: {rating.get('dataset_and_code_score')}")
                    print(f"  dataset_quality: {rating.get('dataset_quality')}")
                    print(f"  code_quality: {rating.get('code_quality')}")
                    print(f"  size_score.raspberry_pi: {rating.get('size_score', {}).get('raspberry_pi')}")
                    print(f"  size_score.jetson_nano: {rating.get('size_score', {}).get('jetson_nano')}")
                else:
                    print(f"  Error: {response.status_code} - {response.text}")
                break

def main():
    print("Testing autograder scenarios...")
    print(f"API URL: {API_URL}")
    
    token = get_auth_token()
    if not token:
        print("Failed to authenticate!")
        return
    
    print(f"Auth token: {token[:50]}...")
    
    # Run tests
    test_regex_endpoint(token)
    test_by_name_endpoint(token)
    test_rating_endpoint(token)

if __name__ == "__main__":
    main()
