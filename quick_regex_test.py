import requests

API = 'https://2047fz40z1.execute-api.us-east-1.amazonaws.com'

# Test regex endpoint without auth (should get 403, not 422)
r = requests.post(f'{API}/artifact/byRegEx', json={'regex': 'bert'})
print(f'Regex test status: {r.status_code}')
print(f'Response: {r.text[:300]}')

# Expected: 403 (auth required) if route is correct
# Got before: 422 (url required) because wrong route

