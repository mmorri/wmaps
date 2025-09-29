from fastapi.testclient import TestClient
from book_api import app

client = TestClient(app)

print("Testing Book Sales Rank Prediction API (in-process)")
print("=" * 50)

# Root endpoint
print("\n1. Testing root endpoint...")
resp = client.get("/")
print(f"Status: {resp.status_code}")
print(resp.json())

# Health
print("\n2. Testing health check...")
resp = client.get("/health")
print(f"Status: {resp.status_code}")
print(resp.json())

# Categories
print("\n3. Getting available categories...")
resp = client.get("/categories")
print(f"Status: {resp.status_code}")
data = resp.json()
print(f"Available: {data.get('available_categories', [])}")

# Predictions (only if models loaded)
if data.get('available_categories'):
    print("\n4. Testing predictions...")
    tests = [
        {"category": data['available_categories'][0]},
        {"category": data['available_categories'][0], "price": 29.99},
    ]
    for t in tests:
        r = client.post("/predict", json=t)
        print(f"Input={t} -> status {r.status_code}")
        print(r.json())
else:
    print("\nModels not loaded or categories empty; skipping predictions.")

print("\n" + "=" * 50)
print("API testing complete!")
