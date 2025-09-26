import requests
import json

# API base URL
BASE_URL = "http://localhost:8001"

print("Testing Book Sales Rank Prediction API")
print("=" * 50)

# Test 1: Root endpoint
print("\n1. Testing root endpoint...")
response = requests.get(f"{BASE_URL}/")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Health check
print("\n2. Testing health check...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test 3: Get categories
print("\n3. Getting available categories...")
response = requests.get(f"{BASE_URL}/categories")
data = response.json()
print(f"Status: {response.status_code}")
print(f"Available categories: {data['available_categories']}")
print("\nCategory Statistics:")
for cat, stats in data['category_statistics'].items():
    print(f"  {cat}:")
    print(f"    Average Rank: {stats['average_sales_rank']:,.0f}")
    print(f"    Median Rank: {stats['median_sales_rank']:,.0f}")
    print(f"    Samples: {stats['sample_count']}")

# Test 4: Predictions for different categories
print("\n4. Testing predictions for different categories...")
test_cases = [
    {"category": "programming"},
    {"category": "business", "price": 29.99},
    {"category": "cooking", "price": 19.99, "title_length": 100, "has_brand": True},
    {"category": "history", "title_length": 50},
    {"category": "scifi", "price": 14.99}
]

for test_case in test_cases:
    print(f"\n  Testing: {test_case}")
    response = requests.post(f"{BASE_URL}/predict", json=test_case)

    if response.status_code == 200:
        data = response.json()
        print(f"    Category Average Rank: {data['average_sales_rank']:,.0f}")
        print(f"    Random Forest Prediction: {data['predictions']['random_forest']:,.0f}")
        print(f"    XGBoost Prediction: {data['predictions']['xgboost']:,.0f}")
        print(f"    Average Prediction: {data['predictions']['average_prediction']:,.0f}")
    else:
        print(f"    Error: {response.status_code} - {response.text}")

# Test 5: Invalid category
print("\n5. Testing invalid category (should fail)...")
response = requests.post(f"{BASE_URL}/predict", json={"category": "invalid_category"})
print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"Error message: {response.json()['detail']}")

print("\n" + "=" * 50)
print("API testing complete!")
print("\nAPI is running at: http://localhost:8001")
print("Interactive docs available at: http://localhost:8001/docs")