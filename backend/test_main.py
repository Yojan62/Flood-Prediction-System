# Imports the TestClient for making simulated requests to the API.
from fastapi.testclient import TestClient
# Imports the 'app' instance from the main application file.
from main import app

# Creates a TestClient instance using the FastAPI app.
# This client will be used to send requests in the tests.
client = TestClient(app)

# --- Tests for the /api/subscribe endpoint ---

def test_subscribe_user_success():
    """
    Tests the POST /api/subscribe endpoint with valid data.
    It should return a 200 OK status and the placeholder success message.
    """
    # Defines the valid data payload for the request body.
    subscription_data = {
        "email": "test@example.com",
        "location_id": 1 # Assumes a location ID exists or will exist later.
    }

    # Uses the TestClient to send a POST request to the endpoint with the JSON data.
    response = client.post("/api/subscribe", json=subscription_data)

    # Asserts that the HTTP status code in the response is 200 (OK).
    assert response.status_code == 200

    # Defines the expected JSON response from the placeholder endpoint.
    expected_response = {"status": "success", "message": "Subscription request received (not saved yet)."}
    # Asserts that the JSON body of the response matches the expected response.
    assert response.json() == expected_response

def test_subscribe_user_invalid_data_missing_field():
    """
    Tests the POST /api/subscribe endpoint with invalid data (missing location_id).
    FastAPI should automatically return a 422 Unprocessable Entity status code
    because the request body doesn't match the SubscriptionCreate schema.
    """
    # Defines an invalid data payload (missing the required 'location_id' field).
    invalid_subscription_data = {
        "email": "test@example.com"
    }

    # Sends the POST request with the invalid data.
    response = client.post("/api/subscribe", json=invalid_subscription_data)

    # Asserts that the HTTP status code is 422, indicating a validation error.
    assert response.status_code == 422

def test_subscribe_user_invalid_data_wrong_type():
    """
    Tests the POST /api/subscribe endpoint with invalid data (wrong data type for location_id).
    FastAPI should automatically return a 422 Unprocessable Entity status code.
    """
    # Defines an invalid data payload ('location_id' should be an integer).
    invalid_subscription_data = {
        "email": "test@example.com",
        "location_id": "not-an-integer" # Incorrect data type.
    }

    # Sends the POST request with the invalid data.
    response = client.post("/api/subscribe", json=invalid_subscription_data)

    # Asserts that the HTTP status code is 422.
    assert response.status_code == 422

# --- (Add tests for other endpoints like /api/locations later) ---

# Example test for the create_location endpoint (requires careful setup/teardown if interacting with DB)
# def test_create_location_success():
#     location_data = {"name": "Test City", "latitude": 10.0, "longitude": 20.0}
#     response = client.post("/api/locations", json=location_data)
#     assert response.status_code == 200
#     data = response.json()
#     assert data["name"] == location_data["name"]
#     assert "location_id" in data
#     # TODO: Add cleanup logic to remove the test city from the database after the test