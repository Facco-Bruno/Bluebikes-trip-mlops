import requests


def test_prediction_api():
    url = "http://localhost:8000/predict"
    payload = {"trips": [{"start_station_id": "72", "end_station_id": "79"}]}

    response = requests.post(url, json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "results" in json_response
    assert len(json_response["results"]) == 1
    assert "predicted_duration" in json_response["results"][0]
