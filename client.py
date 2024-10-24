import requests

# Sample input data for prediction
body = {
    "Temperature": 80.943049718378,   # Example temperature
    "Humidity": 64.7400433475117,      # Example humidity
    "Wind_Speed": 14.184830644531,    # Example wind speed
    "Precipitation": 0.916884446744698,   # Example precipitation
    "Cloud_Cover": 77.3647625221292,    # Example cloud cover
    "Pressure": 980.79673864618      # Example pressure
}

response = requests.post(url='http://127.0.0.1:8000/score', json=body)
print(response.json())