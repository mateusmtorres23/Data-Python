import requests
import pandas as pd
import joblib
from pathlib import Path

LATITUDE = -9.1269   #Coordinates for
LONGITUDE = -36.4589 #Lagoa do Ouro - PE - BR
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MODEL_FEATURES = ["humidity_pc", "precip_mm", "wind_speed_kmh", "hour", "month", "day_of_year"]

def get_live_forecast(days=3):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "America/Sao_Paulo",
        "forecast_days": days
    }
    
    try:
        response = requests.get(FORECAST_URL, params=params)
        response.raise_for_status() # Raises error for 400s/500s codes
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"API Error: {e}")

    data = response.json()
    df = pd.DataFrame(data["hourly"])
    
    column_map = {
        "time": "time",
        "temperature_2m": "api_temp", 
        "relative_humidity_2m": "humidity_pc",
        "precipitation": "precip_mm",
        "wind_speed_10m": "wind_speed_kmh"
    }
    df = df.rename(columns=column_map)
    
    df["time"] = pd.to_datetime(df["time"])
    
    return df

def prepare_forecast_features(df):

    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    
    X = df[MODEL_FEATURES]
    
    return X

def load_trained_model(path):

    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first!")
    
    model = joblib.load(path)
    return model

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "models" / "weather_model.joblib"
    
    print("Starting Prediction Pipeline")
    
    print("Fetching live forecast...")
    forecast_df = get_live_forecast(days=3)
    
    X_future = prepare_forecast_features(forecast_df)
    
    print(f"Loading model from: {MODEL_PATH}")
    model = load_trained_model(MODEL_PATH)
    
    print("Generating predictions...")
    predictions = model.predict(X_future)
    
    forecast_df["my_model_temp"] = predictions
    
    forecast_df["correction"] = forecast_df["my_model_temp"] - forecast_df["api_temp"]
    
    output_cols = ["time", "api_temp", "my_model_temp", "correction"]
    
    print("\n--- Forecast Results ---")
    print(forecast_df[output_cols].to_string(index=False))
    
    print("\n--- Summary ---")
    avg_correction = forecast_df["correction"].mean()
    print(f"On average, your model adjusts the API forecast by {avg_correction:+.2f}°C")