import requests
import pandas as pd
from pathlib import Path


LATITUDE = -9.1269   #Coordinates for
LONGITUDE = -36.4589 #Lagoa do Ouro-PE-BR
START_DATE = "2024-01-01"
END_DATE = "2025-12-01"
ARCH_URL = "https://archive-api.open-meteo.com/v1/archive" 

def extract_weather_data(lat, lon, st_dt, end_dt, url):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": st_dt,
        "end_date": end_dt,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "America/Sao_Paulo" 
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    return response.json()

def transform_data(json_data):
    hourly_data = json_data["hourly"]

    df = pd.DataFrame(hourly_data)

    df["time"] = pd.to_datetime(df["time"])

    column_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity_pc",
        "precipitation": "precip_mm",
        "wind_speed_10m": "wind_speed_kmh"
    }

    df = df.rename(columns=column_map)

    df = df.dropna()

    return df

def load_data(df, path_str):
    path = Path(path_str)

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"Data saved to: {path}\nShape: {df.shape}")

if __name__ == "__main__":
    print("Starting ETL Process")
    
    raw_json = extract_weather_data(LATITUDE, LONGITUDE, START_DATE, END_DATE, ARCH_URL)
    
    data = transform_data(raw_json)
    
    script_dir = Path(__file__).parent
    output_path = script_dir / "data" / "weather.csv"
    
    load_data(data, output_path)
    print("ETL Process Complete.")