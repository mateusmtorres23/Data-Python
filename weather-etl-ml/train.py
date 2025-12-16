import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    df = pd.read_csv(file_path, parse_dates=["time"])
    print(f"Data loaded. Shape: {df.shape}")
    return df

def prepare_features(df):

    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    X = df.drop(columns=["time", "temperature"])
    y = df["temperature"]
    
    return X, y

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print("metrics")
    print(f"MAE:  {mae:.4f} °C")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f} °C")
    
    return model

def save_model(model, output_path):

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"model saved successfully to: {output_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / "data" / "weather.csv"
    MODEL_PATH = BASE_DIR / "models" / "weather_model.joblib"
    
    print("Starting Training Pipeline...")
    
    raw_df = load_data(DATA_PATH)
    
    X, y = prepare_features(raw_df)
    
    trained_model = train_model(X, y)
    
    save_model(trained_model, MODEL_PATH)
    
    print("Pipeline Complete.")