import joblib
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))  # project root
MODEL_PATH = os.path.join(BASE, "f1_model.pkl")

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print("Model load failed:", e)
        from train_model import train_and_save_model  # fixed import
        train_and_save_model(model_path=MODEL_PATH)
        return joblib.load(MODEL_PATH)

def prepare_input(grid, points):
    return pd.DataFrame({"grid": [grid], "points": [points]})
