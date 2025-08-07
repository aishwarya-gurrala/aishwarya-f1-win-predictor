import pandas as pd
from xgboost import XGBClassifier
import joblib
import os

def load_and_prepare():
    base = os.path.dirname(os.path.dirname(__file__))  # project root
    results_path = os.path.join(base, "data", "results.csv")
    races_path = os.path.join(base, "data", "races.csv")

    print(f"Current working dir: {os.getcwd()}")
    print(f"Looking for results.csv at: {results_path} -> exists? {os.path.exists(results_path)}")
    print(f"Looking for races.csv at: {races_path} -> exists? {os.path.exists(races_path)}")

    if not os.path.exists(results_path) or not os.path.exists(races_path):
        missing = []
        if not os.path.exists(results_path):
            missing.append("results.csv")
        if not os.path.exists(races_path):
            missing.append("races.csv")
        raise FileNotFoundError(f"Missing required data file(s): {', '.join(missing)} in {os.path.join(base,'data')}")

    results = pd.read_csv(results_path)
    races = pd.read_csv(races_path)
    df = results.merge(races, on="raceId", how="left")
    df = df[df["positionOrder"].notnull()]
    df["won"] = df["positionOrder"].apply(lambda x: 1 if x == 1 else 0)
    df = df.dropna(subset=["grid", "points"])
    X = df[["grid", "points"]]
    y = df["won"]
    return X, y

def train_and_save_model(model_path="../f1_model.pkl"):
    X, y = load_and_prepare()
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
