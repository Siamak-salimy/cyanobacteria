# -*- coding: utf-8 -*-
"""Prediction using the saved model - used by API/dashboard/CLI"""
from pathlib import Path
import joblib
import pandas as pd
from config import MODEL_PATH, ALERT_LEVELS
from features import engineer_features

_artifact = None


def load_artifact():
    global _artifact
    if _artifact is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\nRun train.py first."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def _alert_level(value: float) -> str:
    for threshold, label in ALERT_LEVELS:
        if value < threshold:
            return label
    return ALERT_LEVELS[-1][1]


def predict_cyanobacteria(ec, ph, turbidity, sulfate,
                          nitrite, nitrate, phosphate, temperature):
    art = load_artifact()
    row = pd.DataFrame([{
        "EC": ec, "pH": ph, "turbidity": turbidity,
        "sulfate": sulfate, "nitrite": nitrite,
        "nitrate": nitrate, "phosphate": phosphate,
        "temperature": temperature,
    }])
    row_eng = engineer_features(row)
    pred = float(art["model"].predict(row_eng)[0])
    rmse = art["rmse_real"]
    lo, hi = max(0.0, pred - rmse), pred + rmse
    return {
        "prediction": round(pred, 1),
        "interval":   (round(lo, 1), round(hi, 1)),
        "alert":      _alert_level(pred),
        "best_model": art["best_name"],
    }