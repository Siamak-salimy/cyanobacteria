# -*- coding: utf-8 -*-
"""Central project configuration (stlite-compatible)"""
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
DATA_FILE  = BASE_DIR / "talab.xlsx"
SHEET_NAME = "فصل اول"

MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cyanobacteria_model.pkl"

COL_MAP = {
    "ردیف": "id",
    "سیانوباکتر ها": "cyanobacteria",
    "هدايت الكتريكي": "EC",
    "PH": "pH",
    "کدورت": "turbidity",
    "سولفات": "sulfate",
    "نيتريت": "nitrite",
    "نيترات": "nitrate",
    "فسفات": "phosphate",
    "دما نمونه": "temperature",
}

FEATURE_COLS = ["EC", "pH", "turbidity", "sulfate",
                "nitrite", "nitrate", "phosphate", "temperature"]

TARGET_COL = "cyanobacteria"

ALERT_LEVELS = [
    (1000,         "کم"),
    (10000,        "متوسط"),
    (30000,        "بالا"),
    (float("inf"), "شکوفایی"),
]