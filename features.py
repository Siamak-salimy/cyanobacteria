# -*- coding: utf-8 -*-
"""مهندسی ویژگی غیرخطی — مشترک بین train و predict"""
import numpy as np
import pandas as pd


def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """ورودی: ۸ ستون خام. خروجی: ۱۴ ویژگی (۸ اصلی + ۶ مهندسی‌شده)"""
    out = df_in.copy()
    eps = 1e-6
    out["NP_ratio"]    = out["nitrate"] / (out["phosphate"] + eps)
    out["temp_x_turb"] = out["temperature"] * out["turbidity"]
    out["temp_sq"]     = out["temperature"] ** 2
    out["log_sulfate"] = np.log1p(out["sulfate"])
    out["log_EC"]      = np.log1p(out["EC"])
    out["turb_sq"]     = out["turbidity"] ** 2
    return out