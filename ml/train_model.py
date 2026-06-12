"""
03_train_model.py
sensor_cleaned.csv를 읽어 ML 모델 학습 후 저장
XGBoost / LightGBM / RandomForest / GradientBoosting 4개 모델 비교 후 최적 선택
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

CSV_PATH  = "sensor_cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "minute_sin", "minute_cos",
    "is_weekend",
    "time_group",
    "temp_ma10", "temp_ma30", "temp_ma60", "temp_ma120",
    "humi_ma10", "humi_ma30", "humi_ma60", "humi_ma120",
    "pm25_ma10", "pm25_ma30", "pm25_ma60", "pm25_ma120",
    "temp_diff",  "humi_diff",  "pm25_diff",
    "temp_diff5", "humi_diff5", "pm25_diff5",
    "temp_diff10","humi_diff10","pm25_diff10",
    "temp_accel", "humi_accel", "pm25_accel",
    "temp_std10", "humi_std10", "pm25_std10",
    "temp_humi",  "pm25_hour",
    "has_event",
]

PM25_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "time_group",
    "temp_ma60", "humi_ma60",
    "temp_ma120", "humi_ma120",
    "temp_diff",  "humi_diff",
    "temp_diff5", "humi_diff5",
    "temp_std10", "humi_std10",
    "temp_humi",
    "has_event",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["recorded_at"])
    print(f"[로드] {len(df):,} 행  /  기간: {df['recorded_at'].min()} ~ {df['recorded_at'].max()}")
    return df


def make_hourly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"temperature": "target_temp", "humidity": "target_humi"}
    if "pm25" in df.columns:
        cols["pm25"] = "target_pm25"

    agg_dict = {new: (old, "mean") for old, new in cols.items()}
    agg_dict["count"] = ("temperature", "count")

    pattern = df.groupby(["hour", "is_weekend"]).agg(**agg_dict).reset_index().round(2)
    path = os.path.join(MODEL_DIR, "hourly_pattern.json")
    pattern.to_json(path, orient="records", force_ascii=False, indent=2)
    print(f"[패턴] 시간대별 패턴 저장: {path}  ({len(pattern)} 그룹)")
    return pattern


def prepare_xy(df: pd.DataFrame):
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[경고] 없는 피처 (건너뜀): {missing}")

    X  = df[available].copy()
    yt = df["temperature"].copy()
    yh = df["humidity"].copy()
    yp = df["pm25"].copy() if "pm25" in df.columns else None

    valid = X.notna().all(axis=1) & yt.notna() & yh.notna()
    X, yt, yh = X[valid], yt[valid], yh[valid]
    if yp is not None:
        yp = yp[valid]

    print(f"[준비] 학습 샘플: {len(X):,}  /  피처: {available}")
    return X, yt, yh, yp, available

# PM2.5 전용 피처 (누수 피처 제외)
PM25_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "temp_ma60",
    "humi_ma60",
    "temp_diff",
    "humi_diff",
    "has_event",
]

def train_model(X, y, target_name: str):
    print(f"\n  [{target_name}] 모델 학습 중...")

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=8,
            min_samples_leaf=3, random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8, random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=-1, n_jobs=-1,
        ),
    }

    best_model = None
    best_name  = None
    best_mae   = float("inf")
    all_metrics = {}

    for name, model in models.items():
        cv_scores = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
        mean_mae  = cv_scores.mean()
        all_metrics[name] = round(float(mean_mae), 4)
        print(f"    {name:<20} CV MAE: {mean_mae:.4f} ± {cv_scores.std():.4f}")

        if mean_mae < best_mae:
            best_mae   = mean_mae
            best_model = model
            best_name  = name

    print(f"    ✅ 선택된 모델: {best_name} (MAE: {best_mae:.4f})")
    best_model.fit(X, y)

    pred = best_model.predict(X)
    mae  = mean_absolute_error(y, pred)
    r2   = r2_score(y, pred)
    print(f"    최종 MAE: {mae:.4f}  /  R²: {r2:.4f}")

    return best_model, {
        "model":    best_name,
        "cv_mae":   round(best_mae, 4),
        "mae":      round(mae, 4),
        "r2":       round(r2, 4),
        "all_models": all_metrics,
    }


def print_feature_importance(model, feature_names: list, target: str):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\n  [{target}] 피처 중요도:")
    for feat, val in imp.items():
        bar = "█" * int(val * 40)
        print(f"    {feat:<15} {val:.4f}  {bar}")


def main():
    print("===== XGBoost/LightGBM 포함 모델 학습 시작 =====\n")

    df = load_data()
    make_hourly_pattern(df)

    X, yt, yh, yp, feature_names = prepare_xy(df)

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    # 온도 모델
    print("\n[온도 모델]")
    model_temp, metrics_temp = train_model(X_scaled, yt, "온도")
    print_feature_importance(model_temp, feature_names, "온도")

    # 습도 모델
    print("\n[습도 모델]")
    model_humi, metrics_humi = train_model(X_scaled, yh, "습도")
    print_feature_importance(model_humi, feature_names, "습도")

    # PM2.5 모델 (누수 피처 제외)
    metrics_pm25 = {}
    if yp is not None:
        print("\n[PM2.5 모델] (누수 피처 제외)")
        pm25_feats = [f for f in PM25_FEATURES if f in feature_names]
        X_pm25 = X_scaled[pm25_feats]
        model_pm25, metrics_pm25 = train_model(X_pm25, yp, "PM2.5")
        print_feature_importance(model_pm25, pm25_feats, "PM2.5")
        joblib.dump(model_pm25, os.path.join(MODEL_DIR, "model_pm25.pkl"))
        joblib.dump(pm25_feats, os.path.join(MODEL_DIR, "feature_names_pm25.pkl"))

    # 저장
    joblib.dump(model_temp,    os.path.join(MODEL_DIR, "model_temp.pkl"))
    joblib.dump(model_humi,    os.path.join(MODEL_DIR, "model_humi.pkl"))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    import datetime
    meta = {
        "trained_at":   datetime.datetime.now().isoformat(),
        "n_samples":    int(len(X)),
        "features":     feature_names,
        "metrics_temp": metrics_temp,
        "metrics_humi": metrics_humi,
        "metrics_pm25": metrics_pm25,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n===== 학습 완료 =====")
    print(f"저장 위치: {MODEL_DIR}/")
    print(f"\n모델 성능 요약:")
    for target, metrics in [("온도", metrics_temp), ("습도", metrics_humi), ("PM2.5", metrics_pm25)]:
        if metrics:
            print(f"  {target}: {metrics.get('model')} (CV MAE: {metrics.get('cv_mae')})")


if __name__ == "__main__":
    main()