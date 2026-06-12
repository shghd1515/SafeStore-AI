"""
04_autoencoder.py
scikit-learn 기반 Autoencoder 이상치 감지
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

CSV_PATH  = "sensor_cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "temperature", "humidity", "pm25",
    "hour_sin", "hour_cos",
    "temp_diff", "humi_diff", "pm25_diff",
    "temp_ma60", "humi_ma60", "pm25_ma60",
]

def train_autoencoder():
    print("===== Autoencoder 학습 시작 =====")
    df = pd.read_csv(CSV_PATH)

    available = [f for f in FEATURES if f in df.columns]
    X = df[available].dropna()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Autoencoder: 입력 → 압축 → 복원
    n_features = X_scaled.shape[1]
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(8, 4, 8),  # 압축 → 복원
        activation="relu",
        max_iter=500,
        random_state=42,
        verbose=False,
    )
    autoencoder.fit(X_scaled, X_scaled)

    # 재구성 오차 계산
    X_pred = autoencoder.predict(X_scaled)
    errors = np.mean((X_scaled - X_pred) ** 2, axis=1)

    # 임계값: 상위 5%를 이상치로 설정
    threshold = float(np.percentile(errors, 95))
    print(f"  학습 샘플: {len(X):,}")
    print(f"  임계값 (95th percentile): {threshold:.6f}")
    print(f"  평균 재구성 오차: {errors.mean():.6f}")

    # 저장
    joblib.dump(autoencoder, os.path.join(MODEL_DIR, "autoencoder.pkl"))
    joblib.dump(scaler,      os.path.join(MODEL_DIR, "autoencoder_scaler.pkl"))
    joblib.dump({
        "threshold":  threshold,
        "features":   available,
        "mean_error": float(errors.mean()),
    }, os.path.join(MODEL_DIR, "autoencoder_meta.pkl"))

    print("===== Autoencoder 학습 완료 =====")
    return autoencoder, scaler, threshold, available

if __name__ == "__main__":
    train_autoencoder()
