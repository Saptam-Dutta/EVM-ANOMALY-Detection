import pandas as pd
from sklearn.ensemble import IsolationForest


def calculate_evm(df):
    df = df.copy()

    # EVM calculations
    df["CV"] = df["EV"] - df["AC"]
    df["SV"] = df["EV"] - df["PV"]

    df["CPI"] = df["EV"] / df["AC"]
    df["SPI"] = df["EV"] / df["PV"]

    return df


def detect_anomalies(df):
    df = df.copy()

    features = [
        "CV",
        "SV",
        "CPI",
        "SPI"
    ]

    model = IsolationForest(
        contamination=0.1,
        random_state=42
    )

    df["Anomaly"] = model.fit_predict(df[features])

    df["Status"] = df["Anomaly"].map({
        1: "NORMAL",
        -1: "ANOMALY"
    })

    return df