"""Step 5 — Retrain on recent data and write a Kaggle submission.

Once the backtest in step 4 says the model beats the baselines, it retrain
without a held-out fold and predict the 16 test
days. Output matches data/sample_submission.csv: columns id, sales.

N_ROUNDS is fixed here (not early-stopped) to the median `best_iter` that
step 4 reported. Set it after the run on step 4. That keeps the final fit honest:
the iteration budget was chosen on data whose labels this model never sees
tuned against it.
"""

from __future__ import annotations

from importlib import import_module

import lightgbm as lgb
import numpy as np
import pandas as pd

from fc_lib import ARTIFACTS, DATA

_m04 = import_module("04_model_gbm")
CATEGORICAL = _m04.CATEGORICAL
PARAMS = _m04.PARAMS
TRAIN_SINCE = _m04.TRAIN_SINCE

# <-- set this to the "median best_iter" line printed by 04_model_gbm.py
# (0.4006 CV RMSLE run reported 479; round to 480)
N_ROUNDS = 480


def main() -> None:
    df = pd.read_parquet(ARTIFACTS / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    feats = [c for c in df.columns
             if c not in ("id", "date", "sales", "y", "split")]

    train = df[df["split"] == "train"]
    if TRAIN_SINCE is not None:
        train = train[train["date"] >= pd.Timestamp(TRAIN_SINCE)]
    test = df[df["split"] == "test"].copy()
    print(f"fit on {len(train):,} rows since {train['date'].min().date()}, "
          f"{N_ROUNDS} rounds")

    dtrain = lgb.Dataset(train[feats], train["y"],
                         categorical_feature=CATEGORICAL)
    model = lgb.train(PARAMS, dtrain, num_boost_round=N_ROUNDS)

    pred = np.expm1(model.predict(test[feats]))
    test["sales"] = np.clip(pred, 0, None)

    sub = test[["id", "sales"]].sort_values("id")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert set(sub["id"]) == set(sample["id"]), "id mismatch vs sample_submission"
    assert len(sub) == len(sample)

    out = ARTIFACTS / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}  rows={len(sub)}")
    print(sub["sales"].describe().to_string())
    print("\nUpload to Kaggle: 'Store Sales - Time Series Forecasting'.")


if __name__ == "__main__":
    main()
