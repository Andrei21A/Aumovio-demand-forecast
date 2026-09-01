"""Step 4 — A gradient-boosted model, scored on the SAME backtest as step 2.

TRAINING WINDOW.  By default  only train on rows from `TRAIN_SINCE` onward
(~1.5 years). For a 16-day forecast, sales from 2013 carry almost no signal
that 2016-2017 doesn't carry better, and the shorter frame keeps this
runnable on a laptop. Set TRAIN_SINCE = None to use everything and compare
the CV score — that experiment is itself a lesson.


Reads artifacts/features.parquet (run 03_features.py first).
Writes artifacts/feature_importance.csv and reports/figures/importance.png.

WHAT TO LOOK FOR
  * cv RMSLE vs the step-2 baseline table (dow_mean_8w ~ 0.47). A 10-20%
    relative improvement is a healthy result for this dataset.
  * Fold-to-fold stability. Big swings = the model is sensitive to the period.
  * The importance chart. If lag/rolling features dominate, the series is
    driven by its own recent history; if calendar/promo dominate, by context.
  * Try deleting a feature group in 03 and re-running — does CV get worse?
    That loop (hypothesis -> feature -> backtest) is the actual job.
"""

from __future__ import annotations

import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd

from fc_lib import ARTIFACTS, FIGURES, HORIZON, ensure_dirs, rmsle, sliding_windows

# Only train on rows from here on. Set to None to train on the full history.
TRAIN_SINCE = "2016-01-01"

CATEGORICAL = ["family", "city", "state", "type", "store_nbr", "cluster"]

PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.08,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=200,
    max_bin=127,          # smaller histograms -> less RAM, marginally faster
    num_threads=0,
    verbose=-1,
)
NUM_BOOST_ROUND = 1500
EARLY_STOPPING = 60


def load_features() -> tuple[pd.DataFrame, list[str]]:
    """Read the parquet, cast dtypes, return (train_frame, feature_cols)."""
    df = pd.read_parquet(ARTIFACTS / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["split"] == "train"].drop(columns=["split", "id"])
    if TRAIN_SINCE is not None:
        df = df[df["date"] >= pd.Timestamp(TRAIN_SINCE)]

    # Shrink: float32 for everything continuous, category codes for the rest.
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    feats = [c for c in df.columns if c not in ("date", "sales", "y")]
    return df.reset_index(drop=True), feats


def main() -> None:
    ensure_dirs()
    if not (ARTIFACTS / "features.parquet").exists():
        raise SystemExit("Run scripts/03_features.py first (no features.parquet).")

    df, feats = load_features()
    print(f"training rows: {len(df):,}  ({df['date'].min().date()} .. "
          f"{df['date'].max().date()})   features: {len(feats)}")

    windows = sliding_windows(df["date"].max(), n_folds=3, horizon=HORIZON)
    rows, importances, best_iters = [], [], []

    for w in windows:
        tr_mask = df["date"] <= w.train_end
        va_mask = (df["date"] >= w.valid_start) & (df["date"] <= w.valid_end)
        # Early-stopping slice: the tail of training, one horizon wide, so the
        # iteration count is never chosen by looking at the validation fold.
        es_mask = tr_mask & (df["date"] > w.train_end - pd.Timedelta(days=HORIZON))
        fit_mask = tr_mask & ~es_mask

        dtrain = lgb.Dataset(df.loc[fit_mask, feats], df.loc[fit_mask, "y"],
                             categorical_feature=CATEGORICAL, free_raw_data=True)
        dvalid = lgb.Dataset(df.loc[es_mask, feats], df.loc[es_mask, "y"],
                             reference=dtrain, free_raw_data=True)
        model = lgb.train(
            PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid], valid_names=["es"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred = np.expm1(model.predict(df.loc[va_mask, feats],
                                      num_iteration=model.best_iteration))
        score = rmsle(df.loc[va_mask, "sales"].values, pred)
        rows.append({"fold": w.fold,
                     "valid": f"{w.valid_start.date()}..{w.valid_end.date()}",
                     "best_iter": model.best_iteration,
                     "rmsle": score})
        best_iters.append(model.best_iteration)
        importances.append(pd.Series(
            model.feature_importance("gain"), index=feats))
        print(f"fold {w.fold}  {w.valid_start.date()}..{w.valid_end.date()}  "
              f"best_iter={model.best_iteration:4d}  RMSLE={score:.4f}", flush=True)
        del dtrain, dvalid, model
        gc.collect()

    res = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(res.to_string(index=False))
    print(f"\nLightGBM CV RMSLE: {res['rmsle'].mean():.4f} "
          f"+/- {res['rmsle'].std():.4f}")
    print(f"median best_iter: {int(np.median(best_iters))}  "
          "(use this for N_ROUNDS in 05_make_submission.py)")
    print("Compare against the best baseline from 02_baselines.py (~0.47).")

    imp = (pd.concat(importances, axis=1).mean(axis=1)
           .sort_values(ascending=False).rename("gain"))
    imp.to_csv(ARTIFACTS / "feature_importance.csv")
    fig, ax = plt.subplots(figsize=(8, 10))
    imp.head(25).iloc[::-1].plot.barh(ax=ax)
    ax.set(title="LightGBM feature importance (mean gain over folds)",
           xlabel="gain")
    fig.savefig(FIGURES / "importance.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {ARTIFACTS / 'feature_importance.csv'} and "
          f"{FIGURES / 'importance.png'}")
    print("\ntop 15 features by gain:")
    print(imp.head(15).to_string())


if __name__ == "__main__":
    main()
