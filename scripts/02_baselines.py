"""Step 2 — Baselines + a trustworthy backtest.

So step 2 does two things:

  A. Establishes DUMB baselines. Any real model must beat these or it is not
     earning its complexity.
  B. Scores them with a *sliding time window* backtest (fc_lib.sliding_windows)
     so you see not just an average RMSLE but how much it varies fold to fold.

Baselines implemented (all per (store_nbr, family) series):
  * mean_all      - historical mean sales. The "no information" floor.
  * last_value    - carry the last observed day forward (random-walk).
  * ma_28         - mean of the last 28 days, carried flat.
  * dow_mean_8w   - mean by day-of-week over the last 8 weeks. Captures the
                    weekly cycle, which is the strongest pattern in step 1.

WHAT TO LOOK FOR
  * Which baseline wins?
  * The spread across folds. If fold RMSLE ranges 0.45-0.65, a model that
    scores 0.02 "better" on one split has proven nothing.
  * The step-3/4 model should be compared against THIS table, fold for fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fc_lib import HORIZON, load_raw, rmsle, sliding_windows


def predict_baselines(train_part: pd.DataFrame, valid_keys: pd.DataFrame,
                      valid_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return one prediction column per baseline, aligned to valid_keys.

    valid_keys: the (date, store_nbr, family) rows we must predict.
    train_part: everything the model is allowed to see (date <= train_end).
    """
    g = train_part.groupby(["store_nbr", "family"], observed=True)
    out = valid_keys.copy()

    # mean_all -------------------------------------------------------------
    mean_all = g["sales"].mean().rename("mean_all")
    out = out.merge(mean_all, on=["store_nbr", "family"], how="left")

    # last_value ---------------------------------------------------------
    last_val = (train_part.sort_values("date")
                .groupby(["store_nbr", "family"], observed=True)["sales"]
                .last().rename("last_value"))
    out = out.merge(last_val, on=["store_nbr", "family"], how="left")

    # ma_28 ------------------------------------------------------------
    cutoff = train_part["date"].max() - pd.Timedelta(days=28)
    ma28 = (train_part[train_part["date"] > cutoff]
            .groupby(["store_nbr", "family"], observed=True)["sales"]
            .mean().rename("ma_28"))
    out = out.merge(ma28, on=["store_nbr", "family"], how="left")

    # dow_mean_8w ----------------------------------------------------
    cutoff8 = train_part["date"].max() - pd.Timedelta(days=56)
    recent = train_part[train_part["date"] > cutoff8].copy()
    recent["dow"] = recent["date"].dt.dayofweek
    dow = (recent.groupby(["store_nbr", "family", "dow"], observed=True)["sales"]
           .mean().rename("dow_mean_8w").reset_index())
    out["dow"] = out["date"].dt.dayofweek
    out = out.merge(dow, on=["store_nbr", "family", "dow"], how="left")

    # Fill any series with no recent history: fall back to the coarser number.
    out["ma_28"] = out["ma_28"].fillna(out["mean_all"])
    out["dow_mean_8w"] = out["dow_mean_8w"].fillna(out["ma_28"])
    out["last_value"] = out["last_value"].fillna(out["mean_all"])
    for c in ["mean_all", "last_value", "ma_28", "dow_mean_8w"]:
        out[c] = out[c].fillna(0.0).clip(lower=0)
    return out


def main() -> None:
    raw = load_raw()
    train = raw["train"].copy()
    train["date"] = pd.to_datetime(train["date"])

    methods = ["mean_all", "last_value", "ma_28", "dow_mean_8w"]
    windows = sliding_windows(train["date"].max(), n_folds=4, horizon=HORIZON)

    records = []
    for w in windows:
        train_part = train[train["date"] <= w.train_end]
        valid = train[(train["date"] >= w.valid_start)
                      & (train["date"] <= w.valid_end)].copy()

        preds = predict_baselines(
            train_part,
            valid[["date", "store_nbr", "family"]],
            pd.DatetimeIndex(valid["date"].unique()),
        )
        valid = valid.reset_index(drop=True)
        row = {"fold": w.fold,
               "train_end": w.train_end.date(),
               "valid": f"{w.valid_start.date()}..{w.valid_end.date()}",
               "n": len(valid)}
        for m in methods:
            row[m] = rmsle(valid["sales"].values, preds[m].values)
        records.append(row)
        print(f"fold {w.fold}  train<= {w.train_end.date()}  "
              + "  ".join(f"{m}={row[m]:.4f}" for m in methods))

    res = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print("RMSLE by fold (lower is better):")
    print(res[["fold", "valid", "n", *methods]].to_string(index=False))
    print("\nmean +/- std across folds:")
    summary = pd.DataFrame({
        "mean": res[methods].mean(),
        "std": res[methods].std(),
    }).sort_values("mean")
    print(summary.round(4).to_string())
    print(f"\nBest baseline: {summary.index[0]}  "
          f"(RMSLE {summary['mean'].iloc[0]:.4f}). "
          "Beat this in step 4 or your model is not pulling its weight.")


if __name__ == "__main__":
    main()
