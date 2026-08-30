"""Step 3 — Feature engineering: turn the raw grid into a modelling table.

Feature groups built here:

  calendar    year, month, day, day-of-week, day-of-year, week, is_weekend,
              payday (Ecuador pays on the 15th and the last day of the month),
              days since / until payday.
  store       city, state, type, cluster  (static per store_nbr).
  promo       onpromotion (known for the future too), plus its recent history.
  holidays    national / regional / local holiday flags and an event flag,
              honouring `transferred` (a moved holiday did NOT happen that day)
              and the Transfer rows that mark where it landed.
  oil         interpolated daily WTI price + its 7-day change.
  lags        sales `HORIZON` days ago and further back.
  rolling     mean / std / min / max of sales over trailing windows, every one
              SHIFTED BY `HORIZON` days.

THE ONE RULE THAT MATTERS — no leakage:
  When you predict 2017-08-31 you only know sales through 2017-08-15. So any
  feature derived from the target must look back at least `HORIZON` (16) days.
  `lag_16` is the freshest sales number you are allowed to use. Break this and
  your backtest score will be fantastic and your real score will be garbage.

Writes artifacts/features.parquet (train rows + test rows, tagged by `split`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fc_lib import ARTIFACTS, HORIZON, ensure_dirs, load_raw, oil_daily

LAGS = [HORIZON, HORIZON + 1, HORIZON + 2, HORIZON + 5, HORIZON + 12, HORIZON + 19]
ROLL_WINDOWS = [7, 14, 28, 56, 84]


# --------------------------------------------------------------------------- #
def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["year"] = d.year
    df["month"] = d.month
    df["day"] = d.day
    df["dayofweek"] = d.dayofweek
    df["dayofyear"] = d.dayofyear
    df["weekofyear"] = d.isocalendar().week.astype("int16")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    # Payday: the 15th and the last calendar day of the month.
    is_month_end = df["date"] == (df["date"] + pd.offsets.MonthEnd(0))
    df["is_payday"] = ((df["day"] == 15) | is_month_end).astype("int8")
    day = df["day"].to_numpy()
    dim = df["date"].dt.days_in_month.to_numpy()
    # Next payday is the 15th (on/before it) or month-end (after it); 0 on a payday.
    df["days_to_payday"] = np.where(day <= 15, 15 - day, dim - day).astype("int16")
    # Previous payday was the 15th (if we're past it) or the previous month-end,
    # which was exactly `day` days ago.
    df["days_since_payday"] = np.where(day >= 15, day - 15, day).astype("int16")
    return df


def holiday_flags(holidays: pd.DataFrame, stores: pd.DataFrame,
                  keys: pd.DataFrame) -> pd.DataFrame:
    
    h = holidays.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["transferred"] = h["transferred"].astype(bool)

    # A 'Holiday' row with transferred=True did NOT give people the day off.
    # 'Transfer' rows mark the day it was actually observed. Bridge / Additional
    # are genuine extra days off. 'Work Day' is a compensating work day.
    real_off = h[
        ((h["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"]))
         & ~((h["type"] == "Holiday") & h["transferred"]))
    ]
    events = h[h["type"] == "Event"]
    workdays = h[h["type"] == "Work Day"]

    nat = set(real_off.loc[real_off["locale"] == "National", "date"])
    reg = set(zip(real_off.loc[real_off["locale"] == "Regional", "locale_name"],
                  real_off.loc[real_off["locale"] == "Regional", "date"]))
    loc = set(zip(real_off.loc[real_off["locale"] == "Local", "locale_name"],
                  real_off.loc[real_off["locale"] == "Local", "date"]))
    event_dates = set(events["date"])
    workday_dates = set(workdays["date"])

    k = keys.merge(stores[["store_nbr", "city", "state"]], on="store_nbr", how="left")
    k["is_holiday_national"] = k["date"].isin(nat).astype("int8")
    k["is_holiday_regional"] = [
        (s, d) in reg for s, d in zip(k["state"], k["date"])
    ]
    k["is_holiday_local"] = [
        (c, d) in loc for c, d in zip(k["city"], k["date"])
    ]
    k["is_holiday_regional"] = k["is_holiday_regional"].astype("int8")
    k["is_holiday_local"] = k["is_holiday_local"].astype("int8")
    k["is_holiday_any"] = (
        k[["is_holiday_national", "is_holiday_regional", "is_holiday_local"]]
        .max(axis=1).astype("int8")
    )
    k["is_event"] = k["date"].isin(event_dates).astype("int8")
    k["is_workday_makeup"] = k["date"].isin(workday_dates).astype("int8")
    return k.drop(columns=["city", "state"])


def add_lags_and_rolls(df: pd.DataFrame) -> pd.DataFrame:
    """Per (store_nbr, family) time-series features on `sales`.

    df must be sorted by date within each series and contain the FULL daily
    grid for both train and test so shifts line up on calendar days.
    """
    df = df.sort_values(["store_nbr", "family", "date"])
    g = df.groupby(["store_nbr", "family"], observed=True)["sales"]

    for lag in LAGS:
        df[f"sales_lag_{lag}"] = g.shift(lag).astype("float32")

    # Rolling stats on a series already shifted by HORIZON, so the window can
    # never touch a day inside the forecast horizon.
    shifted = g.shift(HORIZON)
    grp = shifted.groupby([df["store_nbr"], df["family"]], observed=True)
    for w in ROLL_WINDOWS:
        r = grp.rolling(w, min_periods=max(2, w // 2))
        df[f"sales_rmean_{w}"] = r.mean().reset_index(level=[0, 1], drop=True).astype("float32")
        df[f"sales_rstd_{w}"] = r.std().reset_index(level=[0, 1], drop=True).astype("float32")
        df[f"sales_rmax_{w}"] = r.max().reset_index(level=[0, 1], drop=True).astype("float32")

    # Promo history (onpromotion is known for the future, so no shift needed
    # for the current value; we still add a trailing sum for momentum).
    gp = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"]
    df["promo_roll_14"] = (
        gp.shift(1).groupby([df["store_nbr"], df["family"]], observed=True)
        .rolling(14, min_periods=3).sum()
        .reset_index(level=[0, 1], drop=True).astype("float32")
    )
    return df


# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_dirs()
    raw = load_raw()
    train, test = raw["train"].copy(), raw["test"].copy()
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    train["split"] = "train"
    test["split"] = "test"
    test["sales"] = np.nan
    df = pd.concat([train, test], ignore_index=True)

    # --- static store attributes ---
    df = df.merge(raw["stores"], on="store_nbr", how="left")
    for c in ["city", "state", "type"]:
        df[c] = df[c].astype("category")

    # --- calendar ---
    df = calendar_features(df)

    # --- holidays ---
    keys = df[["date", "store_nbr"]].drop_duplicates()
    hf = holiday_flags(raw["holidays"], raw["stores"], keys)
    df = df.merge(hf, on=["date", "store_nbr"], how="left")

    # --- oil ---
    oil = oil_daily(raw["oil"], df["date"].min(), df["date"].max())
    df = df.merge(oil, on="date", how="left")

    # --- lags / rolling ---
    df = add_lags_and_rolls(df)

    # --- target on the log scale (what the model will actually fit) ---
    df["y"] = np.log1p(df["sales"])

    out = ARTIFACTS / "features.parquet"
    df.to_parquet(out, index=False)
    feat_cols = [c for c in df.columns if c not in
                 ("id", "date", "sales", "y", "split")]
    print(f"wrote {out}  shape={df.shape}")
    print(f"{len(feat_cols)} feature columns:")
    print("  " + ", ".join(feat_cols))
    na = df.loc[df.split == "test", feat_cols].isna().mean().sort_values(ascending=False)
    print("\ntop feature NaN rates on TEST rows (lags need warm-up history):")
    print(na.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
