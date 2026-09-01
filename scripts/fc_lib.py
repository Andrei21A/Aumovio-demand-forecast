"""Shared helpers for the demand-forecasting scripts.

Nothing here is specific to a model. It is the plumbing every step reuses:
loading the raw CSVs efficiently, the competition metric (RMSLE), and a
*time-aware* backtest splitter — the single most important tool in
forecasting, because a random train/test split leaks the future into the
past and flatters every model build.

Import it from the numbered scripts:

    from fc_lib import load_raw, rmsle, sliding_windows
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"  # parquet feature tables, model dumps

# The test window is 16 days (2017-08-16 .. 2017-08-31). We reuse that number
# everywhere: every backtest fold predicts exactly this far ahead, so the
# score we measure offline is comparable to the real task.
HORIZON = 16


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw() -> dict[str, pd.DataFrame]:
    """Read every CSV once, with sane dtypes.

    train.csv is ~122 MB / 3M rows. We let DuckDB do the parse and the dtype
    downcast in C, then hand back a pandas frame. `family` and `store_nbr`
    become pandas categoricals — that keeps memory down and lets LightGBM
    treat them as native categorical splits later.
    """
    con = duckdb.connect()

    def q(sql: str) -> pd.DataFrame:
        return con.sql(sql).df()

    train = q(f"""
        SELECT id,
               date::DATE            AS date,
               store_nbr::SMALLINT   AS store_nbr,
               family,
               sales::DOUBLE         AS sales,
               onpromotion::INTEGER  AS onpromotion
        FROM read_csv_auto('{DATA / "train.csv"}')
    """)
    test = q(f"""
        SELECT id,
               date::DATE            AS date,
               store_nbr::SMALLINT   AS store_nbr,
               family,
               onpromotion::INTEGER  AS onpromotion
        FROM read_csv_auto('{DATA / "test.csv"}')
    """)
    stores = q(f"SELECT * FROM read_csv_auto('{DATA / 'stores.csv'}')")
    txns = q(f"""
        SELECT date::DATE AS date, store_nbr::SMALLINT AS store_nbr,
               transactions::INTEGER AS transactions
        FROM read_csv_auto('{DATA / "transactions.csv"}')
    """)
    oil = q(f"""
        SELECT date::DATE AS date, dcoilwtico
        FROM read_csv_auto('{DATA / "oil.csv"}')
    """)
    holidays = q(f"SELECT * FROM read_csv_auto('{DATA / 'holidays_events.csv'}')")
    con.close()

    for df in (train, test):
        df["family"] = df["family"].astype("category")

    return {
        "train": train,
        "test": test,
        "stores": stores,
        "transactions": txns,
        "oil": oil,
        "holidays": holidays,
    }


def oil_daily(oil: pd.DataFrame, start, end) -> pd.DataFrame:
    """Oil price on a *complete* daily calendar.

    oil.csv only has weekday rows and some of those are NaN. Markets don't
    move on weekends, so forward/backward interpolation on a full calendar is
    the honest fill. We also add a 7-day trend because the *direction* of the
    price often matters more than the level.
    """
    cal = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    oil = oil.copy()
    oil["date"] = pd.to_datetime(oil["date"])
    out = cal.merge(oil, on="date", how="left")
    out["dcoilwtico"] = out["dcoilwtico"].interpolate("linear", limit_direction="both")
    out["oil_7d_chg"] = out["dcoilwtico"].diff(7)
    return out


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def rmsle(y_true, y_pred) -> float:
    """Root Mean Squared Logarithmic Error — the competition metric.

    RMSLE = sqrt( mean( (log1p(pred) - log1p(true))^2 ) )

    Properties worth internalising:
      * It measures *ratio* error, not absolute error. Predicting 10 when the
        truth is 20 is penalised the same as predicting 100 vs 200.
      * It punishes under-prediction more than over-prediction.
      * Because of the log, the usual move is to train on log1p(sales) with a
        plain squared-error loss — then RMSLE is just RMSE in that space.

    Sales are never negative here, but model outputs can be, so we clip.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# --------------------------------------------------------------------------- #
# Time-aware backtest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    """One backtest fold: train on everything strictly before `valid_start`,
    predict the `HORIZON` days starting at `valid_start`."""

    fold: int
    train_end: pd.Timestamp   # last date the model is allowed to see
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def sliding_windows(
    max_date: pd.Timestamp,
    n_folds: int = 4,
    horizon: int = HORIZON,
    step: int = HORIZON,
) -> list[Window]:
    """Build `n_folds` consecutive out-of-sample windows ending at `max_date`.

    Fold k trains on [start .. cut_k] and validates on the next `horizon`
    days. Folds march backwards from the end of the data so the last fold's
    validation window is the freshest data — the closest proxy for
    the real 2017-08-16 task.

        train ................|  valid(16d)                     fold 0 (oldest)
        train ....................|  valid(16d)                 fold 1
        train ........................|  valid(16d)             fold 2
        train ............................|  valid(16d)         fold 3 (newest)

    train_end and valid_start are adjacent days, but the *features* must still
    respect the horizon (see 03_features.py): when it predicts 2017-08-31 it
    only knows sales up to 2017-08-15, so any lag shorter than 16 days is
    unusable and every rolling feature has to be shifted by 16.
    """
    max_date = pd.Timestamp(max_date)
    windows = []
    for k in range(n_folds):
        offset = (n_folds - 1 - k) * step
        valid_end = max_date - pd.Timedelta(days=offset)
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        train_end = valid_start - pd.Timedelta(days=1)
        windows.append(
            Window(
                fold=k,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return windows


def ensure_dirs() -> None:
    for d in (REPORTS, FIGURES, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
