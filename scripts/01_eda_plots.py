"""Step 1 — Visual EDA

`explore_data.py` already checked the data is clean. This script asks the
different question a forecaster cares about: what patterns exist, and how
strong are they?

WHAT TO LOOK FOR 
  1 total_sales_trend.png
      - Is there a long-run trend (growth/decline)?  -> a time index feature.
      - Level shifts or spikes on specific dates?     -> holiday / event flags.
      - The 2016-04-16 earthquake bump is the classic example here.
  2 weekly_seasonality.png / monthly_seasonality.png
      - Which weekdays are high? How big is Sun vs Wed?  -> day-of-week feature.
      - Payday effect: Ecuador pays on the 15th and last day of the month.
  3 yearly_seasonality.png
      - December ramp, January hangover, mid-year dips -> month / day-of-year.
  4 family_heatmap.png
      - Families live on totally different scales (GROCERY I vs BOOKS).
        -> per-series normalisation, or let the model split on `family`.
      - Some families are ~always zero -> consider dropping or special-casing.
  5 autocorrelation.png
      - How many days back does the past predict the future? The lags with
        high ACF/PACF are exactly the lag features worth engineering.
  6 promo_effect.png
      - Does `onpromotion` actually lift sales, and by how much?
  7 oil_vs_sales.png
      - Ecuador's economy is oil-linked. Is there a visible relationship, or
        is it too slow-moving to matter for a 16-day forecast?
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fc_lib import FIGURES, ensure_dirs, load_raw, oil_daily
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def save(fig, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIGURES.parents[1])}")


def main() -> None:
    ensure_dirs()
    raw = load_raw()
    train = raw["train"].copy()
    train["date"] = pd.to_datetime(train["date"])


    daily = train.groupby("date", as_index=False)["sales"].sum()

    # 1 -- trend -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily["date"], daily["sales"], lw=0.7)
    ax.plot(daily["date"], daily["sales"].rolling(28, center=True).mean(),
            color="crimson", lw=2, label="28-day moving average")
    ax.axvline(pd.Timestamp("2016-04-16"), color="k", ls="--", lw=1,
               label="2016 earthquake")
    ax.set(title="Total daily sales — trend & level shifts", ylabel="sales")
    ax.legend()
    save(fig, "total_sales_trend.png")

    # 2 -- weekly & monthly seasonality ----------------------------------
    daily["dow"] = daily["date"].dt.day_name()
    daily["dom"] = daily["date"].dt.day
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    daily.groupby("dow")["sales"].mean().reindex(order).plot.bar(ax=axes[0])
    axes[0].set(title="Mean sales by day of week", xlabel="")
    daily.groupby("dom")["sales"].mean().plot.bar(ax=axes[1])
    axes[1].set(title="Mean sales by day of month (payday = 15 & last)",
                xlabel="day of month")
    save(fig, "weekly_seasonality.png")

    # 3 -- yearly shape (overlay each year) -----------------------------
    daily["doy"] = daily["date"].dt.dayofyear
    daily["year"] = daily["date"].dt.year
    fig, ax = plt.subplots(figsize=(13, 4))
    for yr, g in daily.groupby("year"):
        ax.plot(g["doy"], g["sales"].rolling(7, center=True).mean(), label=str(yr))
    ax.set(title="Yearly seasonality (7-day smoothed, overlaid by year)",
           xlabel="day of year", ylabel="sales")
    ax.legend()
    save(fig, "yearly_seasonality.png")

    # 4 -- family scale differences ------------------------------------
    fam = (train.groupby("family", observed=True)["sales"]
           .mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(9, 9))
    fam.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set(title="Mean sales per family (note the scale spread)", xlabel="sales")
    save(fig, "family_scale.png")

    # 5 -- autocorrelation of the total series -------------------------
    # De-trend a little with a 1st difference so ACF reflects seasonality,
    # not the slow drift.
    s = daily.set_index("date")["sales"].asfreq("D").interpolate()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    plot_acf(s.diff().dropna(), lags=40, ax=axes[0])
    axes[0].set_title("ACF of differenced daily sales (spikes at 7,14,21 = weekly)")
    plot_pacf(s.diff().dropna(), lags=40, ax=axes[1], method="ywm")
    axes[1].set_title("PACF (which specific lags carry independent signal)")
    save(fig, "autocorrelation.png")

    # 6 -- promo effect ----------------------------------------------
    # Compare same family with/without promo, on the per-row level.
    pr = train.assign(promo=lambda d: d["onpromotion"] > 0)
    eff = (pr.groupby(["family", "promo"], observed=True)["sales"].mean()
           .unstack("promo"))
    eff.columns = ["no_promo", "promo"]
    eff["lift_x"] = eff["promo"] / eff["no_promo"].replace(0, np.nan)
    eff = eff.sort_values("lift_x", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 9))
    eff["lift_x"].plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.axvline(1.0, color="k", lw=1)
    ax.set(title="Promo sales lift (x times the no-promo mean)", xlabel="lift")
    save(fig, "promo_effect.png")
    print("\n  promo lift, top 10 families:")
    print(eff.head(10).round(2).to_string())

    # 7 -- oil vs sales --------------------------------------------
    oil = oil_daily(raw["oil"], daily["date"].min(), daily["date"].max())
    m = daily.merge(oil, on="date")
    fig, ax1 = plt.subplots(figsize=(13, 4))
    ax1.plot(m["date"], m["sales"].rolling(28, center=True).mean(),
             color="tab:blue", label="sales (28d MA)")
    ax1.set_ylabel("sales", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(m["date"], m["dcoilwtico"], color="tab:red", label="oil (WTI)")
    ax2.set_ylabel("oil price USD", color="tab:red")
    ax1.set_title(
        f"Sales vs oil price  |  corr(level) = "
        f"{m['sales'].corr(m['dcoilwtico']):.2f}"
    )
    save(fig, "oil_vs_sales.png")

    print("\nDone. Open the PNGs in reports/figures/ and answer the questions "
          "in this file's docstring before moving to step 2.")


if __name__ == "__main__":
    main()
