"""
PHASE 1 - STEP 4: KOPARGAON BASIS MODEL
=========================================
The problem: Kopargaon only reports prices on ~40% of trading days.
Lasalgaon (the benchmark yard) reports far more often.

The fix ("basis modelling"): on days Kopargaon DOES report, its price sits
close to a stable ratio of Lasalgaon's price (median ~0.93 - Kopargaon
trades about 7% below Lasalgaon). We learn that ratio from history, then
apply it to Lasalgaon's price to estimate Kopargaon's price EVEN ON DAYS
KOPARGAON NEVER REPORTED.

Two versions, weakest to strongest, both walk-forward (only ever using
data from BEFORE the day being predicted - no future leakage):

  1. GLOBAL RATIO   - one median ratio learned from all history to date
  2. ROLLING RATIO  - median ratio from just the last 90 matched days
                      (adapts if the relationship drifts over time)

We compare both against a naive "assume Kopargaon = Lasalgaon" baseline
(ratio = 1.0), to prove the basis approach is actually adding value and
not just noise.

Finally: combine this ratio with the Lasalgaon horizon forecast from
train_model.py to produce a full Kopargaon P10/P50/P90 forecast - the
actual deliverable for the app.
"""

import pandas as pd
import numpy as np

ROLLING_WINDOW = 90  # number of past MATCHED days used for the rolling ratio


def load_matched_series():
    df = pd.read_csv('onion_clean_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    las = df[df.mandi == 'LASALGAON'][['date', 'price']].rename(columns={'price': 'las_price'})
    kop = df[df.mandi == 'KOPARGAON'][['date', 'price']].rename(columns={'price': 'kop_price'})
    m = pd.merge(kop, las, on='date', how='inner').sort_values('date').reset_index(drop=True)
    m['ratio'] = m['kop_price'] / m['las_price']
    return m


def walk_forward_basis(m, test_frac=0.2):
    n = len(m)
    test_start = int(n * (1 - test_frac))

    rows = []
    for i in range(test_start, n):
        past = m.iloc[:i]  # only data strictly BEFORE this point - no leakage
        if len(past) < 10:
            continue

        global_ratio = past['ratio'].median()
        rolling_ratio = past['ratio'].tail(ROLLING_WINDOW).median()

        actual = m.iloc[i]['kop_price']
        las_today = m.iloc[i]['las_price']

        rows.append({
            'date': m.iloc[i]['date'],
            'actual_kop_price': actual,
            'las_price': las_today,
            'pred_naive_ratio1': las_today * 1.0,
            'pred_global_ratio': las_today * global_ratio,
            'pred_rolling_ratio': las_today * rolling_ratio,
            'global_ratio_used': global_ratio,
            'rolling_ratio_used': rolling_ratio,
        })

    return pd.DataFrame(rows)


def mape(actual, pred):
    return (abs(actual - pred) / actual).mean() * 100


if __name__ == '__main__':
    m = load_matched_series()
    print(f"Matched Kopargaon-Lasalgaon days: {len(m)}")
    print(f"Date range: {m.date.min().date()} to {m.date.max().date()}")
    print()

    result = walk_forward_basis(m)
    print(f"Test window: {result.date.min().date()} to {result.date.max().date()} "
          f"({len(result)} days)")
    print()

    mape_naive = mape(result['actual_kop_price'], result['pred_naive_ratio1'])
    mape_global = mape(result['actual_kop_price'], result['pred_global_ratio'])
    mape_rolling = mape(result['actual_kop_price'], result['pred_rolling_ratio'])

    print("=" * 70)
    print("BASIS MODEL RESULTS - predicting Kopargaon price from Lasalgaon price")
    print("=" * 70)
    print(f"{'Method':30s} {'MAPE':>10s}")
    print(f"{'Naive (Kopargaon = Lasalgaon)':30s} {mape_naive:9.2f}%")
    print(f"{'Global median ratio':30s} {mape_global:9.2f}%")
    print(f"{'Rolling 90-day median ratio':30s} {mape_rolling:9.2f}%")
    print()
    best = min(mape_naive, mape_global, mape_rolling)
    if best == mape_rolling:
        print("BEST: Rolling ratio - the spread drifts over time, adapting helps.")
    elif best == mape_global:
        print("BEST: Global ratio - the spread is stable, no need to adapt.")
    else:
        print("BEST: Naive (ratio=1) - the basis approach isn't adding value here.")
        print("      This would be an honest negative result - report it as such.")

    result.to_csv('kopargaon_basis_results.csv', index=False)
    print("\nSaved kopargaon_basis_results.csv")

    # -----------------------------------------------------------------
    # Combine with the Lasalgaon horizon forecast to get the real deliverable:
    # a Kopargaon P10/P50/P90 forecast, usable even on days Kopargaon
    # itself never reported.
    # -----------------------------------------------------------------
    try:
        las_forecast = pd.read_csv('model_results.csv')
        las_forecast['date'] = pd.to_datetime(las_forecast['date'])
        latest_ratio = m['ratio'].tail(ROLLING_WINDOW).median()

        kop_forecast = las_forecast.copy()
        for col in ['p10', 'p50', 'p90']:
            kop_forecast[col] = kop_forecast[col] * latest_ratio
        kop_forecast = kop_forecast.rename(
            columns={'p10': 'kop_p10', 'p50': 'kop_p50', 'p90': 'kop_p90'})

        kop_forecast[['date', 'kop_p10', 'kop_p50', 'kop_p90']].to_csv(
            'kopargaon_forecast.csv', index=False)

        print(f"\nUsing latest rolling ratio ({latest_ratio:.3f}) applied to "
              f"Lasalgaon's 7-day forecast:")
        print("Saved kopargaon_forecast.csv - Kopargaon forecast for EVERY day "
              "Lasalgaon has a forecast, regardless of whether Kopargaon reported.")
        print()
        print(kop_forecast[['date', 'kop_p10', 'kop_p50', 'kop_p90']].tail(5).to_string(index=False))
    except FileNotFoundError:
        print("\n(model_results.csv not found - run train_model.py first to get "
              "the combined Kopargaon forecast)")
