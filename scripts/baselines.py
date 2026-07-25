"""
PHASE 1 - STEP 2: BASELINES
============================
Before any machine learning, we need a bar to beat. Two dead-simple methods:

  1. NAIVE: tomorrow's price = today's price
  2. MA7:    tomorrow's price = average of last 7 trading days

We evaluate with WALK-FORWARD validation: for every day in the test window,
only use data that would have actually been available at that point in time.
This is the only honest way to test a time series - a random train/test split
leaks the future and inflates the score.

Horizon tested: 7 days ahead (the PS's shortest promised horizon).
"""

import pandas as pd
import numpy as np

MANDI = 'LASALGAON'
HORIZON = 7
TEST_FRACTION = 0.2  # last 20% of the series is held out for testing


def load_series(mandi):
    df = pd.read_csv('onion_clean_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    s = df[df.mandi == mandi].sort_values('date').reset_index(drop=True)
    return s


def walk_forward_eval(s, horizon, test_frac):
    n = len(s)
    test_start = int(n * (1 - test_frac))
    prices = s['price'].values
    dates = s['date'].values

    naive_errs, ma7_errs = [], []
    rows = []

    for i in range(test_start, n - horizon):
        actual = prices[i + horizon]
        naive_pred = prices[i]                       # last known price
        ma7_pred = prices[max(0, i - 6):i + 1].mean()  # last 7 days incl today

        naive_errs.append(abs(actual - naive_pred) / actual)
        ma7_errs.append(abs(actual - ma7_pred) / actual)

        rows.append({
            'date_forecast_made': dates[i], 'date_target': dates[i + horizon],
            'actual': actual, 'naive_pred': naive_pred, 'ma7_pred': ma7_pred
        })

    result = pd.DataFrame(rows)
    return result, np.mean(naive_errs) * 100, np.mean(ma7_errs) * 100


if __name__ == '__main__':
    s = load_series(MANDI)
    print(f"Mandi: {MANDI} | total usable days: {len(s)}")
    print(f"Series: {s.date.min().date()} to {s.date.max().date()}")
    print()

    result, naive_mape, ma7_mape = walk_forward_eval(s, HORIZON, TEST_FRACTION)

    print(f"Test window: {result.date_forecast_made.min().date()} to "
          f"{result.date_forecast_made.max().date()}  ({len(result)} forecasts)")
    print()
    print(f"{'Method':10s} {'MAPE (7-day ahead)':>20s}")
    print(f"{'Naive':10s} {naive_mape:19.2f}%")
    print(f"{'MA7':10s} {ma7_mape:19.2f}%")
    print()
    print(f"BASELINE TO BEAT: {min(naive_mape, ma7_mape):.2f}% MAPE at horizon={HORIZON} days")

    result.to_csv('baseline_results.csv', index=False)
    print("\nSaved baseline_results.csv")
