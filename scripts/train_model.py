"""
PHASE 1 - STEP 3: LIGHTGBM QUANTILE MODEL
===========================================
Key design choices (each one matters, don't simplify these away):

1. TARGET = ratio, not price. We predict price[t+horizon] / price[t], not the
   raw price. A tree model can never predict outside the range it trained on -
   if we predicted price directly and onion later spikes past our training
   max, the model silently caps out exactly when accuracy matters most.

2. THREE MODELS PER HORIZON (P10 / P50 / P90) using quantile objective, so
   the output is an honest range, not a fake single number.

3. WALK-FORWARD split - same test window as baselines.py, so the comparison
   is apples to apples.

4. Features only use information available at the time the forecast is made
   (lags and rolling stats are all shifted correctly - no leakage).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

MANDI = 'LASALGAON'
HORIZON = 7
TEST_FRACTION = 0.2
QUANTILES = {'p10': 0.1, 'p50': 0.5, 'p90': 0.9}


def load_series(mandi):
    df = pd.read_csv('onion_clean_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df[df.mandi == mandi].sort_values('date').reset_index(drop=True)


def make_features(s, horizon):
    df = s.copy()
    p = df['price']

    # Lags (trading-day lags, since our series is trading days not calendar days)
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f'lag_{lag}'] = p.shift(lag)

    # Rolling stats - shift(1) first so "today" is never included in its own window
    for win in [7, 14, 30]:
        df[f'roll_mean_{win}'] = p.shift(1).rolling(win).mean()
        df[f'roll_std_{win}'] = p.shift(1).rolling(win).std()

    # Momentum: how much has price moved recently
    df['change_7d'] = p.shift(1) / p.shift(8) - 1
    df['change_30d'] = p.shift(1) / p.shift(31) - 1

    # Calendar features
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow'] = df['date'].dt.dayofweek

    # TARGET: ratio of future price to today's price (ratio-based, not absolute)
    df['target_ratio'] = p.shift(-horizon) / p

    feature_cols = [c for c in df.columns if c.startswith(('lag_', 'roll_', 'change_'))
                     ] + ['month_sin', 'month_cos', 'dow']
    return df, feature_cols


def train_and_eval(mandi, horizon, test_frac):
    s = load_series(mandi)
    df, feat_cols = make_features(s, horizon)

    df_model = df.dropna(subset=feat_cols + ['target_ratio']).reset_index(drop=True)
    n = len(df_model)
    split = int(n * (1 - test_frac))
    train, test = df_model.iloc[:split], df_model.iloc[split:]

    print(f"Train: {len(train)} rows ({train.date.min().date()} to {train.date.max().date()})")
    print(f"Test:  {len(test)} rows  ({test.date.min().date()} to {test.date.max().date()})")

    preds = {}
    for name, q in QUANTILES.items():
        model = lgb.LGBMRegressor(
            objective='quantile', alpha=q,
            n_estimators=300, max_depth=4, learning_rate=0.03,
            min_child_samples=15, verbose=-1
        )
        model.fit(train[feat_cols], train['target_ratio'])
        preds[name] = model.predict(test[feat_cols])

    result = test[['date', 'price']].copy()
    result['target_date'] = None  # filled below using positional offset in trading days
    result['actual_future_price'] = test['price'] * test['target_ratio']
    for name in QUANTILES:
        result[name] = test['price'].values * preds[name]

    # Metrics
    mape = (abs(result['actual_future_price'] - result['p50']) / result['actual_future_price']).mean() * 100
    coverage = ((result['actual_future_price'] >= result['p10']) &
                (result['actual_future_price'] <= result['p90'])).mean() * 100

    return result, mape, coverage, feat_cols, train, test


if __name__ == '__main__':
    result, mape, coverage, feat_cols, train, test = train_and_eval(MANDI, HORIZON, TEST_FRACTION)

    print()
    print("=" * 70)
    print(f"LIGHTGBM RESULTS - {MANDI}, {HORIZON}-day horizon")
    print("=" * 70)
    print(f"MAPE (P50 vs actual):     {mape:.2f}%")
    print(f"P10-P90 coverage:         {coverage:.1f}%  (target: ~80%)")
    print()
    print("Compare to baseline.py:")
    print(f"  Naive MAPE was 13.79%  ->  LightGBM is {'BETTER' if mape < 13.79 else 'WORSE'} "
          f"by {abs(13.79 - mape):.2f} points")

    result.to_csv('model_results.csv', index=False)
    print("\nSaved model_results.csv")
    print("\nSample of predictions (last 10 test days):")
    print(result[['date', 'price', 'p10', 'p50', 'p90', 'actual_future_price']].tail(10).to_string(index=False))
