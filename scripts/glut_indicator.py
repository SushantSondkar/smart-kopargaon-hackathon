"""
PHASE 1 - STEP 6: DEMAND / GLUT INDICATOR
============================================
Answers the PS's "Demand analysis" requirement, which nothing built so far
actually covers - forecasting and net-realization both serve "price
prediction" and "mandi comparison", but demand analysis needs the arrival
(supply) side specifically.

METHOD:
For each mandi, we compare today's arrival quantity to its own recent
"normal" level (trailing 30-day average of arrival days). If today's
arrivals are well above normal, that's a glut signal - a farmer holding
stock should expect downward price pressure. Well below normal = scarcity,
upward pressure.

We express this as a z-score (how many standard deviations from normal),
which is easy to turn into a simple traffic-light flag for the UI:
  z > +1.0   -> HIGH SUPPLY  (glut warning)
  z < -1.0   -> LOW SUPPLY   (scarcity / good time to sell)
  otherwise  -> NORMAL

VALIDATION: we don't just assert this matters - we test whether a HIGH
SUPPLY day actually predicts a price drop in the following days, using
only past data at each point (walk-forward, same discipline as the other
models).
"""

import pandas as pd
import numpy as np

ROLLING_WINDOW = 30   # trading days used to define "normal" arrivals
FORWARD_DAYS = 7       # how many days ahead we check price reaction


def load_arrivals():
    df = pd.read_csv('arrivals_clean.csv')
    df['date'] = pd.to_datetime(df['d'])
    df = df.rename(columns={'market_clean': 'mandi'})
    return df[['mandi', 'date', 'ArrivalQty']].sort_values(['mandi', 'date'])


def load_prices():
    df = pd.read_csv('onion_clean_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df[['mandi', 'date', 'price']]


def build_glut_indicator(mandi):
    arr = load_arrivals()
    arr = arr[arr.mandi == mandi].reset_index(drop=True)

    # Rolling mean/std computed using only PAST data (shift(1) first)
    arr['roll_mean'] = arr['ArrivalQty'].shift(1).rolling(ROLLING_WINDOW, min_periods=10).mean()
    arr['roll_std'] = arr['ArrivalQty'].shift(1).rolling(ROLLING_WINDOW, min_periods=10).std()
    arr['glut_zscore'] = (arr['ArrivalQty'] - arr['roll_mean']) / arr['roll_std']

    def flag(z):
        if pd.isna(z):
            return 'INSUFFICIENT_HISTORY'
        if z > 1.0:
            return 'HIGH_SUPPLY'
        if z < -1.0:
            return 'LOW_SUPPLY'
        return 'NORMAL'

    arr['supply_flag'] = arr['glut_zscore'].apply(flag)
    return arr


def validate_against_price(mandi, glut_df):
    """Does a HIGH_SUPPLY day actually predict a price drop over the next
    FORWARD_DAYS trading days? Walk-forward - only past info used."""
    prices = load_prices()
    prices = prices[prices.mandi == mandi].sort_values('date').reset_index(drop=True)

    merged = pd.merge_asof(glut_df.sort_values('date'),
                            prices.sort_values('date'),
                            on='date', direction='nearest', tolerance=pd.Timedelta('3D'))
    merged = merged.dropna(subset=['price']).reset_index(drop=True)

    # future price change over FORWARD_DAYS trading rows in the price series
    price_only = prices.reset_index(drop=True)
    price_map = price_only.set_index('date')['price']
    future_change = []
    dates_sorted = price_only['date'].tolist()
    for i, row in merged.iterrows():
        try:
            idx = dates_sorted.index(row['date'])
            if idx + FORWARD_DAYS < len(dates_sorted):
                future_price = price_only.iloc[idx + FORWARD_DAYS]['price']
                pct_change = (future_price - row['price']) / row['price'] * 100
                future_change.append(pct_change)
            else:
                future_change.append(np.nan)
        except ValueError:
            future_change.append(np.nan)
    merged['fwd_price_change_pct'] = future_change

    return merged.dropna(subset=['fwd_price_change_pct'])


if __name__ == '__main__':
    MANDI = 'KOPARGAON'

    glut = build_glut_indicator(MANDI)
    glut.to_csv('glut_indicator.csv', index=False)

    print("=" * 70)
    print(f"GLUT / DEMAND-PRESSURE INDICATOR - {MANDI}")
    print("=" * 70)
    print(f"Total days: {len(glut)}")
    print()
    print("Distribution of supply flags:")
    print(glut['supply_flag'].value_counts().to_string())
    print()

    validated = validate_against_price(MANDI, glut)
    print(f"Days with a matching price + valid {FORWARD_DAYS}-day-forward check: {len(validated)}")
    print()
    print(f"Average {FORWARD_DAYS}-day-forward price change, by supply flag on the day:")
    summary = validated.groupby('supply_flag')['fwd_price_change_pct'].agg(['mean', 'count'])
    print(summary.to_string())
    print()

    if 'HIGH_SUPPLY' in summary.index and 'LOW_SUPPLY' in summary.index:
        high = summary.loc['HIGH_SUPPLY', 'mean']
        low = summary.loc['LOW_SUPPLY', 'mean']
        print(f"HIGH_SUPPLY days were followed by {high:+.2f}% avg price change")
        print(f"LOW_SUPPLY days were followed by {low:+.2f}% avg price change")
        if high < low:
            print("CONFIRMED: high supply predicts price drops, low supply predicts price rises.")
            print("The direction is exactly what real supply-demand economics predicts.")
        else:
            print("WARNING: direction doesn't match expectation - flag this before using in demo.")

    validated.to_csv('glut_validation.csv', index=False)
    print("\nSaved glut_indicator.csv and glut_validation.csv")
