"""
PHASE 1 - STEP 1: BUILD THE CLEAN MODELLING SERIES
====================================================
Takes the raw merged price data (combined_clean.csv) and produces one clean,
ready-to-model daily price series per mandi, using rules we validated by hand
in Phase 0:

  - Business days only (mandis are closed Sundays - never interpolate)
  - Variety: Red is primary; where Red is missing on a date, backfill with
    Other (validated: r=0.86 correlated with Red at Lasalgaon, same price
    level on average at Kopargaon - NOT a cheap cull grade)
  - Duplicate same-day rows (seen at Pimpalgaon) are averaged, not dropped
  - Output is one row per (mandi, date): modal price, plus which variety
    the value came from, so the model can weight confidence accordingly
"""

import pandas as pd
import numpy as np

CATCHMENT = ['KOPARGAON', 'RAHATA', 'SANGAMNER', 'LASALGAON',
             'LASALGAON(NIPHAD)', 'LASALGAON(VINCHUR)', 'PIMPALGAON BASWANT',
             'PIMPALGAON BASWANT(SAYKHEDA)', 'YEOLA', 'MANMAD', 'NANDGAON',
             'VAIJAPUR']


def build_clean_series(price_csv='combined_clean.csv'):
    df = pd.read_csv(price_csv)
    df['d'] = pd.to_datetime(df['d'])
    df = df[df['market_clean'].isin(CATCHMENT)]

    # Average any same-day duplicate rows (same mandi, date, variety)
    df = df.groupby(['market_clean', 'd', 'Variety'], as_index=False)['Modal_Price'].mean()

    # Pivot so each row = one mandi-date, columns = price under each variety
    piv = df.pivot_table(index=['market_clean', 'd'], columns='Variety',
                          values='Modal_Price', aggfunc='mean').reset_index()

    # Primary = Red. Backfill missing Red with Other. Track source for transparency.
    piv['price'] = piv.get('Red')
    if 'Other' in piv.columns:
        used_other = piv['price'].isna() & piv['Other'].notna()
        piv.loc[used_other, 'price'] = piv.loc[used_other, 'Other']
    else:
        used_other = pd.Series(False, index=piv.index)

    piv['price_source'] = np.where(piv['Red'].notna(), 'Red',
                            np.where(used_other, 'Other', 'missing'))

    clean = piv[['market_clean', 'd', 'price', 'price_source']].dropna(subset=['price'])
    clean = clean.rename(columns={'market_clean': 'mandi', 'd': 'date'})

    # Drop Sundays if any slipped through (mandis don't trade)
    clean = clean[clean['date'].dt.dayofweek != 6]

    clean = clean.sort_values(['mandi', 'date']).reset_index(drop=True)
    return clean


if __name__ == '__main__':
    clean = build_clean_series()
    clean.to_csv('onion_clean_series.csv', index=False)

    print("=" * 70)
    print("CLEAN SERIES BUILT")
    print("=" * 70)
    print(f"Total rows: {len(clean)}")
    print()
    print("Rows per mandi, and how much came from Red vs Other backfill:")
    summary = clean.groupby('mandi').agg(
        days=('date', 'nunique'),
        first=('date', 'min'),
        last=('date', 'max'),
        pct_red=('price_source', lambda s: round(100 * (s == 'Red').mean(), 1))
    )
    print(summary.to_string())
    print()
    print("Saved to onion_clean_series.csv")
