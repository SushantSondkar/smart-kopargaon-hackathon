"""
PHASE 1 - STEP 5: NET REALIZATION DECISION LAYER
===================================================
This is the core "aha" feature from the problem statement: don't just show
the farmer the highest price mandi - show what he'd actually walk away with
after transport, commission (adat), loading labour (hamali), and weighing
charges (tolai). The highest GROSS price is very often not the highest NET.

*** IMPORTANT - READ BEFORE USING IN THE DEMO ***
The rates in cost_params.csv (adat %, hamali, tolai) and the transport rate
below are ASSUMED PLACEHOLDER VALUES based on typical Maharashtra APMC
practice - NOT yet confirmed with a real APMC. Before presenting real rupee
numbers to judges, call Kopargaon APMC (or Sangamner/Lasalgaon) and update
cost_params.csv with real figures. The MATHS and RANKING LOGIC below are
correct regardless - only the input numbers need verification.

Distances are straight-line road-distance estimates, not measured - also
flagged for verification.

FORMULA (documented so it's easy to defend on stage):
    gross          = quintals x price_per_quintal
    adat_cost      = gross x (adat_pct / 100)
    hamali_cost    = quintals x hamali_per_qtl
    tolai_cost     = quintals x tolai_per_qtl
    transport_cost = distance_km x transport_rate_per_km_per_qtl x quintals
    net            = gross - adat_cost - hamali_cost - tolai_cost - transport_cost
"""

import pandas as pd

# Assumed - typical rate for tractor-trolley/mini-truck hire in this belt.
# VERIFY before demo: call a transporter or APMC office.
TRANSPORT_RATE_PER_KM_PER_QTL = 2.0


MAX_STALENESS_DAYS = 21  # a price older than this vs the anchor date is not
                          # a fair same-day comparison - excluded, not silently used


def load_latest_prices():
    """
    FIX (v2): comparing each mandi's own 'latest' price was unfair - two
    mandis (Lasalgaon(Niphad), Rahata) happened to have much more recent
    reports (mid-July) than the rest of the catchment (early May), which
    would have made them look artificially dominant just because of WHEN
    they last reported, not because their price was actually higher on a
    like-for-like day.

    Fix: anchor every mandi to the same reference date (the median of each
    mandi's own latest-report date). Pull each mandi's most recent price ON
    OR BEFORE that anchor. If a mandi's most recent price is still older
    than MAX_STALENESS_DAYS relative to the anchor, it's excluded from the
    comparison rather than silently shown as if current.
    """
    df = pd.read_csv('onion_clean_series.csv')
    df['date'] = pd.to_datetime(df['date'])

    latest_per_mandi = df.sort_values('date').groupby('mandi')['date'].max()
    anchor_date = latest_per_mandi.median()

    on_or_before = df[df['date'] <= anchor_date]
    latest = on_or_before.sort_values('date').groupby('mandi').tail(1).copy()
    latest['staleness_days'] = (anchor_date - latest['date']).dt.days

    excluded = latest[latest['staleness_days'] > MAX_STALENESS_DAYS]
    included = latest[latest['staleness_days'] <= MAX_STALENESS_DAYS]

    if len(excluded):
        print(f"Anchor date for this comparison: {anchor_date.date()}")
        print(f"EXCLUDED (no price within {MAX_STALENESS_DAYS} days of anchor):")
        for _, row in excluded.iterrows():
            print(f"   {row['mandi']:30s} last price was {row['staleness_days']} days stale "
                  f"(reported {row['date'].date()})")
        print()

    result = included[['mandi', 'date', 'price']].rename(columns={'price': 'price_per_qtl'})
    return result, anchor_date


def load_kopargaon_forecast_price():
    """Prefer the basis-model FORECAST for Kopargaon over its stale actual
    price, since Kopargaon under-reports and its 'latest actual' may be old."""
    try:
        fc = pd.read_csv('kopargaon_forecast.csv')
        fc['date'] = pd.to_datetime(fc['date'])
        latest = fc.sort_values('date').iloc[-1]
        return latest['date'], latest['kop_p50']
    except FileNotFoundError:
        return None, None


def compute_net_realization(qty_quintals):
    prices, anchor_date = load_latest_prices()
    costs = pd.read_csv('cost_params.csv')

    # Swap in the forecast-based Kopargaon price if available - more current
    # and more reliable than its own sparse actual reporting.
    kop_date, kop_forecast_price = load_kopargaon_forecast_price()
    if kop_forecast_price is not None and 'KOPARGAON' in prices.mandi.values:
        prices.loc[prices.mandi == 'KOPARGAON', 'price_per_qtl'] = kop_forecast_price
        prices.loc[prices.mandi == 'KOPARGAON', 'date'] = kop_date

    m = pd.merge(prices, costs, on='mandi', how='inner')

    m['gross'] = qty_quintals * m['price_per_qtl']
    m['adat_cost'] = m['gross'] * (m['adat_pct'] / 100)
    m['hamali_cost'] = qty_quintals * m['hamali_per_qtl']
    m['tolai_cost'] = qty_quintals * m['tolai_per_qtl']
    m['transport_cost'] = m['distance_km_from_kopargaon'] * TRANSPORT_RATE_PER_KM_PER_QTL * qty_quintals
    m['total_deductions'] = m['adat_cost'] + m['hamali_cost'] + m['tolai_cost'] + m['transport_cost']
    m['net'] = m['gross'] - m['total_deductions']
    m['net_per_qtl'] = m['net'] / qty_quintals

    m = m.sort_values('net', ascending=False).reset_index(drop=True)
    m['rank_by_net'] = m.index + 1

    # Also rank by gross price alone, to show how often the two disagree
    m_by_gross = m.sort_values('gross', ascending=False).reset_index(drop=True)
    gross_rank_map = {row['mandi']: i + 1 for i, row in m_by_gross.iterrows()}
    m['rank_by_gross'] = m['mandi'].map(gross_rank_map)

    return m


if __name__ == '__main__':
    QTY = 40  # example: farmer with 40 quintals to sell

    print("=" * 90)
    print(f"NET REALIZATION COMPARISON - {QTY} quintals of onion, all 12 catchment mandis")
    print("=" * 90)
    print("NOTE: cost rates are ASSUMED placeholders pending APMC verification (see cost_params.csv)")
    print()

    result = compute_net_realization(QTY)

    display = result[['mandi', 'date', 'price_per_qtl', 'distance_km_from_kopargaon',
                       'total_deductions', 'net', 'net_per_qtl',
                       'rank_by_net', 'rank_by_gross']].copy()
    display['date'] = display['date'].dt.date
    for col in ['price_per_qtl', 'total_deductions', 'net', 'net_per_qtl']:
        display[col] = display[col].round(0).astype(int)

    print(display.to_string(index=False))
    print()

    top_net = result.iloc[0]
    top_gross = result.sort_values('gross', ascending=False).iloc[0]

    result['rank_jump'] = result['rank_by_gross'] - result['rank_by_net']
    biggest_mover = result.loc[result['rank_jump'].idxmax()]

    print(f"*** DEMO TALKING POINT ***")
    print(f"{biggest_mover['mandi']} moves from rank #{biggest_mover['rank_by_gross']} by gross "
          f"price to rank #{biggest_mover['rank_by_net']} by net realization "
          f"({int(biggest_mover['rank_jump'])} places better) once real costs are counted in.")
    print()

    if top_net['mandi'] != top_gross['mandi']:
        diff = top_net['net'] - result[result.mandi == top_gross['mandi']]['net'].values[0]
        print(f"Highest GROSS price: {top_gross['mandi']} (Rs {top_gross['price_per_qtl']:.0f}/qtl)")
        print(f"Highest NET realization: {top_net['mandi']} (Rs {top_net['net_per_qtl']:.0f}/qtl net)")
        print(f"A farmer chasing the highest sticker price would have earned "
              f"Rs {diff:.0f} LESS than going to {top_net['mandi']} instead.")
    else:
        print(f"Top rank by gross and by net agree here: {top_net['mandi']}")
        print("(The rank-jump further down the table is still the real story - see above.)")

    result.to_csv('net_realization_example.csv', index=False)
    print("\nSaved net_realization_example.csv")
