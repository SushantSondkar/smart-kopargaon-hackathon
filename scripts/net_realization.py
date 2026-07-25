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
from glut_indicator import build_glut_indicator

# Assumed - typical rate for tractor-trolley/mini-truck hire in this belt.
# VERIFY before demo: call a transporter or APMC office.
TRANSPORT_RATE_PER_KM_PER_QTL = 2.0


MAX_STALENESS_DAYS = 21  # a price older than this vs the anchor date is not
                          # a fair same-day comparison - excluded, not silently used


def get_market_condition_level(mandi):
    """
    Your APMC contact confirmed adat/hamali/tolai rates vary with market
    conditions - the EXACT rate isn't in any dataset (it's a private
    farmer-agent transaction, never centrally recorded by Agmarknet).

    What we CAN do: use the glut indicator (separately validated - HIGH_
    SUPPLY days really do precede price drops) as a proxy for negotiating
    leverage. More onions arriving -> more competition among sellers ->
    agents likely lean toward the HIGH end of the cost range. Scarcity ->
    agents compete for sellers -> likely LOW end.

    THIS PICKS A POINT WITHIN AN ALREADY-ASSUMED RANGE USING A REAL,
    VALIDATED SIGNAL. It is a reasonable assumption about how APMC
    economics typically work, NOT a proven relationship, and NOT a
    substitute for confirming actual rates with an APMC contact.

    Falls back to 'typical' if this mandi has no arrival data.
    """
    try:
        glut = build_glut_indicator(mandi)
        if glut.empty:
            return 'typical', None
        latest_flag = glut.sort_values('date').iloc[-1]['supply_flag']
        mapping = {'HIGH_SUPPLY': 'high', 'LOW_SUPPLY': 'low',
                   'NORMAL': 'typical', 'INSUFFICIENT_HISTORY': 'typical'}
        return mapping.get(latest_flag, 'typical'), latest_flag
    except (FileNotFoundError, KeyError):
        return 'typical', None


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

    # Rates vary with market conditions (confirmed by APMC contact) - so we
    # compute LOW/TYPICAL/HIGH net realization, same P10/P50/P90 discipline
    # used for the price forecast, rather than pretending costs are fixed.
    for level in ['low', 'typical', 'high']:
        adat_cost = m['gross'] * (m[f'adat_pct_{level}'] / 100)
        hamali_cost = qty_quintals * m[f'hamali_per_qtl_{level}']
        tolai_cost = qty_quintals * m[f'tolai_per_qtl_{level}']
        transport_cost = m['distance_km_from_kopargaon'] * TRANSPORT_RATE_PER_KM_PER_QTL * qty_quintals
        m[f'net_{level}'] = m['gross'] - adat_cost - hamali_cost - tolai_cost - transport_cost

    # Headline "net" = glut-aware pick within the range, not always 'typical'.
    # On a HIGH_SUPPLY day at that mandi, lean toward the high-cost (lower
    # net) scenario; on LOW_SUPPLY, lean toward the low-cost (higher net)
    # scenario. See get_market_condition_level() docstring for the caveat.
    condition_levels, condition_flags = [], []
    for mandi in m['mandi']:
        level, flag = get_market_condition_level(mandi)
        condition_levels.append(level)
        condition_flags.append(flag)
    m['market_condition'] = condition_flags
    m['net'] = [row[f'net_{lvl}'] for row, lvl in zip(m.to_dict('records'), condition_levels)]

    _low_raw = m['net_low'].copy()
    _high_raw = m['net_high'].copy()
    m['net_low'] = pd.concat([_low_raw, _high_raw], axis=1).min(axis=1)    # worst case (highest costs)
    m['net_high'] = pd.concat([_low_raw, _high_raw], axis=1).max(axis=1)   # best case (lowest costs)
    m['net_per_qtl'] = m['net'] / qty_quintals
    m['total_deductions'] = m['gross'] - m['net']

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
    print("NOTE: cost RANGES are ASSUMED placeholders pending APMC verification.")
    print("      The headline 'net' picks a point in that range based on each mandi's")
    print("      current supply signal (glut_indicator.py) - an assumption about")
    print("      market behavior, not a proven or verified rate. See docstring.")
    print()

    result = compute_net_realization(QTY)

    display = result[['mandi', 'date', 'price_per_qtl', 'distance_km_from_kopargaon',
                       'market_condition', 'net_low', 'net', 'net_high', 'net_per_qtl',
                       'rank_by_net', 'rank_by_gross']].copy()
    display['date'] = display['date'].dt.date
    for col in ['price_per_qtl', 'net_low', 'net', 'net_high', 'net_per_qtl']:
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
