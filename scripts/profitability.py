"""
PHASE 1 - STEP 7: PROFITABILITY ESTIMATION
=============================================
PS requirement: "Profitability estimation" - one of the 6 core features.

This doesn't need new data or a new model - it's arithmetic on top of
net_realization.py's output. Takes what a farmer spent to grow the crop
and tells him, per mandi, whether he's making money or losing it, and by
how much.

*** IMPORTANT - cost-of-cultivation figures below are ASSUMED PLACEHOLDERS,
same caveat as cost_params.csv. Typical Maharashtra onion cultivation runs
roughly Rs 60,000-90,000/acre depending on irrigation type (drip vs flood)
and seed cost, with yields 80-120 quintals/acre. VERIFY with your own
farmer contact or campus survey before using real figures in the demo -
this is exactly the kind of number a judge from this belt will sanity-check.

FORMULA:
    cost_per_qtl (break-even) = total_cultivation_cost / expected_yield_qtl
    margin_per_qtl (at a mandi) = net_realization_per_qtl - cost_per_qtl
    total_profit (at a mandi)   = margin_per_qtl * quantity_sold_qtl
"""

import pandas as pd
from net_realization import compute_net_realization

# ASSUMED - verify with a real farmer or APMC contact before the demo
DEFAULT_CULTIVATION_COST_PER_ACRE = 75000   # Rs, typical drip-irrigated onion
DEFAULT_YIELD_PER_ACRE = 100                 # quintals/acre, typical Nashik-belt onion


def compute_profitability(area_acres, qty_quintals,
                           cultivation_cost_per_acre=DEFAULT_CULTIVATION_COST_PER_ACRE,
                           yield_per_acre=DEFAULT_YIELD_PER_ACRE):

    total_cost = area_acres * cultivation_cost_per_acre
    expected_yield = area_acres * yield_per_acre
    breakeven_price_per_qtl = total_cost / expected_yield

    net = compute_net_realization(qty_quintals)
    net['breakeven_per_qtl'] = breakeven_price_per_qtl
    net['margin_per_qtl'] = net['net_per_qtl'] - breakeven_price_per_qtl
    net['total_profit'] = net['margin_per_qtl'] * qty_quintals
    net['is_profitable'] = net['margin_per_qtl'] > 0

    net = net.sort_values('total_profit', ascending=False).reset_index(drop=True)
    return net, total_cost, expected_yield, breakeven_price_per_qtl


if __name__ == '__main__':
    AREA_ACRES = 2
    QTY_QUINTALS = 40   # what he's actually taking to market this trip

    print("=" * 90)
    print(f"PROFITABILITY ESTIMATION - {AREA_ACRES} acres onion, selling {QTY_QUINTALS} quintals")
    print("=" * 90)
    print("NOTE: cultivation cost/yield are ASSUMED placeholders - verify before demo")
    print()

    result, total_cost, expected_yield, breakeven = compute_profitability(AREA_ACRES, QTY_QUINTALS)

    print(f"Total cultivation cost: Rs {total_cost:,.0f}")
    print(f"Expected total yield:   {expected_yield:.0f} quintals")
    print(f"Break-even price:       Rs {breakeven:.0f} per quintal")
    print()

    display = result[['mandi', 'net_per_qtl', 'breakeven_per_qtl', 'margin_per_qtl',
                       'total_profit', 'is_profitable', 'rank_by_net']].copy()
    for col in ['net_per_qtl', 'breakeven_per_qtl', 'margin_per_qtl', 'total_profit']:
        display[col] = display[col].round(0).astype(int)
    print(display.to_string(index=False))
    print()

    profitable_count = result['is_profitable'].sum()
    print(f"Profitable at {profitable_count} of {len(result)} mandis at current prices.")

    best = result.iloc[0]
    worst = result.iloc[-1]
    print(f"\nBest option: {best['mandi']} -> Rs {best['total_profit']:,.0f} profit "
          f"({'PROFITABLE' if best['is_profitable'] else 'LOSS'})")
    print(f"Worst option: {worst['mandi']} -> Rs {worst['total_profit']:,.0f} "
          f"({'PROFITABLE' if worst['is_profitable'] else 'LOSS'})")

    result.to_csv('profitability_example.csv', index=False)
    print("\nSaved profitability_example.csv")
