"""
PHASE 1 - STEP 8: PRICE ALERTS
================================
PS requirement: "Price alerts" - one of the 6 core features.

Unlike the other scripts, this isn't a one-time analysis - it's a rule
engine meant to run every day as part of the nightly refresh (once the
backend exists). What's built here is the CORE LOGIC plus a backtest
showing it actually fires sensibly on real historical data - proof it
works, not just a design sketch.

THREE ALERT TYPES (cover the PS's examples: threshold, target price,
sell/arrival window):
  1. PRICE_ABOVE   - forecast crosses above a price the farmer set
  2. PRICE_BELOW   - forecast crosses below a price (early warning to sell
                      before it drops further)
  3. GLUT_WARNING  - the demand indicator (Step 6) flags HIGH_SUPPLY -
                      "arrivals are unusually high, expect price pressure"

In the real app, a farmer sets rules like "tell me if Kopargaon onion goes
above Rs 1500" via the UI - these get stored (see alert_rule table in the
original schema plan) and checked each time the forecast refreshes.
"""

import pandas as pd


def check_price_alert(rule, current_price):
    """rule: dict with 'type' (PRICE_ABOVE/PRICE_BELOW) and 'threshold'."""
    if rule['type'] == 'PRICE_ABOVE' and current_price >= rule['threshold']:
        return True, (f"Kopargaon कांदा भाव Rs {current_price:.0f}/qtl झाला आहे "
                       f"(तुमचा लक्ष्य: Rs {rule['threshold']:.0f}) - विकण्याची चांगली वेळ!")
    if rule['type'] == 'PRICE_BELOW' and current_price <= rule['threshold']:
        return True, (f"WARNING: Kopargaon onion price dropped to Rs {current_price:.0f}/qtl "
                       f"(your alert: below Rs {rule['threshold']:.0f})")
    return False, None


def check_glut_alert(supply_flag):
    if supply_flag == 'HIGH_SUPPLY':
        return True, ("आवक जास्त आहे - पुढील काही दिवसात भाव दबावाखाली येऊ शकतो. "
                       "(High arrivals detected - price pressure likely in the coming days)")
    return False, None


def backtest_alerts(price_series_csv='kopargaon_forecast.csv',
                     glut_csv='glut_indicator.csv'):
    """Run the alert logic against real historical data to prove it fires
    sensibly, not just on paper. Uses the Kopargaon forecast (P50) and the
    glut flags we already validated."""

    forecast = pd.read_csv(price_series_csv)
    forecast['date'] = pd.to_datetime(forecast['date'])

    glut = pd.read_csv(glut_csv)
    glut['date'] = pd.to_datetime(glut['date'])

    # Example rules a farmer might set - illustrative thresholds based on
    # the price range we've actually seen at Kopargaon (roughly Rs 700-1300
    # in the recent period)
    rules = [
        {'type': 'PRICE_ABOVE', 'threshold': 1000},
        {'type': 'PRICE_BELOW', 'threshold': 750},
    ]

    fired_price = []
    for _, row in forecast.iterrows():
        for rule in rules:
            fired, msg = check_price_alert(rule, row['kop_p50'])
            if fired:
                fired_price.append({'date': row['date'], 'rule': rule['type'],
                                     'threshold': rule['threshold'],
                                     'price': row['kop_p50'], 'message': msg})

    fired_glut = []
    for _, row in glut.iterrows():
        fired, msg = check_glut_alert(row['supply_flag'])
        if fired:
            fired_glut.append({'date': row['date'], 'message': msg})

    return pd.DataFrame(fired_price), pd.DataFrame(fired_glut), len(forecast), len(glut)


if __name__ == '__main__':
    price_alerts, glut_alerts, n_forecast_days, n_glut_days = backtest_alerts()

    print("=" * 90)
    print("PRICE ALERT ENGINE - BACKTEST ON REAL DATA")
    print("=" * 90)
    print("Example rules tested: alert if Kopargaon forecast >= Rs 1000, "
          "or <= Rs 750")
    print()
    print(f"Forecast days checked: {n_forecast_days}")
    print(f"PRICE alerts that would have fired: {len(price_alerts)}")
    if len(price_alerts):
        print(price_alerts[['date', 'rule', 'price']].tail(5).to_string(index=False))
    print()

    print(f"Glut-flagged days checked: {n_glut_days}")
    print(f"GLUT_WARNING alerts that would have fired: {len(glut_alerts)}")
    if len(glut_alerts):
        print(f"Sample message:\n  {glut_alerts.iloc[-1]['message']}")
    print()

    price_alerts.to_csv('alert_backtest_price.csv', index=False)
    glut_alerts.to_csv('alert_backtest_glut.csv', index=False)
    print("Saved alert_backtest_price.csv and alert_backtest_glut.csv")
    print()
    print("NOTE: in the real app, rules come from the farmer via the UI and")
    print("are checked automatically each time the nightly forecast refreshes -")
    print("this script proves the underlying logic works on real data.")
