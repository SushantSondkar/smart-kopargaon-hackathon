# Smart Kopargaon Hackathon - Phase 1 Handoff

## What's in this folder

### data/
- `combined_clean.csv` - all raw price data (6 crops, 6.5 years), cleaned and name-normalized (Phase 0 output)
- `arrivals_clean.csv` - onion arrival quantity data, 2024-2026, Ahmednagar + Nashik
- `onion_clean_series.csv` - the modelling-ready onion series (Red/Other backfilled, business days only) - THIS is what the model trains on
- `baseline_results.csv` - naive/MA7 predictions on the test window
- `model_results.csv` - LightGBM predictions (P10/P50/P90) on the test window

### scripts/
Run in this order:
1. `build_series.py` - turns combined_clean.csv into onion_clean_series.csv
2. `baselines.py` - naive + MA7 baseline, prints MAPE
3. `train_model.py` - trains the LightGBM quantile model, prints MAPE + coverage

## How to run (after `pip install pandas numpy lightgbm scikit-learn`)

```
cd scripts
python3 build_series.py
python3 baselines.py
python3 train_model.py
```

## Current results (Lasalgaon, onion, Red variety)

| Horizon | Naive MAPE | LightGBM MAPE |
|---|---|---|
| 7 days  | 13.79% | 13.18% |
| 15 days | 22.09% | 22.96% |
| 30 days | 35.80% | 30.08% |

LightGBM's advantage grows with horizon - strongest case is the 30-day
storage-decision use case, not the 7-day "what's tomorrow's price" case.

P10-P90 band coverage: ~77.5% (target ~80%, close and honest).

## Known limitations (say these out loud in the PPT, don't hide them)
- No arrival-quantity data before 2024 (only price)
- Market names change spelling/format across years (handled in cleaning)
- Ahmednagar district renamed to Ahilyanagar mid-2026 (handled)
- Kopargaon only reports Red onion ~36% of days - Other-variety backfill used
- Cross-mandi (Manmad/Pimpalgaon) features tested, did NOT improve accuracy - dropped

## Not yet built
- Kopargaon basis (spread) model - derives Kopargaon forecast from Lasalgaon
- Net-realization / decision layer maths
- Other 4 crops (maize, soyabean, wheat, tomato) - same pipeline, not yet run
- Glut indicator from arrival data
