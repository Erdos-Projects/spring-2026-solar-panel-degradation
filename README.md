# Solar Power 24-Hour Forecasting — System 2107, Arbuckle CA

Day-ahead solar power forecasting using NWP weather features and ML models,
with financial analysis using CAISO real-time electricity prices (LMP).

---

## Repository Structure

```
solar-forecast-2107/
│
├── README.md                        ← You are here
│
├── 01_data_pipeline.py              ← STEP 1: Download data & build CSVs
├── 02_model_comparison.ipynb        ← STEP 2: Train & evaluate all models
├── 03_financial_analysis.ipynb      ← STEP 3: MWOL & bidding strategy
│
├── data/                            ← Created by 01_data_pipeline.py
│   ├── df_train_features.csv        — Training features (2022-03-23→2023-11-09)
│   ├── df_val_features.csv          — Validation features (2024-01-01→2024-05-31)
│   ├── df_test_features.csv         — Test features (2024-06-01→2024-10-31)
│   ├── lmp_2024.csv                 — CAISO Day-Ahead LMP prices
│   ├── feature_list.txt             — Ordered list of 57 feature names
│   └── data_audit.txt               — Sanity checks & row counts
│
└── figures/                         ← Saved by notebooks (auto-created)
    ├── figT1_top3_together.png
    ├── figF1_mwol_total.png
    └── ...
```

---

## How to Run

### Step 1 — Download data and build feature CSVs
```bash
python 01_data_pipeline.py
```
This takes ~15 min (Open-Meteo download ~2 min, LMP download ~8 min).
Run once. All subsequent steps load from `data/`.

### Step 2 — Train models and compare
Open and run `02_model_comparison.ipynb` top to bottom.
~5–10 min depending on `USE_MLP` setting.

### Step 3 — Financial analysis
Open and run `03_financial_analysis.ipynb` top to bottom.
Requires `02_model_comparison.ipynb` to have been run in the same kernel session.

---

## Data Sources

| Data | Source | Resolution |
|---|---|---|
| Solar meter (AC output) | [PVDAQ / OEDI (DOE)](https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/2107_OEDI/) | 15-min |
| Weather forecasts | [Open-Meteo Previous Runs API](https://previous-runs-api.open-meteo.com) (GFS Seamless) | 15-min |
| Electricity prices (LMP) | [CAISO OASIS API](http://oasis.caiso.com) (Day-Ahead Market, NP15 hub) | Hourly |

---

## Models

| Model | Type | Test RMSE% | MWOL ($) |
|---|---|---|---|
| RandomForest | Ensemble (bagging) | ~15.7% | ~$738 |
| XGBoost | Gradient boosting | ~16.9% | ~$767 |
| LightGBM | Gradient boosting | ~17.3% | ~$793 |
| Stacking | Meta-learner | ~16.6% | ~$1,096 |
| MLP | Neural network | ~19.8% | ~$936 |
| Ridge | Linear + L2 reg | ~18.1% | ~$1,624 |
| LinearReg | OLS | ~18.1% | ~$1,624 |
| Persistence | Baseline | ~25.2% | ~$1,161 |
| Climatology | Baseline | ~20.6% | ~$1,166 |
| SmartPersistence | Baseline | ~24.3% | ~$907 |

---

## Key Results

- **Best model:** RandomForest — RMSE% 15.7%, MWOL $738 over Jun–Oct 2024
- **vs best baseline:** Saves ~$428 MWOL compared to SmartPersistence
- **Optimal bid strategy:** Apply a ~15–20% discount below forecast during
  positive-LMP hours to reduce MWOL by a further 40–60%
- **Cloud events:** July 16 type events (sudden afternoon cloud dropout)
  account for the majority of residual error — irreducible with day-ahead NWP

---

## Requirements

```bash
pip install pandas numpy requests scikit-learn lightgbm xgboost matplotlib
```

Python 3.10+ required.

---

## System Details

- **Site:** PVDAQ System 2107, Arbuckle, California (38.9963°N, 122.1341°W)
- **Capacity:** ~707 kW installed AC
- **Meter:** Revenue-grade AC output at 15-minute resolution
- **Forecast horizon:** 24 hours ahead (day-ahead market)
