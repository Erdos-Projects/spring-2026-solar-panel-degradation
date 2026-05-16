"""
01_data_pipeline.py
===================
Solar Power Forecasting — System 2107, Arbuckle, CA

PURPOSE
-------
Downloads all raw data, cleans it, engineers features, and saves
train / validation / test CSV files ready for model training.

Run this script once before opening any notebook.

OUTPUTS (saved to ./data/)
--------------------------
  data/df_train_features.csv   — 2022-03-23 → 2023-11-09
  data/df_val_features.csv     — 2024-01-01 → 2024-05-31
  data/df_test_features.csv    — 2024-06-01 → 2024-10-31
  data/feature_list.txt        — ordered list of feature column names
  data/data_audit.txt          — row counts, NaN checks, sanity stats

DATA SOURCES
------------
  1. Meter data   : PVDAQ / OEDI (DOE Open Energy Data Initiative)
  2. Weather      : Open-Meteo Previous Runs API (GFS Seamless, 1-day ahead)
  3. LMP prices   : CAISO OASIS API (Day-Ahead Market, NP15 hub)
                    → saved to data/lmp_2024.csv for financial analysis

USAGE
-----
  python 01_data_pipeline.py

REQUIREMENTS
------------
  pip install pandas numpy requests scikit-learn
"""

import os
import sys
import requests
import zipfile
import io
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET             = "actual_power_kw"
SYSTEM_CAPACITY_KW = 707.0
LAT                = 38.9963
LON                = -122.1341
PWR_COL            = "meter_revenue_grade_ac_output_meter_149578"
VAL_END            = pd.Timestamp("2024-06-01")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DOWNLOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_weather():
    """Download 15-min NWP weather forecasts from Open-Meteo (GFS Seamless)."""
    print("\n[1/3] Downloading Open-Meteo weather forecasts...")
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude":    LAT,
        "longitude":   LON,
        "start_date":  "2022-03-23",
        "end_date":    "2024-10-31",
        "timezone":    "PST8PDT",        # local Pacific time — matches meter timestamps
        "minutely_15": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "wind_speed_10m", "wind_speed_80m",
            "wind_direction_10m", "wind_direction_80m", "wind_gusts_10m",
            "shortwave_radiation", "direct_radiation", "direct_normal_irradiance",
            "diffuse_radiation", "global_tilted_irradiance", "sunshine_duration",
            "precipitation", "snowfall", "rain", "cape", "visibility",
            "weather_code", "cloud_cover", "pressure_msl", "surface_pressure",
        ],
        "models":       "gfs_seamless",
        "previous_day": 1,              # 1-day-ahead forecast (realistic operational)
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    weather_df = (pd.DataFrame(resp.json()["minutely_15"])
                    .rename(columns={"time": "measured_on"}))
    weather_df["measured_on"] = pd.to_datetime(weather_df["measured_on"])
    print(f"   Weather rows  : {len(weather_df):,}")
    print(f"   Date range    : {weather_df['measured_on'].min()} → "
          f"{weather_df['measured_on'].max()}")
    return weather_df


def download_meter():
    """Download 15-min AC output meter data from PVDAQ / OEDI (DOE)."""
    print("\n[2/3] Downloading PVDAQ meter data...")
    urls = {
        "train": ("https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize"
                  "/2107_OEDI/data/2107_meter_15m_data.csv"),
        "2024":  ("https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize"
                  "/2107_OEDI/data/2107_meter_15m_data_2024.csv"),
    }
    dfs = {}
    for key, url in urls.items():
        df = pd.read_csv(url)
        df["measured_on"] = pd.to_datetime(df["measured_on"])
        df = df.rename(columns={PWR_COL: TARGET})
        # Physical cap: system capacity ~707 kW; values >1000 kW are sensor errors
        df = df[df[TARGET] <= 1000].copy()
        dfs[key] = df
        print(f"   meter_{key:<5} : {len(df):,} rows  "
              f"max={df[TARGET].max():.1f} kW")
    return dfs["train"], dfs["2024"]


def download_lmp():
    """Download hourly Day-Ahead LMP from CAISO OASIS API (NP15 hub).

    Downloads in 30-day chunks. Takes ~5-8 min.
    Saves to data/lmp_2024.csv.
    """
    print("\n[3/3] Downloading CAISO LMP (this takes ~5-8 min)...")
    base    = "http://oasis.caiso.com/oasisapi/SingleZip"
    cur     = pd.Timestamp("2024-01-01")
    end     = pd.Timestamp("2024-11-01")
    records = []

    while cur < end:
        nxt = min(cur + pd.Timedelta(days=30), end)
        params = {
            "queryname":     "PRC_LMP",
            "startdatetime": cur.strftime("%Y%m%dT00:00-0000"),
            "enddatetime":   nxt.strftime("%Y%m%dT00:00-0000"),
            "version":       1,
            "market_run_id": "DAM",
            "node":          "TH_NP15_GEN-APND",
            "resultformat":  6,
        }
        try:
            r = requests.get(base, params=params, timeout=90)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
                chunk = pd.read_csv(z.open(csv_name))
            chunk = chunk[chunk["LMP_TYPE"] == "LMP"].copy()
            chunk["Interval Start"] = (
                pd.to_datetime(chunk["INTERVALSTARTTIME_GMT"], utc=True)
                  .dt.tz_convert("US/Pacific")
                  .dt.tz_localize(None)
            )
            chunk = chunk.rename(columns={"MW": "LMP"})[["Interval Start", "LMP"]]
            records.append(chunk)
            print(f"   {cur.date()} → {nxt.date()} : {len(chunk)} rows")
        except Exception as e:
            print(f"   WARNING {cur.date()} → {nxt.date()} : {e}")
        cur = nxt
        time.sleep(2)

    if not records:
        print("   ERROR: No LMP data downloaded. Skipping.")
        return None

    lmp_df = (pd.concat(records, ignore_index=True)
                .drop_duplicates("Interval Start")
                .sort_values("Interval Start")
                .reset_index(drop=True))
    lmp_df.to_csv("data/lmp_2024.csv", index=False)
    print(f"   Saved → data/lmp_2024.csv  ({len(lmp_df):,} rows)")
    return lmp_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA CLEANING & MERGING
# ══════════════════════════════════════════════════════════════════════════════

def clean_and_merge(meter_train, meter_2024, weather_df):
    """Merge meter + weather, restrict to weather coverage window, inner join."""
    print("\n[Cleaning] Merging meter + weather data...")

    # Restrict meter_train to weather coverage start (2022-03-23)
    # Pre-2022 rows have no weather features → useless for training
    weather_start = weather_df["measured_on"].min()
    meter_train   = meter_train[meter_train["measured_on"] >= weather_start].copy()

    # Inner merge: keeps only rows with valid weather for both periods
    df_train_raw = meter_train.merge(weather_df, on="measured_on", how="inner")
    df_2024_raw  = meter_2024.merge(weather_df,  on="measured_on", how="inner")

    # Drop any rows missing target
    df_train_raw = df_train_raw.dropna(subset=[TARGET])
    df_2024_raw  = df_2024_raw.dropna(subset=[TARGET])

    # Split 2024 into val and test
    df_val_raw  = df_2024_raw[df_2024_raw["measured_on"] <  VAL_END].copy()
    df_test_raw = df_2024_raw[df_2024_raw["measured_on"] >= VAL_END].copy()

    print(f"   df_train_raw : {len(df_train_raw):,} rows  "
          f"({df_train_raw['measured_on'].min().date()} → "
          f"{df_train_raw['measured_on'].max().date()})")
    print(f"   df_val_raw   : {len(df_val_raw):,} rows   "
          f"({df_val_raw['measured_on'].min().date()} → "
          f"{df_val_raw['measured_on'].max().date()})")
    print(f"   df_test_raw  : {len(df_test_raw):,} rows   "
          f"({df_test_raw['measured_on'].min().date()} → "
          f"{df_test_raw['measured_on'].max().date()})")

    # Sanity: zero weather NaNs expected after inner merge
    assert df_train_raw["shortwave_radiation"].isna().sum() == 0, \
        "Weather NaNs found in train after inner merge!"

    return df_train_raw, df_val_raw, df_test_raw, df_2024_raw


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "direct_normal_irradiance",
    "diffuse_radiation", "global_tilted_irradiance", "sunshine_duration",
    "precipitation", "snowfall", "rain", "cape", "visibility",
    "cloud_cover", "pressure_msl", "surface_pressure",
]

def build_label_encoder(df_train_raw, df_2024_raw):
    """Fit LabelEncoder on union of all weather_code values."""
    all_wc = pd.concat([
        df_train_raw["weather_code"].fillna(0),
        df_2024_raw["weather_code"].fillna(0),
    ]).astype(int)
    le = LabelEncoder()
    le.fit(all_wc)
    return le


def add_features(df, le_wc):
    """Compute all 57 features on a sorted, indexed DataFrame.

    Features:
      - 23 raw NWP weather columns
      - 6  cyclical time encodings (sin/cos hour, sin/cos doy, hour int, month)
      - 6  lag features (24h, 48h, 7d, 14d, 21d, 3-day average)
      - 5  rolling statistics (shortwave 1h/3h mean+std, power 24h mean+std)
      - 4  wind u/v components at 10m and 80m
      - 12 irradiance ratios and cloud-event features
      - 1  encoded weather_code
    """
    df = df.copy()

    # ── Cyclical time ─────────────────────────────────────────────────────────
    hour = df.index.hour + df.index.minute / 60
    doy  = df.index.dayofyear
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["sin_doy"]  = np.sin(2 * np.pi * doy  / 365)
    df["cos_doy"]  = np.cos(2 * np.pi * doy  / 365)
    df["hour"]     = df.index.hour
    df["month"]    = df.index.month

    # ── Lag features ─────────────────────────────────────────────────────────
    # lag_1/lag_4 excluded — cause "sunset spike" pathology
    for lag in [96, 192, 672, 1344, 2016]:   # 24h, 48h, 7d, 14d, 21d
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    df["lag_3day_avg"] = pd.concat(
        [df[TARGET].shift(96), df[TARGET].shift(192), df[TARGET].shift(288)],
        axis=1
    ).mean(axis=1)

    # ── Rolling statistics ────────────────────────────────────────────────────
    df["sw_roll1h_mean"]   = df["shortwave_radiation"].rolling(4,  min_periods=1).mean()
    df["sw_roll3h_mean"]   = df["shortwave_radiation"].rolling(12, min_periods=1).mean()
    df["sw_roll1h_std"]    = df["shortwave_radiation"].rolling(4,  min_periods=1).std().fillna(0)
    df["pow_roll24h_mean"] = df[TARGET].shift(1).rolling(96,  min_periods=24).mean()
    df["pow_roll24h_std"]  = df[TARGET].shift(1).rolling(96,  min_periods=24).std().fillna(0)

    # ── Wind u/v components ───────────────────────────────────────────────────
    for spd, dirn, suffix in [
        ("wind_speed_10m", "wind_direction_10m", "10m"),
        ("wind_speed_80m", "wind_direction_80m", "80m"),
    ]:
        rad = np.deg2rad(df[dirn].fillna(0))
        df[f"wind_u_{suffix}"] = df[spd] * np.cos(rad)
        df[f"wind_v_{suffix}"] = df[spd] * np.sin(rad)

    # ── Irradiance ratios ─────────────────────────────────────────────────────
    df["dni_ghi_ratio"]  = np.where(df["shortwave_radiation"] > 10,
                                     df["direct_normal_irradiance"] /
                                     df["shortwave_radiation"].clip(lower=1), 0)
    df["diffuse_frac"]   = np.where(df["shortwave_radiation"] > 10,
                                     df["diffuse_radiation"] /
                                     df["shortwave_radiation"].clip(lower=1), 0)
    df["clearsky_idx"]   = (1 - df["cloud_cover"].fillna(0) / 100).clip(0, 1)
    df["gti_x_clearsky"] = df["global_tilted_irradiance"] * df["clearsky_idx"]
    df["hour_x_sw"]      = df.index.hour * df["shortwave_radiation"]
    df["hour_x_gti"]     = df.index.hour * df["global_tilted_irradiance"]

    # ── Cloud-event features ──────────────────────────────────────────────────
    df["cloud_change_1h"] = df["cloud_cover"].diff(4).fillna(0)
    df["cloud_change_3h"] = df["cloud_cover"].diff(12).fillna(0)
    df["cloud_event"]     = (df["cloud_change_1h"] > 20).astype(float)
    df["sw_drop_1h"]      = df["shortwave_radiation"].diff(4).clip(upper=0).abs().fillna(0)

    # ── Solar elevation proxy ─────────────────────────────────────────────────
    hour_angle  = (df.index.hour + df.index.minute/60 - 12) * (np.pi / 12)
    declination = 23.45 * np.sin(2*np.pi*(df.index.dayofyear - 81)/365) * np.pi/180
    df["solar_elev_proxy"] = (np.sin(np.deg2rad(LAT)) * np.sin(declination) +
                               np.cos(np.deg2rad(LAT)) * np.cos(declination) *
                               np.cos(hour_angle)).clip(lower=0)

    # ── Clearness change ratio ────────────────────────────────────────────────
    sw_yest = df["shortwave_radiation"].shift(96)
    df["sw_today_vs_yest"] = np.where(
        sw_yest > 10,
        df["shortwave_radiation"] / sw_yest.clip(lower=1),
        1.0
    ).clip(0, 3)

    # ── Weather code ──────────────────────────────────────────────────────────
    df["weather_code_enc"] = le_wc.transform(df["weather_code"].fillna(0).astype(int))

    return df


def build_feature_list():
    TIME_FEATS  = ["sin_hour","cos_hour","sin_doy","cos_doy","hour","month"]
    LAG_FEATS   = ["lag_96","lag_192","lag_672","lag_1344","lag_2016","lag_3day_avg"]
    ROLL_FEATS  = ["sw_roll1h_mean","sw_roll3h_mean","sw_roll1h_std",
                   "pow_roll24h_mean","pow_roll24h_std"]
    WIND_FEATS  = ["wind_u_10m","wind_v_10m","wind_u_80m","wind_v_80m"]
    RATIO_FEATS = ["dni_ghi_ratio","diffuse_frac","clearsky_idx","gti_x_clearsky",
                   "hour_x_sw","hour_x_gti","cloud_change_1h","cloud_change_3h",
                   "cloud_event","sw_drop_1h","solar_elev_proxy","sw_today_vs_yest"]
    return (WEATHER_COLS + TIME_FEATS + LAG_FEATS + ROLL_FEATS +
            WIND_FEATS + RATIO_FEATS + ["weather_code_enc"])


def prep_block(df_raw, le_wc, features):
    """Set index, compute features, drop NaN rows."""
    df = df_raw.copy()
    if "measured_on" in df.columns:
        df = df.set_index("measured_on")
    df = df.sort_index()
    df = add_features(df, le_wc)
    df = df.dropna(subset=features + [TARGET])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA AUDIT
# ══════════════════════════════════════════════════════════════════════════════

def run_audit(df_train_feat, df_val_feat, df_test_feat, features, outfile="data/data_audit.txt"):
    """Write sanity check statistics to a text file."""
    lines = []
    lines.append("DATA AUDIT REPORT")
    lines.append("=" * 60)
    for name, df in [("TRAIN", df_train_feat),
                     ("VAL",   df_val_feat),
                     ("TEST",  df_test_feat)]:
        y = df[TARGET]
        lines.append(f"\n{name} SET")
        lines.append(f"  Rows            : {len(df):,}")
        lines.append(f"  Date range      : {df.index.min().date()} → "
                     f"{df.index.max().date()}")
        lines.append(f"  Power max (kW)  : {y.max():.1f}")
        lines.append(f"  Power mean (kW) : {y.mean():.1f}")
        lines.append(f"  Daytime mean    : {y[y>0].mean():.1f} kW  "
                     f"(rows where actual > 0)")
        lines.append(f"  NaN in features : {df[features].isna().sum().sum()}")
        lines.append(f"  lag_96 max      : {df['lag_96'].max():.1f}")
        lines.append(f"  lag_96 NaNs     : {df['lag_96'].isna().sum()}")
    lines.append("\n" + "=" * 60)
    with open(outfile, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[Audit] Saved → {outfile}")
    for line in lines:
        print("  " + line)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SOLAR FORECASTING — DATA PIPELINE")
    print("  System 2107, Arbuckle CA")
    print("=" * 60)

    # 1. Download
    weather_df          = download_weather()
    meter_train_raw, \
    meter_2024_raw      = download_meter()
    lmp_df              = download_lmp()   # optional for financial analysis

    # 2. Clean & merge
    df_train_raw, \
    df_val_raw,   \
    df_test_raw,  \
    df_2024_raw         = clean_and_merge(meter_train_raw, meter_2024_raw, weather_df)

    # 3. Feature engineering
    print("\n[Features] Building feature set...")
    le_wc               = build_label_encoder(df_train_raw, df_2024_raw)
    features            = build_feature_list()
    print(f"   Total features  : {len(features)}")

    df_train_feat       = prep_block(df_train_raw, le_wc, features)
    df_2024_feat        = prep_block(df_2024_raw,  le_wc, features)
    df_val_feat         = df_2024_feat[df_2024_feat.index < VAL_END].copy()
    df_test_feat        = df_2024_feat[df_2024_feat.index >= VAL_END].copy()

    # 4. Audit
    run_audit(df_train_feat, df_val_feat, df_test_feat, features)

    # 5. Save CSVs
    print("\n[Save] Writing feature CSVs...")
    df_train_feat.to_csv("data/df_train_features.csv")
    df_val_feat.to_csv("data/df_val_features.csv")
    df_test_feat.to_csv("data/df_test_features.csv")
    with open("data/feature_list.txt", "w") as f:
        f.write("\n".join(features))
    print("   data/df_train_features.csv")
    print("   data/df_val_features.csv")
    print("   data/df_test_features.csv")
    print("   data/feature_list.txt")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Run notebooks in order:")
    print("  02_model_comparison.ipynb  — train & evaluate all models")
    print("  03_financial_analysis.ipynb — MWOL, bidding strategy")
    print("=" * 60)


if __name__ == "__main__":
    main()
