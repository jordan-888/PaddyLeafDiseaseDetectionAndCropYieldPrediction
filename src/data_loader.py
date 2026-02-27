"""
Data Loader — India Paddy Yield (Real Data)
Loads Custom_Crops_yield_Historical_Dataset.csv, filters rice,
aggregates district → state, engineers lag features.
No synthetic generation whatsoever.
"""

import pandas as pd
import numpy as np
import os
from .config import (
    RAW_DATA_PATH, CLEAN_DATA_PATH, TARGET_CROP,
    TARGET_COLUMN, DATA_DIR
)


# ---------------------------------------------------------------------------
# Column mapping: raw → standardised
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    'State Name':                 'State',
    'Year':                       'Year',
    'Area_ha':                    'Area_ha',
    'Yield_kg_per_ha':            'Yield_kg_per_ha',
    'N_req_kg_per_ha':            'N_req_kg_per_ha',
    'P_req_kg_per_ha':            'P_req_kg_per_ha',
    'K_req_kg_per_ha':            'K_req_kg_per_ha',
    'Temperature_C':              'Temperature_C',
    'Humidity_%':                 'Humidity_%',
    'pH':                         'pH',
    'Rainfall_mm':                'Rainfall_mm',
    'Wind_Speed_m_s':             'Wind_Speed_m_s',
    'Solar_Radiation_MJ_m2_day':  'Solar_Radiation_MJ_m2_day',
}


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and basic-filter the raw dataset."""
    print(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Filter to rice only
    df = df[df['Crop'].str.lower() == TARGET_CROP].copy()
    print(f"  After rice filter: {df.shape}")
    return df


def aggregate_to_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate district-level rows → state-year level.
    Uses Area_ha as weight for yield; mean for other numeric features.
    """
    print("Aggregating district → state level...")

    # Weighted yield (area-weighted average)
    df['_weighted_yield'] = df['Yield_kg_per_ha'] * df['Area_ha']
    df['_total_area']     = df['Area_ha']

    numeric_cols = [
        'N_req_kg_per_ha', 'P_req_kg_per_ha', 'K_req_kg_per_ha',
        'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm',
        'Wind_Speed_m_s', 'Solar_Radiation_MJ_m2_day',
        '_weighted_yield', '_total_area'
    ]

    agg_dict = {c: 'mean' for c in numeric_cols}
    agg_dict['_weighted_yield'] = 'sum'
    agg_dict['_total_area']     = 'sum'

    state_df = df.groupby(['State Name', 'Year']).agg(agg_dict).reset_index()

    # Recover area-weighted yield
    state_df['Yield_kg_per_ha'] = (
        state_df['_weighted_yield'] / state_df['_total_area']
    )
    state_df.drop(columns=['_weighted_yield', '_total_area'], inplace=True)
    state_df.rename(columns={'State Name': 'State'}, inplace=True)

    print(f"  After aggregation: {state_df.shape}")
    return state_df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-aware lag features, sorted by State → Year.
    Lags: Yield_t1, Yield_t2, Rainfall_t1, Rainfall_t2, Temp_t1
    Rows lacking sufficient history are dropped.
    """
    print("Engineering lag features...")
    df = df.sort_values(['State', 'Year']).copy()

    grp = df.groupby('State')
    df['Yield_t1']    = grp['Yield_kg_per_ha'].shift(1)
    df['Yield_t2']    = grp['Yield_kg_per_ha'].shift(2)
    df['Rainfall_t1'] = grp['Rainfall_mm'].shift(1)
    df['Rainfall_t2'] = grp['Rainfall_mm'].shift(2)
    df['Temp_t1']     = grp['Temperature_C'].shift(1)

    before = len(df)
    df.dropna(subset=['Yield_t1', 'Yield_t2', 'Rainfall_t1',
                      'Rainfall_t2', 'Temp_t1'], inplace=True)
    print(f"  Dropped {before - len(df)} rows lacking lag history")
    print(f"  Final shape: {df.shape}")
    return df.reset_index(drop=True)


def load_and_prepare(save: bool = True) -> pd.DataFrame:
    """
    Full pipeline:
      raw CSV → filter rice → aggregate state → lag features → save
    Returns clean DataFrame ready for model training.
    """
    raw = load_raw_data()
    state_df = aggregate_to_state(raw)
    clean_df  = add_lag_features(state_df)

    # Enforce numeric types
    numeric_cols = [c for c in clean_df.columns if c != 'State']
    clean_df[numeric_cols] = clean_df[numeric_cols].apply(
        pd.to_numeric, errors='coerce'
    )
    clean_df.dropna(inplace=True)

    print(f"\nFinal dataset: {clean_df.shape}")
    print(f"States: {sorted(clean_df['State'].unique())}")
    print(f"Year: {clean_df['Year'].min()} – {clean_df['Year'].max()}")
    print(f"Yield range: {clean_df['Yield_kg_per_ha'].min():.1f} – "
          f"{clean_df['Yield_kg_per_ha'].max():.1f} kg/ha")

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        clean_df.to_csv(CLEAN_DATA_PATH, index=False)
        print(f"\nSaved clean dataset → {CLEAN_DATA_PATH}")

    return clean_df


def load_clean_data() -> pd.DataFrame:
    """Load the pre-processed clean dataset (must run load_and_prepare first)."""
    if not os.path.exists(CLEAN_DATA_PATH):
        print("Clean dataset not found — running preparation pipeline...")
        return load_and_prepare()
    print(f"Loading clean dataset from: {CLEAN_DATA_PATH}")
    return pd.read_csv(CLEAN_DATA_PATH)


if __name__ == '__main__':
    df = load_and_prepare()
    print(df.head())
    print(df.dtypes)
