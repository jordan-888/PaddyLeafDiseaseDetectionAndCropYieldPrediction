"""
Data Loading Module
Loads crop yield data and adds synthetic features for Chennai region
"""

import pandas as pd
import numpy as np
from typing import Optional
from .config import (
    DATA_PATH, 
    CHENNAI_SYNTHETIC_DATA, 
    SUPPORTED_CROPS,
    TARGET
)


def load_yield_data(
    filepath: str = DATA_PATH,
    crop_filter: Optional[list] = None,
    add_synthetic: bool = True
) -> pd.DataFrame:
    """
    Load crop yield dataset and optionally add synthetic features.
    
    Args:
        filepath: Path to yield_df.csv
        crop_filter: List of crop names to filter (None = all crops)
        add_synthetic: Whether to add synthetic Chennai data
        
    Returns:
        pd.DataFrame: Cleaned dataset with synthetic features
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Drop unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Filter by crop type if specified
    if crop_filter:
        df = df[df['Item'].isin(crop_filter)]
        print(f"Filtered to {len(df)} records for crops: {crop_filter}")
    
    # Handle missing values
    print("\nHandling missing values...")
    print(f"Missing values before:\n{df.isnull().sum()}")
    
    # Drop rows with missing target variable
    df = df.dropna(subset=[TARGET])
    
    # Fill missing numerical values with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} with median: {median_val:.2f}")
    
    print(f"\nDataset after cleaning: {len(df)} records")
    
    # Add synthetic features
    if add_synthetic:
        df = add_synthetic_features(df)
    
    return df


def add_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic features for Chennai region (humidity, soil nutrients, pH).
    
    Uses normal distribution with parameters from CHENNAI_SYNTHETIC_DATA.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added synthetic features
    """
    print("\nAdding synthetic Chennai features...")
    np.random.seed(42)  # For reproducibility
    
    n_samples = len(df)
    
    for feature, params in CHENNAI_SYNTHETIC_DATA.items():
        # Generate values from normal distribution
        values = np.random.normal(
            loc=params['mean'],
            scale=params['std'],
            size=n_samples
        )
        
        # Clip to valid range
        values = np.clip(values, params['min'], params['max'])
        
        df[feature] = values
        print(f"  Added {feature}: mean={values.mean():.2f}, "
              f"std={values.std():.2f}, range=[{values.min():.2f}, {values.max():.2f}]")
    
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get summary information about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Dataset statistics
    """
    info = {
        'total_records': len(df),
        'num_features': len(df.columns) - 1,  # Excluding target
        'crops': df['Item'].unique().tolist(),
        'num_crops': df['Item'].nunique(),
        'areas': df['Area'].unique().tolist(),
        'num_areas': df['Area'].nunique(),
        'year_range': (df['Year'].min(), df['Year'].max()),
        'target_stats': {
            'mean': df[TARGET].mean(),
            'std': df[TARGET].std(),
            'min': df[TARGET].min(),
            'max': df[TARGET].max()
        }
    }
    
    return info


def print_dataset_summary(df: pd.DataFrame):
    """
    Print a formatted summary of the dataset.
    
    Args:
        df: Input DataFrame
    """
    info = get_dataset_info(df)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total Records: {info['total_records']:,}")
    print(f"Number of Features: {info['num_features']}")
    print(f"Number of Crops: {info['num_crops']}")
    print(f"Number of Areas: {info['num_areas']}")
    print(f"Year Range: {info['year_range'][0]} - {info['year_range'][1]}")
    print(f"\nTarget Variable ({TARGET}):")
    print(f"  Mean: {info['target_stats']['mean']:,.2f} hg/ha")
    print(f"  Std:  {info['target_stats']['std']:,.2f} hg/ha")
    print(f"  Min:  {info['target_stats']['min']:,.2f} hg/ha")
    print(f"  Max:  {info['target_stats']['max']:,.2f} hg/ha")
    print(f"\nSupported Crops ({info['num_crops']}):")
    for crop in sorted(info['crops']):
        count = len(df[df['Item'] == crop])
        print(f"  - {crop}: {count:,} records")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the data loader
    df = load_yield_data()
    print_dataset_summary(df)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
