"""
Utility Functions
Helper functions for the crop yield prediction module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def convert_hg_to_tonnes(hg_per_ha: float) -> float:
    """
    Convert hectograms per hectare to tonnes per hectare.
    
    Args:
        hg_per_ha: Yield in hectograms per hectare
        
    Returns:
        Yield in tonnes per hectare
    """
    return hg_per_ha / 10000


def convert_tonnes_to_hg(tonnes_per_ha: float) -> float:
    """
    Convert tonnes per hectare to hectograms per hectare.
    
    Args:
        tonnes_per_ha: Yield in tonnes per hectare
        
    Returns:
        Yield in hectograms per hectare
    """
    return tonnes_per_ha * 10000


def format_yield(yield_hg: float, unit: str = 'hg/ha') -> str:
    """
    Format yield value for display.
    
    Args:
        yield_hg: Yield value
        unit: Unit string
        
    Returns:
        Formatted string
    """
    if unit == 'tonnes/ha':
        yield_val = convert_hg_to_tonnes(yield_hg)
        return f"{yield_val:.2f} tonnes/ha"
    else:
        return f"{yield_hg:,.2f} hg/ha"


def calculate_yield_change(old_yield: float, new_yield: float) -> Dict:
    """
    Calculate yield change statistics.
    
    Args:
        old_yield: Previous yield
        new_yield: New yield
        
    Returns:
        Dictionary with change statistics
    """
    absolute_change = new_yield - old_yield
    percent_change = (absolute_change / old_yield) * 100 if old_yield != 0 else 0
    
    return {
        'absolute_change': absolute_change,
        'percent_change': percent_change,
        'direction': 'increase' if absolute_change > 0 else 'decrease' if absolute_change < 0 else 'no change'
    }


def create_feature_summary(input_features: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from input features.
    
    Args:
        input_features: Dictionary of features
        
    Returns:
        DataFrame with feature summary
    """
    summary = pd.DataFrame([
        {'Feature': k, 'Value': v}
        for k, v in input_features.items()
    ])
    
    return summary


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10, save_path: str = None):
    """
    Plot feature importance as a horizontal bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Yield (hg/ha)')
    plt.ylabel('Predicted Yield (hg/ha)')
    plt.title('Predicted vs Actual Crop Yield')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot residuals (prediction errors).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Optional path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Yield (hg/ha)')
    axes[0].set_ylabel('Residuals (hg/ha)')
    axes[0].set_title('Residual Plot')
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals (hg/ha)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def generate_prediction_report(results: List[Dict], output_path: str = None) -> pd.DataFrame:
    """
    Generate a report from multiple predictions.
    
    Args:
        results: List of prediction result dictionaries
        output_path: Optional path to save CSV report
        
    Returns:
        DataFrame with prediction report
    """
    report_data = []
    
    for result in results:
        if 'error' not in result:
            row = {
                **result['input_features'],
                'predicted_yield': result['predicted_yield'],
                'ci_lower': result['confidence_interval_95'][0],
                'ci_upper': result['confidence_interval_95'][1],
                'std_deviation': result['std_deviation']
            }
            report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    
    if output_path:
        report_df.to_csv(output_path, index=False)
        print(f"Report saved to: {output_path}")
    
    return report_df


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions\n")
    
    # Test unit conversion
    hg_val = 50000
    tonnes_val = convert_hg_to_tonnes(hg_val)
    print(f"{hg_val} hg/ha = {tonnes_val} tonnes/ha")
    
    # Test yield change
    change = calculate_yield_change(40000, 50000)
    print(f"\nYield change: {change}")
    
    # Test formatting
    print(f"\nFormatted: {format_yield(50000)}")
    print(f"Formatted (tonnes): {format_yield(50000, 'tonnes/ha')}")
