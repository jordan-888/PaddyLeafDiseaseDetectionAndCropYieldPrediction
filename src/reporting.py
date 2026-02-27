"""
Reporting — Feature Importance, Predictions vs Actual, Residuals, SHAP
Saves all plots to reports/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (no display needed)
import warnings
warnings.filterwarnings('ignore')

from .config import REPORT_DIR
from .model import YieldPredictor


def _ensure_reports_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def plot_feature_importance(predictor: YieldPredictor, top_n: int = 15,
                             save: bool = True):
    _ensure_reports_dir()
    df = predictor.get_feature_importance()
    if df is None:
        print("Feature importance not available.")
        return

    df = df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    ax.barh(df['feature'][::-1], df['importance'][::-1], color=colors[::-1])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance — Top {top_n}\n'
                 f'({predictor.model_type.replace("_"," ").title()})',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(REPORT_DIR, 'feature_importance.png')
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                                split_label: str = 'Test', save: bool = True):
    _ensure_reports_dir()
    fig, ax = plt.subplots(figsize=(8, 8))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, alpha=0.45, edgecolors='k',
               linewidths=0.3, c='steelblue', s=40, label='Predictions')
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual Yield (kg/ha)', fontsize=12)
    ax.set_ylabel('Predicted Yield (kg/ha)', fontsize=12)
    ax.set_title(f'Predictions vs Actual — {split_label} Set',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(REPORT_DIR, 'predictions_vs_actual.png')
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   save: bool = True):
    _ensure_reports_dir()
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, c='darkorange',
                    edgecolors='k', linewidths=0.3, s=35)
    axes[0].axhline(0, color='red', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Predicted Yield (kg/ha)', fontsize=11)
    axes[0].set_ylabel('Residual (Predicted − Actual)', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=40, color='steelblue',
                 edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='red', linewidth=1.5, linestyle='--')
    axes[1].set_xlabel('Residual (kg/ha)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Residual Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(REPORT_DIR, 'residuals.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_shap(predictor: YieldPredictor, X_test: np.ndarray,
              feature_names: list, save: bool = True):
    """SHAP summary plot — only for XGBoost."""
    if predictor.model_type != 'xgboost':
        print("SHAP skipped (only for XGBoost).")
        return
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return

    _ensure_reports_dir()
    explainer = shap.TreeExplainer(predictor.model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(REPORT_DIR, 'shap_summary.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved → {path}")
    plt.close('all')


# ---------------------------------------------------------------------------
def generate_all_reports(predictor: YieldPredictor,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: list = None):
    print("\n--- Generating Reports ---")
    y_pred = predictor.predict(X_test)
    plot_feature_importance(predictor)
    plot_predictions_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_shap(predictor, X_test,
              feature_names or predictor.feature_names or [])
    print(f"All plots saved to: {REPORT_DIR}")
