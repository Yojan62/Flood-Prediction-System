"""
Utility functions for metrics, plotting, and logging.
"""
import io
import sys
import numpy as np
import requests
import matplotlib.pyplot as plt
from contextlib import contextmanager, redirect_stderr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Logging Helpers ---
class _StderrFilter(io.TextIOBase):
    def __init__(self, underlying):
        self._under = underlying
    def write(self, s):
        if "[LightGBM] [Warning] No further splits with positive gain" in s:
            return len(s)
        return self._under.write(s)
    def flush(self):
        return self._under.flush()

@contextmanager
def redirect_lgb_stderr():
    """Context manager to suppress LightGBM warnings."""
    try:
        filt = _StderrFilter(sys.stderr)
        with redirect_stderr(filt):
            yield
    finally:
        pass

# --- API Helpers ---
def safe_get_json(url, params, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# --- Metric Helpers ---
def metrics_with_peak(y_true, y_pred, thr):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2v = r2_score(y_true, y_pred)
    
    peak_mask = y_true >= thr
    if peak_mask.sum() >= 2:
        r2_peak = r2_score(y_true[peak_mask], y_pred[peak_mask])
    else:
        r2_peak = float('nan')
        
    return {'mae': mae, 'rmse': rmse, 'r2': r2v, 'r2_peak': r2_peak, 'thr': thr}

# --- Plotting Helpers ---

def plot_obs_vs_pred(y_true, y_pred, title="Observed vs Predicted", out_file="obs_vs_pred.png"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi],[lo, hi], 'r--')
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved plot: {out_file}")

def save_backtest_plot(dates, y_true, y_pred, thr, out_file="backtest.png"):
    plt.figure(figsize=(14,6))
    plt.plot(dates, y_true, label='Observed', linewidth=1.2)
    plt.plot(dates, y_pred, '--', label='Predicted', linewidth=1.4)
    plt.axhline(thr, color='orange', linestyle=':', label=f'High Risk Threshold')
    plt.legend()
    plt.grid(True)
    plt.title('Backtest: Observed vs Predicted')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved plot: {out_file}")

def plot_feature_importance(model, feature_names, out_file="feature_importance.png"):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            print("Model does not support feature importance plotting.")
            return

        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), top_importances, align='center', color='#556B2F')
        plt.yticks(range(len(indices)), top_features)
        plt.gca().invert_yaxis()
        plt.xlabel('Relative Importance')
        plt.title('Top 10 Features Driving the Model')
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"Saved plot: {out_file}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

def plot_residuals(y_true, y_pred, out_file="residuals.png"):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, s=10, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Predicted Discharge (m³/s)')
    plt.ylabel('Error (Residuals)')
    plt.title('Residual Plot (Prediction Errors)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved plot: {out_file}")

def plot_zoom_flood(dates, y_true, y_pred, thr, out_file="zoom.png"):
    try:
        max_idx = np.argmax(y_true)
        start_idx = max(0, max_idx - 30)
        end_idx = min(len(dates), max_idx + 30)
        
        zoom_dates = dates[start_idx:end_idx]
        zoom_true = y_true[start_idx:end_idx]
        zoom_pred = y_pred[start_idx:end_idx]

        plt.figure(figsize=(12, 5))
        plt.plot(zoom_dates, zoom_true, label='Observed', linewidth=2)
        plt.plot(zoom_dates, zoom_pred, '--', label='Predicted', linewidth=2, color='red')
        plt.axhline(thr, color='orange', linestyle=':', label='High Risk')
        plt.legend()
        plt.grid(True)
        plt.title(f'Zoomed View: Peak Flood Event')
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"Saved plot: {out_file}")
    except Exception as e:
        print(f"Could not plot zoom: {e}")

def plot_hydrograph(df, out_file="hydrograph.png"):
    try:
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Plot Discharge on primary y-axis (Left)
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('River Discharge (m³/s)', color=color)
        ax1.plot(df.index, df['observed'], color=color, label='Observed Flow', linewidth=2)
        ax1.plot(df.index, df['predicted'], color='green', linestyle='--', label='Predicted Flow', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Create secondary y-axis for Rain (Right)
        ax2 = ax1.twinx() 
        color = 'tab:gray'
        ax2.set_ylabel('Rainfall (mm)', color=color)
        # We invert the axis so rain "falls" from the top
        ax2.bar(df.index, df['rainfall_mm'], color=color, alpha=0.3, label='Rainfall', width=1.0)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, df['rainfall_mm'].max() * 3) 
        ax2.invert_yaxis()
        
        plt.title('Hydrograph: Rainfall vs. River Response')
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"Saved plot: {out_file}")
    except Exception as e:
        print(f"Could not plot hydrograph: {e}")

def plot_seasonal_accuracy(df, out_file="seasonal_accuracy.png"):
    try:
        df['month'] = df.index.month
        monthly_r2 = []
        months = range(1, 13)
        
        for m in months:
            subset = df[df['month'] == m]
            if len(subset) > 5:
                score = r2_score(subset['observed'], subset['predicted'])
                monthly_r2.append(max(0, score)) # Clip negative scores for chart readability
            else:
                monthly_r2.append(0)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(months, monthly_r2, color='#556B2F')
        plt.xlabel('Month')
        plt.ylabel('Accuracy (R2 Score)')
        plt.title('Model Performance by Month')
        plt.xticks(months, ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Highlight Monsoon months
        for i in range(5, 9): # Jun-Sep indices
            bars[i].set_color('#d32f2f') # Red for monsoon
            
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"Saved plot: {out_file}")
    except Exception as e:
        print(f"Could not plot seasonal accuracy: {e}")

def plot_error_distribution(y_true, y_pred, out_file="error_dist.png"):
    try:
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.xlabel('Prediction Error (m³/s)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f"Saved plot: {out_file}")
    except Exception as e:
        print(f"Could not plot error distribution: {e}")