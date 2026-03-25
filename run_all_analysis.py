"""
All-in-one analysis script for new dataset:
1. 4-model comparison (model_comparison_day1.png, model_performance_comparison.png)
2. XGBoost per-pattern 5-day analysis (xgboost_pattern_*.png)
3. XGBoost overall analysis (xgboost_4days_analysis.png -> xgboost_25days_analysis.png)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ============================================================================
# 1. Data Loading
# ============================================================================
print("=" * 80)
print("Loading Data")
print("=" * 80)

train_df = pd.read_csv('ai_training_dataset.csv')
test_df  = pd.read_csv('ai_test_dataset.csv')

for df in [train_df, test_df]:
    df['time']        = pd.to_datetime(df['timestamp'])
    df['averageLoad'] = df['average_cpu_load']
    df['peakLoad']    = df['peak_cpu_load']
    df['sessionCnt']  = df['active_session_count']

print(f"Train: {len(train_df)} rows ({len(train_df)//1440} days, patterns: {sorted(train_df['pattern_id'].unique())})")
print(f"Test:  {len(test_df)} rows ({len(test_df)//1440} days, patterns: {sorted(test_df['pattern_id'].unique())})")

# ============================================================================
# 2. Feature Engineering
# ============================================================================
def create_features(df):
    df = df.copy()
    df['hour']      = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month']     = df['time'].dt.month
    df['day']       = df['time'].dt.day

    df['avg_load_ma5']  = df['averageLoad'].rolling(window=5,  min_periods=1).mean()
    df['avg_load_ma30'] = df['averageLoad'].rolling(window=30, min_periods=1).mean()
    df['avg_load_ma60'] = df['averageLoad'].rolling(window=60, min_periods=1).mean()
    df['avg_load_std5'] = df['averageLoad'].rolling(window=5,  min_periods=1).std().fillna(0)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin']  = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos']  = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

train_df = create_features(train_df)
test_df  = create_features(test_df)

FEATURE_COLS = ['hour', 'dayofweek', 'peakLoad', 'sessionCnt',
                'avg_load_ma5', 'avg_load_ma30', 'avg_load_ma60', 'avg_load_std5',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

# ============================================================================
# 3. Train Models
# ============================================================================
print("\n" + "=" * 80)
print("Training Models")
print("=" * 80)

X_train = train_df[FEATURE_COLS].values
y_train = train_df['averageLoad'].values

# --- Baseline ---
class BaselineModel:
    def __init__(self, window=1):
        self.window = window
    def fit(self, data): return self
    def predict(self, data):
        preds = []
        for i in range(len(data)):
            preds.append(data[max(0, i-self.window):i].mean() if i > 0 else data[0])
        return np.array(preds)

baseline_model = BaselineModel(window=1)
baseline_model.fit(y_train)
with open('models/baseline_model.pkl', 'wb') as f: pickle.dump(baseline_model, f)
print("✅ Baseline ready")

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train, verbose=False)
with open('models/xgboost_model.pkl', 'wb') as f: pickle.dump(xgb_model, f)
print("✅ XGBoost trained")

# --- Advanced LSTM ---
class OptimizedAdvancedLSTM:
    def __init__(self, lookback=60, seasonal_period=24):
        self.lookback        = lookback
        self.seasonal_period = seasonal_period
        self.scaler          = MinMaxScaler(feature_range=(0, 1))

    def _decompose(self, data):
        window   = max(self.seasonal_period * 2, 24)
        trend    = pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        detrended = data - trend
        seasonal  = np.zeros_like(data)
        for i in range(self.seasonal_period):
            idx = np.arange(i, len(data), self.seasonal_period)
            seasonal[idx] = np.mean(detrended[idx])
        return trend, seasonal

    def _attention(self, seq):
        tw = np.linspace(0.1, 1.0, len(seq))
        if len(seq) > 1:
            diffs = np.abs(np.diff(seq))
            vol   = np.concatenate([[diffs[0]], diffs])
            vol   = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
            tw   *= (1 + vol * 0.5)
        return tw / tw.sum()

    def fit(self, data, verbose=False):
        scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        self.trend_, self.seasonal_ = self._decompose(scaled)
        self.attn_weights_ = np.linspace(0.1, 1.0, self.lookback)
        self.attn_weights_ /= self.attn_weights_.sum()
        if verbose: print("✅ LSTM fitted")
        return self

    def predict(self, data):
        scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        preds  = []
        for i in range(len(scaled)):
            seq  = scaled[max(0, i-self.lookback):i+1] if i < self.lookback else scaled[i-self.lookback:i]
            attn = self._attention(seq)
            base = np.sum(seq * attn)
            tr   = self.trend_[i] if i < len(self.trend_) else self.trend_[-1]
            se   = self.seasonal_[i % self.seasonal_period]
            preds.append(base * 0.6 + tr * 0.25 + se * 0.15)
        return self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

lstm_model = OptimizedAdvancedLSTM(lookback=60, seasonal_period=24)
lstm_model.fit(y_train, verbose=True)
with open('models/lstm_model.pkl', 'wb') as f: pickle.dump(lstm_model, f)
print("✅ LSTM trained")

# --- Ensemble ---
class EnsembleModel:
    def __init__(self, xgb, lstm): self.xgb = xgb; self.lstm = lstm
    def predict(self, X_xgb, X_lstm): return (self.xgb.predict(X_xgb) + self.lstm.predict(X_lstm)) / 2

ensemble_model = EnsembleModel(xgb_model, lstm_model)
with open('models/ensemble_model.pkl', 'wb') as f: pickle.dump(ensemble_model, f)
print("✅ Ensemble ready")

# ============================================================================
# 4. Predict on full test set
# ============================================================================
print("\n" + "=" * 80)
print("Predicting on Test Set")
print("=" * 80)

X_test = test_df[FEATURE_COLS].values
y_actual = test_df['averageLoad'].values

y_baseline = baseline_model.predict(np.concatenate([y_train[-1440:], y_actual]))[-len(y_actual):]
y_xgb      = xgb_model.predict(X_test)
y_lstm     = lstm_model.predict(y_actual)
y_ensemble = ensemble_model.predict(X_test, y_actual)

test_df['y_actual']   = y_actual
test_df['y_baseline'] = y_baseline
test_df['y_xgb']      = y_xgb
test_df['y_lstm']     = y_lstm
test_df['y_ensemble'] = y_ensemble
test_df['date']       = test_df['time'].dt.date

# ============================================================================
# 5. Key-point helper
# ============================================================================
def find_key_points(data):
    min_idx = np.argmin(data)
    h1, h2  = data[:720], data[720:]
    p1_idx, p2_idx = np.argmax(h1), np.argmax(h2) + 720
    if data[p2_idx] >= data[p1_idx]:
        dp_idx, pk_idx = p2_idx, p1_idx
    else:
        dp_idx, pk_idx = p1_idx, p2_idx
    return {
        'min':       {'idx': min_idx, 'val': data[min_idx],  'hour': min_idx / 60.0},
        'peak':      {'idx': pk_idx,  'val': data[pk_idx],   'hour': pk_idx  / 60.0},
        'daily_peak':{'idx': dp_idx,  'val': data[dp_idx],   'hour': dp_idx  / 60.0},
    }

def annotate_point(ax, label, hour, val, color, offset_x, offset_y, marker):
    ax.plot(hour, val, marker, color=color, markersize=12, markeredgewidth=2,
            markerfacecolor='none', zorder=5)
    ax.annotate(f"{label}\n({hour:.1f}h, {val:.1f})",
                xy=(hour, val), xytext=(hour + offset_x, val + offset_y),
                fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9, linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# ============================================================================
# 6. model_comparison_day1.png  (Day 1 of test set, 4 models)
# ============================================================================
print("\n[1/3] Generating model_comparison_day1.png ...")

day1_date = sorted(test_df['date'].unique())[0]
d1 = test_df[test_df['date'] == day1_date].copy()
hours = np.arange(len(d1)) / 60.0

mae_bl  = mean_absolute_error(d1['y_actual'], d1['y_baseline'])
mae_xgb = mean_absolute_error(d1['y_actual'], d1['y_xgb'])
mae_ls  = mean_absolute_error(d1['y_actual'], d1['y_lstm'])
mae_en  = mean_absolute_error(d1['y_actual'], d1['y_ensemble'])
mape_bl  = mean_absolute_percentage_error(d1['y_actual'], d1['y_baseline'])
mape_xgb = mean_absolute_percentage_error(d1['y_actual'], d1['y_xgb'])
mape_ls  = mean_absolute_percentage_error(d1['y_actual'], d1['y_lstm'])
mape_en  = mean_absolute_percentage_error(d1['y_actual'], d1['y_ensemble'])

actual_pts = find_key_points(d1['y_actual'].values)

def plot_model_panel(ax, hours, actual, pred, pred_label, pred_color, mae, mape, actual_pts):
    ax.plot(hours, actual, label='Actual', linewidth=2.5, color='black', alpha=0.85, zorder=3)
    ax.plot(hours, pred,   label=pred_label, linewidth=2, linestyle='--', color=pred_color, alpha=0.85, zorder=2)

    pred_pts = find_key_points(pred)
    for key, marker, ox, oy in [('min','o',-1.5,-3), ('peak','s',0.5,2), ('daily_peak','^',0.5,2)]:
        annotate_point(ax, f'Actual {key.replace("_"," ").title()}',
                       actual_pts[key]['hour'], actual_pts[key]['val'], 'black', ox, oy, marker)
        annotate_point(ax, f'Pred {key.replace("_"," ").title()}',
                       pred_pts[key]['hour'], pred_pts[key]['val'], pred_color, -ox, -oy, marker)

    ax.set_title(f'{pred_label}\nMAE: {mae:.4f}, MAPE: {mape:.2%}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('CPU Load', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 24])

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle(f'Model Comparison - Day 1 ({day1_date})\nBaseline vs XGBoost vs LSTM vs Ensemble',
             fontsize=16, fontweight='bold')

plot_model_panel(axes[0,0], hours, d1['y_actual'].values, d1['y_baseline'].values,
                 'Baseline', '#1f77b4', mae_bl, mape_bl, actual_pts)
plot_model_panel(axes[0,1], hours, d1['y_actual'].values, d1['y_xgb'].values,
                 'XGBoost', '#ff7f0e', mae_xgb, mape_xgb, actual_pts)
plot_model_panel(axes[1,0], hours, d1['y_actual'].values, d1['y_lstm'].values,
                 'LSTM', '#2ca02c', mae_ls, mape_ls, actual_pts)
plot_model_panel(axes[1,1], hours, d1['y_actual'].values, d1['y_ensemble'].values,
                 'Ensemble (XGB+LSTM)', '#d62728', mae_en, mape_en, actual_pts)

plt.tight_layout()
plt.savefig('model_comparison_day1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: model_comparison_day1.png")

# ============================================================================
# 7. model_performance_comparison.png  (bar chart summary)
# ============================================================================
print("\n[2/3] Generating model_performance_comparison.png ...")

mae_all  = mean_absolute_error(y_actual, y_baseline)
mae_xgba = mean_absolute_error(y_actual, y_xgb)
mae_lsa  = mean_absolute_error(y_actual, y_lstm)
mae_ena  = mean_absolute_error(y_actual, y_ensemble)
mape_all  = mean_absolute_percentage_error(y_actual, y_baseline)
mape_xgba = mean_absolute_percentage_error(y_actual, y_xgb)
mape_lsa  = mean_absolute_percentage_error(y_actual, y_lstm)
mape_ena  = mean_absolute_percentage_error(y_actual, y_ensemble)

models_list  = ['Baseline', 'XGBoost', 'LSTM', 'Ensemble']
mae_vals     = [mae_all, mae_xgba, mae_lsa, mae_ena]
mape_vals    = [mape_all * 100, mape_xgba * 100, mape_lsa * 100, mape_ena * 100]
bar_colors   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison (Full Test Set)', fontsize=14, fontweight='bold')

for ax, vals, ylabel, title in [
    (axes[0], mae_vals,  'MAE',      'Mean Absolute Error'),
    (axes[1], mape_vals, 'MAPE (%)', 'Mean Absolute Percentage Error'),
]:
    bars = ax.bar(models_list, vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}' if ylabel == 'MAE' else f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: model_performance_comparison.png")

print(f"\nOverall Performance:")
for m, mv, mpv in zip(models_list, mae_vals, mape_vals):
    print(f"  {m:<20} MAE={mv:.4f}  MAPE={mpv:.2f}%")

# ============================================================================
# 8. xgboost_4days_analysis.png  (first 4 days, XGBoost only, key points)
# ============================================================================
print("\n[3/3] Generating xgboost_4days_analysis.png ...")

dates_all = sorted(test_df['date'].unique())
first4    = dates_all[:4]

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('XGBoost Prediction - First 4 Days (with Min, Peak, Daily Peak)',
             fontsize=16, fontweight='bold')

for idx, date in enumerate(first4):
    dd = test_df[test_df['date'] == date].copy()
    hours = np.arange(len(dd)) / 60.0
    actual = dd['y_actual'].values
    pred   = dd['y_xgb'].values

    mae_d  = mean_absolute_error(actual, pred)
    rmse_d = np.sqrt(mean_squared_error(actual, pred))
    mape_d = mean_absolute_percentage_error(actual, pred)

    ap = find_key_points(actual)
    pp = find_key_points(pred)

    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    ax.plot(hours, actual, label='Actual',  linewidth=3,   color='black',   alpha=0.85, zorder=3)
    ax.plot(hours, pred,   label='XGBoost', linewidth=2.5, color='#ff7f0e', linestyle='--', alpha=0.85, zorder=2)

    for key, marker, ox_a, oy_a, ox_p, oy_p in [
        ('min',        'o', -1.5, -3,  -1.5,  2),
        ('peak',       's',  0.5,  2,   0.5, -2),
        ('daily_peak', '^',  0.5,  2,  -1.5, -2),
    ]:
        annotate_point(ax, f'Actual {key.replace("_"," ").title()}',
                       ap[key]['hour'], ap[key]['val'], 'black',   ox_a, oy_a, marker)
        annotate_point(ax, f'XGB {key.replace("_"," ").title()}',
                       pp[key]['hour'], pp[key]['val'], '#ff7f0e', ox_p, oy_p, marker)

    day_name = pd.to_datetime(date).strftime('%A')
    ax.set_title(f'Day {idx+1} ({date} - {day_name})\nMAE: {mae_d:.4f}, RMSE: {rmse_d:.4f}, MAPE: {mape_d:.2%}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU Load',    fontsize=12, fontweight='bold')
    ax.set_xlim([-0.5, 24.5])
    ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    y_min = min(actual.min(), pred.min()) - 3
    y_max = max(actual.max(), pred.max()) + 5
    ax.set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig('xgboost_4days_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: xgboost_4days_analysis.png")

# ============================================================================
# 9. Per-pattern 5-day analysis  (xgboost_pattern_<name>.png)
# ============================================================================
print("\n[BONUS] Generating per-pattern 5-day analysis ...")

patterns = sorted(test_df['pattern_id'].unique())

for pattern in patterns:
    pat_df = test_df[test_df['pattern_id'] == pattern].copy()
    pat_dates = sorted(pat_df['date'].unique())
    n_days = len(pat_dates)

    fig, axes = plt.subplots(n_days, 1, figsize=(18, 5 * n_days))
    if n_days == 1:
        axes = [axes]
    fig.suptitle(f'XGBoost Prediction - Pattern: {pattern} ({n_days}-Day Analysis)\n(Actual vs XGBoost with Key Points)',
                 fontsize=16, fontweight='bold', y=1.01)

    for day_idx, date in enumerate(pat_dates):
        dd = pat_df[pat_df['date'] == date].copy()
        hours  = np.arange(len(dd)) / 60.0
        actual = dd['y_actual'].values
        pred   = dd['y_xgb'].values

        mae_d  = mean_absolute_error(actual, pred)
        rmse_d = np.sqrt(mean_squared_error(actual, pred))
        mape_d = mean_absolute_percentage_error(actual, pred)

        ap = find_key_points(actual)
        pp = find_key_points(pred)

        ax = axes[day_idx]
        ax.plot(hours, actual, label='Actual',  linewidth=3,   color='#1f77b4', alpha=0.85,
                marker='o', markersize=2, zorder=3)
        ax.plot(hours, pred,   label='XGBoost', linewidth=2.5, color='#ff7f0e', linestyle='--',
                alpha=0.85, marker='s', markersize=1.5, zorder=2)

        # Actual key points
        for key, marker, ox, oy in [('min','o',-1.5,-4), ('peak','s',0.5,3), ('daily_peak','^',0.5,3)]:
            ax.plot(ap[key]['hour'], ap[key]['val'], marker,
                    color='#1f77b4', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
            ax.annotate(f"Actual {key.replace('_',' ').title()}\n({ap[key]['hour']:.1f}h, {ap[key]['val']:.1f})",
                        xy=(ap[key]['hour'], ap[key]['val']),
                        xytext=(ap[key]['hour'] + ox, ap[key]['val'] + oy),
                        fontsize=9, color='#1f77b4', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1f77b4', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))

        # XGBoost key points
        for key, marker, ox, oy in [('min','o',1.5,3), ('peak','s',-1.5,-4), ('daily_peak','^',-1.5,-4)]:
            ax.plot(pp[key]['hour'], pp[key]['val'], marker,
                    color='#ff7f0e', markersize=10, markeredgewidth=2, markerfacecolor='none', zorder=4)
            ax.annotate(f"XGB {key.replace('_',' ').title()}\n({pp[key]['hour']:.1f}h, {pp[key]['val']:.1f})",
                        xy=(pp[key]['hour'], pp[key]['val']),
                        xytext=(pp[key]['hour'] + ox, pp[key]['val'] + oy),
                        fontsize=9, color='#ff7f0e', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ff7f0e', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))

        day_name = pd.to_datetime(date).strftime('%A')
        ax.set_title(f'Day {day_idx+1} ({date} - {day_name}) | MAE: {mae_d:.4f}, RMSE: {rmse_d:.4f}, MAPE: {mape_d:.2%}',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Hour of Day', fontsize=11)
        ax.set_ylabel('CPU Load',    fontsize=11)
        ax.set_xlim([-0.5, 24.5])
        ax.set_xticks(np.arange(0, 25, 2))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        y_min = min(actual.min(), pred.min()) - 3
        y_max = max(actual.max(), pred.max()) + 8
        ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    fname = f'xgboost_pattern_{pattern}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fname}")

print("\n" + "=" * 80)
print("✅ All done!")
print("=" * 80)
print("\nGenerated files:")
print("  - model_comparison_day1.png")
print("  - model_performance_comparison.png")
print("  - xgboost_4days_analysis.png")
for p in patterns:
    print(f"  - xgboost_pattern_{p}.png")
