"""
XGBoost Feature Engineering Comparison
Basic Features vs Advanced Features (Lag, Statistics, Rate of Change)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=" * 80)
print("XGBoost Feature Engineering Comparison")
print("=" * 80)

# ============================================================================
# 1. Data Loading
# ============================================================================
print("\n1. Loading Data")
train_df = pd.read_csv('ai_training_dataset.csv')
test_df = pd.read_csv('ai_test_dataset.csv')

train_df['time'] = pd.to_datetime(train_df['timestamp'])
train_df['averageLoad'] = train_df['average_cpu_load']
train_df['peakLoad'] = train_df['peak_cpu_load']
train_df['sessionCnt'] = train_df['active_session_count']

test_df['time'] = pd.to_datetime(test_df['timestamp'])
test_df['averageLoad'] = test_df['average_cpu_load']
test_df['peakLoad'] = test_df['peak_cpu_load']
test_df['sessionCnt'] = test_df['active_session_count']

print(f"Train data: {len(train_df)} rows ({len(train_df)//1440} days, patterns: {sorted(train_df['pattern_id'].unique())})")
print(f"Test data: {len(test_df)} rows ({len(test_df)//1440} days, patterns: {sorted(test_df['pattern_id'].unique())})")

# ============================================================================
# 2. Basic Feature Engineering
# ============================================================================
print("\n2. Basic Feature Engineering")

def create_basic_features(df):
    """Basic Features: Time, Day of Week, Moving Average"""
    df = df.copy()
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    
    df['avg_load_ma5'] = df['averageLoad'].rolling(window=5, min_periods=1).mean()
    df['avg_load_ma30'] = df['averageLoad'].rolling(window=30, min_periods=1).mean()
    df['avg_load_ma60'] = df['averageLoad'].rolling(window=60, min_periods=1).mean()
    df['avg_load_std5'] = df['averageLoad'].rolling(window=5, min_periods=1).std().fillna(0)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

train_basic = create_basic_features(train_df.copy())
test_basic = create_basic_features(test_df.copy())

basic_feature_cols = ['hour', 'dayofweek', 'peakLoad', 'sessionCnt', 
                      'avg_load_ma5', 'avg_load_ma30', 'avg_load_ma60', 'avg_load_std5',
                      'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

print(f"Number of basic features: {len(basic_feature_cols)}")
print(f"Basic Features: {basic_feature_cols}")

# ============================================================================
# 3. Advanced Feature Engineering
# ============================================================================
print("\n3. Advanced Feature Engineering (Lag, Statistics, Rate of Change)")

def create_advanced_features(df):
    """Advanced Features: Lag, Statistics, Rate of Change, etc."""
    df = df.copy()
    
    # Basic Features
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    
    # Moving Average
    df['avg_load_ma5'] = df['averageLoad'].rolling(window=5, min_periods=1).mean()
    df['avg_load_ma10'] = df['averageLoad'].rolling(window=10, min_periods=1).mean()
    df['avg_load_ma30'] = df['averageLoad'].rolling(window=30, min_periods=1).mean()
    df['avg_load_ma60'] = df['averageLoad'].rolling(window=60, min_periods=1).mean()
    
    # Volatility (Standard Deviation)
    df['avg_load_std5'] = df['averageLoad'].rolling(window=5, min_periods=1).std().fillna(0)
    df['avg_load_std10'] = df['averageLoad'].rolling(window=10, min_periods=1).std().fillna(0)
    df['avg_load_std30'] = df['averageLoad'].rolling(window=30, min_periods=1).std().fillna(0)
    
    # Lag Features (Past Values)
    for lag in [1, 5, 10, 30, 60]:
        df[f'avg_load_lag{lag}'] = df['averageLoad'].shift(lag).fillna(method='bfill')
        df[f'peak_load_lag{lag}'] = df['peakLoad'].shift(lag).fillna(method='bfill')
        df[f'session_lag{lag}'] = df['sessionCnt'].shift(lag).fillna(method='bfill')
    
    # Rate of Change
    df['avg_load_change_1'] = df['averageLoad'].diff(1).fillna(0)
    df['avg_load_change_5'] = df['averageLoad'].diff(5).fillna(0)
    df['avg_load_change_10'] = df['averageLoad'].diff(10).fillna(0)
    
    # Percentage Change
    df['avg_load_pct_change_1'] = df['averageLoad'].pct_change(1).fillna(0)
    df['avg_load_pct_change_5'] = df['averageLoad'].pct_change(5).fillna(0)
    
    # Max/Min in Window
    df['avg_load_max5'] = df['averageLoad'].rolling(window=5, min_periods=1).max()
    df['avg_load_min5'] = df['averageLoad'].rolling(window=5, min_periods=1).min()
    df['avg_load_max30'] = df['averageLoad'].rolling(window=30, min_periods=1).max()
    df['avg_load_min30'] = df['averageLoad'].rolling(window=30, min_periods=1).min()
    
    # Range
    df['avg_load_range5'] = df['avg_load_max5'] - df['avg_load_min5']
    df['avg_load_range30'] = df['avg_load_max30'] - df['avg_load_min30']
    
    # Cyclical Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Interaction Features
    df['load_session_ratio'] = df['averageLoad'] / (df['sessionCnt'] + 1)
    df['peak_avg_ratio'] = df['peakLoad'] / (df['averageLoad'] + 1)
    
    return df

train_advanced = create_advanced_features(train_df.copy())
test_advanced = create_advanced_features(test_df.copy())

# Advanced Feature List (exclude non-numeric and target columns)
exclude_cols = ['time', 'timestamp', 'averageLoad', 'peakLoad', 'sessionCnt', 
                'date', 'y_actual', 'y_pred_xgb', 'pattern_id',
                'average_cpu_load', 'peak_cpu_load', 'active_session_count']
advanced_feature_cols = [col for col in train_advanced.columns if col not in exclude_cols]

print(f"Number of advanced features: {len(advanced_feature_cols)}")
print(f"Added Features:")
print(f"  - Lag Features: avg_load_lag1/5/10/30/60, peak_load_lag1/5/10/30/60, session_lag1/5/10/30/60")
print(f"  - Change Features: avg_load_change_1/5/10, avg_load_pct_change_1/5")
print(f"  - Max/Min Features: avg_load_max5/30, avg_load_min5/30")
print(f"  - Range Features: avg_load_range5/30")
print(f"  - Interaction Features: load_session_ratio, peak_avg_ratio")

# ============================================================================
# 4. Train XGBoost with Basic Features
# ============================================================================
print("\n" + "=" * 80)
print("4. Training XGBoost with Basic Features")
print("=" * 80)

X_train_basic = train_basic[basic_feature_cols].values
y_train = train_basic['averageLoad'].values

xgb_basic = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_basic.fit(X_train_basic, y_train, verbose=False)
print("✅ Basic Feature XGBoost training completed")

# Prediction with Basic Features
X_test_basic = test_basic[basic_feature_cols].values
y_pred_basic = xgb_basic.predict(X_test_basic)

mae_basic = mean_absolute_error(test_df['averageLoad'].values, y_pred_basic)
rmse_basic = np.sqrt(mean_squared_error(test_df['averageLoad'].values, y_pred_basic))
mape_basic = mean_absolute_percentage_error(test_df['averageLoad'].values, y_pred_basic)

print(f"Basic Feature Performance:")
print(f"  MAE:  {mae_basic:.6f}")
print(f"  RMSE: {rmse_basic:.6f}")
print(f"  MAPE: {mape_basic:.4%}")

# ============================================================================
# 5. Train XGBoost with Advanced Features
# ============================================================================
print("\n" + "=" * 80)
print("5. Training XGBoost with Advanced Features")
print("=" * 80)

X_train_advanced = train_advanced[advanced_feature_cols].values
xgb_advanced = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_advanced.fit(X_train_advanced, y_train, verbose=False)
print("✅ Advanced Feature XGBoost training completed")

# Prediction with Advanced Features
X_test_advanced = test_advanced[advanced_feature_cols].values
y_pred_advanced = xgb_advanced.predict(X_test_advanced)

mae_advanced = mean_absolute_error(test_df['averageLoad'].values, y_pred_advanced)
rmse_advanced = np.sqrt(mean_squared_error(test_df['averageLoad'].values, y_pred_advanced))
mape_advanced = mean_absolute_percentage_error(test_df['averageLoad'].values, y_pred_advanced)

print(f"Advanced Feature Performance:")
print(f"  MAE:  {mae_advanced:.6f}")
print(f"  RMSE: {rmse_advanced:.6f}")
print(f"  MAPE: {mape_advanced:.4%}")

# ============================================================================
# 6. Performance Comparison
# ============================================================================
print("\n" + "=" * 80)
print("6. Performance Comparison")
print("=" * 80)

improvement_mae = (mae_basic - mae_advanced) / mae_basic * 100
improvement_rmse = (rmse_basic - rmse_advanced) / rmse_basic * 100
improvement_mape = (mape_basic - mape_advanced) / mape_basic * 100

print(f"\n{'Metric':<15} {'Basic Feature':<20} {'Advanced Feature':<20} {'Improvement':<15}")
print("-" * 70)
print(f"{'MAE':<15} {mae_basic:<20.6f} {mae_advanced:<20.6f} {improvement_mae:>+.2f}%")
print(f"{'RMSE':<15} {rmse_basic:<20.6f} {rmse_advanced:<20.6f} {improvement_rmse:>+.2f}%")
print(f"{'MAPE':<15} {mape_basic:<20.4%} {mape_advanced:<20.4%} {improvement_mape:>+.2f}%")

# ============================================================================
# 7. Daily Performance Comparison
# ============================================================================
print("\n" + "=" * 80)
print("7. Daily Performance Comparison")
print("=" * 80)

test_df['date'] = test_df['time'].dt.date
test_df['y_pred_basic'] = y_pred_basic
test_df['y_pred_advanced'] = y_pred_advanced

dates = sorted(test_df['date'].unique())

print(f"\n{'Date':<15} {'Basic MAE':<15} {'Advanced MAE':<15} {'Improvement':<15}")
print("-" * 60)

for date in dates:
    day_data = test_df[test_df['date'] == date]
    actual = day_data['averageLoad'].values
    pred_basic = day_data['y_pred_basic'].values
    pred_advanced = day_data['y_pred_advanced'].values
    
    mae_day_basic = mean_absolute_error(actual, pred_basic)
    mae_day_advanced = mean_absolute_error(actual, pred_advanced)
    improvement_day = (mae_day_basic - mae_day_advanced) / mae_day_basic * 100
    
    print(f"{str(date):<15} {mae_day_basic:<15.6f} {mae_day_advanced:<15.6f} {improvement_day:>+.2f}%")

# ============================================================================
# 8. Visualization
# ============================================================================
print("\n" + "=" * 80)
print("8. Visualization")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('XGBoost Feature Engineering Comparison\nBasic Features vs Advanced Features', 
             fontsize=16, fontweight='bold')

# 1. MAE Comparison
ax = axes[0, 0]
models = ['Basic Features', 'Advanced Features']
mae_values = [mae_basic, mae_advanced]
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('Mean Absolute Error Comparison', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. RMSE Comparison
ax = axes[0, 1]
rmse_values = [rmse_basic, rmse_advanced]
bars = ax.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax.set_title('Root Mean Squared Error Comparison', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. MAPE Comparison
ax = axes[1, 0]
mape_values = [mape_basic * 100, mape_advanced * 100]
bars = ax.bar(models, mape_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Mean Absolute Percentage Error Comparison', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, mape_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. Improvement Comparison
ax = axes[1, 1]
improvements = [improvement_mae, improvement_rmse, improvement_mape]
metrics = ['MAE', 'RMSE', 'MAPE']
colors_improvement = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
bars = ax.bar(metrics, improvements, color=colors_improvement, alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax.set_title('Advanced Feature Improvement', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, improvements):
    height = bar.get_height()
    y_pos = height + (0.5 if height > 0 else -1.5)
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{val:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
            fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_engineering_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Graph saved: feature_engineering_comparison.png")
plt.close()

# ============================================================================
# 9. Day 1 Detailed Comparison (1h - 4h)
# ============================================================================
print("\n" + "=" * 80)
print("9. Day 1 Detailed Comparison (1h - 4h)")
print("=" * 80)

day1_data = test_df[test_df['date'] == dates[0]].copy()

# Extract 1h - 4h (60 min to 240 min)
start_idx = 60
end_idx = 240

day1_zoom = day1_data.iloc[start_idx:end_idx].copy()
minutes = np.arange(len(day1_zoom))
hours_zoom = minutes / 60.0 + 1.0

fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(hours_zoom, day1_zoom['averageLoad'].values, label='Actual', linewidth=3, color='black', alpha=0.85, zorder=3)
ax.plot(hours_zoom, day1_zoom['y_pred_basic'].values, label='Basic Features', linewidth=2.5, 
        linestyle='--', color='#1f77b4', alpha=0.8, zorder=2)
ax.plot(hours_zoom, day1_zoom['y_pred_advanced'].values, label='Advanced Features', linewidth=2.5, 
        linestyle='--', color='#ff7f0e', alpha=0.8, zorder=2)

mae_day1_basic = mean_absolute_error(day1_zoom['averageLoad'].values, day1_zoom['y_pred_basic'].values)
mae_day1_advanced = mean_absolute_error(day1_zoom['averageLoad'].values, day1_zoom['y_pred_advanced'].values)
mape_day1_basic = mean_absolute_percentage_error(day1_zoom['averageLoad'].values, day1_zoom['y_pred_basic'].values)
mape_day1_advanced = mean_absolute_percentage_error(day1_zoom['averageLoad'].values, day1_zoom['y_pred_advanced'].values)

ax.set_title(f'Day 1 ({dates[0]}) 1h-4h Comparison\nBasic Features: MAE={mae_day1_basic:.6f}, MAPE={mape_day1_basic:.2%} | Advanced Features: MAE={mae_day1_advanced:.6f}, MAPE={mape_day1_advanced:.2%}', 
             fontsize=13, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax.set_ylabel('CPU Load', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
ax.grid(True, alpha=0.3)
ax.set_xticks(np.arange(1, 4.5, 0.5))
ax.set_xticklabels([f'{h:.1f}' for h in np.arange(1, 4.5, 0.5)])

plt.tight_layout()
plt.savefig('day1_feature_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Graph saved: day1_feature_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("✅ Completed!")
print("=" * 80)
print("\nGenerated files:")
print("  - feature_engineering_comparison.png")
print("  - day1_feature_comparison.png")
