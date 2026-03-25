"""
테스트 데이터 4일치에 대한 XGBoost 예측 결과
각 일별로 min, peak, daily peak 3개 지점 표시
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=" * 80)
print("XGBoost 4일치 분석 (Min, Peak, Daily Peak 표시)")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n1. 데이터 로드")
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

print(f"Train 데이터: {len(train_df)} 행")
print(f"Test 데이터: {len(test_df)} 행 ({len(test_df) // 1440} 일)")

# ============================================================================
# 2. 특성 엔지니어링
# ============================================================================
def create_features(df):
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

train_df = create_features(train_df)
test_df = create_features(test_df)

# ============================================================================
# 3. XGBoost 모델 로드
# ============================================================================
print("\n2. XGBoost 모델 로드")
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
print("✅ XGBoost 모델 로드 완료")

# ============================================================================
# 4. 특수점 찾기 함수
# ============================================================================
def find_key_points(data):
    """
    최저점, 피크점, 데일리피크점 찾기
    """
    min_idx = np.argmin(data)
    min_val = data[min_idx]
    
    first_half = data[:720]
    peak1_idx = np.argmax(first_half)
    peak1_val = first_half[peak1_idx]
    
    second_half = data[720:]
    peak2_idx = np.argmax(second_half) + 720
    peak2_val = data[peak2_idx]
    
    if peak2_val >= peak1_val:
        daily_peak_idx = peak2_idx
        daily_peak_val = peak2_val
        first_peak_idx = peak1_idx
        first_peak_val = peak1_val
    else:
        daily_peak_idx = peak1_idx
        daily_peak_val = peak1_val
        first_peak_idx = peak2_idx
        first_peak_val = peak2_val
    
    return {
        'min': {'idx': min_idx, 'val': min_val, 'hour': min_idx / 60.0},
        'peak': {'idx': first_peak_idx, 'val': first_peak_val, 'hour': first_peak_idx / 60.0},
        'daily_peak': {'idx': daily_peak_idx, 'val': daily_peak_val, 'hour': daily_peak_idx / 60.0}
    }

# ============================================================================
# 5. 예측 및 그래프 생성
# ============================================================================
print("\n3. 4일치 데이터 예측 및 그래프 생성")

feature_cols = ['hour', 'dayofweek', 'peakLoad', 'sessionCnt', 
                'avg_load_ma5', 'avg_load_ma30', 'avg_load_ma60', 'avg_load_std5',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

X_test = test_df[feature_cols].values
y_pred_xgb = xgb_model.predict(X_test)

test_df['y_actual'] = test_df['averageLoad'].values
test_df['y_pred_xgb'] = y_pred_xgb
test_df['date'] = test_df['time'].dt.date

dates = sorted(test_df['date'].unique())
print(f"테스트 데이터 날짜: {dates}")

# 4일치 그래프 생성
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('XGBoost Prediction - 4 Days Analysis\n(with Min, Peak, Daily Peak Points)', 
             fontsize=18, fontweight='bold')

for day_idx, date in enumerate(dates[:4]):
    day_data = test_df[test_df['date'] == date].copy()
    
    if len(day_data) == 0:
        continue
    
    hours = np.arange(len(day_data)) / 60.0
    actual_vals = day_data['y_actual'].values
    xgb_vals = day_data['y_pred_xgb'].values
    
    actual_points = find_key_points(actual_vals)
    xgb_points = find_key_points(xgb_vals)
    
    mae_day = mean_absolute_error(actual_vals, xgb_vals)
    rmse_day = np.sqrt(mean_squared_error(actual_vals, xgb_vals))
    mape_day = mean_absolute_percentage_error(actual_vals, xgb_vals)
    
    # 서브플롯 위치
    row = day_idx // 2
    col = day_idx % 2
    ax = axes[row, col]
    
    # 라인 플롯
    ax.plot(hours, actual_vals, label='Actual', linewidth=3, color='black', alpha=0.85, zorder=3)
    ax.plot(hours, xgb_vals, label='XGBoost', linewidth=2.5, color='#ff7f0e', 
            linestyle='--', alpha=0.85, zorder=2)
    
    # ============================================================================
    # Actual 특수점 표시
    # ============================================================================
    # Min
    ax.plot(actual_points['min']['hour'], actual_points['min']['val'], 'o', 
            color='black', markersize=14, markeredgewidth=2.5, markerfacecolor='none', zorder=5)
    ax.annotate(f"Min\n({actual_points['min']['hour']:.1f}h, {actual_points['min']['val']:.1f})", 
                xy=(actual_points['min']['hour'], actual_points['min']['val']),
                xytext=(actual_points['min']['hour']-1.5, actual_points['min']['val']-3),
                fontsize=10, color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9, linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Peak
    ax.plot(actual_points['peak']['hour'], actual_points['peak']['val'], 's', 
            color='black', markersize=12, markeredgewidth=2.5, markerfacecolor='none', zorder=5)
    ax.annotate(f"Peak\n({actual_points['peak']['hour']:.1f}h, {actual_points['peak']['val']:.1f})", 
                xy=(actual_points['peak']['hour'], actual_points['peak']['val']),
                xytext=(actual_points['peak']['hour']+0.5, actual_points['peak']['val']+2),
                fontsize=10, color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9, linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Daily Peak
    ax.plot(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val'], '^', 
            color='black', markersize=14, markeredgewidth=2.5, markerfacecolor='none', zorder=5)
    ax.annotate(f"Daily Peak\n({actual_points['daily_peak']['hour']:.1f}h, {actual_points['daily_peak']['val']:.1f})", 
                xy=(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val']),
                xytext=(actual_points['daily_peak']['hour']+0.5, actual_points['daily_peak']['val']+2),
                fontsize=10, color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9, linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # ============================================================================
    # XGBoost 특수점 표시
    # ============================================================================
    # Min
    ax.plot(xgb_points['min']['hour'], xgb_points['min']['val'], 'o', 
            color='#ff7f0e', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=4)
    ax.annotate(f"XGB Min\n({xgb_points['min']['hour']:.1f}h, {xgb_points['min']['val']:.1f})", 
                xy=(xgb_points['min']['hour'], xgb_points['min']['val']),
                xytext=(xgb_points['min']['hour']-1.5, xgb_points['min']['val']+2),
                fontsize=9, color='#ff7f0e', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ff7f0e', alpha=0.85, linewidth=1),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))
    
    # Peak
    ax.plot(xgb_points['peak']['hour'], xgb_points['peak']['val'], 's', 
            color='#ff7f0e', markersize=10, markeredgewidth=2, markerfacecolor='none', zorder=4)
    ax.annotate(f"XGB Peak\n({xgb_points['peak']['hour']:.1f}h, {xgb_points['peak']['val']:.1f})", 
                xy=(xgb_points['peak']['hour'], xgb_points['peak']['val']),
                xytext=(xgb_points['peak']['hour']-1.5, xgb_points['peak']['val']-2),
                fontsize=9, color='#ff7f0e', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ff7f0e', alpha=0.85, linewidth=1),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))
    
    # Daily Peak
    ax.plot(xgb_points['daily_peak']['hour'], xgb_points['daily_peak']['val'], '^', 
            color='#ff7f0e', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=4)
    ax.annotate(f"XGB Daily Peak\n({xgb_points['daily_peak']['hour']:.1f}h, {xgb_points['daily_peak']['val']:.1f})", 
                xy=(xgb_points['daily_peak']['hour'], xgb_points['daily_peak']['val']),
                xytext=(xgb_points['daily_peak']['hour']+0.5, xgb_points['daily_peak']['val']-2),
                fontsize=9, color='#ff7f0e', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ff7f0e', alpha=0.85, linewidth=1),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))
    
    # 그래프 설정
    day_name = pd.to_datetime(date).strftime('%A')
    title = f"Day {day_idx+1} ({date} - {day_name})\nMAE: {mae_day:.4f}, RMSE: {rmse_day:.4f}, MAPE: {mape_day:.2%}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU Load', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.5, 24.5])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
    ax.set_xticks(np.arange(0, 25, 2))
    
    y_min = min(actual_vals.min(), xgb_vals.min()) - 3
    y_max = max(actual_vals.max(), xgb_vals.max()) + 5
    ax.set_ylim([y_min, y_max])

plt.tight_layout()
plt.savefig('xgboost_4days_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장: xgboost_4days_analysis.png")
plt.close()

# ============================================================================
# 6. 4일치 성능 요약
# ============================================================================
print("\n" + "=" * 80)
print("4일치 성능 요약")
print("=" * 80)

for day_idx, date in enumerate(dates[:4]):
    day_data = test_df[test_df['date'] == date].copy()
    
    if len(day_data) == 0:
        continue
    
    actual_vals = day_data['y_actual'].values
    xgb_vals = day_data['y_pred_xgb'].values
    
    mae_day = mean_absolute_error(actual_vals, xgb_vals)
    rmse_day = np.sqrt(mean_squared_error(actual_vals, xgb_vals))
    mape_day = mean_absolute_percentage_error(actual_vals, xgb_vals)
    
    day_name = pd.to_datetime(date).strftime('%A')
    print(f"\nDay {day_idx+1} ({date} - {day_name}):")
    print(f"  MAE:  {mae_day:.4f}")
    print(f"  RMSE: {rmse_day:.4f}")
    print(f"  MAPE: {mape_day:.2%}")

print("\n" + "=" * 80)
print("✅ 완료!")
print("=" * 80)
