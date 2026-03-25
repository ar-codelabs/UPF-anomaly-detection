"""
1시~4시 구간 확대 비교: XGBoost vs Baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=" * 80)
print("1시~4시 구간 확대 비교")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
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
# 3. 모델 로드
# ============================================================================
print("\n모델 로드 중...")
with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
print("✅ XGBoost 모델 로드 완료")

# ============================================================================
# 4. Day 1 데이터 추출
# ============================================================================
day1_data = test_df[test_df['time'].dt.date == pd.to_datetime('2024-01-01').date()].copy()
print(f"Day 1 데이터: {len(day1_data)} 행")

# ============================================================================
# 5. 예측
# ============================================================================
y_actual = day1_data['averageLoad'].values

# Baseline (Naive Forecast - 가장 최근값)
class BaselineModel:
    def __init__(self, window=1):
        self.window = window
    
    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            if i < self.window:
                pred = data[:i+1].mean()
            else:
                pred = data[i-self.window:i].mean()
            predictions.append(pred)
        return np.array(predictions)

baseline_model = BaselineModel(window=1)
y_baseline = baseline_model.predict(train_df['averageLoad'].values[-1440:])
y_baseline = y_baseline[-len(y_actual):]

# XGBoost
feature_cols = ['hour', 'dayofweek', 'peakLoad', 'sessionCnt', 
                'avg_load_ma5', 'avg_load_ma30', 'avg_load_ma60', 'avg_load_std5',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
X_test = day1_data[feature_cols].values
y_xgb = xgb_model.predict(X_test)

# ============================================================================
# 6. 1시~4시 구간 추출 (60분 * 3시간 = 180분)
# ============================================================================
# 1시 = 60분, 4시 = 240분
start_idx = 60  # 1시 시작
end_idx = 240   # 4시 끝

y_actual_zoom = y_actual[start_idx:end_idx]
y_baseline_zoom = y_baseline[start_idx:end_idx]
y_xgb_zoom = y_xgb[start_idx:end_idx]

# 분 단위로 변환 (1시부터 시작)
minutes = np.arange(len(y_actual_zoom))

# ============================================================================
# 7. 시각화
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Zoomed Comparison: 1:00 AM - 4:00 AM (2024-01-01)\nXGBoost vs Baseline', 
             fontsize=16, fontweight='bold')

# 상단: 전체 비교
ax = axes[0]
ax.plot(minutes, y_actual_zoom, label='Actual', linewidth=3, color='black', marker='o', markersize=4, alpha=0.9)
ax.plot(minutes, y_baseline_zoom, label='Baseline (Naive)', linewidth=2.5, 
        linestyle='--', color='#1f77b4', marker='s', markersize=3, alpha=0.8)
ax.plot(minutes, y_xgb_zoom, label='XGBoost', linewidth=2.5, 
        linestyle='--', color='#ff7f0e', marker='^', markersize=3, alpha=0.8)

mae_baseline = mean_absolute_error(y_actual_zoom, y_baseline_zoom)
mae_xgb = mean_absolute_error(y_actual_zoom, y_xgb_zoom)
mape_baseline = mean_absolute_percentage_error(y_actual_zoom, y_baseline_zoom)
mape_xgb = mean_absolute_percentage_error(y_actual_zoom, y_xgb_zoom)

ax.set_title(f'Full View (1:00 - 4:00)\nBaseline: MAE={mae_baseline:.4f}, MAPE={mape_baseline:.2%} | XGBoost: MAE={mae_xgb:.4f}, MAPE={mape_xgb:.2%}', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Minutes from 1:00 AM', fontsize=11)
ax.set_ylabel('CPU Load', fontsize=11)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# 하단: 에러 비교
errors_baseline = np.abs(y_actual_zoom - y_baseline_zoom)
errors_xgb = np.abs(y_actual_zoom - y_xgb_zoom)

ax = axes[1]
ax.plot(minutes, errors_baseline, label='Baseline Error', linewidth=2.5, 
        color='#1f77b4', marker='s', markersize=3, alpha=0.8)
ax.plot(minutes, errors_xgb, label='XGBoost Error', linewidth=2.5, 
        color='#ff7f0e', marker='^', markersize=3, alpha=0.8)
ax.fill_between(minutes, errors_baseline, alpha=0.2, color='#1f77b4')
ax.fill_between(minutes, errors_xgb, alpha=0.2, color='#ff7f0e')

ax.set_title('Absolute Error Comparison', fontsize=12, fontweight='bold')
ax.set_xlabel('Minutes from 1:00 AM', fontsize=11)
ax.set_ylabel('Absolute Error', fontsize=11)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zoom_1to4_hours_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장: zoom_1to4_hours_comparison.png")
plt.close()

# ============================================================================
# 8. 상세 통계
# ============================================================================
print("\n" + "=" * 80)
print("1시~4시 구간 상세 분석")
print("=" * 80)

print(f"\n실제값 (Actual):")
print(f"  평균: {y_actual_zoom.mean():.4f}")
print(f"  최소: {y_actual_zoom.min():.4f}")
print(f"  최대: {y_actual_zoom.max():.4f}")
print(f"  표준편차: {y_actual_zoom.std():.4f}")

print(f"\nBaseline 예측:")
print(f"  평균: {y_baseline_zoom.mean():.4f}")
print(f"  최소: {y_baseline_zoom.min():.4f}")
print(f"  최대: {y_baseline_zoom.max():.4f}")
print(f"  MAE: {mae_baseline:.4f}")
print(f"  MAPE: {mape_baseline:.2%}")

print(f"\nXGBoost 예측:")
print(f"  평균: {y_xgb_zoom.mean():.4f}")
print(f"  최소: {y_xgb_zoom.min():.4f}")
print(f"  최대: {y_xgb_zoom.max():.4f}")
print(f"  MAE: {mae_xgb:.4f}")
print(f"  MAPE: {mape_xgb:.2%}")

print(f"\n개선도:")
print(f"  MAE 개선: {(mae_baseline - mae_xgb) / mae_baseline * 100:.1f}%")
print(f"  MAPE 개선: {(mape_baseline - mape_xgb) / mape_baseline * 100:.1f}%")

print("\n" + "=" * 80)
print("✅ 완료!")
print("=" * 80)
