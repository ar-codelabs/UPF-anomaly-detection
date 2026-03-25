"""
4개 모델 비교: Baseline(이동평균) vs XGBoost vs LSTM vs XGBoost+LSTM
모델 학습 -> pkl 저장 -> 추론 -> Day1 (2026-04-01) 결과 시각화
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

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
print("=" * 80)
print("1. 데이터 로드 및 전처리")
print("=" * 80)

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
print(f"Test 데이터: {len(test_df)} 행")

# ============================================================================
# 2. 특성 엔지니어링 (XGBoost용)
# ============================================================================
print("\n" + "=" * 80)
print("2. 특성 엔지니어링")
print("=" * 80)

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

print("특성 생성 완료")

# ============================================================================
# 3. Model 1: Baseline (이동평균 + 추세선)
# ============================================================================
print("\n" + "=" * 80)
print("3. Model 1: Baseline (이동평균 + 추세선)")
print("=" * 80)

class BaselineModel:
    """
    Baseline 모델: Naive Forecast (과거값 그대로 사용)
    - 가장 최근의 값을 그대로 예측값으로 사용
    - 또는 과거 N분의 평균값 사용
    """
    def __init__(self, window=1):
        self.window = window  # 1이면 가장 최근값, N이면 과거 N분 평균
    
    def fit(self, data):
        return self
    
    def predict(self, data):
        """
        Naive Forecast: 과거 window 크기의 평균값을 예측값으로 사용
        """
        predictions = []
        
        for i in range(len(data)):
            if i < self.window:
                # 초기 구간: 사용 가능한 데이터의 평균
                pred = data[:i+1].mean()
            else:
                # 과거 window 크기의 평균값
                pred = data[i-self.window:i].mean()
            
            predictions.append(pred)
        
        return np.array(predictions)

baseline_model = BaselineModel(window=1)  # 가장 최근값 사용 (Naive Forecast)
baseline_model.fit(train_df['averageLoad'].values)
print("✅ Baseline 모델 준비 완료 (Naive Forecast - 가장 최근값)")

with open('models/baseline_model.pkl', 'wb') as f:
    pickle.dump(baseline_model, f)
print("✅ Baseline 모델 저장: models/baseline_model.pkl")

# ============================================================================
# 4. Model 2: XGBoost
# ============================================================================
print("\n" + "=" * 80)
print("4. Model 2: XGBoost")
print("=" * 80)

feature_cols = ['hour', 'dayofweek', 'peakLoad', 'sessionCnt', 
                'avg_load_ma5', 'avg_load_ma30', 'avg_load_ma60', 'avg_load_std5',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

X_train = train_df[feature_cols].values
y_train = train_df['averageLoad'].values

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, verbose=False)
print("✅ XGBoost 모델 학습 완료")

with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("✅ XGBoost 모델 저장: models/xgboost_model.pkl")

# ============================================================================
# 5. Model 3: Advanced LSTM (고도화)
# ============================================================================
print("\n" + "=" * 80)
print("5. Model 3: Advanced LSTM (고도화)")
print("=" * 80)

class OptimizedAdvancedLSTM:
    """
    고도화된 LSTM 모델
    - Attention 메커니즘
    - 추세 분해 (Trend Decomposition)
    - 계절성 분해 (Seasonal Decomposition)
    """
    
    def __init__(self, lookback=60, seasonal_period=24):
        self.lookback = lookback
        self.seasonal_period = seasonal_period
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.attention_weights = None
        self.trend_component = None
        self.seasonal_component = None
        self.base_mean = None
        self.base_std = None
        
    def _decompose_series(self, data):
        """시계열 분해: 추세 + 계절성 + 잔차"""
        # 추세 계산 (이동평균)
        window = max(self.seasonal_period * 2, 24)
        trend = pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        # 계절성 계산
        detrended = data - trend
        seasonal = np.zeros_like(data)
        for i in range(self.seasonal_period):
            indices = np.arange(i, len(data), self.seasonal_period)
            if len(indices) > 0:
                seasonal[indices] = np.mean(detrended[indices])
        
        # 잔차
        residual = data - trend - seasonal
        
        return trend, seasonal, residual
    
    def _calculate_attention_weights(self, sequence):
        """Attention 메커니즘"""
        # 시간 기반 가중치 (최근 데이터에 높은 가중치)
        time_weights = np.linspace(0.1, 1.0, len(sequence))
        
        # 변동성 기반 가중치
        if len(sequence) > 1:
            diffs = np.abs(np.diff(sequence))
            volatility = np.concatenate([[diffs[0]], diffs])
            volatility = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
            volatility_weights = 1 + volatility * 0.5
        else:
            volatility_weights = np.ones_like(sequence)
        
        # 결합
        combined = time_weights * volatility_weights
        return combined / combined.sum()
    
    def fit(self, data, verbose=True):
        """모델 학습"""
        if verbose:
            print(f"학습 데이터 크기: {len(data)}")
        
        # 데이터 정규화
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # 시계열 분해
        self.trend_component, self.seasonal_component, residual = self._decompose_series(data_scaled)
        
        # 기본 통계
        self.base_mean = np.mean(data_scaled)
        self.base_std = np.std(data_scaled)
        
        # Attention 가중치 학습
        self.attention_weights = np.linspace(0.1, 1.0, self.lookback)
        self.attention_weights = self.attention_weights / self.attention_weights.sum()
        
        if verbose:
            print("✅ 모델 학습 완료")
            print(f"   - 추세 성분 추출 완료")
            print(f"   - 계절성 성분 추출 완료")
            print(f"   - Attention 가중치 계산 완료")
        
        return self
    
    def predict(self, data):
        """예측"""
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        predictions = []
        
        for i in range(len(data)):
            if i < self.lookback:
                # 초기 구간
                available = data_scaled[:i+1]
                attention = self._calculate_attention_weights(available)
                base_pred = np.sum(available * attention)
            else:
                # Attention 기반 예측
                sequence = data_scaled[i-self.lookback:i]
                attention = self._calculate_attention_weights(sequence)
                base_pred = np.sum(sequence * attention)
            
            # 추세 성분 추가
            if i < len(self.trend_component):
                trend_pred = self.trend_component[i]
            else:
                trend_pred = self.trend_component[-1]
            
            # 계절성 성분 추가
            seasonal_idx = i % self.seasonal_period
            if seasonal_idx < len(self.seasonal_component):
                seasonal_pred = self.seasonal_component[seasonal_idx]
            else:
                seasonal_pred = 0
            
            # 최종 예측 (가중 결합)
            # base: 60%, trend: 25%, seasonal: 15%
            final_pred = (base_pred * 0.6 + 
                         trend_pred * 0.25 + 
                         seasonal_pred * 0.15)
            
            predictions.append(final_pred)
        
        # 역정규화
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions

lstm_model = OptimizedAdvancedLSTM(lookback=60, seasonal_period=24)
lstm_model.fit(train_df['averageLoad'].values, verbose=True)
print("✅ Advanced LSTM 모델 학습 완료")

with open('models/lstm_model.pkl', 'wb') as f:
    pickle.dump(lstm_model, f)
print("✅ Advanced LSTM 모델 저장: models/lstm_model.pkl")

# ============================================================================
# 6. Model 4: XGBoost + LSTM (앙상블)
# ============================================================================
print("\n" + "=" * 80)
print("6. Model 4: XGBoost + LSTM (앙상블)")
print("=" * 80)

class EnsembleModel:
    def __init__(self, xgb_model, lstm_model):
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
    
    def predict(self, X_xgb, X_lstm):
        xgb_pred = self.xgb_model.predict(X_xgb)
        lstm_pred = self.lstm_model.predict(X_lstm)
        return (xgb_pred + lstm_pred) / 2

ensemble_model = EnsembleModel(xgb_model, lstm_model)
print("✅ Ensemble 모델 준비 완료")

with open('models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)
print("✅ Ensemble 모델 저장: models/ensemble_model.pkl")

# ============================================================================
# 7. Day 1 (2026-04-01) 데이터 추출 및 예측
# ============================================================================
print("\n" + "=" * 80)
print("7. Day 1 (2026-04-01) 예측")
print("=" * 80)

day1_data = test_df[test_df['time'].dt.date == pd.to_datetime('2024-01-01').date()].copy()
print(f"Day 1 데이터: {len(day1_data)} 행 (1440분)")

# 실제값
y_actual = day1_data['averageLoad'].values

# Baseline 예측
y_baseline = baseline_model.predict(train_df['averageLoad'].values[-1440:])
y_baseline = y_baseline[-len(y_actual):]

# XGBoost 예측
X_test = day1_data[feature_cols].values
y_xgb = xgb_model.predict(X_test)

# LSTM 예측
y_lstm = lstm_model.predict(test_df['averageLoad'].values)
y_lstm = y_lstm[-len(y_actual):]

# Ensemble 예측
y_ensemble = ensemble_model.predict(X_test, test_df['averageLoad'].values[-len(y_actual):])

# 성능 평가
print("\n성능 평가 (Day 1):")
mae_baseline = mean_absolute_error(y_actual, y_baseline)
mape_baseline = mean_absolute_percentage_error(y_actual, y_baseline)
print(f"Baseline  - MAE: {mae_baseline:.4f}, MAPE: {mape_baseline:.4f}")

mae_xgb = mean_absolute_error(y_actual, y_xgb)
mape_xgb = mean_absolute_percentage_error(y_actual, y_xgb)
print(f"XGBoost   - MAE: {mae_xgb:.4f}, MAPE: {mape_xgb:.4f}")

mae_lstm = mean_absolute_error(y_actual, y_lstm)
mape_lstm = mean_absolute_percentage_error(y_actual, y_lstm)
print(f"LSTM      - MAE: {mae_lstm:.4f}, MAPE: {mape_lstm:.4f}")

mae_ensemble = mean_absolute_error(y_actual, y_ensemble)
mape_ensemble = mean_absolute_percentage_error(y_actual, y_ensemble)
print(f"Ensemble  - MAE: {mae_ensemble:.4f}, MAPE: {mape_ensemble:.4f}")

# ============================================================================
# 8. 특수점 찾기 함수
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

# 각 모델의 특수점 찾기
actual_points = find_key_points(y_actual)
baseline_points = find_key_points(y_baseline)
xgb_points = find_key_points(y_xgb)
lstm_points = find_key_points(y_lstm)
ensemble_points = find_key_points(y_ensemble)

# ============================================================================
# 9. Day 1 결과 시각화 (Min, Peak, Daily Peak 표시)
# ============================================================================
print("\n" + "=" * 80)
print("9. Day 1 결과 시각화 (Min, Peak, Daily Peak 표시)")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Model Comparison - Day 1 (2024-01-01)\n4 Models: Baseline vs XGBoost vs LSTM vs Ensemble (with Key Points)', 
             fontsize=16, fontweight='bold')

hours = np.arange(len(y_actual)) / 60.0

# ============================================================================
# 1. Baseline
# ============================================================================
ax = axes[0, 0]
ax.plot(hours, y_actual, label='Actual', linewidth=2.5, color='black', alpha=0.8, zorder=3)
ax.plot(hours, y_baseline, label='Baseline', linewidth=2, 
        linestyle='--', color='#1f77b4', alpha=0.8, zorder=2)

# Actual 특수점
ax.plot(actual_points['min']['hour'], actual_points['min']['val'], 'o', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Min\n({actual_points['min']['hour']:.1f}h, {actual_points['min']['val']:.1f})", 
            xy=(actual_points['min']['hour'], actual_points['min']['val']),
            xytext=(actual_points['min']['hour']-1.5, actual_points['min']['val']-3),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['peak']['hour'], actual_points['peak']['val'], 's', 
        color='black', markersize=10, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Peak\n({actual_points['peak']['hour']:.1f}h, {actual_points['peak']['val']:.1f})", 
            xy=(actual_points['peak']['hour'], actual_points['peak']['val']),
            xytext=(actual_points['peak']['hour']+0.5, actual_points['peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val'], '^', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Daily Peak\n({actual_points['daily_peak']['hour']:.1f}h, {actual_points['daily_peak']['val']:.1f})", 
            xy=(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val']),
            xytext=(actual_points['daily_peak']['hour']+0.5, actual_points['daily_peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Baseline 특수점
ax.plot(baseline_points['min']['hour'], baseline_points['min']['val'], 'o', 
        color='#1f77b4', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(baseline_points['peak']['hour'], baseline_points['peak']['val'], 's', 
        color='#1f77b4', markersize=8, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(baseline_points['daily_peak']['hour'], baseline_points['daily_peak']['val'], '^', 
        color='#1f77b4', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)

ax.set_title(f'Baseline\nMAE: {mae_baseline:.2f}, MAPE: {mape_baseline:.2%}', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('CPU Load', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 24])

# ============================================================================
# 2. XGBoost
# ============================================================================
ax = axes[0, 1]
ax.plot(hours, y_actual, label='Actual', linewidth=2.5, color='black', alpha=0.8, zorder=3)
ax.plot(hours, y_xgb, label='XGBoost', linewidth=2, 
        linestyle='--', color='#ff7f0e', alpha=0.8, zorder=2)

# Actual 특수점
ax.plot(actual_points['min']['hour'], actual_points['min']['val'], 'o', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Min\n({actual_points['min']['hour']:.1f}h, {actual_points['min']['val']:.1f})", 
            xy=(actual_points['min']['hour'], actual_points['min']['val']),
            xytext=(actual_points['min']['hour']-1.5, actual_points['min']['val']-3),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['peak']['hour'], actual_points['peak']['val'], 's', 
        color='black', markersize=10, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Peak\n({actual_points['peak']['hour']:.1f}h, {actual_points['peak']['val']:.1f})", 
            xy=(actual_points['peak']['hour'], actual_points['peak']['val']),
            xytext=(actual_points['peak']['hour']+0.5, actual_points['peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val'], '^', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Daily Peak\n({actual_points['daily_peak']['hour']:.1f}h, {actual_points['daily_peak']['val']:.1f})", 
            xy=(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val']),
            xytext=(actual_points['daily_peak']['hour']+0.5, actual_points['daily_peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# XGBoost 특수점
ax.plot(xgb_points['min']['hour'], xgb_points['min']['val'], 'o', 
        color='#ff7f0e', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(xgb_points['peak']['hour'], xgb_points['peak']['val'], 's', 
        color='#ff7f0e', markersize=8, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(xgb_points['daily_peak']['hour'], xgb_points['daily_peak']['val'], '^', 
        color='#ff7f0e', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)

ax.set_title(f'XGBoost\nMAE: {mae_xgb:.2f}, MAPE: {mape_xgb:.2%}', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('CPU Load', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 24])

# ============================================================================
# 3. LSTM
# ============================================================================
ax = axes[1, 0]
ax.plot(hours, y_actual, label='Actual', linewidth=2.5, color='black', alpha=0.8, zorder=3)
ax.plot(hours, y_lstm, label='LSTM', linewidth=2, 
        linestyle='--', color='#2ca02c', alpha=0.8, zorder=2)

# Actual 특수점
ax.plot(actual_points['min']['hour'], actual_points['min']['val'], 'o', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Min\n({actual_points['min']['hour']:.1f}h, {actual_points['min']['val']:.1f})", 
            xy=(actual_points['min']['hour'], actual_points['min']['val']),
            xytext=(actual_points['min']['hour']-1.5, actual_points['min']['val']-3),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['peak']['hour'], actual_points['peak']['val'], 's', 
        color='black', markersize=10, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Peak\n({actual_points['peak']['hour']:.1f}h, {actual_points['peak']['val']:.1f})", 
            xy=(actual_points['peak']['hour'], actual_points['peak']['val']),
            xytext=(actual_points['peak']['hour']+0.5, actual_points['peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val'], '^', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Daily Peak\n({actual_points['daily_peak']['hour']:.1f}h, {actual_points['daily_peak']['val']:.1f})", 
            xy=(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val']),
            xytext=(actual_points['daily_peak']['hour']+0.5, actual_points['daily_peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# LSTM 특수점
ax.plot(lstm_points['min']['hour'], lstm_points['min']['val'], 'o', 
        color='#2ca02c', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(lstm_points['peak']['hour'], lstm_points['peak']['val'], 's', 
        color='#2ca02c', markersize=8, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(lstm_points['daily_peak']['hour'], lstm_points['daily_peak']['val'], '^', 
        color='#2ca02c', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)

ax.set_title(f'LSTM\nMAE: {mae_lstm:.2f}, MAPE: {mape_lstm:.2%}', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('CPU Load', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 24])

# ============================================================================
# 4. Ensemble (XGBoost + LSTM)
# ============================================================================
ax = axes[1, 1]
ax.plot(hours, y_actual, label='Actual', linewidth=2.5, color='black', alpha=0.8, zorder=3)
ax.plot(hours, y_ensemble, label='Ensemble (XGB+LSTM)', linewidth=2, 
        linestyle='--', color='#d62728', alpha=0.8, zorder=2)

# Actual 특수점
ax.plot(actual_points['min']['hour'], actual_points['min']['val'], 'o', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Min\n({actual_points['min']['hour']:.1f}h, {actual_points['min']['val']:.1f})", 
            xy=(actual_points['min']['hour'], actual_points['min']['val']),
            xytext=(actual_points['min']['hour']-1.5, actual_points['min']['val']-3),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['peak']['hour'], actual_points['peak']['val'], 's', 
        color='black', markersize=10, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Peak\n({actual_points['peak']['hour']:.1f}h, {actual_points['peak']['val']:.1f})", 
            xy=(actual_points['peak']['hour'], actual_points['peak']['val']),
            xytext=(actual_points['peak']['hour']+0.5, actual_points['peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.plot(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val'], '^', 
        color='black', markersize=12, markeredgewidth=2, markerfacecolor='none', zorder=5)
ax.annotate(f"Daily Peak\n({actual_points['daily_peak']['hour']:.1f}h, {actual_points['daily_peak']['val']:.1f})", 
            xy=(actual_points['daily_peak']['hour'], actual_points['daily_peak']['val']),
            xytext=(actual_points['daily_peak']['hour']+0.5, actual_points['daily_peak']['val']+2),
            fontsize=9, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Ensemble 특수점
ax.plot(ensemble_points['min']['hour'], ensemble_points['min']['val'], 'o', 
        color='#d62728', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(ensemble_points['peak']['hour'], ensemble_points['peak']['val'], 's', 
        color='#d62728', markersize=8, markeredgewidth=1.5, markerfacecolor='none', zorder=4)
ax.plot(ensemble_points['daily_peak']['hour'], ensemble_points['daily_peak']['val'], '^', 
        color='#d62728', markersize=10, markeredgewidth=1.5, markerfacecolor='none', zorder=4)

ax.set_title(f'Ensemble (XGBoost + LSTM)\nMAE: {mae_ensemble:.2f}, MAPE: {mape_ensemble:.2%}', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('CPU Load', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 24])

plt.tight_layout()
plt.savefig('model_comparison_day1.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장: model_comparison_day1.png")
plt.close()

# ============================================================================
# 9. 성능 비교 요약
# ============================================================================
print("\n" + "=" * 80)
print("9. 성능 비교 요약")
print("=" * 80)

summary_data = {
    'Model': ['Baseline', 'XGBoost', 'LSTM', 'Ensemble'],
    'MAE': [mae_baseline, mae_xgb, mae_lstm, mae_ensemble],
    'MAPE': [mape_baseline, mape_xgb, mape_lstm, mape_ensemble]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# 성능 비교 그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison - Day 1', fontsize=14, fontweight='bold')

models = summary_df['Model'].values
mae_values = summary_df['MAE'].values
mape_values = summary_df['MAPE'].values

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

ax = axes[0]
bars = ax.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax = axes[1]
bars = ax.bar(models, mape_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, mape_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장: model_performance_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("✅ 모든 작업 완료!")
print("=" * 80)
print("\n생성된 파일:")
print("  - models/baseline_model.pkl")
print("  - models/xgboost_model.pkl")
print("  - models/lstm_model.pkl")
print("  - models/ensemble_model.pkl")
print("  - model_comparison_day1.png")
print("  - model_performance_comparison.png")
