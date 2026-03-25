"""
XGBoost 경량 모델 최적화
pkl -> ONNX fp32 -> ONNX fp16 변환 + 벤치마크 비교

설치:
    pip install onnxmltools skl2onnx onnxruntime onnxconverter-common onnx
"""

import os
import time
import pickle
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
print("=" * 70)
print("1. 데이터 로드")
print("=" * 70)

train_df = pd.read_csv('ai_training_dataset.csv')
test_df  = pd.read_csv('ai_test_dataset.csv')

for df in [train_df, test_df]:
    df['time']        = pd.to_datetime(df['timestamp'])
    df['averageLoad'] = df['average_cpu_load']
    df['peakLoad']    = df['peak_cpu_load']
    df['sessionCnt']  = df['active_session_count']

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
N_FEATURES = len(FEATURE_COLS)

X_test   = test_df[FEATURE_COLS].values.astype(np.float32)
y_actual = test_df['averageLoad'].values

print(f"테스트 데이터: {len(X_test)} rows, {N_FEATURES} features")

# ============================================================================
# 2. pkl 모델 로드
# ============================================================================
print("\n" + "=" * 70)
print("2. XGBoost pkl 모델 로드")
print("=" * 70)

with open('models/xgboost_model.pkl', 'rb') as f:
    xgb_pkl = pickle.load(f)

y_pred_pkl = xgb_pkl.predict(X_test)
mae_pkl    = mean_absolute_error(y_actual, y_pred_pkl)
mape_pkl   = mean_absolute_percentage_error(y_actual, y_pred_pkl)
size_pkl   = os.path.getsize('models/xgboost_model.pkl') / 1024  # KB

print(f"pkl 크기: {size_pkl:.1f} KB")
print(f"pkl MAE:  {mae_pkl:.6f}")
print(f"pkl MAPE: {mape_pkl:.4%}")

# ============================================================================
# 3. ONNX fp32 변환
# ============================================================================
print("\n" + "=" * 70)
print("3. ONNX fp32 변환")
print("=" * 70)

try:
    import onnxmltools
    from skl2onnx.common.data_types import FloatTensorType
    import onnx

    initial_type = [('float_input', FloatTensorType([None, N_FEATURES]))]
    onnx_model_fp32 = onnxmltools.convert_xgboost(xgb_pkl, initial_types=initial_type)

    os.makedirs('models', exist_ok=True)
    with open('models/xgboost_model_fp32.onnx', 'wb') as f:
        f.write(onnx_model_fp32.SerializeToString())

    print("✅ ONNX fp32 변환 완료: models/xgboost_model_fp32.onnx")

except ImportError as e:
    print(f"❌ 패키지 미설치: {e}")
    print("   pip install onnxmltools skl2onnx onnx 실행 후 재시도")
    exit(1)

# ============================================================================
# 4. ONNX fp16 양자화
# ============================================================================
print("\n" + "=" * 70)
print("4. ONNX fp16 양자화")
print("=" * 70)

try:
    from onnxconverter_common import float16

    model_fp32_loaded = onnx.load('models/xgboost_model_fp32.onnx')
    model_fp16        = float16.convert_float_to_float16(model_fp32_loaded)
    onnx.save(model_fp16, 'models/xgboost_model_fp16.onnx')

    print("✅ ONNX fp16 양자화 완료: models/xgboost_model_fp16.onnx")

except ImportError as e:
    print(f"❌ 패키지 미설치: {e}")
    print("   pip install onnxconverter-common 실행 후 재시도")
    exit(1)

# ============================================================================
# 5. ONNX 추론 및 정확도 검증
# ============================================================================
print("\n" + "=" * 70)
print("5. ONNX 추론 및 정확도 검증")
print("=" * 70)

import onnxruntime as ort

sess_fp32 = ort.InferenceSession('models/xgboost_model_fp32.onnx')
sess_fp16 = ort.InferenceSession('models/xgboost_model_fp16.onnx')

input_name_fp32 = sess_fp32.get_inputs()[0].name
input_name_fp16 = sess_fp16.get_inputs()[0].name

y_pred_fp32 = sess_fp32.run(None, {input_name_fp32: X_test})[0].flatten()
y_pred_fp16 = sess_fp16.run(None, {input_name_fp16: X_test.astype(np.float16)})[0].flatten()

mae_fp32  = mean_absolute_error(y_actual, y_pred_fp32)
mape_fp32 = mean_absolute_percentage_error(y_actual, y_pred_fp32)
mae_fp16  = mean_absolute_error(y_actual, y_pred_fp16)
mape_fp16 = mean_absolute_percentage_error(y_actual, y_pred_fp16)
size_fp32 = os.path.getsize('models/xgboost_model_fp32.onnx') / 1024
size_fp16 = os.path.getsize('models/xgboost_model_fp16.onnx') / 1024

print(f"ONNX fp32 크기: {size_fp32:.1f} KB  MAE: {mae_fp32:.6f}  MAPE: {mape_fp32:.4%}")
print(f"ONNX fp16 크기: {size_fp16:.1f} KB  MAE: {mae_fp16:.6f}  MAPE: {mape_fp16:.4%}")

# ============================================================================
# 6. 추론 레이턴시 벤치마크 (1,000회 반복)
# ============================================================================
print("\n" + "=" * 70)
print("6. 추론 레이턴시 벤치마크 (단일 샘플, 1,000회 반복)")
print("=" * 70)

WARMUP = 50
REPEAT = 1000
single_fp32 = X_test[:1]
single_fp16 = X_test[:1].astype(np.float16)

# pkl
for _ in range(WARMUP): xgb_pkl.predict(single_fp32)
t0 = time.perf_counter()
for _ in range(REPEAT): xgb_pkl.predict(single_fp32)
lat_pkl = (time.perf_counter() - t0) / REPEAT * 1000  # ms

# ONNX fp32
for _ in range(WARMUP): sess_fp32.run(None, {input_name_fp32: single_fp32})
t0 = time.perf_counter()
for _ in range(REPEAT): sess_fp32.run(None, {input_name_fp32: single_fp32})
lat_fp32 = (time.perf_counter() - t0) / REPEAT * 1000

# ONNX fp16
for _ in range(WARMUP): sess_fp16.run(None, {input_name_fp16: single_fp16})
t0 = time.perf_counter()
for _ in range(REPEAT): sess_fp16.run(None, {input_name_fp16: single_fp16})
lat_fp16 = (time.perf_counter() - t0) / REPEAT * 1000

print(f"pkl      레이턴시: {lat_pkl:.4f} ms")
print(f"ONNX fp32 레이턴시: {lat_fp32:.4f} ms  (pkl 대비 {lat_pkl/lat_fp32:.1f}배 빠름)")
print(f"ONNX fp16 레이턴시: {lat_fp16:.4f} ms  (pkl 대비 {lat_pkl/lat_fp16:.1f}배 빠름)")

# ============================================================================
# 7. 결과 요약 출력
# ============================================================================
print("\n" + "=" * 70)
print("7. 최적화 결과 요약")
print("=" * 70)

results = {
    'Model':     ['XGBoost pkl', 'XGBoost ONNX fp32', 'XGBoost ONNX fp16'],
    'Size(KB)':  [size_pkl,  size_fp32,  size_fp16],
    'MAE':       [mae_pkl,   mae_fp32,   mae_fp16],
    'MAPE(%)':   [mape_pkl*100, mape_fp32*100, mape_fp16*100],
    'Latency(ms)':[lat_pkl,  lat_fp32,   lat_fp16],
}
df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))

print(f"\n크기 감소 (pkl → fp16): {(1 - size_fp16/size_pkl)*100:.1f}%")
print(f"속도 향상 (pkl → fp16): {lat_pkl/lat_fp16:.1f}배")
print(f"MAPE 차이 (pkl → fp16): {abs(mape_pkl - mape_fp16)*100:.4f}%p")

# ============================================================================
# 8. 시각화
# ============================================================================
print("\n" + "=" * 70)
print("8. 시각화")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('XGBoost Lightweight Model Optimization\npkl vs ONNX fp32 vs ONNX fp16',
             fontsize=14, fontweight='bold')

model_labels = ['pkl', 'ONNX fp32', 'ONNX fp16']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 크기 비교
ax = axes[0]
bars = ax.bar(model_labels, df_result['Size(KB)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_title('Model Size (KB)', fontsize=12, fontweight='bold')
ax.set_ylabel('Size (KB)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, df_result['Size(KB)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# MAPE 비교
ax = axes[1]
bars = ax.bar(model_labels, df_result['MAPE(%)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_title('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('MAPE (%)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, df_result['MAPE(%)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.4f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 레이턴시 비교
ax = axes[2]
bars = ax.bar(model_labels, df_result['Latency(ms)'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_title('Inference Latency (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, df_result['Latency(ms)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
os.makedirs('images', exist_ok=True)
plt.savefig('images/onnx_optimization_benchmark.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 그래프 저장: images/onnx_optimization_benchmark.png")

# ============================================================================
# 9. 결과를 파일로 저장
# ============================================================================
df_result.to_csv('models/optimization_benchmark.csv', index=False)
print("✅ 벤치마크 결과 저장: models/optimization_benchmark.csv")

print("\n" + "=" * 70)
print("✅ 완료!")
print("=" * 70)
print("\n생성된 파일:")
print("  - models/xgboost_model_fp32.onnx")
print("  - models/xgboost_model_fp16.onnx")
print("  - models/optimization_benchmark.csv")
print("  - images/onnx_optimization_benchmark.png")
