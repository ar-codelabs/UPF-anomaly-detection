# UPF CPU 전력 최적화 시스템

시계열 예측 기반 CPU 코어 및 클록 속도 동적 조정을 통한 전력 절약

## 프로젝트 개요

현재 시간과 트래픽 부하를 기반으로 미래의 CPU 사용량을 예측하고, 예측 결과에 따라 CPU 코어 수와 클록 속도를 동적으로 조정하여 전력을 절약합니다.

### 핵심 기능

1. **시계열 예측**: LSTM 모델로 CPU 사용량 및 네트워크 트래픽 예측
2. **동적 조정**: 예측 기반 CPU 코어 on/off 및 클록 속도 조정
3. **전력 절약**: 최적화 전후 전력 소비 비교 및 절약률 계산

## 데이터셋: BitBrains

### 실제 데이터 다운로드 (추천)
- **출처**: https://atlarge-research.com/gwa-t-12/
- **다운로드**: **rnd trace (156MB)** 추천
  - fastStorage trace (128MB)도 사용 가능
  - rnd가 더 다양한 워크로드 패턴 제공
- **특징**: 1,750개 VM의 실제 데이터센터 워크로드
- **포함 정보**:
  - ✅ CPU cores (동적 변화)
  - ✅ CPU capacity (MHz) - 클록 속도
  - ✅ CPU usage (MHz, %)
  - ✅ Network RX/TX (KB/s) - 트래픽 부하
  - Memory, Disk I/O
- **형식**: CSV (세미콜론-탭 구분자: `;\t`)
- **구조**: 날짜별 폴더 (2013-7, 2013-8, 2013-9) → VM별 CSV 파일 (1.csv, 2.csv, ...)

### 데이터 사용 방법

**1단계: 다운로드 및 압축 해제**
```bash
# rnd trace 다운로드 후
# data/ 폴더에 압축 해제
# 결과: data/2013-7/, data/2013-8/, data/2013-9/
```

**2단계: Python에서 로드**
```python
from download_bitbrains_data import load_real_bitbrains_data, select_single_vm

# 여러 VM 데이터 로드
df_all = load_real_bitbrains_data(
    data_dir='data',
    num_vms=10,  # 10개 VM 로드
    date_folders=['2013-7']  # 또는 ['2013-7', '2013-8', '2013-9']
)

# 단일 VM 선택 (시계열 예측용)
df = select_single_vm(df_all, vm_id=None, min_samples=5000)
```

**3단계: 노트북에서 사용**
- `UPF_CPU_Power_Optimization_BitBrains.ipynb` 열기
- 셀 2에서 `USE_REAL_DATA = True` 설정
- 실행하면 자동으로 실제 데이터 로드

### 샘플 데이터 생성 (테스트용)
실제 데이터 없이 즉시 시작하려면:
```bash
python download_bitbrains_data.py
```
또는 노트북에서 `USE_REAL_DATA = False` 설정

## 파일 구조

```
.
├── lstm_predictor.py                   # LSTM 시계열 예측 모델
├── power_optimizer.py                  # CPU 전력 최적화 로직
├── download_bitbrains_data.py          # BitBrains 데이터 생성/로드
├── UPF_CPU_Power_Optimization_BitBrains.ipynb  # 실행 노트북
├── requirements.txt                    # 의존성
├── README.md                           # 메인 문서
├── README_power_optimization.md        # 상세 문서
└── QUICKSTART.md                       # 빠른 시작 가이드
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 준비

**옵션 A: 실제 BitBrains 데이터 사용 (추천)**
1. https://atlarge-research.com/gwa-t-12/ 에서 **rnd trace (156MB)** 다운로드
2. `data/` 폴더에 압축 해제 → `data/2013-7/`, `data/2013-8/`, `data/2013-9/` 생성됨
3. 노트북에서 `USE_REAL_DATA = True` 설정

**옵션 B: 샘플 데이터 생성 (테스트용)**
```bash
python download_bitbrains_data.py
```
노트북에서 `USE_REAL_DATA = False` 설정

### 2. Jupyter Notebook 실행
```bash
jupyter notebook UPF_CPU_Power_Optimization_BitBrains.ipynb
```

### 3. 실행 순서
1. 데이터 로드 (실제 또는 샘플)
2. 데이터 시각화
3. 데이터 전처리 및 시퀀스 생성
4. LSTM 모델 학습
5. 예측 성능 평가
6. CPU 전력 최적화 시뮬레이션
7. 결과 저장 및 분석

## 모델 아키텍처

### LSTM Predictor
- **입력**: 과거 12 타임스텝 (1시간, 5분 간격)
- **출력**: 다음 타임스텝의 CPU 사용률 및 네트워크 트래픽
- **구조**: 
  - Bidirectional LSTM (2 layers, 64 hidden units)
  - Fully connected layers with dropout
  - MSE loss, Adam optimizer

### CPU Power Optimizer
- **입력**: 예측된 CPU 사용률 및 네트워크 부하
- **출력**: 최적 CPU 코어 수 및 클록 속도
- **로직**:
  - 부하 수준 분류 (low/medium/high/critical)
  - 네트워크 트래픽 고려
  - 전력 소비 모델: P = k × cores × (clock/base)² × usage

## 최적화 전략

| 부하 수준 | CPU 사용률 | 코어 수 | 클록 속도 | 목적 |
|----------|-----------|---------|----------|------|
| Low | < 30% | 4-6 | 1800 MHz | 최대 전력 절약 |
| Medium | 30-60% | 6-10 | 2400 MHz | 균형 |
| High | 60-80% | 10-14 | 2800 MHz | 성능 우선 |
| Critical | > 80% | 14-16 | 3200 MHz | 최대 성능 |

### 네트워크 트래픽 보정
- 높은 트래픽 (> 15 MB/s): 코어/클록 20% 증가
- 낮은 트래픽 (< 5 MB/s): 코어/클록 20% 감소

## 성능 메트릭

### 예측 정확도
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (결정계수)

### 전력 절약
- 평균 전력 절약 (Watts)
- 평균 절약률 (%)
- 총 에너지 절약 (kWh)

## 하이퍼파라미터

```python
# 시퀀스 설정
SEQ_LENGTH = 12        # 입력 시퀀스 길이 (1시간)
PRED_HORIZON = 1       # 예측 간격 (5분)

# 모델 설정
BATCH_SIZE = 128       # 배치 크기
EPOCHS = 50            # 에폭 수
LEARNING_RATE = 0.001  # 학습률
HIDDEN_SIZE = 64       # LSTM hidden size
NUM_LAYERS = 2         # LSTM 레이어 수

# CPU 설정
MIN_CORES = 4          # 최소 코어
MAX_CORES = 16         # 최대 코어
MIN_CLOCK = 1800       # 최소 클록 (MHz)
MAX_CLOCK = 3200       # 최대 클록 (MHz)
```

## 실제 배포 예시

```python
# 1. 모델 로드
model = LSTMPredictor(input_size=8, output_size=2)
model.load_state_dict(torch.load('best_lstm_predictor.pth'))
model.eval()

# 2. 실시간 데이터 수집 (최근 1시간)
recent_data = collect_recent_metrics(hours=1)  # 12개 샘플
X_input = scaler_X.transform(recent_data)
X_input = torch.FloatTensor(X_input).unsqueeze(0)

# 3. 예측
with torch.no_grad():
    prediction = model(X_input)
    prediction = scaler_y.inverse_transform(prediction.numpy())

predicted_cpu = prediction[0, 0]
predicted_network = prediction[0, 1]

# 4. CPU 설정 결정
optimizer = CPUPowerOptimizer()
config = optimizer.decide_cpu_config(predicted_cpu, predicted_network)

print(f"권장 설정: {config['cores']} cores @ {config['clock_mhz']} MHz")
print(f"이유: {config['reason']}")

# 5. CPU 조정 실행
adjust_cpu_cores(config['cores'])
adjust_cpu_clock(config['clock_mhz'])
```

## AWS SageMaker 배포

### 1. 모델 학습 및 저장
```python
# 노트북에서 학습 후
torch.save(model.state_dict(), 'model.pth')
```

### 2. SageMaker Endpoint 생성
```python
import sagemaker
from sagemaker.pytorch import PyTorchModel

# 모델 업로드
pytorch_model = PyTorchModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    framework_version='2.0',
    py_version='py310',
    entry_point='inference.py'
)

# Endpoint 배포
predictor = pytorch_model.deploy(
    instance_type='ml.t3.medium',
    initial_instance_count=1
)
```

### 3. 실시간 예측
```python
# 예측 요청
response = predictor.predict(recent_data)
predicted_cpu, predicted_network = response
```

## 예상 결과

- **예측 정확도**: MAPE < 10% (CPU 사용률)
- **전력 절약**: 평균 15-25% 절약
- **응답 시간**: < 100ms (실시간 예측)

## 비정상 시나리오 테스트

정상 워크로드로 학습된 모델이 비정상 상황에서 어떻게 반응하는지 테스트할 수 있습니다.

### 테스트 가능한 시나리오

1. **traffic_spike**: 트래픽 급증 (DDoS 공격 시뮬레이션)
   - 네트워크 트래픽 5-10배 증가
   - CPU 사용률도 함께 증가

2. **cpu_overload**: CPU 과부하 (비정상 프로세스)
   - CPU 사용률 80-95%로 급증
   - 네트워크는 정상 유지

3. **gradual_increase**: 점진적 부하 증가
   - 선형적으로 부하 증가
   - 모델의 추세 학습 능력 테스트

4. **sudden_drop**: 갑작스런 부하 감소 (서비스 중단)
   - CPU와 네트워크 거의 0으로
   - 이상 상황 감지 테스트

5. **oscillation**: 진동 패턴 (불안정한 워크로드)
   - 사인파 패턴으로 진동
   - 불안정한 상황 대응 테스트

### 사용 방법

```python
from anomaly_test_generator import run_anomaly_test

# 트래픽 급증 테스트
results, analysis = run_anomaly_test(
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    df=df,
    scenario='traffic_spike',
    start_idx=1000
)
```

노트북의 섹션 9에서 모든 시나리오를 테스트할 수 있습니다.

### 테스트 결과 분석

- **예측 오차**: 비정상 구간에서 예측 오차가 얼마나 증가하는가?
- **반응 속도**: 비정상 감지 후 몇 타임스텝 만에 대응하는가?
- **CPU 조정**: 적절한 코어/클록 조정 결정을 내리는가?
- **시각화**: 실제 vs 예측, CPU 설정 변화 그래프 자동 생성

## 참고 자료

- BitBrains Dataset: https://atlarge-research.com/gwa-t-12/
- LSTM for Time Series Forecasting
- CPU Power Management in Data Centers
- AWS SageMaker PyTorch Deployment

