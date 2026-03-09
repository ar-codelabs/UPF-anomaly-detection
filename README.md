# UPF 이상 탐지 시스템

5G Core PFCP 프로토콜 기반 UPF 이상 탐지 및 공격 분류 시스템

## 프로젝트 개요

3GPP 표준 기반 5G Core의 PFCP (Packet Forwarding Control Protocol) 트래픽을 분석하여 DoS 공격 및 이상 패턴을 탐지합니다.

### 세 가지 접근법

1. **LSTM Autoencoder (비지도 학습)**
   - 정상 데이터로만 학습
   - Reconstruction error 기반 이상 탐지
   - 새로운 공격 패턴 탐지 가능
   - 레이블 불필요

2. **LSTM 분류기 (지도 학습)**
   - 레이블 기반 학습
   - 높은 정확도
   - 알려진 공격 패턴에 효과적
   - Bidirectional LSTM 사용

3. **XGBoost (지도 학습)**
   - Lag features 기반 빠른 학습
   - Feature importance 제공
   - CPU만으로 충분
   - 해석 가능성 높음

## 데이터셋

**5GC PFCP Intrusion Detection Dataset**
- 출처: IEEE Dataport
- 링크: https://ieee-dataport.org/documents/5gc-pfcp-intrusion-detection-dataset-0
- 포함 공격:
  - PFCP Session Establishment DoS
  - PFCP Session Deletion DoS
  - PFCP Session Modification DoS (DROP)
  - PFCP Session Modification DoS (DUPL)

## 파일 구조

```
.
├── models.py                           # LSTM Autoencoder & Classifier 정의
├── utils.py                            # 유틸리티 함수
├── train_autoencoder.py                # Autoencoder 학습
├── train_classifier.py                 # 분류기 학습
├── train_xgboost.py                    # XGBoost 학습
├── UPF_Anomaly_Detection_Complete.ipynb # 완전한 실행 노트북
├── notebook_cells.py                   # 노트북 셀 내용 (LSTM)
├── xgboost_cells.py                    # 노트북 셀 내용 (XGBoost)
├── requirements.txt                    # 의존성
└── README.md                           # 문서
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터셋 다운로드
IEEE Dataport에서 5GC PFCP Dataset 다운로드

### 2. Jupyter Notebook 실행
```bash
jupyter notebook UPF_Anomaly_Detection_Complete.ipynb
```

### 3. 노트북 실행 순서
1. 데이터 로드 및 전처리
2. 시퀀스 생성
3. LSTM Autoencoder 학습 및 평가
4. LSTM 분류기 학습 및 평가
5. XGBoost 학습 및 평가
6. 세 가지 방법 비교
7. 앙상블 (보너스)

## 모델 비교

| 특징 | LSTM Autoencoder | LSTM Classifier | XGBoost |
|------|------------------|-----------------|---------|
| 학습 방식 | 비지도 학습 | 지도 학습 | 지도 학습 |
| 레이블 필요 | ❌ | ✅ | ✅ |
| 새로운 공격 탐지 | ✅ | ❌ | ❌ |
| 학습 시간 | 중간 | 중간 | 빠름 |
| GPU 필요 | 권장 | 권장 | 불필요 |
| 해석 가능성 | 낮음 | 중간 | 높음 |
| Feature Engineering | 불필요 | 불필요 | 필요 (Lag features) |

## 추천 사용 시나리오

- **레이블이 없는 데이터**: LSTM Autoencoder 사용
- **높은 정확도가 필요한 경우**: LSTM Classifier 사용
- **빠른 프로토타입 개발**: XGBoost 사용
- **해석 가능성이 중요한 경우**: XGBoost 사용 (Feature Importance 제공)
- **실제 운영 환경**: 세 가지 방법을 앙상블로 결합하여 사용

## 주요 기능

### LSTM Autoencoder
- 정상 트래픽 패턴 학습
- Reconstruction error 계산
- Threshold 기반 이상 탐지
- 95 percentile threshold 자동 설정

### LSTM 분류기
- Bidirectional LSTM
- Multi-class 분류
- Cross-entropy loss
- Adam optimizer

### XGBoost
- Lag features 자동 생성
- Early stopping
- Feature importance 분석
- 빠른 학습 및 예측

## 성능 메트릭

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (이진 분류)
- Confusion Matrix
- Feature Importance (XGBoost)

## 하이퍼파라미터

```python
# LSTM 공통
SEQ_LENGTH = 10        # 시퀀스 길이
BATCH_SIZE = 128       # 배치 크기
EPOCHS = 50            # 에폭 수
LEARNING_RATE = 0.001  # 학습률
HIDDEN_SIZE = 64       # LSTM hidden size
NUM_LAYERS = 2         # LSTM 레이어 수

# XGBoost
N_LAGS = 5             # Lag features 수
MAX_DEPTH = 6          # 트리 깊이
N_ESTIMATORS = 200     # 트리 개수
```

## 실제 배포

### Autoencoder 사용
```python
# 모델 로드
autoencoder.load_state_dict(torch.load('best_autoencoder.pth'))

# 예측
with torch.no_grad():
    reconstructed = autoencoder(new_data)
    error = torch.mean((new_data - reconstructed) ** 2)
    is_anomaly = error > threshold
```

### 분류기 사용
```python
# 모델 로드
classifier.load_state_dict(torch.load('best_classifier.pth'))

# 예측
with torch.no_grad():
    output = classifier(new_data)
    prediction = torch.argmax(output, dim=1)
```

### XGBoost 사용
```python
# 모델 로드
import pickle
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# 예측
y_pred = xgb_model.predict(X_new)
y_proba = xgb_model.predict_proba(X_new)
```

## 참고 논문

- Amponis et al., "Threatening the 5G core via PFCP DOS attacks", EURASIP Journal, 2022
- LSTM Autoencoder for Anomaly Detection in Time Series
- XGBoost: A Scalable Tree Boosting System

