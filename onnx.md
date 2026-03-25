# 5G UPF 런타임 환경 리서치 및 경량화 모델 배포 전략

**작성일**: 2026-03-23  
**작성자**: AI Research Team  
**목적**: CPU Load Forecasting 모델을 5G UPF 네트워크 장비에 배포하기 위한 런타임 환경 분석 및 경량화 실험 결과

---

## Executive Summary (주요 발견사항)

### 핵심 결론

1. **현재 모델 크기**: XGBoost pkl 모델이 **477.1 KB** — 이미 경량이나 추론 속도 개선 여지 있음
2. **타겟 환경**: 5G UPF는 주로 **Linux 컨테이너 기반**으로 배포되며, x86-64 또는 ARM64 아키텍처 사용
3. **최적 배포 방식**: **ONNX Runtime** 기반 배포가 가장 적합 (크로스 플랫폼, 경량, 빠른 추론)
4. **최적 모델**: **XGBoost ONNX fp32 (373.6 KB, MAPE 0.60%)** — pkl 대비 속도 19.9배 향상, 정확도 완전 동일
5. **fp16 주의**: XGBoost에서 fp16 양자화 시 MAPE 0.39%p 손실 발생 → fp32 권장

### 액션 아이템

- [x] XGBoost 모델 ONNX 변환 (pkl → ONNX fp32 → ONNX fp16)
- [x] ONNX 추론 정확도 검증 (MAPE 비교)
- [x] 추론 성능 벤치마크 수행 (1,000회 반복)
- [x] 크기 / 속도 / 정확도 비교 그래프 생성
- [ ] Docker 컨테이너 배포 가이드 작성
- [ ] ONNX Runtime 추론 서버 구현 (FastAPI)

---

## 1. 오픈소스 5G UPF 구현체 분석

### 1.1 Open5GS (가장 성숙한 오픈소스)

**기본 정보:**
- **구현 언어**: C (97.5%)
- **라이선스**: Apache 2.0
- **프로젝트 성숙도**: 프로덕션 레벨

**시스템 요구사항:**
- **OS**: Ubuntu 18.04+, Debian 10+ (Linux 기반)
- **CPU 아키텍처**: x86-64, ARM64 모두 지원
- **메모리**: 2-4GB RAM (테스트 환경 기준)
- **배포 방식**: Docker, VirtualBox, 베어메탈

### 1.2 free5GC (Go 기반 구현)

**기본 정보:**
- **구현 언어**: Go (93.5%), Shell (6.1%)
- **라이선스**: Apache 2.0
- **특징**: 경량 바이너리, 빠른 빌드

**시스템 요구사항:**
- **OS**: Linux (커널 5.0.0-23-generic 이상, 5.4+ 권장)
- **커널 모듈**: gtp5g (GTP-U 프로토콜 처리용 커널 모듈)
- **배포 방식**: 컨테이너, 베어메탈

### 1.3 배포 환경 특성 비교

| 특성 | Open5GS | free5GC |
|------|---------|---------|
| 구현 언어 | C | Go |
| 메모리 사용량 | 중간 (C 네이티브) | 낮음 (Go 효율적) |
| 빌드 속도 | 느림 (C 컴파일) | 빠름 (Go 빌드) |
| 커널 의존성 | 낮음 | 높음 (gtp5g 모듈) |
| CPU 아키텍처 | x86-64, ARM64 | Linux 지원 모든 아키텍처 |

---

## 2. 실제 배포 환경 특성

### 2.1 컨테이너화 배포 (현재 주류)

#### Kubernetes 기반 배포

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: upf-pod
spec:
  containers:
  - name: upf
    image: upf:latest
    resources:
      requests:
        cpu: "1000m"        # 1 코어
        memory: "2Gi"       # 2GB RAM
      limits:
        cpu: "2000m"        # 최대 2 코어
        memory: "4Gi"       # 최대 4GB RAM
```

### 2.2 엣지 컴퓨팅 환경 (MEC)

| 항목 | Edge DC | Central DC |
|------|---------|------------|
| **CPU** | 2-8 코어 | 16-64 코어 |
| **메모리** | 4-16GB | 64-256GB |
| **레이턴시** | <5ms | 10-50ms |
| **리소스** | 제한적 | 풍부 |

---

## 3. ML 모델 배포 런타임 옵션 비교

### 3.1 런타임 옵션 상세 분석

#### Option A: Python + scikit-learn / XGBoost (현재 방식)

**장점:**
- ✅ 개발 속도 빠름, 디버깅 쉬움
- ✅ 라이브러리 풍부, 유지보수 용이

**단점:**
- ❌ 추론 속도 느림 (Python 인터프리터 오버헤드)
- ❌ 의존성 많음 (numpy, pandas 등)

**시스템 요구사항:**
```
CPU: x86-64, ARM64
메모리: 최소 512MB
OS: Linux with Python 3.9+
```

#### Option B: ONNX Runtime (권장) ⭐⭐⭐⭐⭐

**장점:**
- ✅ 크로스 플랫폼 (x86-64, ARM64, ARM)
- ✅ 경량 런타임 (10-50MB)
- ✅ 빠른 추론 (C++ 기반)
- ✅ 양자화 지원 (float32 → float16/int8)
- ✅ 컨테이너 친화적

**단점:**
- ⚠️ 모델 변환 필요 (XGBoost pkl → ONNX)
- ⚠️ 디버깅 복잡도 증가

**시스템 요구사항:**
```
CPU: x86-64, ARM64, ARM (Cortex-A)
메모리: 최소 512MB
디스크: 50MB (런타임) + 1-5MB (모델)
OS: Linux, Windows, macOS
```

### 3.2 런타임 옵션 비교 표

| 항목 | Python + XGBoost | ONNX Runtime | TensorFlow Lite | Native C/C++ |
|------|-----------------|--------------|-----------------|--------------|
| **모델 크기** | 477 KB | 374 KB | 100KB-1MB | 50KB-500KB |
| **런타임 크기** | 500MB+ | 10-50MB | 1-5MB | 0MB |
| **추론 시간** | ~0.17ms | ~0.009ms | <5ms | <1ms |
| **개발 난이도** | ⭐ (쉬움) | ⭐⭐⭐ (중간) | ⭐⭐⭐⭐ (어려움) | ⭐⭐⭐⭐⭐ (매우 어려움) |
| **권장도** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 4. 시나리오별 배포 전략

### 시나리오 A: 상용 네트워크 장비 (Intel x86 서버)

**추천 배포 방식: ONNX Runtime (컨테이너)**

```dockerfile
FROM python:3.10-slim

RUN pip install onnxruntime==1.16.0

COPY models/xgboost_model_fp32.onnx /app/model.onnx
COPY inference_server.py /app/

WORKDIR /app
CMD ["python", "inference_server.py"]
```

### 시나리오 B: 엣지 컴퓨팅 박스 (ARM 기반)

**추천 배포 방식: ONNX Runtime fp32**

- 모델 크기: 373.6 KB (충분히 경량)
- 추론 시간: 0.009ms (실시간 처리 가능)
- 메모리 사용: <512MB

---

## 5. 권장 아키텍처

### 5.1 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    5G UPF Container                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Main UPF Process (Go/C)                      │  │
│  │  - GTP-U 패킷 처리 (데이터 플레인)                  │  │
│  │  - 세션 관리 / QoS 제어                              │  │
│  │                                                       │  │
│  │  ┌────────────────────────────────────────┐          │  │
│  │  │  Monitoring Agent                      │          │  │
│  │  │  - CPU 메트릭 수집 (1분 간격)          │          │  │
│  │  │  - Feature engineering (12개)          │          │  │
│  │  │  - Historical data buffer (24시간)     │          │  │
│  │  └──────────────┬─────────────────────────┘          │  │
│  └─────────────────┼──────────────────────────────────────┘  │
│                    │ gRPC/REST                               │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      ML Inference Service (Sidecar Container)        │  │
│  │                                                       │  │
│  │  Runtime: ONNX Runtime (C++ API)                     │  │
│  │  Model: xgboost_model_fp32.onnx (373.6 KB)           │  │
│  │  Memory: ~512MB                                       │  │
│  │  Latency: 0.009ms                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 API 인터페이스 설계

#### REST API (권장)

```python
# inference_server.py
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
from datetime import datetime

app = FastAPI()

# 모델 로딩 (시작 시 한 번만)
session    = ort.InferenceSession("models/xgboost_model_fp32.onnx")
input_name = session.get_inputs()[0].name

@app.post("/predict")
async def predict(features: list[float]):
    """
    CPU load 예측
    Input: 12개 features
    Output: 다음 1분 평균 CPU load
    """
    input_array = np.array([features], dtype=np.float32)
    outputs     = session.run(None, {input_name: input_array})
    prediction  = float(outputs[0][0])
    return {
        "prediction": prediction,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## 6. 경량화 실험 결과 (실측)

> 벤치마크 환경: macOS, Python 3.9, ONNX Runtime  
> 스크립트: `optimize_xgb_onnx.py`  
> 테스트 데이터: 36,000 샘플 (5 패턴 × 5일), 레이턴시 측정: 단일 샘플 × 1,000회 반복

### 6.1 XGBoost ONNX 변환 결과 (실측)

**원본 모델 설정:**
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# 결과: 477.1 KB (pkl)
```

**ONNX 변환 + float16 양자화 결과:**

| 모델 | 크기 | MAE | MAPE | 추론 레이턴시 (단일) | 비고 |
|------|------|-----|------|---------------------|------|
| XGBoost pkl | 477.1 KB | 0.2833 | 0.6049% | 0.1688 ms | 원본 |
| XGBoost ONNX fp32 | 373.6 KB | 0.2833 | 0.6049% | 0.0085 ms | **권장** ✅ |
| XGBoost ONNX fp16 | 373.9 KB | 0.4588 | 0.9937% | 0.0090 ms | 정확도 손실 ⚠️ |

**핵심 발견:**
- ONNX fp32 변환으로 크기 21.6% 감소 (477.1 KB → 373.6 KB)
- **추론 속도 19.9배 향상** (0.1688ms → 0.0085ms)
- fp32는 정확도 완전 동일 (MAPE 변화 없음)
- **fp16은 MAPE 0.39%p 손실** — XGBoost 트리 구조에서 float16 양자화 효과 제한적
- → **ONNX fp32가 최적 배포 모델**

### 6.2 전체 모델 비교 요약 (실측)

| 모델 | 크기 | MAPE | 추론 레이턴시 | 배포 권장도 |
|------|------|------|-------------|-----------|
| XGBoost pkl | 477.1 KB | 0.60% | 0.1688 ms | ⭐⭐⭐ 개발용 |
| **XGBoost ONNX fp32** | **373.6 KB** | **0.60%** | **0.0085 ms** | **⭐⭐⭐⭐⭐ 권장** |
| XGBoost ONNX fp16 | 373.9 KB | 0.99% | 0.0090 ms | ⭐⭐ 정확도 손실 |

**권장사항:**
- **일반 배포**: XGBoost ONNX fp32 (373.6 KB, MAPE 0.60%) — 정확도 동일, 속도 19.9배
- **fp16 비권장**: 크기 감소 효과 없고 MAPE 0.39%p 손실 발생

---

## 7. 성능 벤치마크 결과 (실측)

> 벤치마크 환경: macOS, Python 3.9, ONNX Runtime  
> 레이턴시 측정: 단일 샘플 × 1,000회 반복 (워밍업 50회 포함)

### 7.1 추론 레이턴시 벤치마크

| 모델 | 레이턴시 (ms) | pkl 대비 속도 향상 |
|------|-------------|-----------------|
| XGBoost pkl | 0.1688 ms | 1x (기준) |
| **XGBoost ONNX fp32** | **0.0085 ms** | **19.9배** |
| XGBoost ONNX fp16 | 0.0090 ms | 18.8배 |

**핵심 발견:**
- ONNX 변환만으로 약 20배 속도 향상 (Python 인터프리터 오버헤드 제거)
- 모든 ONNX 모델이 <0.01ms 레이턴시로 실시간 처리에 충분
- fp16이 fp32보다 오히려 약간 느림 (변환 오버헤드)

### 7.2 크기 vs 정확도 트레이드오프 (실측)

| 모델 | 크기 | MAPE | Δ MAPE | 크기 감소율 |
|------|------|------|--------|-----------|
| XGBoost pkl | 477.1 KB | 0.6049% | — | — |
| XGBoost ONNX fp32 | 373.6 KB | 0.6049% | 0.0000%p | 21.7% |
| XGBoost ONNX fp16 | 373.9 KB | 0.9937% | +0.3888%p | 21.6% |

### 7.3 벤치마크 그래프

![ONNX 최적화 벤치마크](images/onnx_optimization_benchmark.png)

*좌: 모델 크기(KB) / 중: MAPE(%) / 우: 추론 레이턴시(ms) — pkl vs ONNX fp32 vs ONNX fp16 비교*

### 7.4 벤치마크 재현 스크립트

```bash
# 의존성 설치
pip install onnxmltools skl2onnx onnxruntime onnxconverter-common onnx

# 실행
python optimize_xgb_onnx.py
```

결과 파일 위치:
- `models/xgboost_model_fp32.onnx` — ONNX fp32 모델
- `models/xgboost_model_fp16.onnx` — ONNX fp16 모델
- `models/optimization_benchmark.csv` — 벤치마크 수치 결과
- `images/onnx_optimization_benchmark.png` — 벤치마크 그래프

---

## 8. 배포용 추론 코드

### 8.1 ONNX fp32 추론 (권장)

```python
import onnxruntime as ort
import numpy as np

# 모델 로드 (373.6 KB, 정확도 동일)
session    = ort.InferenceSession('models/xgboost_model_fp32.onnx')
input_name = session.get_inputs()[0].name

# 실시간 예측 (단일 샘플, 0.0085ms)
features = np.array([[
    hour, dayofweek, peakLoad, sessionCnt,
    avg_load_ma5, avg_load_ma30, avg_load_ma60, avg_load_std5,
    hour_sin, hour_cos, day_sin, day_cos
]], dtype=np.float32)

prediction = session.run(None, {input_name: features})[0][0]
print(f"예측 CPU 부하: {prediction:.2f}%")
```

### 8.2 pkl 추론 (개발/테스트용)

```python
import pickle
import numpy as np

with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

features = np.array([[...]])  # shape: (1, 12)
prediction = model.predict(features)[0]
print(f"예측 CPU 부하: {prediction:.2f}%")
```

**요구사항 (ONNX fp32)**: CPU 1코어, RAM 512MB, 레이턴시 <1ms

---

## 9. 다음 단계 (Action Items)

### Phase 1: 모델 경량화 ✅ 완료

**완료된 작업:**
```bash
python optimize_xgb_onnx.py
```

**생성된 모델 파일 (`models/`):**
- `xgboost_model.pkl` (477.1 KB) — 원본
- `xgboost_model_fp32.onnx` (373.6 KB) — **권장 배포 모델**
- `xgboost_model_fp16.onnx` (373.9 KB) — 정확도 손실로 비권장

### Phase 2: 성능 벤치마크 ✅ 완료

벤치마크 결과:
- `models/optimization_benchmark.csv` — 수치 결과
- `images/onnx_optimization_benchmark.png` — 시각화

### Phase 3: 배포 준비 (우선순위: 높음, 미완료)

**Task 3.1: ONNX Runtime 추론 서버 구현**
```bash
# FastAPI + ONNX Runtime
pip install fastapi uvicorn
uvicorn inference_server:app --host 0.0.0.0 --port 8080
```

**Task 3.2: Docker 이미지 빌드**
```bash
docker build -t upf-ml-inference:latest -f Dockerfile.onnx .
```

**Task 3.3: Kubernetes 매니페스트 작성**
```bash
kubectl apply -f deployment/k8s-deployment.yaml
```

### Phase 4: 실제 환경 테스트 (우선순위: 중간, 미완료)

- [ ] 로컬 UPF 환경 셋업 (free5GC 또는 Open5GS)
- [ ] ML 서비스 통합 테스트
- [ ] 부하 테스트 (1,000 RPS 기준)

---

## 10. 리스크 및 제한사항

### 10.1 기술적 리스크

1. **fp16 정확도 손실** ⚠️ 확인됨
   - XGBoost ONNX fp16: MAPE +0.39%p 증가
   - 원인: XGBoost 트리 구조에서 float16 정밀도 손실
   - 완화: **fp32 사용 권장**

2. **런타임 호환성 문제**
   - 리스크: ONNX Runtime이 특정 CPU에서 미지원
   - 완화: 배포 전 타겟 환경에서 테스트 필수

3. **모델 업데이트**
   - 리스크: 재학습 시 ONNX 변환 파이프라인 재실행 필요
   - 완화: CI/CD 파이프라인 자동화

### 10.2 제한사항

1. **Feature Engineering 의존성**
   - 이동평균 계산을 위해 과거 60분 데이터 필요
   - 메모리 버퍼: ~3KB (60 samples × 8 bytes × 6 features)

2. **Cold Start 문제**
   - UPF 재시작 시 초기 60분은 이동평균 feature 불완전
   - 완화: `min_periods=1` 설정으로 부분 데이터 허용 (현재 적용됨)

3. **실시간 재학습 불가**
   - 경량 모델은 추론 전용, 학습은 중앙에서 수행
   - 모델 업데이트는 별도 프로세스 필요

---

## 11. 참고 자료

- **ONNX Runtime**: https://onnxruntime.ai
- **onnxmltools (XGBoost 변환)**: https://github.com/onnx/onnxmltools
- **skl2onnx**: https://github.com/onnx/sklearn-onnx
- **ONNX Runtime Quantization**: https://onnxruntime.ai/docs/performance/quantization.html
- **Open5GS**: https://open5gs.org
- **free5GC**: https://free5gc.org

---

## 12. 결론

### 핵심 요약 (실측 기반)

1. **5G UPF는 Linux 컨테이너 기반**으로 배포되며, x86-64 또는 ARM64 환경에서 실행됨
2. **ONNX Runtime 기반 배포**가 가장 적합 — 0.0085ms 추론 레이턴시 달성 (pkl 대비 19.9배)
3. **XGBoost ONNX fp32 (373.6 KB, MAPE 0.60%)** 가 최적 배포 모델 — 정확도 완전 동일
4. **fp16 양자화는 XGBoost에서 비권장** — 크기 감소 효과 없고 MAPE 0.39%p 손실 발생
5. 원본 XGBoost가 이미 경량(477 KB)이므로 크기보다 **속도 향상이 주요 이점**

### 배포 모델 선정 가이드

| 시나리오 | 권장 모델 | 파일 | 크기 | MAPE |
|----------|----------|------|------|------|
| **일반 배포 (추천)** | XGBoost ONNX fp32 | `xgboost_model_fp32.onnx` | 373.6 KB | 0.60% |
| **개발/테스트** | XGBoost pkl | `xgboost_model.pkl` | 477.1 KB | 0.60% |
| **비권장** | XGBoost ONNX fp16 | `xgboost_model_fp16.onnx` | 373.9 KB | 0.99% |

### 권장 Action Plan

**단기 — 완료됨 ✅:**
- [x] XGBoost ONNX 변환 및 fp16 양자화 구현 (`optimize_xgb_onnx.py`)
- [x] 성능 벤치마크 수행 (레이턴시, 크기, 정확도)
- [x] 벤치마크 그래프 생성 (`images/onnx_optimization_benchmark.png`)

**중기 — 다음 단계:**
- [ ] ONNX Runtime 추론 서버 구현 (FastAPI + ONNX Runtime)
- [ ] Docker 이미지 빌드
- [ ] 로컬 UPF 환경에서 통합 테스트

**장기:**
- [ ] 실제 5G 네트워크 환경에서 파일럿 배포
- [ ] 프로덕션 환경 모니터링 및 최적화
- [ ] Auto-scaling 연동

---

**문서 작성**: 2026-03-23  
**최종 업데이트**: 2026-03-23 — 벤치마크 실측 결과 반영
