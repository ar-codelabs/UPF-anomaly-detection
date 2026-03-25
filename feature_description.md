# Feature Engineering 설명

> XGBoost CPU 부하 예측 모델에 사용된 Feature 목록 및 설명

---

## 1. 원본 데이터 컬럼

학습/테스트 데이터셋(`ai_training_dataset.csv`, `ai_test_dataset.csv`)에 실제로 존재하는 컬럼들이다.
1분 단위 스냅샷 방식으로 수집된다.

| 컬럼 | 타입 | 범위 | 설명 |
|------|------|------|------|
| `timestamp` | datetime | - | 측정 시각 (1분 단위) |
| `average_cpu_load` | float | 0 ~ 100% | 1분 평균 CPU 부하 (**예측 대상**) |
| `peak_cpu_load` | float | 0 ~ 100% | 1분 최대 CPU 부하 (평균보다 1~3% 높음) |
| `active_session_count` | int | 1,000 ~ 250,000 | 해당 시점에 연결된 활성 세션 수 |
| `pattern_id` | str | - | 트래픽 패턴 식별자 |

### 데이터 수집 구조

```
하루 데이터 포인트: 1분 × 60분 × 24시간 = 1,440 rows/일
```

- `active_session_count`는 수집 횟수가 아니라 **그 시점에 연결된 세션의 총 개수**
- 수집 간격을 5분으로 늘리면 세션 수 자체는 비슷하지만 데이터 포인트가 1/5로 줄어들어 단기 변화를 못 잡게 됨

```
t=04:00  →  active_session_count =   5,000  (새벽 최저)
t=09:00  →  active_session_count = 150,000  (오전 피크)
t=19:00  →  active_session_count = 230,000  (저녁 피크)
```

---

## 2. 기본 Feature (12개)

원본 컬럼에서 파생된 Feature. 모델 학습에 실제로 사용되는 입력값이다.

### 시간 Feature

| Feature | 설명 | 범위 |
|---------|------|------|
| `hour` | 현재 시각 | 0 ~ 23 |
| `dayofweek` | 요일 | 0(월) ~ 6(일) |
| `month` | 월 | 1 ~ 12 |
| `day` | 일 | 1 ~ 31 |

### 원본 측정값 (그대로 사용)

| Feature | 원본 컬럼 | 설명 |
|---------|----------|------|
| `peakLoad` | `peak_cpu_load` | 해당 분의 최대 CPU 부하 (%) |
| `sessionCnt` | `active_session_count` | 현재 활성 세션 수 |

### 이동평균 (Moving Average)

직전 N분간의 **평균** CPU 부하. 노이즈를 줄이고 단기 추세를 반영한다.

> Lag와의 차이: ma5는 5개 값의 평균, lag5는 5분 전 단일 값

| Feature | 윈도우 | 설명 |
|---------|--------|------|
| `avg_load_ma5` | 5분 | 직전 5분 평균 CPU 부하 |
| `avg_load_ma30` | 30분 | 직전 30분 평균 CPU 부하 |
| `avg_load_ma60` | 60분 | 직전 60분 평균 CPU 부하 |

### 변동성

| Feature | 설명 |
|---------|------|
| `avg_load_std5` | 직전 5분 CPU 부하의 표준편차. 값이 클수록 부하가 불안정함 |

### 순환 인코딩 (Cyclical Encoding)

시각과 요일은 주기성을 가진다. 단순 숫자로 표현하면 23시와 0시가 멀어 보이는 문제가 생기므로 sin/cos로 변환해 원형 연속성을 유지한다.

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

sin 하나만 쓰면 6시와 18시가 같은 값이 되므로, sin + cos 두 개를 함께 써서 24시간 중 어느 시각이든 유일한 좌표를 만든다.

```
23시 → sin=-0.26, cos=0.97
 0시 → sin= 0.00, cos=1.00   ← 수치적으로 가까움
 1시 → sin= 0.26, cos=0.97
```

| Feature | 설명 |
|---------|------|
| `hour_sin` | 시각의 sin 인코딩 (24시간 주기) |
| `hour_cos` | 시각의 cos 인코딩 (24시간 주기) |
| `day_sin` | 요일의 sin 인코딩 (7일 주기) |
| `day_cos` | 요일의 cos 인코딩 (7일 주기) |

---

## 3. 고도화 Feature (기본 Feature 포함 43개)

기본 Feature에 아래 항목들이 추가된다.

### 이동평균 추가

| Feature | 윈도우 | 설명 |
|---------|--------|------|
| `avg_load_ma10` | 10분 | 직전 10분 평균 CPU 부하 |

### 변동성 추가

| Feature | 윈도우 | 설명 |
|---------|--------|------|
| `avg_load_std10` | 10분 | 직전 10분 CPU 부하 표준편차 |
| `avg_load_std30` | 30분 | 직전 30분 CPU 부하 표준편차 |

### Lag Feature (과거 시점 단일 값)

이동평균(ma)이 N분간의 평균을 보는 것과 달리, Lag는 N분 전의 단일 값을 그대로 가져온다.

```
현재 t=10분일 때:
avg_load_ma5  = (t6+t7+t8+t9+t10) / 5  → 5개 평균
avg_load_lag5 = t5                       → 5분 전 값 하나
```

모델이 추세(ma)와 과거 패턴(lag) 두 가지를 동시에 학습할 수 있다.

| Feature | 시점 | 설명 |
|---------|------|------|
| `avg_load_lag1` | 1분 전 | 1분 전 평균 CPU 부하 |
| `avg_load_lag5` | 5분 전 | 5분 전 평균 CPU 부하 |
| `avg_load_lag10` | 10분 전 | 10분 전 평균 CPU 부하 |
| `avg_load_lag30` | 30분 전 | 30분 전 평균 CPU 부하 |
| `avg_load_lag60` | 60분 전 | 60분 전 평균 CPU 부하 |
| `peak_load_lag1` | 1분 전 | 1분 전 피크 CPU 부하 |
| `peak_load_lag5` | 5분 전 | 5분 전 피크 CPU 부하 |
| `peak_load_lag10` | 10분 전 | 10분 전 피크 CPU 부하 |
| `peak_load_lag30` | 30분 전 | 30분 전 피크 CPU 부하 |
| `peak_load_lag60` | 60분 전 | 60분 전 피크 CPU 부하 |
| `session_lag1` | 1분 전 | 1분 전 세션 수 |
| `session_lag5` | 5분 전 | 5분 전 세션 수 |
| `session_lag10` | 10분 전 | 10분 전 세션 수 |
| `session_lag30` | 30분 전 | 30분 전 세션 수 |
| `session_lag60` | 60분 전 | 60분 전 세션 수 |

### 변화량 (Rate of Change)

현재 값과 N분 전 값의 차이. 부하가 올라가는 중인지 내려가는 중인지 방향을 나타낸다.

| Feature | 설명 |
|---------|------|
| `avg_load_change_1` | 1분 전 대비 CPU 부하 변화량 |
| `avg_load_change_5` | 5분 전 대비 CPU 부하 변화량 |
| `avg_load_change_10` | 10분 전 대비 CPU 부하 변화량 |

### 퍼센트 변화율 (Percentage Change)

변화량을 비율로 표현. 절대값이 아닌 상대적 변화를 반영한다.

| Feature | 설명 |
|---------|------|
| `avg_load_pct_change_1` | 1분 전 대비 CPU 부하 % 변화율 |
| `avg_load_pct_change_5` | 5분 전 대비 CPU 부하 % 변화율 |

### 윈도우 최대/최소

| Feature | 설명 |
|---------|------|
| `avg_load_max5` | 직전 5분 내 최대 CPU 부하 |
| `avg_load_min5` | 직전 5분 내 최소 CPU 부하 |
| `avg_load_max30` | 직전 30분 내 최대 CPU 부하 |
| `avg_load_min30` | 직전 30분 내 최소 CPU 부하 |

### 범위 (Range)

윈도우 내 최대 - 최소. 단기 변동폭을 나타낸다.

| Feature | 설명 |
|---------|------|
| `avg_load_range5` | 직전 5분 CPU 부하 범위 (max - min) |
| `avg_load_range30` | 직전 30분 CPU 부하 범위 (max - min) |

### 상호작용 Feature (Interaction Feature)

두 값의 관계를 비율로 표현한다. 모델이 스스로 나눗셈 관계를 학습하기 어렵기 때문에 미리 만들어준다.

| Feature | 계산식 | 설명 |
|---------|--------|------|
| `load_session_ratio` | `averageLoad / (sessionCnt + 1)` | 세션 1개당 CPU 부하. 평소보다 급등하면 비정상 트래픽 가능성 |
| `peak_avg_ratio` | `peakLoad / (averageLoad + 1)` | 평균 대비 피크 비율. 1에 가까울수록 안정적, 클수록 순간 스파이크 심함 |

---

## 4. Feature Engineering 성능 비교

| 지표 | 기본 Feature | 고도화 Feature | 개선도 |
|------|-------------|---------------|--------|
| MAE | 0.2833 | 0.1593 | **+43.76%** |
| RMSE | 0.3831 | 0.2514 | **+34.39%** |
| MAPE | 0.60% | 0.34% | **+43.07%** |

---

**작성일**: 2026-03-25
