# AWS SageMaker 배포 가이드

## 1. 개요

이 가이드는 UPF CPU 트래픽 부하 예측 모델을 AWS SageMaker에 배포하는 방법을 설명합니다.

---

## 2. 사전 준비

### 2.1 AWS 계정 설정
```bash
# AWS CLI 설정
aws configure
# AWS_ACCESS_KEY_ID: [your-access-key]
# AWS_SECRET_ACCESS_KEY: [your-secret-key]
# Default region: us-east-1 (또는 원하는 리전)
```

### 2.2 필요한 IAM 권한
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "s3:*",
        "ecr:*",
        "logs:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### 2.3 S3 버킷 생성
```bash
aws s3 mb s3://upf-traffic-prediction-bucket --region us-east-1
```

---

## 3. 모델 준비

### 3.1 모델 저장
```python
import joblib
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

# 모델 저장
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(gb_model, 'gb_model.pkl')
```

### 3.2 추론 스크립트 작성
```python
# inference.py
import json
import joblib
import numpy as np

# 모델 로드
xgb_model = joblib.load('xgboost_model.pkl')
gb_model = joblib.load('gb_model.pkl')

def model_fn(model_dir):
    """모델 로드"""
    xgb_model = joblib.load(f'{model_dir}/xgboost_model.pkl')
    gb_model = joblib.load(f'{model_dir}/gb_model.pkl')
    return {'xgb': xgb_model, 'gb': gb_model}

def input_fn(request_body, request_content_type):
    """입력 처리"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['features']).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """예측"""
    xgb_pred = model['xgb'].predict(input_data)
    gb_pred = model['gb'].predict(input_data)
    ensemble_pred = 0.5 * xgb_pred + 0.5 * gb_pred
    return ensemble_pred

def output_fn(prediction, accept):
    """출력 처리"""
    if accept == 'application/json':
        return json.dumps({
            'predicted_load': float(prediction[0]),
            'cpu_cores': int(cpu_core_control(prediction[0]))
        }), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

def cpu_core_control(predicted_load):
    """CPU 코어 제어"""
    if predicted_load < 30:
        return 2
    elif predicted_load < 80:
        return 4
    else:
        return 8
```

### 3.3 Docker 이미지 생성 (선택사항)
```dockerfile
# Dockerfile
FROM python:3.8

RUN pip install xgboost scikit-learn joblib

COPY inference.py /opt/ml/code/inference.py
COPY xgboost_model.pkl /opt/ml/model/
COPY gb_model.pkl /opt/ml/model/

ENV SAGEMAKER_PROGRAM inference.py

ENTRYPOINT ["python", "-m", "sagemaker_containers.beta.framework_entry_point"]
```

---

## 4. SageMaker 배포

### 4.1 Python SDK를 이용한 배포
```python
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model import Model

# SageMaker 세션 생성
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

# 모델 업로드
model_data = session.upload_data(
    path='model.tar.gz',
    bucket=bucket,
    key_prefix='upf-traffic-prediction'
)

# SKLearn 모델 생성
sklearn_model = SKLearnModel(
    entry_point='inference.py',
    role=role,
    framework_version='0.23-1',
    py_version='py3',
    model_data=model_data,
    sagemaker_session=session
)

# 엔드포인트 배포
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='upf-traffic-prediction-endpoint'
)
```

### 4.2 AWS CLI를 이용한 배포
```bash
# 모델 생성
aws sagemaker create-model \
    --model-name upf-traffic-prediction-model \
    --primary-container Image=382416733822.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn:0.23-1-cpu-py3,ModelDataUrl=s3://upf-traffic-prediction-bucket/model.tar.gz

# 엔드포인트 설정 생성
aws sagemaker create-endpoint-config \
    --endpoint-config-name upf-traffic-prediction-config \
    --production-variants VariantName=Primary,ModelName=upf-traffic-prediction-model,InitialInstanceCount=1,InstanceType=ml.m5.large

# 엔드포인트 생성
aws sagemaker create-endpoint \
    --endpoint-name upf-traffic-prediction-endpoint \
    --endpoint-config-name upf-traffic-prediction-config
```

---

## 5. 실시간 예측

### 5.1 Python을 이용한 예측
```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

# 입력 데이터 준비
features = [
    9,      # hour
    2,      # dayofweek
    85,     # peakLoad
    1050,   # sessionCnt
    88.5,   # avg_load_ma5
    87.2,   # avg_load_ma30
    86.8,   # avg_load_ma60
    2.1,    # avg_load_std5
    0.95,   # hour_sin
    0.31,   # hour_cos
    -0.78,  # day_sin
    0.62,   # day_cos
    89,     # avg_load_lag1
    87,     # avg_load_lag5
    85,     # avg_load_lag10
    84,     # avg_load_lag30
    82      # avg_load_lag60
]

# 예측 요청
response = client.invoke_endpoint(
    EndpointName='upf-traffic-prediction-endpoint',
    ContentType='application/json',
    Body=json.dumps({'features': features})
)

# 결과 파싱
result = json.loads(response['Body'].read().decode())
print(f"예측 부하: {result['predicted_load']:.2f}")
print(f"권장 CPU 코어: {result['cpu_cores']}")
```

### 5.2 배치 예측
```python
from sagemaker.processing import ScriptProcessor

# 배치 변환 작업
transformer = sklearn_model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://upf-traffic-prediction-bucket/batch-predictions'
)

transformer.transform(
    data='s3://upf-traffic-prediction-bucket/test_data.csv',
    content_type='text/csv',
    split_type='Line'
)
```

---

## 6. 모니터링 및 로깅

### 6.1 CloudWatch 로그 확인
```bash
# 엔드포인트 로그 확인
aws logs tail /aws/sagemaker/Endpoints/upf-traffic-prediction-endpoint --follow
```

### 6.2 모델 성능 모니터링
```python
import cloudwatch

cloudwatch_client = boto3.client('cloudwatch')

# 예측 오차 메트릭 발행
cloudwatch_client.put_metric_data(
    Namespace='UPF-TrafficPrediction',
    MetricData=[
        {
            'MetricName': 'PredictionError',
            'Value': abs(actual - predicted),
            'Unit': 'None'
        }
    ]
)
```

### 6.3 모델 모니터링 (Model Monitor)
```python
from sagemaker.model_monitor import DataCaptureConfig

# 데이터 캡처 설정
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://upf-traffic-prediction-bucket/data-capture'
)

# 엔드포인트 생성 시 적용
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    data_capture_config=data_capture_config
)
```

---

## 7. 자동 재학습

### 7.1 Lambda 함수로 월간 재학습
```python
# lambda_function.py
import boto3
import sagemaker
from sagemaker.estimator import Estimator

def lambda_handler(event, context):
    """월간 모델 재학습"""
    
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # 학습 작업 생성
    estimator = Estimator(
        image_uri='382416733822.dkr.ecr.us-east-1.amazonaws.com/sagemaker-sklearn:0.23-1-cpu-py3',
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path='s3://upf-traffic-prediction-bucket/models'
    )
    
    # 학습 시작
    estimator.fit('s3://upf-traffic-prediction-bucket/training-data')
    
    # 모델 등록
    model_uri = estimator.model_data
    
    return {
        'statusCode': 200,
        'body': f'Training completed. Model: {model_uri}'
    }
```

### 7.2 EventBridge로 월간 스케줄 설정
```bash
# EventBridge 규칙 생성
aws events put-rule \
    --name upf-monthly-retraining \
    --schedule-expression "cron(0 0 1 * ? *)"  # 매월 1일 00:00

# Lambda 함수 연결
aws events put-targets \
    --rule upf-monthly-retraining \
    --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:123456789012:function:upf-retraining"
```

---

## 8. 비용 최적화

### 8.1 자동 스케일링
```python
autoscaling = boto3.client('application-autoscaling')

# 자동 스케일링 설정
autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/upf-traffic-prediction-endpoint/variant/Primary',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=4
)

# 스케일링 정책
autoscaling.put_scaling_policy(
    PolicyName='upf-scaling-policy',
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/upf-traffic-prediction-endpoint/variant/Primary',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

### 8.2 스팟 인스턴스 사용
```python
# 학습 시 스팟 인스턴스 사용
estimator = Estimator(
    ...
    use_spot_instances=True,
    max_run=3600,
    max_wait=5400
)
```

---

## 9. 보안

### 9.1 VPC 설정
```python
estimator = Estimator(
    ...
    subnets=['subnet-12345678'],
    security_group_ids=['sg-12345678'],
    encrypt_inter_container_traffic=True
)
```

### 9.2 모델 암호화
```python
estimator = Estimator(
    ...
    output_kms_key='arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012'
)
```

---

## 10. 문제 해결

### 10.1 엔드포인트 생성 실패
```bash
# 로그 확인
aws logs tail /aws/sagemaker/Endpoints/upf-traffic-prediction-endpoint --follow

# 엔드포인트 상태 확인
aws sagemaker describe-endpoint --endpoint-name upf-traffic-prediction-endpoint
```

### 10.2 예측 오류
```python
# 입력 데이터 검증
import numpy as np

features = np.array([...])
print(f"Shape: {features.shape}")
print(f"Data type: {features.dtype}")
print(f"Min/Max: {features.min()}/{features.max()}")
```

### 10.3 성능 저하
```bash
# 엔드포인트 메트릭 확인
aws cloudwatch get-metric-statistics \
    --namespace AWS/SageMaker \
    --metric-name ModelLatency \
    --dimensions Name=EndpointName,Value=upf-traffic-prediction-endpoint \
    --start-time 2026-04-01T00:00:00Z \
    --end-time 2026-04-08T00:00:00Z \
    --period 3600 \
    --statistics Average,Maximum
```

---

## 11. 체크리스트

- [ ] AWS 계정 설정 완료
- [ ] IAM 권한 설정 완료
- [ ] S3 버킷 생성 완료
- [ ] 모델 저장 완료
- [ ] 추론 스크립트 작성 완료
- [ ] 엔드포인트 배포 완료
- [ ] 실시간 예측 테스트 완료
- [ ] 모니터링 설정 완료
- [ ] 자동 재학습 설정 완료
- [ ] 보안 설정 완료
- [ ] 비용 최적화 완료
- [ ] 문서화 완료

---

## 12. 참고 자료

- [AWS SageMaker 공식 문서](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker 가격](https://aws.amazon.com/sagemaker/pricing/)
