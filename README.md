# NYC Yellow Taxi 데이터 분석 프로젝트

NYC Yellow Taxi 운행 데이터를 이용해 전처리, 탐색적·통계적 분석, 시각화,
승객 수 예측 모델 학습을 하나의 흐름으로 수행하는 팀 프로젝트입니다.

## 팀원

- 정희중: 통계 분석
- 임창우: 머신러닝 모델
- 유재권: 데이터 시각화
- 박소영: 데이터 전처리

## 주요 기능

### 데이터 전처리

- 같은 Parquet 데이터를 pandas와 Polars로 로딩
- `passenger_count`, `RatecodeID` 등 주요 컬럼의 결측치 처리
- 거리·요금·팁·승객 수의 음수 데이터 제거
- `RatecodeID`, `PULocationID`, `DOLocationID`를 정수형으로 변환

### 통계 분석

- 주요 변수의 평균, 표준편차, 분위수 등 기술통계 출력
- 피처와 라벨 사이의 피어슨 상관계수 계산
- 단독 승객과 동승 승객의 팁 금액 차이에 대한 Welch t-test
- 승객 그룹과 팁 등급의 카이제곱검정 및 Cramér's V 계산

### 시각화

- Seaborn 정적 차트
  - 운행 거리 분포
  - 상관계수 히트맵
  - 승객 수별 요금 비교
- Plotly 인터랙티브 차트
  - 운행 거리 분포
  - 상관계수 히트맵
  - 승객 수별 요금 비교

### 머신러닝 모델

`passenger_count`를 예측하는 다중 클래스 분류 문제로 구성했습니다.

사용 피처:

```text
trip_distance
fare_amount
tip_amount
RatecodeID
PULocationID
DOLocationID
```

학습 구성:

- `SimpleImputer(strategy="median")`로 피처 결측치 처리
- `RandomForestClassifier` 사용
- `train_test_split(test_size=0.2, random_state=42)` 적용
- 클래스 비율 유지를 위한 층화 분할
- Accuracy, Weighted F1, 분류 리포트 출력
- 전체 Pipeline을 `passenger_count_model.joblib`로 저장

## 프로젝트 구조

```text
Skala-Team-Practice/
├── data/
│   └── yellow_tripdata_2026-05.parquet  # 원본 Taxi 데이터
├── outputs/                              # 생성된 PNG/HTML 차트
├── src/
│   ├── model.py                          # 모델 학습·평가·저장
│   ├── preprocessing.py                  # pandas/Polars 전처리
│   ├── statistical_analysis.py           # 기술통계 및 가설검정
│   └── visualization.py                  # Seaborn/Plotly 시각화
├── tests/
│   ├── conftest.py                       # pytest 공통 설정
│   ├── test_model.py                     # 모델 테스트
│   ├── test_preprocessing.py             # 전처리 테스트
│   └── test_statistics.py                # 통계 분석 테스트
├── main.py                               # 전체 분석 실행 진입점
├── requirements.txt                      # Python 패키지 목록
└── README.md
```

`data/`, `outputs/`, `*.joblib`은 용량이 큰 데이터와 생성 결과물이므로 Git에서
제외됩니다.

## 설치 방법

Python 3.11 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell에서는 다음 명령으로 가상환경을 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

## 데이터 준비

다음 위치에 NYC Yellow Taxi Parquet 파일을 배치합니다.

```text
data/yellow_tripdata_2026-05.parquet
```

## 전체 프로젝트 실행

프로젝트 최상위 폴더에서 실행합니다.

```bash
python main.py
```

실행 순서:

```text
데이터 전처리
→ 통계 분석
→ 시각화 생성
→ 모델 학습 및 평가
→ 모델 저장
```

전체 약 400만 건을 학습하기 때문에 모델 실행에는 시간이 걸릴 수 있습니다.

## 생성 결과

시각화 결과는 `outputs/`에 저장됩니다.

```text
outputs/
├── distribution_seaborn.png
├── distribution_plotly.html
├── correlation_seaborn.png
├── correlation_plotly.html
├── group_comparison_seaborn.png
└── group_comparison_plotly.html
```

학습 모델은 프로젝트 최상위에 저장됩니다.

```text
passenger_count_model.joblib
```

## 테스트

전체 테스트 실행:

```bash
python -m pytest -q
```

특정 영역만 실행할 수도 있습니다.

```bash
python -m pytest tests/test_model.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_statistics.py -v
```
