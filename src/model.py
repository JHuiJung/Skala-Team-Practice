"""NYC Yellow Taxi 승객 수(passenger_count) 분류 모델 학습 모듈.
코드 작성자: 임창우
전처리된 pandas/Polars DataFrame을 입력받아 scikit-learn Pipeline으로
학습하고 정확도와 weighted F1 점수를 출력한 뒤 joblib 파일로 저장한다.

실행 예:
    python model.py data/yellow_tripdata_2026-05.parquet
    python model.py data/yellow_tripdata_2026-05.parquet --max-samples 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import run_pandas


FEATURE_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
]
TARGET_COLUMN = "passenger_count"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "yellow_tripdata_2026-05.parquet"
DEFAULT_MODEL_PATH = PROJECT_DIR / "passenger_count_model.joblib"


def prepare_model_data(trips: Any) -> tuple[pd.DataFrame, pd.Series]:
    """전처리된 DataFrame에서 모델 입력(X)과 라벨(y)을 만든다."""
    if not isinstance(trips, pd.DataFrame):
        if hasattr(trips, "to_pandas"):
            trips = trips.to_pandas()
        else:
            raise TypeError("trips는 pandas 또는 Polars DataFrame이어야 합니다.")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in trips.columns]
    if missing_columns:
        raise ValueError(f"모델 학습에 필요한 컬럼이 없습니다: {missing_columns}")

    selected = trips[required_columns].copy().replace([np.inf, -np.inf], np.nan)
    selected = selected.dropna(subset=[TARGET_COLUMN])
    if selected.empty:
        raise ValueError("학습 가능한 행이 없습니다.")

    # passenger_count는 사람 수이므로 정수형 클래스 라벨로 사용한다.
    target_numeric = selected[TARGET_COLUMN]
    target_numeric = pd.to_numeric(target_numeric, errors="coerce")
    valid_target = target_numeric.notna() & (target_numeric >= 0)
    selected = selected.loc[valid_target]
    target_numeric = target_numeric.loc[valid_target]
    if selected.empty:
        raise ValueError("유효한 passenger_count 라벨이 없습니다.")

    X = selected[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = target_numeric.round().astype(int)
    return X, y


def build_pipeline(random_state: int = 42) -> Pipeline:
    """결측치 처리와 Random Forest 분류기를 묶은 Pipeline을 반환한다."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_evaluate(
    trips: Any,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: int | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """모델을 학습·평가하고 전체 Pipeline을 joblib로 저장한다."""
    X, y = prepare_model_data(trips)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples는 1 이상의 정수여야 합니다.")
        if len(X) > max_samples:
            sampled = X.sample(n=max_samples, random_state=random_state).index
            X, y = X.loc[sampled], y.loc[sampled]

    if y.nunique() < 2:
        raise ValueError("분류 학습에는 passenger_count 클래스가 2개 이상 필요합니다.")

    # 모든 클래스가 최소 2건일 때만 층화 분할을 사용한다.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }

    print(f"학습 데이터: {len(X_train):,}건 / 테스트 데이터: {len(X_test):,}건")
    print(f"정확도(Accuracy): {metrics['accuracy']:.4f}")
    print(f"F1 점수(Weighted): {metrics['f1_weighted']:.4f}")
    print("\n분류 리포트")
    print(classification_report(y_test, predictions, zero_division=0))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"모델 저장 완료: {model_path.resolve()}")
    return pipeline, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="승객 수 분류 모델 학습")
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA_PATH,
        help=f"Yellow Taxi parquet 파일 경로 (기본값: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="학습된 Pipeline 저장 경로",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="학습에 사용할 최대 행 수(미지정 시 전체 데이터)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned_trips = run_pandas(args.data_path)
    train_and_evaluate(
        cleaned_trips,
        model_path=args.model_path,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
