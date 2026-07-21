"""model.py의 데이터 준비와 학습 Pipeline을 검증하는 테스트."""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (
    FEATURE_COLUMNS,
    build_pipeline,
    prepare_model_data,
    train_and_evaluate,
)


def make_sample_data(rows: int = 120) -> pd.DataFrame:
    """테스트에서 사용할 작은 Taxi 샘플 데이터를 만든다."""
    return pd.DataFrame(
        {
            "trip_distance": [float(index % 15) for index in range(rows)],
            "fare_amount": [10.0 + index % 20 for index in range(rows)],
            "tip_amount": [float(index % 5) for index in range(rows)],
            "RatecodeID": [1 + index % 2 for index in range(rows)],
            "PULocationID": [1 + index % 30 for index in range(rows)],
            "DOLocationID": [1 + (index + 5) % 30 for index in range(rows)],
            "passenger_count": [1 + index % 3 for index in range(rows)],
        }
    )


def test_prepare_model_data_uses_expected_columns() -> None:
    """지정한 6개 피처와 passenger_count 라벨을 분리하는지 확인한다."""
    sample = make_sample_data()

    X, y = prepare_model_data(sample)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(sample)
    assert y.name == "passenger_count"


def test_prepare_model_data_rejects_missing_feature() -> None:
    """필수 피처가 없으면 알아보기 쉬운 오류를 발생시키는지 확인한다."""
    sample = make_sample_data().drop(columns="tip_amount")

    with pytest.raises(ValueError, match="tip_amount"):
        prepare_model_data(sample)


def test_build_pipeline_configuration() -> None:
    """결측치 처리와 Random Forest가 하나의 Pipeline인지 확인한다."""
    pipeline = build_pipeline(random_state=42)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["imputer", "classifier"]
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.random_state == 42
    assert classifier.class_weight == "balanced_subsample"


def test_train_evaluate_and_save_model(tmp_path) -> None:
    """학습·평가 후 joblib 모델을 저장하고 다시 불러올 수 있는지 확인한다."""
    sample = make_sample_data()
    model_path = tmp_path / "test_passenger_count_model.joblib"

    pipeline, metrics = train_and_evaluate(sample, model_path=model_path)

    assert model_path.is_file()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_weighted"] <= 1.0

    loaded_pipeline = joblib.load(model_path)
    X, _ = prepare_model_data(sample)
    predictions = loaded_pipeline.predict(X.head(5))
    assert len(predictions) == 5
