"""
preprocessing.py의 run_pandas/run_polars가 정상적으로 DataFrame을 반환하는지,
결측치/음수 이상치가 남아있지 않은지, ID 컬럼이 int로 캐스팅됐는지 확인하는 테스트.

작성자: 박소영
변경 내역
- 2026-07-21 run_pandas/run_polars 정상 동작 확인 테스트 작성
- 2026-07-21 결측치/음수/타입 캐스팅 검증 테스트를 pandas/polars 별로 분리
"""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from src.preprocessing import (
    ID_COLUMNS_TO_CAST,
    NON_NEGATIVE_COLUMNS,
    run_pandas,
    run_polars,
)

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "yellow_tripdata_2026-05.parquet"
)


@pytest.fixture(scope="module")
def pandas_result() -> pd.DataFrame:
    # 모듈 내 모든 pandas 테스트가 공유하는 run_pandas 결과
    return run_pandas(DATA_PATH)


@pytest.fixture(scope="module")
def polars_result() -> pl.DataFrame:
    # 모듈 내 모든 Polars 테스트가 공유하는 run_polars 결과
    return run_polars(DATA_PATH)


# =========================================================
# pandas
# =========================================================


def test_run_pandas_returns_dataframe(pandas_result: pd.DataFrame) -> None:
    # run_pandas가 비어있지 않은 DataFrame을 반환하는지 확인
    assert isinstance(pandas_result, pd.DataFrame)
    assert not pandas_result.empty


def test_run_pandas_has_no_missing_values(pandas_result: pd.DataFrame) -> None:
    # 결측치 처리 대상 컬럼에 남아있는 결측치가 없는지 확인
    checked_columns = [
        "passenger_count",
        "RatecodeID",
        "congestion_surcharge",
        "Airport_fee",
        "store_and_fwd_flag",
    ]
    assert pandas_result[checked_columns].isna().sum().sum() == 0


def test_run_pandas_has_no_negative_values(pandas_result: pd.DataFrame) -> None:
    # 음수면 안 되는 컬럼에 음수 값이 남아있지 않은지 확인
    for column in NON_NEGATIVE_COLUMNS:
        assert (pandas_result[column] < 0).sum() == 0


def test_run_pandas_casts_id_columns_to_int(pandas_result: pd.DataFrame) -> None:
    # 카테고리성 ID 컬럼이 int 타입으로 캐스팅됐는지 확인
    for column in ID_COLUMNS_TO_CAST:
        assert pd.api.types.is_integer_dtype(pandas_result[column])


# =========================================================
# Polars
# =========================================================


def test_run_polars_returns_dataframe(polars_result: pl.DataFrame) -> None:
    # run_polars가 비어있지 않은 DataFrame을 반환하는지 확인
    assert isinstance(polars_result, pl.DataFrame)
    assert polars_result.height > 0


def test_run_polars_has_no_missing_values(polars_result: pl.DataFrame) -> None:
    # 결측치 처리 대상 컬럼에 남아있는 결측치가 없는지 확인
    checked_columns = [
        "passenger_count",
        "RatecodeID",
        "congestion_surcharge",
        "Airport_fee",
        "store_and_fwd_flag",
    ]
    null_counts = polars_result.select(checked_columns).null_count()
    assert sum(null_counts.row(0)) == 0


def test_run_polars_has_no_negative_values(polars_result: pl.DataFrame) -> None:
    # 음수면 안 되는 컬럼에 음수 값이 남아있지 않은지 확인
    for column in NON_NEGATIVE_COLUMNS:
        assert polars_result.filter(pl.col(column) < 0).height == 0


def test_run_polars_casts_id_columns_to_int(polars_result: pl.DataFrame) -> None:
    # 카테고리성 ID 컬럼이 int 타입으로 캐스팅됐는지 확인
    for column in ID_COLUMNS_TO_CAST:
        assert polars_result.schema[column].is_integer()
