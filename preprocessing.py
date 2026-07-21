"""
NYC Yellow Taxi 운행 데이터(yellow_tripdata_*.parquet)를 읽어와
EDA(기본 정보/결측치 확인)와 결측치 처리를 수행하는 재사용 모듈.
pandas 버전과 Polars 버전을 각각 제공한다.

사용 예:
    from preprocessing import run_pandas, run_polars

    trips_cleaned = run_pandas("data/yellow_tripdata_2026-05.parquet")
    trips_cleaned_pl = run_polars("data/yellow_tripdata_2026-05.parquet")

작성자: 박소영
변경 내역
- 2026-07-21 parquet 데이터 로드 및 pandas/Polars 결측치 처리 함수 작성
- 2026-07-21 practice5.py를 다른 스크립트에서 재사용할 수 있도록 preprocessing.py 모듈로 분리, EDA 함수 추가
"""

from pathlib import Path

import pandas as pd
import polars as pl

# 결측치 처리 방식 (컬럼 성격에 맞춰 채움 값을 다르게 지정)
NUMERIC_FILL_VALUE = 0  # RatecodeID, congestion_surcharge, Airport_fee 등
CATEGORICAL_FILL_VALUE = "N"  # store_and_fwd_flag
PASSENGER_COUNT_FILL_STRATEGY = "median"  # passenger_count는 중앙값으로 대체


# =========================================================
# pandas
# =========================================================


def load_yellow_tripdata_pandas(parquet_path: str | Path) -> pd.DataFrame:
    """pandas로 Yellow Taxi parquet 파일을 읽어 DataFrame으로 반환한다."""
    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet 파일을 찾을 수 없습니다: {parquet_path}")
    return pd.read_parquet(parquet_path)


def eda_pandas(trips: pd.DataFrame, print_output: bool = False) -> pd.Series:
    """pandas 기본 EDA(info/결측치 개수)를 수행하고 컬럼별 결측치 개수를 반환한다."""
    missing_counts = trips.isna().sum()
    if print_output:
        trips.info()
        print(missing_counts)
    return missing_counts


def handle_missing_values_pandas(trips: pd.DataFrame) -> pd.DataFrame:
    """pandas 기반으로 컬럼 성격에 맞춰 결측치를 처리한다."""
    trips = trips.copy()

    passenger_count_median = trips["passenger_count"].median()
    trips["passenger_count"] = trips["passenger_count"].fillna(passenger_count_median)

    trips["RatecodeID"] = trips["RatecodeID"].fillna(NUMERIC_FILL_VALUE)
    trips["congestion_surcharge"] = trips["congestion_surcharge"].fillna(
        NUMERIC_FILL_VALUE
    )
    trips["Airport_fee"] = trips["Airport_fee"].fillna(NUMERIC_FILL_VALUE)
    trips["store_and_fwd_flag"] = trips["store_and_fwd_flag"].fillna(
        CATEGORICAL_FILL_VALUE
    )

    return trips


def run_pandas(parquet_path: str | Path) -> pd.DataFrame:
    """parquet 파일명을 받아 pandas로 로드하고 결측치를 처리한 DataFrame을 반환한다."""
    trips = load_yellow_tripdata_pandas(parquet_path)
    return handle_missing_values_pandas(trips)


# =========================================================
# Polars
# =========================================================


def load_yellow_tripdata_polars(parquet_path: str | Path) -> pl.DataFrame:
    """Polars로 Yellow Taxi parquet 파일을 읽어 DataFrame으로 반환한다."""
    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet 파일을 찾을 수 없습니다: {parquet_path}")
    return pl.read_parquet(parquet_path)


def eda_polars(trips: pl.DataFrame, print_output: bool = False) -> pl.DataFrame:
    """Polars 기본 EDA(schema/결측치 개수)를 수행하고 컬럼별 결측치 개수를 반환한다."""
    null_counts = trips.null_count()
    if print_output:
        print(trips.schema)
        print(null_counts)
    return null_counts


def handle_missing_values_polars(trips: pl.DataFrame) -> pl.DataFrame:
    """Polars 기반으로 컬럼 성격에 맞춰 결측치를 처리한다."""
    passenger_count_median = trips["passenger_count"].median()

    return trips.with_columns(
        pl.col("passenger_count").fill_null(passenger_count_median),
        pl.col("RatecodeID").fill_null(NUMERIC_FILL_VALUE),
        pl.col("congestion_surcharge").fill_null(NUMERIC_FILL_VALUE),
        pl.col("Airport_fee").fill_null(NUMERIC_FILL_VALUE),
        pl.col("store_and_fwd_flag").fill_null(CATEGORICAL_FILL_VALUE),
    )


def run_polars(parquet_path: str | Path) -> pl.DataFrame:
    """parquet 파일명을 받아 Polars로 로드하고 결측치를 처리한 DataFrame을 반환한다."""
    trips = load_yellow_tripdata_polars(parquet_path)
    return handle_missing_values_polars(trips)
