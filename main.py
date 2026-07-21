"""
preprocessing.py 모듈을 불러와 Yellow Taxi 데이터의 결측치를 처리하는 엔트리 포인트.

작성자: 박소영
변경 내역
- 2026-07-21 preprocessing 모듈을 불러와 실행하는 main.py 작성
- 2026-07-21 preprocessing의 run_pandas/run_polars를 사용하도록 수정
"""

from pathlib import Path

from preprocessing import run_pandas, run_polars


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "yellow_tripdata_2026-05.parquet"

    run_pandas(data_path)
    run_polars(data_path)
