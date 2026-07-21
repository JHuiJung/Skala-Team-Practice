"""
preprocessing.py 모듈을 불러와 Yellow Taxi 데이터의 결측치를 처리하고,
src/statistics.py로 기술통계·상관계수·t-test 분석까지 수행하는 엔트리 포인트.

작성자: 박소영
변경 내역
- 2026-07-21 preprocessing 모듈을 불러와 실행하는 main.py 작성
- 2026-07-21 preprocessing의 run_pandas/run_polars를 사용하도록 수정
- 2026-07-21 src.statistics의 통계 분석(run_statistical_analysis)을 실행하도록 추가
"""

from pathlib import Path

from preprocessing import run_pandas, run_polars
from src.statistics import run_statistical_analysis


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "yellow_tripdata_2026-05.parquet"

    trips_pd_cleaned = run_pandas(data_path)
    trips_pl_cleaned = run_polars(data_path)

    stats_result = run_statistical_analysis(trips_pd_cleaned, print_output=True)

    return trips_pd_cleaned, trips_pl_cleaned, stats_result


if __name__ == "__main__":
    main()
