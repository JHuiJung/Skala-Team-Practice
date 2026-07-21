"""
preprocessing.py 모듈을 불러와 Yellow Taxi 데이터의 결측치/이상치 처리와
타입 캐스팅을 수행하는 엔트리 포인트.

작성자: 박소영
변경 내역
- 2026-07-21 preprocessing 모듈을 불러와 실행하는 main.py 작성
- 2026-07-21 preprocessing의 run_pandas/run_polars를 사용하도록 수정
- 2026-07-21 run_pandas 결과 컬럼 확인용 print 추가 (결측치+이상치+캐스팅 반영)
"""

from pathlib import Path

from preprocessing import run_pandas, run_polars


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "yellow_tripdata_2026-05.parquet"

    df_pandas = run_pandas(data_path)
    df_polars = run_polars(data_path)
    
    print("Pandas DataFrame")
    print(df_pandas)
    print("Polars DataFrame")
    print(df_polars)

