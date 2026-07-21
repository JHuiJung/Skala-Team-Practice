"""
preprocessing.py 모듈을 불러와 Yellow Taxi 데이터의 결측치/이상치 처리와
타입 캐스팅을 수행하는 엔트리 포인트.

작성자: 박소영
변경 내역
- 2026-07-21 preprocessing 모듈을 불러와 실행하는 main.py 작성
- 2026-07-21 preprocessing의 run_pandas/run_polars를 사용하도록 수정
- 2026-07-21 run_pandas 결과 컬럼 확인용 print 추가 (결측치+이상치+캐스팅 반영)
- 2026-07-21 전처리된 pandas DataFrame으로 모델 학습 및 평가 실행
- 2026-07-21 Seaborn/Plotly 시각화 결과를 outputs 폴더에 저장
"""

from pathlib import Path

from src.model import train_and_evaluate
from src.preprocessing import run_pandas, run_polars
from src.report import generate_report
from src.statistical_analysis import run_statistical_analysis
from src.visualization import (
    plot_correlation_heatmap_plotly,
    plot_correlation_heatmap_seaborn,
    plot_distance_distribution_plotly,
    plot_distance_distribution_seaborn,
    plot_fare_by_passenger_count_plotly,
    plot_fare_by_passenger_count_seaborn,
)


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "yellow_tripdata_2026-05.parquet"

    df_pandas = run_pandas(data_path)
    df_polars = run_polars(data_path)

    stats_result = run_statistical_analysis(df_pandas, print_output=True)

    # 전처리된 pandas 데이터로 정적 차트와 인터랙티브 차트를 생성한다.
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    print("=== 시각화 생성 중 ===")
    plot_distance_distribution_seaborn(
        df_pandas,
        save_path=output_dir / "distribution_seaborn.png",
    )
    plot_distance_distribution_plotly(
        df_pandas,
        save_path=output_dir / "distribution_plotly.html",
    )
    plot_correlation_heatmap_seaborn(
        df_pandas,
        save_path=output_dir / "correlation_seaborn.png",
    )
    plot_correlation_heatmap_plotly(
        df_pandas,
        save_path=output_dir / "correlation_plotly.html",
    )
    plot_fare_by_passenger_count_seaborn(
        df_pandas,
        save_path=output_dir / "group_comparison_seaborn.png",
    )
    plot_fare_by_passenger_count_plotly(
        df_pandas,
        save_path=output_dir / "group_comparison_plotly.html",
    )
    print(f"시각화 저장 완료: {output_dir}")

    # preprocessing에서 만든 DataFrame을 다시 읽지 않고 모델에 바로 전달한다.
    model_path = project_dir / "passenger_count_model.joblib"
    model, model_metrics = train_and_evaluate(df_pandas, model_path=model_path)

    print("Pandas DataFrame")
    print(df_pandas)
    print("Polars DataFrame")
    print(df_polars)

    # report.md에서는 outputs/ 기준 상대경로로 링크해야 마크다운 뷰어에서 바로 열린다.
    visualization_paths = {
        "trip_distance 분포 (Seaborn)": "outputs/distribution_seaborn.png",
        "trip_distance 분포 (Plotly)": "outputs/distribution_plotly.html",
        "피처 간 상관관계 (Seaborn)": "outputs/correlation_seaborn.png",
        "피처 간 상관관계 (Plotly)": "outputs/correlation_plotly.html",
        "승객 수별 fare_amount 비교 (Seaborn)": "outputs/group_comparison_seaborn.png",
        "승객 수별 fare_amount 비교 (Plotly)": "outputs/group_comparison_plotly.html",
    }
    report_path = generate_report(
        df_pandas,
        df_polars,
        stats_result,
        model_metrics,
        model_path,
        visualization_paths,
        output_path=project_dir / "report.md",
    )
    print(f"리포트 저장 완료: {report_path}")
