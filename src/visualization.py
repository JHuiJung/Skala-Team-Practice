"""
전처리된 Yellow Taxi 데이터(preprocessing.run_pandas 결과)를 받아
Seaborn 정적 차트와 Plotly 인터랙티브 차트를 생성하는 재사용 모듈.

분포(distribution), 상관관계(correlation), 그룹 비교(group comparison)
세 가지 유형을 각각 Seaborn/Plotly로 제공한다.

사용 예:
    from preprocessing import run_pandas
    from visualization import (
        plot_distance_distribution_seaborn,
        plot_correlation_heatmap_plotly,
    )

    trips = run_pandas("data/yellow_tripdata_2026-05.parquet")
    plot_distance_distribution_seaborn(trips, save_path="dist_seaborn.png")
    plot_correlation_heatmap_plotly(trips, save_path="corr_plotly.html")

작성자: 유재권
변경 내역
- 2026-07-21 시각화 모듈 작성 (분포/상관관계/그룹비교 x Seaborn/Plotly)
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# 한글 라벨이 깨지지 않도록 OS별 한글 폰트를 찾아 지정 (macOS/Windows/Linux)
_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _KOREAN_FONT_CANDIDATES:
    if _font_name in _available_fonts:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

# 종합실습2.txt 기준 레이블/피처
LABEL_COLUMN = "passenger_count"
FEATURE_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
]
CORRELATION_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

# Plotly는 원본 데이터 포인트를 그대로 HTML에 담기 때문에, 수백만 행을 통째로
# 넘기면 파일이 수십~수백 MB로 커진다. 포인트 단위 차트(분포/그룹비교)는
# 아래 개수만큼 샘플링해서 그린다. (상관계수 히트맵은 집계값만 사용하므로 해당 없음)
DEFAULT_SAMPLE_SIZE = 20_000


def _sample_for_plotly(
    trips: pd.DataFrame, sample_size: int, random_state: int = 42
) -> pd.DataFrame:
    """Plotly 인터랙티브 차트용으로 대용량 데이터를 무작위 샘플링한다."""
    if len(trips) <= sample_size:
        return trips
    return trips.sample(n=sample_size, random_state=random_state)


# =========================================================
# 분포 (Distribution)
# =========================================================


def plot_distance_distribution_seaborn(
    trips: pd.DataFrame,
    column: str = "trip_distance",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Seaborn 히스토그램으로 컬럼 분포를 그린다."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=trips, x=column, bins=50, kde=True, ax=ax)
    ax.set_title(f"{column} 분포")
    ax.set_xlabel(column)
    ax.set_ylabel("빈도수")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    return fig


def plot_distance_distribution_plotly(
    trips: pd.DataFrame,
    column: str = "trip_distance",
    save_path: str | Path | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> go.Figure:
    """Plotly 인터랙티브 히스토그램으로 컬럼 분포를 그린다."""
    sample = _sample_for_plotly(trips, sample_size)

    fig = px.histogram(
        sample,
        x=column,
        nbins=50,
        title=f"{column} 분포",
        labels={column: column, "count": "빈도수"},
    )
    fig.update_layout(yaxis_title="빈도수")

    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn")
    return fig


# =========================================================
# 상관관계 (Correlation)
# =========================================================


def plot_correlation_heatmap_seaborn(
    trips: pd.DataFrame,
    columns: list[str] = CORRELATION_COLUMNS,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Seaborn 히트맵으로 피처 간 상관계수를 그린다."""
    corr = trips[columns].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("피처 간 상관관계")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    return fig


def plot_correlation_heatmap_plotly(
    trips: pd.DataFrame,
    columns: list[str] = CORRELATION_COLUMNS,
    save_path: str | Path | None = None,
) -> go.Figure:
    """Plotly 인터랙티브 히트맵으로 피처 간 상관계수를 그린다."""
    corr = trips[columns].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="피처 간 상관관계",
        labels={"x": "피처", "y": "피처", "color": "상관계수"},
    )

    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn")
    return fig


# =========================================================
# 그룹 비교 (Group comparison)
# =========================================================


def plot_fare_by_passenger_count_seaborn(
    trips: pd.DataFrame,
    group_col: str = LABEL_COLUMN,
    value_col: str = "fare_amount",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Seaborn 박스플롯으로 그룹(승객 수)별 값 분포를 비교한다."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=trips, x=group_col, y=value_col, ax=ax)
    ax.set_title(f"{group_col}별 {value_col} 비교")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    return fig


def plot_fare_by_passenger_count_plotly(
    trips: pd.DataFrame,
    group_col: str = LABEL_COLUMN,
    value_col: str = "fare_amount",
    save_path: str | Path | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> go.Figure:
    """Plotly 인터랙티브 박스플롯으로 그룹(승객 수)별 값 분포를 비교한다."""
    sample = _sample_for_plotly(trips, sample_size)

    fig = px.box(
        sample,
        x=group_col,
        y=value_col,
        title=f"{group_col}별 {value_col} 비교",
        labels={group_col: group_col, value_col: value_col},
    )

    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn")
    return fig


if __name__ == "__main__":
    from preprocessing import run_pandas

    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "data" / "yellow_tripdata_2026-05.parquet"
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    trips = run_pandas(data_path)

    plot_distance_distribution_seaborn(
        trips, save_path=output_dir / "distribution_seaborn.png"
    )
    plot_distance_distribution_plotly(
        trips, save_path=output_dir / "distribution_plotly.html"
    )
    plot_correlation_heatmap_seaborn(
        trips, save_path=output_dir / "correlation_seaborn.png"
    )
    plot_correlation_heatmap_plotly(
        trips, save_path=output_dir / "correlation_plotly.html"
    )
    plot_fare_by_passenger_count_seaborn(
        trips, save_path=output_dir / "group_comparison_seaborn.png"
    )
    plot_fare_by_passenger_count_plotly(
        trips, save_path=output_dir / "group_comparison_plotly.html"
    )

    print(f"차트를 {output_dir} 에 저장했습니다.")
