"""
main.py 실행 결과(Pandas/Polars 데이터 로딩 비교, src.statistical_analysis 통계 분석,
src.model 학습 결과, src.visualization 차트 목록)를 Jinja2 템플릿으로 렌더링해
report.md(Markdown)로 저장하는 모듈.

사용 예:
    from src.report import generate_report

    report_path = generate_report(
        df_pandas,
        df_polars,
        stats_result,
        model_metrics,
        model_path,
        visualization_paths,
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from jinja2 import Template

# report.md는 HTML이 아니라 순수 Markdown이므로 autoescape는 사용하지 않는다.
REPORT_TEMPLATE = """\
# Yellow Taxi 데이터 분석 리포트

생성 일시: {{ generated_at }}

## 1. 데이터 로딩 비교 (Pandas vs Polars)

| 구분 | 행 수 | 컬럼 수 |
| --- | --- | --- |
| Pandas | {{ pandas_rows }} | {{ pandas_cols }} |
| Polars | {{ polars_rows }} | {{ polars_cols }} |

{{ row_count_match_message }}

## 2. 기술통계 (평균/표준편차/분위수)

{{ descriptive_stats_md }}

## 3. 변수 간 상관계수

{{ correlation_matrix_md }}

## 4. t-test: 단독 승객 vs 동승 승객의 tip_amount 비교

- 단독 승객(1명) 평균 tip_amount: {{ "%.4f"|format(ttest.single_group_mean) }}
- 동승 승객(2명 이상) 평균 tip_amount: {{ "%.4f"|format(ttest.multi_group_mean) }}
- t-statistic: {{ "%.4f"|format(ttest.t_statistic) }}
- p-value: {{ ttest.p_value_display }}

{{ ttest.interpretation }}

{{ ttest.plain_language_summary }}

## 5. Cramér's V: 승객 그룹 x 팁 등급 연관성

{{ contingency_table_md }}

- chi2: {{ "%.4f"|format(cramers_v.chi2_statistic) }}
- p-value: {{ cramers_v.p_value_display }}
- Cramér's V: {{ "%.4f"|format(cramers_v.cramers_v) }} ({{ cramers_v.strength }})

{{ cramers_v.interpretation }}

## 6. 모델 학습 결과 (passenger_count 분류)

- 정확도(Accuracy): {{ "%.4f"|format(model_metrics.accuracy) }}
- F1 점수(Weighted): {{ "%.4f"|format(model_metrics.f1_weighted) }}
- 모델 저장 경로: `{{ model_path }}`

## 7. 시각화 차트

{% for label, path in visualization_paths.items() -%}
- [{{ label }}]({{ path }})
{% endfor %}
"""


def _format_p_value(p_value: float) -> str:
    """표본이 매우 크면 p-value가 float64 정밀도 이하로 언더플로되어 0.0이 될 수 있으므로
    "0"이 아니라 "정밀도 이하로 매우 작음"이라고 표기한다. (src.statistical_analysis와 동일한 규칙)
    """
    return f"{p_value:.4g}" if p_value > 0 else "< 1e-300 (float64 정밀도 이하)"


def generate_report(
    trips_pd_cleaned: pd.DataFrame,
    trips_pl_cleaned: pl.DataFrame,
    stats_result: dict[str, object],
    model_metrics: dict[str, float],
    model_path: str | Path,
    visualization_paths: dict[str, str],
    output_path: str | Path = "report.md",
) -> Path:
    """main.py 실행 결과를 report.md(Markdown)로 렌더링하여 저장하고 저장 경로를 반환한다."""
    ttest_result = stats_result["ttest_result"]
    cramers_v_result = stats_result["cramers_v_result"]

    pandas_rows, pandas_cols = trips_pd_cleaned.shape
    polars_rows, polars_cols = trips_pl_cleaned.shape
    row_count_match_message = (
        "Pandas와 Polars 로딩 결과의 행 수가 일치합니다."
        if pandas_rows == polars_rows
        else f"⚠ Pandas({pandas_rows})와 Polars({polars_rows})의 행 수가 다릅니다."
    )

    context = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pandas_rows": pandas_rows,
        "pandas_cols": pandas_cols,
        "polars_rows": polars_rows,
        "polars_cols": polars_cols,
        "row_count_match_message": row_count_match_message,
        "descriptive_stats_md": stats_result["descriptive_stats"].to_markdown(),
        "correlation_matrix_md": stats_result["correlation_matrix"].to_markdown(),
        "contingency_table_md": cramers_v_result["contingency_table"].to_markdown(),
        "ttest": {
            **ttest_result,
            "p_value_display": _format_p_value(ttest_result["p_value"]),
        },
        "cramers_v": {
            **cramers_v_result,
            "p_value_display": _format_p_value(cramers_v_result["p_value"]),
        },
        "model_metrics": model_metrics,
        "model_path": model_path,
        "visualization_paths": visualization_paths,
    }

    rendered = Template(REPORT_TEMPLATE).render(**context)

    output_path = Path(output_path)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path
