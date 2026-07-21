"""
Yellow Taxi 데이터에 대한 통계 분석 모듈.

작성자 : 정희중

trip_distance, fare_amount, tip_amount, RatecodeID, PULocationID, DOLocationID를
설명 변수(FEATURES)로, passenger_count를 타겟(TARGET)으로 하여 다음을 수행한다.
- 기술통계(평균/표준편차/분위수) 산출
- 설명 변수 + 타겟 간 상관계수 계산
- 단독 승객(1명) vs 동승 승객(2명 이상) 그룹 간 tip_amount 평균 차이에 대한
  scipy.stats.ttest_ind 수행 및 p-value 해석

preprocessing.run_pandas()로 결측치 처리가 끝난 DataFrame을 입력으로 받는 것을 전제로 한다.

사용 예:
    from preprocessing import run_pandas
    from src.statistics import run_statistical_analysis

    trips = run_pandas("data/yellow_tripdata_2026-05.parquet")
    result = run_statistical_analysis(trips, print_output=True)
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

# 설명 변수 / 타겟 정의
FEATURES = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
]
TARGET = "passenger_count"

# t-test 유의수준 (일반적으로 사용되는 5% 기준)
ALPHA = 0.05


def _validate_columns(trips: pd.DataFrame, required_columns: list[str]) -> None:
    """분석에 필요한 컬럼이 DataFrame에 모두 존재하는지 확인한다. 없으면 KeyError를 발생시킨다."""
    missing = [col for col in required_columns if col not in trips.columns]
    if missing:
        raise KeyError(
            f"다음 컬럼이 DataFrame에 없습니다: {missing}. "
            "preprocessing.run_pandas()로 전처리된 DataFrame을 사용하고 있는지 확인하세요."
        )


def compute_descriptive_stats(
    trips: pd.DataFrame, print_output: bool = False
) -> pd.DataFrame:
    """FEATURES + TARGET에 대한 기술통계(평균/표준편차/분위수 등)를 산출한다.

    pandas의 describe()는 count/mean/std/min/25%/50%/75%/max를 한 번에 계산해 주므로
    별도 계산 없이 그대로 활용한다.
    """
    _validate_columns(trips, FEATURES + [TARGET])

    desc = trips[FEATURES + [TARGET]].describe()
    if print_output:
        print("=== 기술통계 (평균/표준편차/분위수) ===")
        print(desc)
    return desc


def compute_correlation_matrix(
    trips: pd.DataFrame, print_output: bool = False
) -> pd.DataFrame:
    """FEATURES + TARGET 간 피어슨 상관계수 행렬을 계산한다."""
    _validate_columns(trips, FEATURES + [TARGET])

    corr = trips[FEATURES + [TARGET]].corr()
    if print_output:
        print("=== 변수 간 상관계수 ===")
        print(corr)
    return corr


def run_passenger_tip_ttest(
    trips: pd.DataFrame, print_output: bool = False
) -> dict[str, float | str]:
    """단독 승객(passenger_count == 1) vs 동승 승객(passenger_count >= 2) 그룹 간
    tip_amount 평균 차이를 scipy.stats.ttest_ind로 검정하고 p-value를 해석한다.

    귀무가설(H0): 두 그룹의 tip_amount 평균은 같다.
    대립가설(H1): 두 그룹의 tip_amount 평균은 다르다.
    """
    _validate_columns(trips, ["passenger_count", "tip_amount"])

    # t-test는 결측치를 처리할 수 없으므로 두 컬럼 모두 값이 있는 행만 사용한다.
    valid_trips = trips.dropna(subset=["passenger_count", "tip_amount"])

    single_group = valid_trips.loc[
        valid_trips["passenger_count"] == 1, "tip_amount"
    ]
    multi_group = valid_trips.loc[
        valid_trips["passenger_count"] >= 2, "tip_amount"
    ]

    # 표본이 2개 미만이면 t-test 자체가 성립하지 않으므로 명시적으로 예외를 발생시킨다.
    if len(single_group) < 2 or len(multi_group) < 2:
        raise ValueError(
            "t-test를 수행하기에 표본 수가 부족합니다. "
            f"single_group={len(single_group)}건, multi_group={len(multi_group)}건."
        )

    # equal_var=False (Welch's t-test): 두 그룹의 표본 수/분산이 크게 다를 수 있어
    # 등분산 가정을 하지 않는 것이 더 안전하다.
    t_statistic, p_value = stats.ttest_ind(
        single_group, multi_group, equal_var=False
    )

    is_significant = p_value < ALPHA
    # 표본 수가 매우 크면 p-value가 float64 정밀도 이하로 언더플로되어 정확히 0.0이 될 수 있다.
    # 이 경우 "0"이 아니라 "정밀도 이하로 매우 작음"이라고 표기해 오해를 방지한다.
    p_value_display = f"{p_value:.4g}" if p_value > 0 else "< 1e-300 (float64 정밀도 이하)"
    interpretation = (
        f"p-value({p_value_display}) < 유의수준({ALPHA}) 이므로 귀무가설을 기각합니다. "
        "단독 승객과 동승 승객 그룹의 tip_amount 평균은 통계적으로 유의미하게 다릅니다."
        if is_significant
        else f"p-value({p_value_display}) >= 유의수준({ALPHA}) 이므로 귀무가설을 기각할 수 없습니다. "
        "두 그룹의 tip_amount 평균 차이가 통계적으로 유의미하다고 보기 어렵습니다."
    )

    single_mean = single_group.mean()
    multi_mean = multi_group.mean()

    # 통계 용어에 익숙하지 않은 팀원도 바로 이해할 수 있도록 결론을 일상 언어로 한 번 더 풀어 쓴다.
    higher_group_label = "동승 승객" if multi_mean > single_mean else "단독 승객"
    plain_language_summary = (
        f"혼자 탄 승객과 2명 이상 같이 탄 승객의 팁 금액 차이"
        f"(약 {single_mean:.2f}달러 vs {multi_mean:.2f}달러)는 단순한 우연이 아니라, "
        f"실제로 {higher_group_label}이 팁을 더 많이 주는 명확한 차이가 존재합니다."
        if is_significant
        else f"혼자 탄 승객과 2명 이상 같이 탄 승객의 팁 금액 차이"
        f"(약 {single_mean:.2f}달러 vs {multi_mean:.2f}달러)는 통계적으로 유의미하지 않아, "
        "우연에 의한 차이일 가능성을 배제할 수 없습니다."
    )

    result = {
        "single_group_mean": single_mean,
        "multi_group_mean": multi_mean,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "is_significant": is_significant,
        "interpretation": interpretation,
        "plain_language_summary": plain_language_summary,
    }

    if print_output:
        print("=== t-test: 단독 승객 vs 동승 승객의 tip_amount 비교 ===")
        print(f"단독 승객(1명) 평균 tip_amount: {result['single_group_mean']:.4f}")
        print(f"동승 승객(2명 이상) 평균 tip_amount: {result['multi_group_mean']:.4f}")
        print(f"t-statistic: {t_statistic:.4f}, p-value: {p_value_display}")
        print("p-value는 두 그룹의 tip_amount 평균이 실제로 같다고 가정했을 때, 지금과 같은 차이(또는 더 큰 차이)가 우연히 관측될 확률을 의미합니다.")
        print(result["interpretation"])
        print(plain_language_summary)

    return result


def run_statistical_analysis(
    trips: pd.DataFrame, print_output: bool = True
) -> dict[str, object]:
    """기술통계, 상관계수, t-test를 모두 수행하고 결과를 하나의 dict로 반환한다."""
    return {
        "descriptive_stats": compute_descriptive_stats(trips, print_output),
        "correlation_matrix": compute_correlation_matrix(trips, print_output),
        "ttest_result": run_passenger_tip_ttest(trips, print_output),
    }
