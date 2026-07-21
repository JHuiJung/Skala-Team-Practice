"""src/statistics.py 통계 분석 함수에 대한 pytest 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    FEATURES,
    TARGET,
    compute_correlation_matrix,
    compute_descriptive_stats,
    compute_passenger_tip_cramers_v,
    run_passenger_tip_ttest,
    run_statistical_analysis,
)


@pytest.fixture
def sample_trips() -> pd.DataFrame:
    """단독 승객(tip 1~2달러) vs 동승 승객(tip 4~5달러)이 뚜렷이 구분되는 합성 데이터.

    분포를 명확히 분리해 두어 t-test/Cramér's V 결과가 난수 시드에 관계없이
    항상 동일하게 나오도록 한다(테스트 재현성 확보).
    """
    single_tip = [1.0] * 50 + [2.0] * 50  # passenger_count == 1
    multi_tip = [4.0] * 50 + [5.0] * 50  # passenger_count in {2, 3}
    tip_amount = single_tip + multi_tip
    passenger_count = [1] * 100 + [2] * 50 + [3] * 50
    n = len(passenger_count)

    return pd.DataFrame(
        {
            "trip_distance": np.linspace(0.5, 10.0, n),
            "fare_amount": np.linspace(5.0, 50.0, n),
            "tip_amount": tip_amount,
            "RatecodeID": [(i % 3) + 1 for i in range(n)],
            "PULocationID": list(range(1, n + 1)),
            "DOLocationID": list(range(n, 0, -1)),
            "passenger_count": passenger_count,
        }
    )


class TestComputeDescriptiveStats:
    def test_returns_all_features_and_target_columns(self, sample_trips):
        desc = compute_descriptive_stats(sample_trips)
        assert list(desc.columns) == FEATURES + [TARGET]

    def test_index_contains_mean_std_and_quantiles(self, sample_trips):
        desc = compute_descriptive_stats(sample_trips)
        assert list(desc.index) == [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]

    def test_mean_value_matches_manual_calculation(self, sample_trips):
        desc = compute_descriptive_stats(sample_trips)
        expected_mean = sample_trips["passenger_count"].mean()
        assert desc.loc["mean", "passenger_count"] == pytest.approx(expected_mean)

    def test_missing_column_raises_keyerror(self, sample_trips):
        trips = sample_trips.drop(columns=["fare_amount"])
        with pytest.raises(KeyError):
            compute_descriptive_stats(trips)


class TestComputeCorrelationMatrix:
    def test_diagonal_is_one(self, sample_trips):
        corr = compute_correlation_matrix(sample_trips)
        for column in corr.columns:
            assert corr.loc[column, column] == pytest.approx(1.0)

    def test_matrix_is_symmetric(self, sample_trips):
        corr = compute_correlation_matrix(sample_trips)
        assert np.allclose(corr.to_numpy(), corr.to_numpy().T)

    def test_missing_column_raises_keyerror(self, sample_trips):
        trips = sample_trips.drop(columns=["RatecodeID"])
        with pytest.raises(KeyError):
            compute_correlation_matrix(trips)


class TestRunPassengerTipTtest:
    def test_detects_significant_difference(self, sample_trips):
        result = run_passenger_tip_ttest(sample_trips)
        assert result["is_significant"]
        assert result["p_value"] < 0.05
        assert result["multi_group_mean"] > result["single_group_mean"]

    def test_plain_language_summary_mentions_higher_group(self, sample_trips):
        result = run_passenger_tip_ttest(sample_trips)
        assert "동승 승객" in result["plain_language_summary"]

    def test_missing_column_raises_keyerror(self, sample_trips):
        trips = sample_trips.drop(columns=["tip_amount"])
        with pytest.raises(KeyError):
            run_passenger_tip_ttest(trips)

    def test_insufficient_samples_raises_valueerror(self):
        trips = pd.DataFrame({"passenger_count": [1, 2], "tip_amount": [1.0, 2.0]})
        with pytest.raises(ValueError):
            run_passenger_tip_ttest(trips)


class TestComputePassengerTipCramersV:
    def test_value_is_between_zero_and_one(self, sample_trips):
        result = compute_passenger_tip_cramers_v(sample_trips)
        assert 0.0 <= result["cramers_v"] <= 1.0

    def test_strength_label_is_known_category(self, sample_trips):
        result = compute_passenger_tip_cramers_v(sample_trips)
        assert result["strength"] in {
            "거의 없음 (negligible)",
            "약함 (weak)",
            "중간 (moderate)",
            "강함 (strong)",
        }

    def test_contingency_table_has_two_passenger_groups(self, sample_trips):
        result = compute_passenger_tip_cramers_v(sample_trips)
        assert result["contingency_table"].shape[0] == 2

    def test_missing_column_raises_keyerror(self, sample_trips):
        trips = sample_trips.drop(columns=["passenger_count"])
        with pytest.raises(KeyError):
            compute_passenger_tip_cramers_v(trips)

    def test_single_passenger_group_only_raises_valueerror(self):
        # 동승(2명 이상) 그룹이 아예 없으면 분할표가 1행짜리가 되어 카이제곱검정이 불가능하다.
        trips = pd.DataFrame(
            {
                "passenger_count": [1, 1, 1, 1],
                "tip_amount": [1.0, 2.0, 3.0, 4.0],
            }
        )
        with pytest.raises(ValueError):
            compute_passenger_tip_cramers_v(trips)


class TestRunStatisticalAnalysis:
    def test_returns_all_expected_result_keys(self, sample_trips):
        result = run_statistical_analysis(sample_trips, print_output=False)
        assert set(result.keys()) == {
            "descriptive_stats",
            "correlation_matrix",
            "ttest_result",
            "cramers_v_result",
        }
        assert isinstance(result["descriptive_stats"], pd.DataFrame)
        assert isinstance(result["correlation_matrix"], pd.DataFrame)
        assert isinstance(result["ttest_result"], dict)
        assert isinstance(result["cramers_v_result"], dict)

    def test_print_output_true_writes_to_stdout(self, sample_trips, capsys):
        run_statistical_analysis(sample_trips, print_output=True)
        captured = capsys.readouterr()
        assert "기술통계" in captured.out
        assert "t-test" in captured.out
        assert "Cramér" in captured.out