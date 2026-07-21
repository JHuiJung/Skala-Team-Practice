"""src/visualization.py의 Seaborn/Plotly 차트 함수에 대한 pytest 테스트."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization import (
    CORRELATION_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    _sample_for_plotly,
    plot_correlation_heatmap_plotly,
    plot_correlation_heatmap_seaborn,
    plot_distance_distribution_plotly,
    plot_distance_distribution_seaborn,
    plot_fare_by_passenger_count_plotly,
    plot_fare_by_passenger_count_seaborn,
)


@pytest.fixture
def sample_trips() -> pd.DataFrame:
    n = 200
    return pd.DataFrame(
        {
            "trip_distance": [float(i % 15) + 0.5 for i in range(n)],
            "fare_amount": [10.0 + (i % 20) for i in range(n)],
            "tip_amount": [float(i % 5) for i in range(n)],
            "RatecodeID": [1 + i % 2 for i in range(n)],
            "PULocationID": [1 + i % 30 for i in range(n)],
            "DOLocationID": [1 + (i + 5) % 30 for i in range(n)],
            "passenger_count": [1 + i % 3 for i in range(n)],
        }
    )


@pytest.fixture(autouse=True)
def _close_all_figures():
    # 테스트마다 만든 matplotlib Figure가 다음 테스트로 새지 않도록 정리한다.
    yield
    plt.close("all")


class TestSampleForPlotly:
    def test_returns_original_when_smaller_than_sample_size(self, sample_trips):
        result = _sample_for_plotly(sample_trips, sample_size=1000)
        pd.testing.assert_frame_equal(result, sample_trips)

    def test_samples_down_when_larger_than_sample_size(self, sample_trips):
        result = _sample_for_plotly(sample_trips, sample_size=50)
        assert len(result) == 50


class TestDistanceDistribution:
    def test_seaborn_returns_figure_and_saves_file(self, sample_trips, tmp_path):
        save_path = tmp_path / "dist.png"
        fig = plot_distance_distribution_seaborn(sample_trips, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.is_file()

    def test_plotly_returns_figure_and_saves_file(self, sample_trips, tmp_path):
        save_path = tmp_path / "dist.html"
        fig = plot_distance_distribution_plotly(
            sample_trips, save_path=save_path, sample_size=50
        )
        assert isinstance(fig, go.Figure)
        assert save_path.is_file()


class TestCorrelationHeatmap:
    def test_seaborn_returns_figure(self, sample_trips):
        fig = plot_correlation_heatmap_seaborn(sample_trips)
        assert isinstance(fig, plt.Figure)

    def test_plotly_z_values_match_correlation_shape(self, sample_trips):
        fig = plot_correlation_heatmap_plotly(sample_trips)
        assert isinstance(fig, go.Figure)
        expected_corr = sample_trips[CORRELATION_COLUMNS].corr()
        assert np.array(fig.data[0].z).shape == expected_corr.shape

    def test_default_correlation_columns_include_features_and_label(self):
        assert CORRELATION_COLUMNS == FEATURE_COLUMNS + [LABEL_COLUMN]


class TestFareByPassengerCount:
    def test_seaborn_returns_figure_and_saves_file(self, sample_trips, tmp_path):
        save_path = tmp_path / "group.png"
        fig = plot_fare_by_passenger_count_seaborn(sample_trips, save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.is_file()

    def test_plotly_returns_figure_and_saves_file(self, sample_trips, tmp_path):
        save_path = tmp_path / "group.html"
        fig = plot_fare_by_passenger_count_plotly(
            sample_trips, save_path=save_path, sample_size=50
        )
        assert isinstance(fig, go.Figure)
        assert save_path.is_file()
