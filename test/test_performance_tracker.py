from pathlib import Path

import pandas as pd
import pytest

from script.performance_tracker import PerformanceTracker


def read_local_database(file_name="base_dados.xlsx") -> pd.DataFrame:
    data_path = Path(__file__).parent.parent.resolve().joinpath("data", file_name)
    data = pd.read_excel(data_path)
    return data


def get_returns_dataframe(raw_data):
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    raw_data = raw_data.set_index('Date').sort_index()
    data = raw_data.pct_change()
    data = data.dropna()
    return data


class TestPerformanceTracker:
    @pytest.fixture
    def performance_tracker(self):
        data = read_local_database(file_name="base_dados.xlsx")
        returns_data = get_returns_dataframe(data)
        stock_data = returns_data[returns_data.columns[0]]
        market_data = returns_data[returns_data.columns[1]]
        performance_tracker = PerformanceTracker(stock_data, market_data)
        return performance_tracker

    def test_annualized_return(self, performance_tracker):
        assert performance_tracker.annualized_return() == pytest.approx(-11.53141059526598)

    def test_std_return(self, performance_tracker):
        assert performance_tracker.annualized_std_return() == pytest.approx(65.809170709555)

    def test_portfolio_beta(self, performance_tracker):
        assert performance_tracker.portfolio_beta() == pytest.approx(-0.001099718414683148)

    def test_sharpe(self, performance_tracker):
        assert performance_tracker.sharpe_ratio() == pytest.approx(0.1490898311509849)

    def test_max_drawdown(self, performance_tracker):
        assert performance_tracker.max_drawdown() == pytest.approx(-51.50672229948996)
