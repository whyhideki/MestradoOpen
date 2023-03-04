import itertools
import json
from pathlib import Path

import pandas as pd


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


def naive_portfolio_returns(returns_df):
    weights_df = pd.DataFrame(index=returns_df.index)
    naive_weight = 1.0 / len(returns_df.columns)
    for stock in returns_df.columns:
        weights_df[stock] = naive_weight
    weights_df.index.name = "Date"

    naive_returns = (weights_df * returns_df).sum(axis=1)
    return naive_returns


def generate_all_parameter_combinations():
    window_sizes = [52]
    step_sizes = [4]
    risk_aversion_range = [i / 100.0 for i in range(0, 101, 10)]
    returns_percentiles = [i for i in range(0, 101, 10)]
    volatility_tolerances = [i / 100.0 for i in range(0, 101, 10)]

    combination = list(itertools.product(window_sizes,
                                         step_sizes,
                                         risk_aversion_range,
                                         returns_percentiles,
                                         volatility_tolerances))
    return combination


def save_portfolio_returns(portfolio_returns, parameters, file_name):
    portfolio_returns.index = portfolio_returns.index.strftime("%Y-%m-%d %H:%M:%S")
    json_to_save = {
        "parameters": parameters,
        "portfolio_returns": portfolio_returns.to_json(),
        "average_return": portfolio_returns.mean(),
        "average_volatility": portfolio_returns.std(),
        "sharpe": portfolio_returns.mean() / portfolio_returns.std()
    }
    json_path = Path(__file__).parent.parent.resolve().joinpath("data", "portfolio_returns.json")
    if json_path.is_file():
        with open(json_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    key = ';'.join([str(x) for x in parameters])
    data[key] = json_to_save
    with open(json_path, 'w+') as file:
        json.dump(data, file, indent=2)
