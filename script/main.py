import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json

from functions import read_local_database, get_returns_dataframe, generate_all_parameter_combinations, \
    save_portfolio_returns
from genetic_algorithm import optimize_normalized_genetic_algorithm_markowitz

data = read_local_database(file_name="base_dados.xlsx")
df = get_returns_dataframe(data)

combination = generate_all_parameter_combinations()

json_path = Path(__file__).parent.parent.resolve().joinpath("data", "portfolio_returns.json")
if json_path.is_file():
    with open(json_path, 'r') as file:
        data = json.load(file)

for parameters in tqdm(combination):
    if ';'.join([str(x) for x in parameters]) in data.keys():
        continue
    window_size = parameters[0]
    step_size = parameters[1]
    risk_aversion = parameters[2]
    returns_percentile = parameters[3]
    volatility_tolerance = parameters[4]

    weights_df = pd.DataFrame(columns=df.columns)

    for i in range(window_size, len(df), step_size):
        historical_data = df.iloc[i - window_size:i]

        population_size = 10
        max_iterations = 10000
        mutation_rate = 0.5
        fitness_threshold = 1e-7
        num_assets = len(historical_data.columns)
        expected_returns = historical_data.mean()
        covariance_matrix = historical_data.cov()

        weights = optimize_normalized_genetic_algorithm_markowitz(population_size, num_assets, expected_returns,
                                                                  covariance_matrix, risk_aversion, max_iterations,
                                                                  mutation_rate, fitness_threshold, returns_percentile,
                                                                  volatility_tolerance)

        weights_dict = {}
        for col in df.columns:
            weights_dict[col] = weights[col]
        weights_dict["Date"] = historical_data.index.max()
        weights_df = weights_df.append(weights_dict, ignore_index=True)

    weights_df = weights_df.set_index("Date")
    weights_df = weights_df.reindex(index=df.index, method='ffill')
    portfolio_returns = (weights_df * df).sum(axis=1)
    save_portfolio_returns(portfolio_returns, parameters, "portfolio_returns.json")
