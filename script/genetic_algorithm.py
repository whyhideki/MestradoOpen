import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


class GeneticAlgorithm:
    def __init__(self, population_size, num_assets, expected_returns, covariance_matrix
                 , risk_aversion, max_iterations, mutation_rate, fitness_threshold
                 , minimum_return=-np.inf, maximum_risk=np.inf, normalization_parameters=None):
        self.num_assets = num_assets
        self.population_size = population_size
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_aversion = risk_aversion
        self.fitness_threshold = fitness_threshold
        self.normalization_parameters = normalization_parameters
        if normalization_parameters:
            self.minimum_return = (minimum_return - self.normalization_parameters["returns_min"]) / \
                                  (self.normalization_parameters["returns_max"] - self.normalization_parameters[
                                      "returns_min"])
            self.maximum_risk = (maximum_risk - self.normalization_parameters["std_min"]) / \
                                (self.normalization_parameters["std_max"] - self.normalization_parameters["std_min"])
        else:
            self.minimum_return = minimum_return
            self.maximum_risk = maximum_risk
        self.iterations = 0
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate
        self.population = np.array([])
        self.initial_population()

    def random_individual(self) -> np.array:
        individual = np.array([random.random() for _ in range(self.num_assets)])
        individual /= individual.sum()
        return individual

    def initial_population(self) -> None:
        self.population = np.array([self.random_individual() for _ in range(self.population_size)])

    def calculate_fitness(self, individual, risk_aversion) -> float:
        individual_expected_returns = np.dot(self.expected_returns, individual)
        individual_standard_deviation = np.sqrt(np.linalg.multi_dot([individual, self.covariance_matrix, individual]))

        if self.normalization_parameters:
            individual_expected_returns = (individual_expected_returns - self.normalization_parameters["returns_min"]) \
                                          / (self.normalization_parameters["returns_max"] -
                                             self.normalization_parameters["returns_min"])
            individual_standard_deviation = (individual_standard_deviation - self.normalization_parameters["std_min"]) \
                                            / (self.normalization_parameters["std_max"] - self.normalization_parameters[
                "std_min"])

        if individual_expected_returns <= self.minimum_return:
            return_penalty = (self.minimum_return - individual_expected_returns) ** 2
        else:
            return_penalty = 0

        if individual_standard_deviation >= self.maximum_risk:
            risk_penalty = (individual_standard_deviation - self.maximum_risk) ** 2
        else:
            risk_penalty = 0

        fitness = risk_aversion * individual_standard_deviation - (1 - risk_aversion) * individual_expected_returns \
                  + return_penalty + risk_penalty
        return fitness

    def chooses_n_ids_from_list(self, n_ids, population) -> list:
        fitness_of_population = np.array([self.calculate_fitness(individual, risk_aversion=self.risk_aversion)
                                          for individual in population])
        fitness_of_population = np.array([1 / stats.percentileofscore(fitness_of_population, individual)
                                          for individual in fitness_of_population])

        ids = []
        for _ in range(n_ids):
            valor = random.random() * fitness_of_population.sum()
            cumulative_probability = 0
            for idx in range(fitness_of_population.size):
                cumulative_probability += fitness_of_population[idx]
                if valor < cumulative_probability:
                    fitness_of_population[idx] = 0
                    ids.append(idx)
                    break
        return ids

    def select_parents(self) -> tuple:
        parents_ids = self.chooses_n_ids_from_list(n_ids=2, population=self.population)
        return tuple(parents_ids)

    def crossover(self, parent_id1, parent_id2) -> np.array:
        beta_valor = random.random()
        child = beta_valor * self.population[parent_id1] + (1 - beta_valor) * self.population[parent_id2]
        return child

    def mutation(self, individual) -> tuple:
        pos1, pos2 = tuple(random.sample(range(0, self.num_assets), k=2))
        child1 = individual.copy()
        child2 = individual.copy()
        child1[pos1] += child1[pos2]
        child1[pos2] = 0
        child2[pos2] += child2[pos1]
        child2[pos1] = 0
        return child1, child2

    def generate_children(self) -> np.array:
        children = []
        while len(children) < self.population_size:
            parent_id1, parend_id2 = self.select_parents()
            child1 = self.crossover(parent_id1, parend_id2)
            children.append(child1)
            if random.random() < self.mutation_rate:
                child2, child3 = self.mutation(child1)
                children.append(child2)
                children.append(child3)
        return np.array(children)

    def stopping_criteria(self) -> bool:
        fitness_std = np.std([self.calculate_fitness(individual, risk_aversion=self.risk_aversion)
                              for individual in self.population])
        if fitness_std < self.fitness_threshold:
            return True
        if self.iterations > self.max_iterations:
            return True

        return False

    def generate_next_population(self, children):
        n_from_children = int(np.floor(self.population_size / 2))
        children_ids = self.chooses_n_ids_from_list(n_from_children, children)
        nex_population_children = children[children_ids]

        n_from_parents = self.population_size - n_from_children
        parents_ids = self.chooses_n_ids_from_list(n_from_parents, self.population)
        next_population_parents = self.population[parents_ids]

        next_population = np.concatenate((nex_population_children, next_population_parents))
        self.population = next_population

    def best_individual(self):
        top_idx = np.argsort([self.calculate_fitness(individual, risk_aversion=self.risk_aversion)
                              for individual in self.population])[:1][0]
        return self.population[top_idx]

    def run(self):
        while not self.stopping_criteria():
            children = self.generate_children()
            self.generate_next_population(children)
            self.iterations += 1
        return self.best_individual()


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


def optimize_normalized_genetic_algorithm_markowitz(population_size, num_assets, expected_returns, covariance_matrix,
                                                    risk_aversion, max_iterations, mutation_rate, fitness_threshold,
                                                    returns_percentile, volatility_tolerance):
    normalization_parameters = {}

    no_risk = 0
    ga = GeneticAlgorithm(population_size, num_assets, expected_returns, covariance_matrix, no_risk,
                          max_iterations, mutation_rate, fitness_threshold)
    best_return_individual = ga.run()
    normalization_parameters["returns_max"] = np.dot(expected_returns, best_return_individual)
    normalization_parameters["std_max"] = np.sqrt(np.linalg.multi_dot([best_return_individual,
                                                                       covariance_matrix,
                                                                       best_return_individual]))
    # print(f"Iterations risk = 0: {ga.iterations}\n"
    #       f"Best Individual return: {normalization_parameters['returns_max']}")
    full_risk = 1
    ga = GeneticAlgorithm(population_size, num_assets, expected_returns, covariance_matrix, full_risk,
                          max_iterations, mutation_rate, fitness_threshold)
    best_volatility_individual = ga.run()
    normalization_parameters["returns_min"] = np.dot(expected_returns, best_volatility_individual)
    normalization_parameters["std_min"] = np.sqrt(np.linalg.multi_dot([best_volatility_individual,
                                                                       covariance_matrix,
                                                                       best_volatility_individual]))
    # print(f"Iterations risk = 1: {ga.iterations}\n"
    #       f"Best Individual std: {normalization_parameters['std_min']}")
    minimum_return = np.percentile(expected_returns, returns_percentile)
    maximum_risk = (1 + volatility_tolerance) * normalization_parameters["std_min"]
    ga = GeneticAlgorithm(population_size, num_assets, expected_returns, covariance_matrix, risk_aversion,
                          max_iterations, mutation_rate, fitness_threshold, minimum_return, maximum_risk,
                          normalization_parameters)
    best_individual = ga.run()
    bi_series = pd.Series(best_individual, index=expected_returns.index)
    # print(f"Iterations normalized: {ga.iterations}\n"
    #       f"Best Individual return: {np.dot(expected_returns, best_individual)}\n"
    #       f"Best Individual Std: {np.sqrt(np.linalg.multi_dot([best_individual, covariance_matrix, best_individual]))}\n"
    #       f"fitness: {0.5 * np.sqrt(np.linalg.multi_dot([best_individual, covariance_matrix, best_individual])) - 0.5 * np.dot(expected_returns, best_individual)}")
    return bi_series


if __name__ == '__main__':
    data = read_local_database(file_name="base_dados.xlsx")
    returns_df = get_returns_dataframe(data)
    returns_df = returns_df[["ITUB4", "MGLU3", "CIEL3"]]
    num_assets = len(returns_df.columns)
    expected_returns = returns_df.mean()
    covariance_matrix = returns_df.cov()

    population_size = 10
    max_iterations = 1000
    mutation_rate = 0.5
    fitness_threshold = 1e-8
    risk_aversion = 0.5

    optimize_normalized_genetic_algorithm_markowitz(population_size, num_assets, expected_returns, covariance_matrix,
                                                    risk_aversion, max_iterations, mutation_rate, fitness_threshold)
