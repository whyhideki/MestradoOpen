import numpy as np
import pytest
import pandas as pd
from pathlib import Path
from script.genetic_algorithm import GeneticAlgorithm


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


class TestGeneticAlgorithm:
    @pytest.fixture()
    def genetic_algorithm_class(self):
        population_size = 3
        risk_aversion = 0.5
        max_iterations = 100
        mutation_rate = 0.5
        fitness_threshold = 1e-7

        data = read_local_database(file_name="base_dados.xlsx")
        returns_df = get_returns_dataframe(data)
        returns_df = returns_df[["ITUB4", "VALE3", "ENBR3", "MGLU3", "BPAC11"]]
        num_assets = len(returns_df.columns)
        expected_returns = returns_df.mean()
        covariance_matrix = returns_df.cov()

        ga = GeneticAlgorithm(population_size, num_assets, expected_returns, covariance_matrix, risk_aversion,
                              max_iterations, mutation_rate, fitness_threshold, normalization_parameters={})
        return ga

    def test_individual(self, genetic_algorithm_class):
        # size
        assert genetic_algorithm_class.population[0].size == genetic_algorithm_class.num_assets
        # positive_restriction
        for individual in genetic_algorithm_class.population:
            assert all(individual >= 0)
            # full investment restriction
            assert pytest.approx(individual.sum()) == 1
            # distinct
            assert np.unique(individual).size == genetic_algorithm_class.num_assets

    def test_population_size(self, genetic_algorithm_class):
        # size
        assert genetic_algorithm_class.population.shape[0] == genetic_algorithm_class.population_size
        # distinct
        distinct_numbers = genetic_algorithm_class.population_size * genetic_algorithm_class.num_assets
        assert np.unique(genetic_algorithm_class.population).size == distinct_numbers

    def test_fitness_function(self, genetic_algorithm_class):
        expected_returns = genetic_algorithm_class.expected_returns
        covariance_matrix = genetic_algorithm_class.covariance_matrix
        individual = genetic_algorithm_class.population[0]
        # fitness is a float value
        assert isinstance(genetic_algorithm_class.calculate_fitness(individual, risk_aversion=0.5), float)
        # risk_aversion = 0 -> fitness = - expected_return
        fitness_risk_aversion_0 = genetic_algorithm_class.calculate_fitness(individual, risk_aversion=0)
        individual_expected_returns = np.dot(expected_returns, individual)
        assert pytest.approx(fitness_risk_aversion_0) == pytest.approx(- individual_expected_returns)
        # risk_aversion = 1 -> fitness = standard_deviation
        individual_standard_deviation = np.sqrt(np.linalg.multi_dot([individual, covariance_matrix, individual]))
        fitness_risk_aversion_1 = genetic_algorithm_class.calculate_fitness(individual, risk_aversion=1)
        assert pytest.approx(fitness_risk_aversion_1) == pytest.approx(individual_standard_deviation)

    def test_parent_selection(self, genetic_algorithm_class):
        id1, id2 = genetic_algorithm_class.select_parents()
        # indexes on population
        assert id1 < genetic_algorithm_class.population_size
        assert id2 < genetic_algorithm_class.population_size
        # different parents
        assert id1 != id2

    def test_crossover(self, genetic_algorithm_class):
        parent_id1, parent_id2 = 0, 1
        individual = genetic_algorithm_class.crossover(parent_id1, parent_id2)
        # check_individual_properties
        assert individual.size == genetic_algorithm_class.num_assets
        assert all(individual >= 0)
        assert pytest.approx(individual.sum()) == 1
        # not in population
        for population_individual in genetic_algorithm_class.population:
            child_plus_population = np.array([individual, population_individual])
            distinct_numbers = 2 * genetic_algorithm_class.num_assets
            assert np.unique(child_plus_population).size == distinct_numbers

    def test_mutation(self, genetic_algorithm_class):
        individual = genetic_algorithm_class.population[0]
        mutated_individual1, mutated_individual2 = genetic_algorithm_class.mutation(individual)

        # check_individual1_properties
        assert mutated_individual1.size == genetic_algorithm_class.num_assets
        assert all(mutated_individual1 >= 0)
        assert pytest.approx(mutated_individual1.sum()) == 1

        # check_individual2_properties
        assert mutated_individual2.size == genetic_algorithm_class.num_assets
        assert all(mutated_individual2 >= 0)
        assert pytest.approx(mutated_individual2.sum()) == 1

    def test_stopping_criteria(self, genetic_algorithm_class):
        genetic_algorithm_class.max_iterations = 10
        genetic_algorithm_class.iterations = genetic_algorithm_class.max_iterations + 1
        # check max_iterations
        assert genetic_algorithm_class.stopping_criteria() is True
        genetic_algorithm_class.max_iterations = 10
        genetic_algorithm_class.iterations = 0
        genetic_algorithm_class.population = np.array([genetic_algorithm_class.random_individual()]*5)
        assert genetic_algorithm_class.stopping_criteria() is True

    def test_children(self, genetic_algorithm_class):
        children = genetic_algorithm_class.generate_children()
        assert children.shape[0] >= genetic_algorithm_class.population_size
        for individual in children:
            assert all(individual >= 0)
            # full investment restriction
            assert pytest.approx(individual.sum()) == 1
            # distinct
            assert np.unique(individual).size == genetic_algorithm_class.num_assets

    def test_best_individual(self, genetic_algorithm_class):
        best_individual = genetic_algorithm_class.best_individual()
        print(best_individual)
        best_fitness = genetic_algorithm_class.calculate_fitness(best_individual,
                                                                 risk_aversion=genetic_algorithm_class.risk_aversion)
        assert all(best_individual >= 0)
        # full investment restriction
        assert pytest.approx(best_individual.sum()) == 1
        # distinct
        assert np.unique(best_individual).size == genetic_algorithm_class.num_assets

        for individual in genetic_algorithm_class.population:
            print(individual)
            individual_fitness = genetic_algorithm_class.calculate_fitness(individual,
                                                                           risk_aversion=genetic_algorithm_class.risk_aversion)
            assert best_fitness <= individual_fitness

    def test_next_population(self, genetic_algorithm_class):
        children = genetic_algorithm_class.generate_children()
        current_population = genetic_algorithm_class.population.copy()
        genetic_algorithm_class.generate_next_population(children)
        assert genetic_algorithm_class.population.shape[0] == genetic_algorithm_class.population_size
        population_plus_new_population = np.array([current_population, genetic_algorithm_class.population])
        distinct_numbers = genetic_algorithm_class.population_size * genetic_algorithm_class.num_assets
        assert np.unique(population_plus_new_population).size > distinct_numbers

    def test_run(self, genetic_algorithm_class):
        genetic_algorithm_class.run()
