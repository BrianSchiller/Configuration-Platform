#!/home/schiller/bachelor/Configurations/venv/bin/python

import smac
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac import Callback
from smac.runhistory import TrialInfo, TrialValue
from smac.main.smbo import SMBO
from nevergrad.optimization.base import ConfiguredOptimizer

import warnings
# Ignore all DeprecationWarnings from the smac package
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from validation import Validator
from test import Tester
import settings
from training import Training
from models import MetaModelOnePlusOne, ChainMetaModelPowell, CMA, Cobyla, MetaModel, MetaModelFmin2

class CustomCallback(Callback):
    def __init__(self) -> None:
        self.trials_counter = 0

    def on_start(self, smbo: SMBO) -> None:
        print("Let's start!")

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        self.trials_counter += 1
        if self.trials_counter % 20 == 0:
            print(f"Evaluated {self.trials_counter} trials so far.")

        return None
    
def run_experiment(model_name: str, budget: int, dimensions: list[int], unique_directory: Path):
    
    trainings_function = Training(Path(*unique_directory.parts[1:]), dimensions=dimensions, budget=budget)
    
    if model_name == "MetaModelOnePlusOne":
        model = MetaModelOnePlusOne(trainings_function)
    if model_name == "ChainMetaModelPowell":
        model = ChainMetaModelPowell(trainings_function)
    if model_name == "CMA":
        model = CMA(trainings_function)
    if model_name == "MetaModel":
        model = MetaModel(trainings_function)
    if model_name == "MetaModelFmin2":
        model = MetaModelFmin2(trainings_function)
    if model_name == "Cobyla":
        model = Cobyla(trainings_function)

    model_output = unique_directory / model.name

    scenario = Scenario(model.configspace, deterministic=True, n_trials=settings.trials, output_directory=model_output)

    print(f"Beginning Configuration of {model.name}")
    print()

    smac = ACFacade(
        scenario,
        model.train, 
        overwrite=True, 
        callbacks=[CustomCallback()],
    )

    incumbent = smac.optimize()
    print(f"Finished configuration of {model.name}")

    #Plot results
    run_history = smac.runhistory
    fitness_values = [entry.cost for entry in run_history._data.values()]
    
    min_loss = min(fitness_values)
    min_loss_index = fitness_values.index(min_loss)

    plt.clf()
    plt.plot(fitness_values, label="Fitness values over iterations")

    # Mark default & incumbent
    plt.scatter(0, fitness_values[0], color='red', marker='o', s=100, label='Default configuration')
    plt.scatter(min_loss_index, min_loss, color='blue', marker='*', s=200, label='Incumbent configuration')
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Loss)')
    plt.title('Optimization Process')
    plt.legend()
    plt.grid(True)

    output_file = unique_directory / f"{model.name}.png"
    plt.savefig(output_file)

    # best_configs = Validator(smac).validate()
    # print(f"Finished Validation")

    best_configs = run_history.get_configs("cost")[:3]
    test_configs = model.configspace.sample_configuration(size = settings.test_size)
    default_config = model.configspace.get_default_configuration()
    best_config = Tester(smac).test(best_configs[:settings.test_size], test_configs, default_config, model_output)
    print(f"Finished Testing! Results can be found in {model_output}")
    
    # Output best_config
    with open(model_output / f"{model.name}_B_{budget}_D_{'_'.join(map(str, dimensions))}.txt", 'w') as file:
        print(best_config, file=file)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Run optimization training.')
    parser.add_argument('--model-name', type=str, help='Optimization model to be used', required=False)
    parser.add_argument('--dimension', type=int, nargs='+', required=True, help='List of dimensions')
    parser.add_argument('--budget', type=int, help='Budget for the optimiser', required=False)
    parser.add_argument('--directory', type=Path, help='Path to the result directory', required=False)
    args = parser.parse_args()

    model = args.model_name
    dimensions = args.dimension
    budget = args.budget
    unique_directory = args.directory

    print(dimensions)

    run_experiment(model, budget, dimensions, unique_directory)

