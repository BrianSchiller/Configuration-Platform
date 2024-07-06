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
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from validation import Validator
from test import Tester
import settings
from training import Training
from models import MetaModelOnePlusOne, ChainMetaModelPowell, CMA, Cobyla, MetaModel, MetaModelFmin2
import plot
    
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

    instances = [f"{problem}_{instance}" for problem in settings.problems for instance in settings.instances]
    index_dict = {instance: [i] for i, instance in enumerate(instances)}

    scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=settings.trials, 
                        output_directory=model_output, 
                        instances=instances,
                        instance_features=index_dict)

    print(f"Beginning Configuration of {model.name}")
    print()

    smac = ACFacade(
        scenario,
        model.train, 
        overwrite=True, 
    )

    incumbent = smac.optimize()
    print(f"Finished configuration of {model.name}")

    # Plot the trajectory of the incumbents
    plot.plot_trajectory(smac.intensifier, unique_directory, model.name)
    
    # Test the found configurations against the default one and randomly sampled ones
    best_configs = smac.runhistory.get_configs("cost")[:settings.test_size]
    test_configs = model.configspace.sample_configuration(size = settings.test_size)
    default_config = model.configspace.get_default_configuration()
    # Adjust the popsize
    if model.name == "MetaModel" or model.name == "CMA" or model.name == "ChainMetaModelPowell":
        default_config["popsize"] = 4 + int(3 * np.log(dimensions[-1]))
    Tester(smac).test(best_configs[:settings.test_size], test_configs, default_config, model_output)
    print(f"Finished Testing! Results can be found in {model_output}")
    print()
    
    # Output best_config
    config_output = model_output / f"{model.name}_B_{budget}_D_{'_'.join(map(str, dimensions))}.txt"
    with open(config_output, 'w') as file:
        print(best_configs[0].get_dictionary(), file=file)

    plot.plot_config_difference(config_output, default_config, model.name, model_output / "plots")


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

    run_experiment(model, budget, dimensions, unique_directory)

