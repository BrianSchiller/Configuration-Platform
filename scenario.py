#!/home/schiller/bachelor/Configurations/venv/bin/python

import smac
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario

import warnings
# Ignore all DeprecationWarnings from the smac package
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from test import Tester
import settings
from training import Training
from models import MetaModelOnePlusOne, ChainMetaModelPowell, CMA, Cobyla, MetaModel, MetaModelFmin2
import plot
    
def run_experiment(model_name: str, budget: int, dimensions: list[int], unique_directory: Path, trials: int):
    
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
                        n_trials=trials, 
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

    # Output best_config
    config_output = model_output / f"{model.name}_B_{budget}_D_{'_'.join(map(str, dimensions))}.txt"
    with open(config_output, 'w') as file:
        print(incumbent.get_dictionary(), file=file)

    # Plot the trajectory of the incumbents
    plot.plot_trajectory(smac.intensifier, smac.runhistory, unique_directory, model.name)
    
    # Test the found configuration against the default one and randomly sampled ones
    test_configs = model.configspace.sample_configuration(size = settings.test_size)
    default_config = model.configspace.get_default_configuration()
    # Adjust the popsize
    if model.name == "MetaModel" or model.name == "CMA" or model.name == "ChainMetaModelPowell":
        default_config["popsize"] = 4 + int(3 * np.log(dimensions[-1]))
    Tester(smac).test(incumbent, test_configs, default_config, model_output)
    print(f"Finished Testing! Results can be found in {model_output}")

    plot.plot_config_difference(config_output, default_config, model.name, model_output / "plots")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Run optimization training.')
    parser.add_argument('--model-name', type=str, help='Optimization model to be used', required=False)
    parser.add_argument('--dimension', type=int, nargs='+', required=True, help='List of dimensions')
    parser.add_argument('--budget', type=int, help='Budget for the optimiser', required=False)
    parser.add_argument('--directory', type=Path, help='Path to the result directory', required=False)
    parser.add_argument('--trials', type=int, help='Number of trials to run', required=False)
    args = parser.parse_args()

    model = args.model_name
    dimensions = args.dimension
    budget = args.budget
    unique_directory = args.directory
    trials = args.trials

    run_experiment(model, budget, dimensions, unique_directory, trials)

