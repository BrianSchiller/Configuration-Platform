import smac
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac import Callback
from smac.runhistory import TrialInfo, TrialValue
from smac.main.smbo import SMBO

import warnings
# Ignore all DeprecationWarnings from the smac package
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import os
import datetime
import argparse
from pathlib import Path

from training import Training
from models import MetaModelOnePlusOne, ChainMetaModelPowell, CMA, Cobyla, MetaModel, MetaModelFmin2
from validation import Validator
from test import Tester
from settings import Settings

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

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Run optimization training.')
    parser.add_argument('--name', type=str, help='Name of the result folder', required=False)
    args = parser.parse_args()

    # Models 
    trainings_function = Training()
    metaModelOnePlusOne = MetaModelOnePlusOne(trainings_function)
    chainMetaModelPowell = ChainMetaModelPowell(trainings_function)
    # cma = CMA(trainings_function)
    # cobyla = Cobyla(trainings_function)
    metaModel = MetaModel(trainings_function)
    metaModelFmin2 = MetaModelFmin2(trainings_function)

    models = [metaModelOnePlusOne, chainMetaModelPowell, metaModel, metaModelFmin2]

    # Output paths
    output_dir = 'Output'
    if not os.path.exists(Path(output_dir)):
        os.makedirs(Path(output_dir))
        print(f"Directory '{output_dir}' created.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.name is not None:
        unique_directory = Path(output_dir) / f"{args.name}-{timestamp}"
    else:
        unique_directory = Path(output_dir) / f"run_{timestamp}"
    unique_directory.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{unique_directory}' created.")

    # For each model create the scenario, run, validate, test and plot
    for model in models:
        model_output = unique_directory / model.name

        scenario = Scenario(model.configspace, deterministic=True, n_trials=Settings.trials, output_directory=model_output)

        print()
        print(f"Beginning Configuration of {model.name}")

        smac = ACFacade(
            scenario,
            model.train, 
            overwrite=True, 
            callbacks=[CustomCallback()],
        )

        incumbent = smac.optimize()
        print(f"Finished configuration of {model.name}")

        best_configs = Validator(smac).validate()
        print(f"Finished Validation")

        test_configs = model.configspace.sample_configuration(Settings.test_size)
        Tester(smac).test(best_configs[:Settings.test_size], test_configs, model_output)
        print(f"Finished Testing! Results can be found in {model_output}")

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

