import os
import datetime
import argparse
import json
from pathlib import Path

from settings import Settings
from scenario import run_experiment

    
def create_dirs() -> Path:
    output_dir = 'Output'
    if not os.path.exists(Path(output_dir)):
        os.makedirs(Path(output_dir))
        print(f"Directory '{output_dir}' created.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if args.name is not None:
        unique_directory = Path(output_dir) / f"{args.name}-{timestamp}"
    else:
        unique_directory = Path(output_dir) / f"run_{timestamp}"
    unique_directory.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{unique_directory}' created.")
    
    return unique_directory
    
def write_scenario_file(output_dir):
    data = {
        "Trials": Settings.trials,
        "Optimisation": {
            "Problems": Settings.problems,
            "Dimensions": Settings.dimension_sets,
            "Instances": Settings.instances,
            "Budget": Settings.budgets,
            "Repetitions": Settings.repetitions,
        },
        "Validation": {
            "Candidates": Settings.val_size,
            "Iterations": Settings.val_iterations,
        },
        "Testing": {
            "Candidates": Settings.test_size,
            "Iterations": Settings.test_iterations,
        }
    }

    with open(f"{output_dir}/scenario.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Run optimization training.')
    parser.add_argument('--name', type=str, help='Name of the result folder', required=False)
    parser.add_argument('--slurm', type=str, help='Whether to run on Slurm', required=False, default=False)
    args = parser.parse_args()

    # Output paths
    unique_directory = create_dirs()
    write_scenario_file(unique_directory)

    # Iterate over sets of relevant dimensions
    # e.g.: sets = [[2], [3], [5], [2,3,5]]
    for dimensions in Settings.dimension_sets:
        for budget in Settings.budgets:
            specific_directory = unique_directory / f"B_{budget}__D_{'_'.join(map(str, dimensions))}"

            # Models 
            metaModelOnePlusOne = "metaModelOnePlusOne"
            chainMetaModelPowell = "chainMetaModelPowell"
            cma = "cma"
            cobyla = "cobyla"
            metaModel = "metaModel"
            metaModelFmin2 = "metaModelFmin2"

            models = [cma, metaModelOnePlusOne, chainMetaModelPowell, metaModel, metaModelFmin2]

            # For each model create the scenario, run, validate, test and plot
            for model in models:
                if args.slurm:
                    import runrunner as rrr

                    slurm_output = specific_directory / model.name / "Slurm"
                    srun_options = ["-N1", "-n1", "--mem-per-cpu=3000"]

                    rrr.add_to_slurm_queue(
                        cmd = f"./scenario.py --model {model} --directory {specific_directory}",
                        name = model.name,
                        base_dir = slurm_output,
                        srun_options = srun_options
                    )

                else:
                    run_experiment(model, budget, dimensions, specific_directory)

