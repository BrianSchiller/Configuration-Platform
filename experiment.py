import os
import datetime
import argparse
import json
from pathlib import Path

import settings
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
        "Trials": settings.trials,
        "Optimisation": {
            "Problems": settings.problems,
            "Dimensions": settings.dimension_sets,
            "Instances": settings.instances,
            "Budget": settings.budgets,
            "Repetitions": settings.repetitions,
        },
        "Validation": {
            "Candidates": settings.val_size,
            "Iterations": settings.val_iterations,
        },
        "Testing": {
            "Candidates": settings.test_size,
            "Iterations": settings.test_iterations,
        }
    }

    with open(f"{output_dir}/scenario.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

def create_job_script(model, budget, dimensions, specific_directory, slurm_output):
    script_content = f"""#!/bin/bash
#SBATCH --job-name={model}_B{budget}_D{'_'.join(map(str, dimensions))}
#SBATCH --output={slurm_output}/{model}.out
#SBATCH --error={slurm_output}/{model}.err
#SBATCH --time=48:00:00
#SBATCH --partition=KathleenE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3000M

# Activate virtual environment
source venv/bin/activate

# Run the experiment
python scenario.py --model {model} --directory {specific_directory} --dimension {' '.join(map(str, dimensions))} --budget {budget}
"""
    return script_content


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
    for dimensions in settings.dimension_sets:
        for budget in settings.budgets:
            specific_directory = unique_directory / f"B_{budget}__D_{'_'.join(map(str, dimensions))}"

            # Models 
            metaModelOnePlusOne = "MetaModelOnePlusOne"
            chainMetaModelPowell = "ChainMetaModelPowell"
            cma = "CMA"
            cobyla = "Cobyla"
            metaModel = "MetaModel"
            metaModelFmin2 = "MetaModelFmin2"

            # models = [cma, metaModelOnePlusOne, chainMetaModelPowell, metaModel, metaModelFmin2]
            models = [cma]

            # For each model create the scenario, run, validate, test and plot
            for model in models:
                if args.slurm:
                    slurm_output = specific_directory / model
                    os.makedirs(slurm_output, exist_ok=True)

                    job_script = create_job_script(model, budget, dimensions, specific_directory, slurm_output)
                    job_script_path = slurm_output / f"{model}_B{budget}_D{'_'.join(map(str, dimensions))}.sh"
                    with open(job_script_path, 'w') as file:
                        file.write(job_script)
                    
                    # Submit the job script
                    os.system(f"sbatch {job_script_path}")

                else:
                    run_experiment(model, budget, dimensions, specific_directory)

