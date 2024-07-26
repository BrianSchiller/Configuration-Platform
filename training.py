from nevergrad.optimization.optimizerlib import base
import nevergrad as ng
import numpy as np
from pathlib import Path
from ioh import get_problem, ProblemClass, logger

import json
import math   
import os
import csv

import settings
import constants as const

class Training:
    def __init__(
        self, 
        output_dir: str,
        problems: list[str] = None, 
        dimensions: list[int] = None, 
        budget: int = None, 
        repetitions: int = None, 
        instances: list[int] = None
    ) -> None:
        self.output_dir = output_dir
        self.dimensions = dimensions
        self.budget = budget

        if problems is not None:
            self.problems = problems
        else:
            self.problems = settings.problems
        
        if repetitions is not None:
            self.repetitions = repetitions
        else:
            self.repetitions = settings.repetitions

        if instances is not None:
            self.instances = instances
        else:
            self.instances = settings.instances

        # TODO: Find a more elegant solution for normalising results
        self.values = {}

    def train(self, optimizer: base.ConfiguredOptimizer, _instance, name: str) -> float:
        total_loss = 0
        lower_bound = -5
        upper_bound = 5

        problem, instance = _instance.split("_")
        problem = int(problem)
        instance = int(instance)

        if settings.store_problem_results:
            results = [["problem", "instance", "dimension", "budget", "loss"]]

        for dimension in self.dimensions:
            dir_name = f"{settings.log_folder}/{self.output_dir}/{name}/D{dimension}_F{problem}"
            ioh_logger = logger.Analyzer(folder_name=dir_name,
                                        algorithm_name=optimizer.name)
            function = get_problem(problem, instance=instance,
                                dimension=dimension,
                                problem_class=ProblemClass.BBOB)
            function.attach_logger(ioh_logger)

            param = ng.p.Array(init=np.random.uniform(lower_bound, upper_bound, (function.meta_data.n_variables,)))
            param.set_bounds(lower_bound, upper_bound)
            algorithm = optimizer(parametrization=param, budget=self.budget)

            algorithm.minimize(function)
            function.reset()
            
            ioh_logger.close()
            with Path(ioh_logger.output_directory + f"/IOHprofiler_{const.PROB_NAMES[problem - 1]}.json").open() as metadata_file:
                metadata = json.load(metadata_file)
                loss = 0
                # Loop over runs and return average for multidimensional configuration
                for index,run in enumerate(metadata['scenarios'][0]['runs']):
                    loss += run['best']['y']
                    if settings.store_problem_results:
                        results.append([problem, instance, dimension, self.budget, run['best']['y']])
                loss = loss / len(metadata['scenarios'][0]['runs'])
                # Can't take log_10 of 0, so we cap at 10**(-10)
                if loss > 10**(-10):
                    total_loss += math.log10(loss)
                else:
                    total_loss += (-10)

        if settings.store_problem_results:
            os.makedirs(os.path.dirname(settings.problem_result_dir), exist_ok=True)
            try:
                file_exists = os.path.isfile(settings.problem_result_dir)
                # If file already exists skip header
                if file_exists:
                    results = results[1:]
                with open(settings.problem_result_dir, 'a', newline='') as file:
                    writer = csv.writer(file)
                    for result in results:
                        writer.writerow(result)
            except Exception as e:
                print(f"Error writing results to CSV: {e}")
        return total_loss