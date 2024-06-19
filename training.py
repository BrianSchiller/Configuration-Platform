from nevergrad.benchmark import Experiment as NevergradExperiment
from nevergrad.optimization.optimizerlib import base
from nevergrad.functions import ArtificialFunction
import nevergrad as ng
import numpy as np
from pathlib import Path
from ioh import get_problem, ProblemClass, Experiment, logger

import json
import math   
import os

from settings import Settings
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

        if problems is not None:
            self.problems = problems
        else:
            self.problems = Settings.problems
        
        if dimensions is not None:
            self.dimensions = dimensions
        else:
            self.dimensions = Settings.dimensions
        
        if budget is not None:
            self.budget = budget
        else:
            self.budget = Settings.budget
        
        if repetitions is not None:
            self.repetitions = repetitions
        else:
            self.repetitions = Settings.repetitions

        if instances is not None:
            self.instances = instances
        else:
            self.instances = Settings.instances

        # TODO: Find a more elegant solution for normalising results
        self.values = {}

    def train(self, optimizer: base.ConfiguredOptimizer, name: str) -> float:
        total_loss = 0
        lower_bound = -5
        upper_bound = 5
        for problem in self.problems:
            for dimension in self.dimensions:
                dir_name = f"Log/{self.output_dir}/{name}_D{dimension}_F{problem}"
                ioh_logger = logger.Analyzer(folder_name=dir_name,
                                            algorithm_name=optimizer.name)
                # ioh_logger.add_run_attributes(optimizer, ['algorithm_seed'])
                
                for instance in self.instances:
                    function = get_problem(problem, instance=instance,
                                        dimension=dimension,
                                        problem_class=ProblemClass.BBOB)
                    function.attach_logger(ioh_logger)

                    seed = np.random.randint(1, 1000000)
                    np.random.seed(seed)
                    param = ng.p.Array(init=np.random.uniform(lower_bound, upper_bound, (function.meta_data.n_variables,)))
                    param.set_bounds(lower_bound, upper_bound)
                    algorithm = optimizer(parametrization=param, budget=self.budget)
                    
                    algorithm.minimize(function)
                    function.reset()
                
                ioh_logger.close()
                with Path(ioh_logger.output_directory + f"/IOHprofiler_{const.PROB_NAMES[problem - 1]}.json").open() as metadata_file:
                    metadata = json.load(metadata_file)
                    loss = 0
                    for run in metadata['scenarios'][0]['runs']:
                        loss += run['best']['y']
                    loss = loss / len(metadata['scenarios'][0]['runs'])
                    total_loss += math.log10(loss)
        return total_loss