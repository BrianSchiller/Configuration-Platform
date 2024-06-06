from nevergrad.benchmark import Experiment as NevergradExperiment
from nevergrad.optimization.optimizerlib import base
from nevergrad.functions import ArtificialFunction
from ioh import get_problem, ProblemClass, Experiment

import csv
import math   

from settings import Settings


class Training:
    def __init__(
        self, problems: list[str] = None, dimensions: list[int] = None, budget: int = None, repetitions: int = None, nevergrad =  False
    ) -> None:
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

        # TODO: Find a more elegant solution for normalising results
        self.values = {}

        if nevergrad:
            # Open the CSV file and read its contents
            with open("normalize.csv", newline='') as csvfile:
                reader = csv.reader(csvfile)

                # Skip the header row
                next(reader)

                # Read the remaining rows and populate the values
                for row in reader:
                    func = row.pop(0)  # The first column is the function name
                    if func not in self.values:
                        self.values[func] = {"min": [], "max": []}
                    for i in range(0, len(row), 2):
                        self.values[func]["min"] = (float(row[i]))
                        self.values[func]["max"] = (float(row[i + 1]))

    def train(self, optimizer: base.ConfiguredOptimizer) -> float:
        exp = Experiment(
            algorithm = optimizer,                   # instance of optimization algorithm
            fids = self.problems,                      # list of problem id's
            iids = [1],                                # list of problem instances
            dims = self.dimensions,                                # list of problem dimensions
            problem_class = ProblemClass.BBOB,  # the problem type, function ids should correspond to problems of this type
            njobs = 1,                          # the number of parrellel jobs for running this experiment
            reps = self.repetitions,
            output_directory = 'TMP'                             # the number of repetitions for each (id x instance x dim)                    
        )
        result = exp.run()
        print(result)


    def train_nevergrad(self, optimizer: base.ConfiguredOptimizer) -> float:
        total_loss = 0
        updated_bounds = False

        for func in self.problems:
            for dim in self.dimensions:
                for _ in range(self.repetitions):
                    bbob_function = ArtificialFunction(name=func, block_dimension=dim, num_blocks=2)
                    result = NevergradExperiment(bbob_function, optimizer=optimizer, budget=self.budget, num_workers=1).run()
                    loss = result['loss']

                    if self.update_min_max(func, loss):
                        print(f"Updated min/max values for {func} with loss {loss}")
                        updated_bounds = True

                    normalized_loss = math.log10(loss)
                    # normalized_loss = self.normalize(func, loss)
                    total_loss += normalized_loss
                    # print(f"{func}, {dim}: {normalized_loss}")

        if updated_bounds:
            self.write_csv()
        
        return total_loss
    
    # Function to normalize the result
    def normalize(self, func, loss):
        min = self.values[func]["min"]
        max = self.values[func]["max"]
        return (loss - min) / (max - min)

    # Function to update min and max values
    def update_min_max(self, func, loss):
        if not self.values.get(func):
            self.values[func] = {"min": loss, "max": loss + 1}
        current_min = self.values[func]["min"]
        current_max = self.values[func]["max"]
        if loss < current_min:
            self.values[func]["min"] = loss
            return True
        if loss > current_max:
            self.values[func]["max"] = loss
            return True
        return False
    
    def write_csv(self):
        with open("normalize.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Function", "Min", "Max"])
            for func, data in self.values.items():
                min_val = data["min"]
                max_val = data["max"]
                writer.writerow([func, min_val, max_val])