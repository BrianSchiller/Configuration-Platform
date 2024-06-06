from ConfigSpace import Configuration, ConfigurationSpace, Float, Categorical, Integer, ForbiddenEqualsClause, ForbiddenAndConjunction
from nevergrad.benchmark import Experiment
from nevergrad.optimization.optimizerlib import base, ParametrizedOnePlusOne, ParametrizedMetaModel, ParametrizedCMA, NonObjectOptimizer, Chaining
from nevergrad.functions import ArtificialFunction
from training import Training

import csv

class Model:
    def __init__(
        self, trainings_function: Training
    ) -> None:
        self.trainings_function = trainings_function


class Cobyla(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        random_restart = Categorical("random_restart", [True, False], default=False)

        cs.add_hyperparameters([random_restart])

        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        Cobyla = NonObjectOptimizer(method="COBYLA", random_restart=config["random_restart"])

        return self.trainings_function.train(Cobyla)
    
    name = 'Cobyla'
    
class MetaModelFmin2(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        algorithm = Categorical("algorithm", ["quad", "neural", "svr", "rf"], default="quad")

        # CmaFmin2
        random_restart = Categorical("random_restart", [True, False], default=False)

        cs.add_hyperparameters([frequency_ratio, algorithm, random_restart])

        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        CmaFmin2 = NonObjectOptimizer(method="CmaFmin2", random_restart=config["random_restart"])

        # Could also make algorithm of ParamMetaModel configurable
        MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2, algorithm=config["algorithm"], frequency_ratio=config["frequency_ratio"])

        return self.trainings_function.train(MetaModelFmin2) 
    
    name = "MetaModelFmin2"
    
class MetaModelOnePlusOne(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        algorithm = Categorical("algorithm", ["quad", "neural", "svr", "rf"], default="quad")

        # ParametrizedOnePlusOne
        noise_handling = Categorical("noise_handling", ["random", "optimistic"], default="random")
        noise_frequency = Float("noise_frequency", bounds=(0, 1), default=0.05)
    
        mutation = Categorical("mutation", 
                                ["gaussian", "cauchy", "discrete", "discreteBSO", "fastga", "doublefastga", "rls", 
                                "portfolio", "lengler", "lengler2", "lengler3", "lenglerhalf", "lenglerfourth"],
                                default="gaussian")
        crossover = Categorical("crossover", [True, False], default=False)
        use_pareto = Categorical("use_pareto", [True, False], default=False)
        sparse = Categorical("sparse", [True, False], default=False)
        smoother = Categorical("smoother", [True, False], default=False)
        
        cs.add_hyperparameters([frequency_ratio, algorithm, noise_handling, noise_frequency, mutation, crossover, use_pareto, sparse, smoother])

        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        OnePlusOne = ParametrizedOnePlusOne(
            noise_handling=(config["noise_handling"],
            config["noise_frequency"]),
            mutation=config["mutation"], 
            crossover=config["crossover"], 
            use_pareto=config["use_pareto"],
            sparse=config["sparse"], 
            smoother=config["smoother"]
        )

        # Could also make algorithm of ParamMetaModel configurable
        MetaModelOnePlusOne = ParametrizedMetaModel(multivariate_optimizer=OnePlusOne, algorithm=config["algorithm"], frequency_ratio=config["frequency_ratio"])

        return self.trainings_function.train(MetaModelOnePlusOne) 
    
    name = "MetaModelOnePlusOne"

class MetaModel(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        algorithm = Categorical("algorithm", ["quad", "neural", "svr", "rf"], default="quad")
        
        # ParametrizedCMA
        scale = Float("scale", bounds=(0.1, 10.0), default=1.0)  # Assuming reasonable bounds for scale
        elitist = Categorical("elitist", [True, False], default=False)
        popsize = Integer("popsize", bounds=(2, 1000), default=10)  # Assuming reasonable bounds for population size
        popsize_factor = Float("popsize_factor", bounds=(1.0, 10.0), default=3.0)
        diagonal = Categorical("diagonal", [True, False], default=False)
        high_speed = Categorical("high_speed", [True, False], default=False)
        fcmaes = Categorical("fcmaes", [True, False], default=False)
        random_init = Categorical("random_init", [True, False], default=False)

        cs.add_hyperparameters([frequency_ratio, algorithm, scale, elitist, popsize, popsize_factor, diagonal, high_speed, fcmaes, random_init])

        forbidden_clause = ForbiddenAndConjunction(
            ForbiddenEqualsClause(diagonal, True),
            ForbiddenEqualsClause(fcmaes, True)
        )
        cs.add_forbidden_clause(forbidden_clause)

        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        CMA = ParametrizedCMA(
            scale=config["scale"],
            elitist=config["elitist"],
            popsize=config["popsize"],
            popsize_factor=config["popsize_factor"],
            diagonal=config["diagonal"],
            high_speed=config["high_speed"],
            fcmaes=config["fcmaes"],
            random_init=config["random_init"],
        )

        MetaModel = ParametrizedMetaModel(multivariate_optimizer=CMA, algorithm=config["algorithm"], frequency_ratio=config["frequency_ratio"])

        return self.trainings_function.train(MetaModel) 
    
    name = "MetaModel"

class CMA(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        scale = Float("scale", bounds=(0.1, 10.0), default=1.0)  # Assuming reasonable bounds for scale
        elitist = Categorical("elitist", [True, False], default=False)
        popsize = Integer("popsize", bounds=(2, 1000), default=10)  # Assuming reasonable bounds for population size
        popsize_factor = Float("popsize_factor", bounds=(1.0, 10.0), default=3.0)
        diagonal = Categorical("diagonal", [True, False], default=False)
        high_speed = Categorical("high_speed", [True, False], default=False)
        fcmaes = Categorical("fcmaes", [True, False], default=False)
        random_init = Categorical("random_init", [True, False], default=False)

        cs.add_hyperparameters([scale, elitist, popsize, popsize_factor, diagonal, high_speed, fcmaes, random_init])

        forbidden_clause = ForbiddenAndConjunction(
            ForbiddenEqualsClause(diagonal, True),
            ForbiddenEqualsClause(fcmaes, True)
        )
        cs.add_forbidden_clause(forbidden_clause)

        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        CMA = ParametrizedCMA(
            scale=config["scale"],
            elitist=config["elitist"],
            popsize=config["popsize"],
            popsize_factor=config["popsize_factor"],
            diagonal=config["diagonal"],
            high_speed=config["high_speed"],
            fcmaes=config["fcmaes"],
            random_init=config["random_init"],
        )

        return self.trainings_function.train(CMA) 
    
    name = "CMA"

class ChainMetaModelPowell(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Powell
        random_restart = Categorical("random_restart", [True, False], default=False)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        algorithm = Categorical("algorithm", ["quad", "neural", "svr", "rf"], default="quad")
        
        # Additional Parameters
        scale = Float("scale", bounds=(0.1, 10.0), default=1.0)  # Assuming reasonable bounds for scale
        elitist = Categorical("elitist", [True, False], default=False)
        popsize = Integer("popsize", bounds=(2, 1000), default=10) 
        popsize_factor = Float("popsize_factor", bounds=(1.0, 10.0), default=3.0)
        diagonal = Categorical("diagonal", [True, False], default=False)
        high_speed = Categorical("high_speed", [True, False], default=False)
        fcmaes = Categorical("fcmaes", [True, False], default=False)
        random_init = Categorical("random_init", [True, False], default=False)

        cs.add_hyperparameters([random_restart, frequency_ratio, algorithm, scale, elitist, popsize, popsize_factor, diagonal, high_speed, fcmaes, random_init])

        forbidden_clause = ForbiddenAndConjunction(
            ForbiddenEqualsClause(diagonal, True),
            ForbiddenEqualsClause(fcmaes, True)
        )
        cs.add_forbidden_clause(forbidden_clause)

        return cs
    
    def train(self, config: Configuration, seed: int = 0) -> float:
        CMA = ParametrizedCMA(
            scale=config["scale"],
            elitist=config["elitist"],
            popsize=config["popsize"],
            popsize_factor=config["popsize_factor"],
            diagonal=config["diagonal"],
            high_speed=config["high_speed"],
            fcmaes=config["fcmaes"],
            random_init=config["random_init"],
        )

        MetaModel = ParametrizedMetaModel(multivariate_optimizer=CMA, algorithm=config["algorithm"], frequency_ratio=config["frequency_ratio"])
        Powell = NonObjectOptimizer(method="Powell", random_restart=config["random_restart"])

        # TODO: Configure chaining budget?
        ChainMetaModelPowell = Chaining([MetaModel, Powell], ["half"])

        return self.trainings_function.train(ChainMetaModelPowell)
    
    name = "ChainMetaModelPowell"
