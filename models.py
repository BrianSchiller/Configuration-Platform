from ConfigSpace import Configuration, ConfigurationSpace, Float, Categorical, Integer, ForbiddenEqualsClause, ForbiddenAndConjunction, NotEqualsCondition
from nevergrad.optimization.base import ConfiguredOptimizer
from nevergrad.optimization.optimizerlib import base, ParametrizedOnePlusOne, ParametrizedMetaModel, ParametrizedCMA, NonObjectOptimizer, Chaining
from training import Training

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
    
    def train(self, config: Configuration, instance, seed: int = 0) -> float:
        Cobyla = NonObjectOptimizer(method="COBYLA", random_restart=config["random_restart"])

        return self.trainings_function.train(Cobyla, self.name)
    
    name = 'Cobyla'
    
class MetaModelFmin2(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        # Basically any other algorithm than quad makes performance skyrocket
        algorithm = Categorical("algorithm", ["quad", "svr", "rf"], default="quad")

        # CmaFmin2
        random_restart = Categorical("random_restart", [True, False], default=False)

        cs.add_hyperparameters([frequency_ratio, algorithm, random_restart])

        return cs
    
    def train(self, config: Configuration, instance, seed: int = 0) -> float:
        CmaFmin2 = NonObjectOptimizer(method="CmaFmin2", random_restart=config["random_restart"])

        # Could also make algorithm of ParamMetaModel configurable
        MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2, algorithm=config["algorithm"], frequency_ratio=config["frequency_ratio"])

        return self.trainings_function.train(MetaModelFmin2, instance, self.name) 
    
    name = "MetaModelFmin2"
    
class MetaModelOnePlusOne(Model):
    name = "MetaModelOnePlusOne"

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        algorithm = Categorical("algorithm", ["quad", "svr", "rf"], default="quad")
        # Option neural leads to non-convergence errors
        # algorithm = Categorical("algorithm", ["quad", "neural", "svr", "rf"], default="quad")

        # ParametrizedOnePlusOne
        noise_handling = Categorical("noise_handling", ["None", "random", "optimistic"], default="None")
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

        # Make noise_frequency an active hyperparameter if noise_handling has not the value "None"
        cond = NotEqualsCondition(cs['noise_frequency'], cs['noise_handling'], "None")
        cs.add_condition(cond)

        return cs
    
    def train(self, config: Configuration, instance, seed: int = 0) -> float:
        if config["noise_handling"] == "None":
            noise_handling = None
        else:
            noise_handling = (config["noise_handling"], config["noise_frequency"])

        OnePlusOne = ParametrizedOnePlusOne(
            noise_handling=noise_handling,
            mutation=config["mutation"], 
            crossover=config["crossover"], 
            use_pareto=config["use_pareto"],
            sparse=config["sparse"], 
            smoother=config["smoother"]
        )

        MetaModelOnePlusOne = ParametrizedMetaModel(multivariate_optimizer=OnePlusOne, algorithm=config["algorithm"], frequency_ratio=config["frequency_ratio"])

        return self.trainings_function.train(MetaModelOnePlusOne, instance, self.name) 
    

class MetaModel(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        algorithm = Categorical("algorithm", ["quad", "svr", "rf"], default="quad")
        # Option neural leads to non-convergence errors
        # algorithm = Categorical("algorithm", ["quad", "neural", "svr", "rf"], default="quad")
        
        # ParametrizedCMA
        scale = Float("scale", bounds=(0.1, 10.0), default=1.0)  # Assuming reasonable bounds for scale
        elitist = Categorical("elitist", [True, False], default=False)
        # default depends on dimension {2: 6, 3: 7, 5: 8, 10: 10, 15: 11}
        popsize = Integer("popsize", bounds=(2, 100), default=8)
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
    
    def train(self, config: Configuration, instance, seed: int = 0) -> float:
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

        return self.trainings_function.train(MetaModel, instance, self.name) 
    
    name = "MetaModel"

class CMA(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        scale = Float("scale", bounds=(0.1, 10.0), default=1.0)  # Assuming reasonable bounds for scale
        elitist = Categorical("elitist", [True, False], default=False)
        # default depends on dimension {2: 6, 3: 7, 5: 8, 10: 10, 15: 11}
        popsize = Integer("popsize", bounds=(2, 100), default=8)  # Assuming reasonable bounds for population size
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
    
    def train(self, config: Configuration, instance, seed: int = 0) -> float:
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

        return self.trainings_function.train(CMA, instance, self.name) 
    
    name = "CMA"

class ChainMetaModelPowell(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)

        # Powell
        random_restart = Categorical("random_restart", [True, False], default=False)

        # Meta Model
        frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        algorithm = Categorical("algorithm", ["quad", "svr", "rf"], default="quad")
        # algorithm = Categorical("algorithm", ["quad", "neural", "svr", "rf"], default="quad")
        
        # Additional Parameters
        scale = Float("scale", bounds=(0.1, 10.0), default=1.0)  # Assuming reasonable bounds for scale
        elitist = Categorical("elitist", [True, False], default=False)
        popsize = Integer("popsize", bounds=(2, 100), default=8) 
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
    
    def train(self, config: Configuration, instance, seed: int = 0) -> float:
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

        return self.trainings_function.train(ChainMetaModelPowell, instance, self.name)

    name = "ChainMetaModelPowell"
