from smac import AlgorithmConfigurationFacade as ACFacade
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
import warnings
# Ignore all DeprecationWarnings from the smac package
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nevergrad.optimization.optimizerlib import ParametrizedOnePlusOne, ParametrizedMetaModel
import matplotlib.pyplot as plt

from models import MetaModelOnePlusOne
from training import Training


# class MetaModelOnePlusOne:
#     @property
#     def configspace(self) -> ConfigurationSpace:
#         cs = ConfigurationSpace(seed=0)

#         # Meta Model
#         frequency_ratio = Float("frequency_ratio", bounds=(0, 1), default=0.9)
        
#         # ParametrizedOnePlusOne
#         noise_handling = Categorical("noise_handling", 
#                                     ["random", "optimistic"],
#                                     default="random")
#         noise_frequency = Float("noise_frequency", bounds=(0, 1), default=0.05)
    
#         mutation = Categorical("mutation", 
#                                 ["gaussian", "cauchy", "discrete", "discreteBSO", "fastga", "doublefastga", "rls", 
#                                 "portfolio", "lengler", "lengler2", "lengler3", "lenglerhalf", "lenglerfourth"],
#                                 default="gaussian")
#         crossover = Categorical("crossover", [True, False], default=False)
#         use_pareto = Categorical("use_pareto", [True, False], default=False)
#         sparse = Categorical("sparse", [True, False], default=False)
#         smoother = Categorical("smoother", [True, False], default=False)
        
#         cs.add_hyperparameters([frequency_ratio, noise_handling, noise_frequency, mutation, crossover, use_pareto, sparse, smoother])

#         return cs

#     def train(self, config: Configuration, seed: int = 0) -> float:
#         OnePlusOne = ParametrizedOnePlusOne(noise_handling=(config["noise_handling"], config["noise_frequency"]),
#                                             mutation=config["mutation"], crossover=config["crossover"], use_pareto=config["use_pareto"],
#                                             sparse=config["sparse"], smoother=config["smoother"])
#         MetaModelOnePlusOne = ParametrizedMetaModel(multivariate_optimizer=OnePlusOne, frequency_ratio=config["frequency_ratio"])

#         bbob_functions = [
#             'sphere', 'ellipsoid', 'rastrigin', 'bucherastrigin', 'rosenbrock',
#         ]

#         total_loss = 0

#         for func in bbob_functions:
#             bbob_function = ArtificialFunction(name=func, block_dimension=5)
#             result = Experiment(bbob_function, optimizer=MetaModelOnePlusOne, budget=10, num_workers=1).run()
#             total_loss += result['loss']
#             print(func + " " + str(result['loss']))
#         #{'loss': 6.881955953960828, 'elapsed_budget': 10, 'elapsed_time': 0.010907888412475586, 'error': '', 'pseudotime': 10.0, 'num_objectives': 1, 'seed': -1, 
#         #'name': 'sphere', 'block_dimension': 5, 'num_blocks': 1, 'useless_variables': 0, 'noise_level': 0, 'noise_dissymmetry': False, 'rotation': False, 
#         #'translation_factor': 1.0, 'hashing': False, 'aggregator': 'max', 'split': False, 'bounded': False, 'expo': 1.0, 'zero_pen': False, 'function_class': 'ArtificialFunction', 
#         #'useful_dimensions': 5, 'discrete': False, 'parametrization': '', 'dimension': 5, 'budget': 10, 'num_workers': 1, 'batch_mode': True, 'optimizer_name': "ParametrizedMetaModel(multivariate_optimizer=ParametrizedOnePlusOne(noise_handling=('random', 0.05)))"}
#         print("total: " + str(total_loss))
#         print()
        
#         return total_loss

if __name__ == "__main__":
    trainings_function = Training(None,[2,3,5],10,3)

    model = MetaModelOnePlusOne(trainings_function)

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, n_trials=10)

    # Now we use SMAC to find the best hyperparameters
    smac = ACFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

    incumbent = smac.optimize()
    print(incumbent)

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print()
    print(f"Default cost: {default_cost}")
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print()
    print(f"Default cost: {default_cost}")
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print()
    print(f"Default cost: {default_cost}")
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print()
    print(f"Default cost: {default_cost}")
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print()
    print(f"Default cost: {default_cost}")
    #print(model.configspace.get_default_configuration())

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print()
    print(f"Incumbent cost: {incumbent_cost}")
    incumbent_cost = smac.validate(incumbent)
    print()
    print(f"Incumbent cost: {incumbent_cost}")
    incumbent_cost = smac.validate(incumbent)
    print()
    print(f"Incumbent cost: {incumbent_cost}")
    incumbent_cost = smac.validate(incumbent)
    print()
    print(f"Incumbent cost: {incumbent_cost}")
    incumbent_cost = smac.validate(incumbent)
    print()
    print(f"Incumbent cost: {incumbent_cost}")
    print(incumbent)


    run_history = smac.runhistory
    # OrderedDict([(TrialKey(config_id=1, instance=None, seed=209652396, budget=None), 
    # TrialValue(cost=126867.7579791603, time=0.07253432273864746, status=<StatusType.SUCCESS: 1>, starttime=1717162944.2159677, endtime=1717162944.2901123, additional_info={})),
    fitness_values = [entry.cost for entry in run_history._data.values()]

    plt.plot(fitness_values, label="Fitness values over iterations")
    
    # Mark default configuration
    plt.scatter(0, fitness_values[0], color='red', marker='o', s=100, label='Default configuration')
    
    # Mark configuration with lowest overall loss (incumbent)
    min_loss = min(fitness_values)
    min_loss_index = fitness_values.index(min_loss)
    plt.scatter(min_loss_index, min_loss, color='blue', marker='*', s=200, label='Incumbent configuration')
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Loss)')
    plt.title('Optimization Process')
    plt.legend()
    plt.grid(True)
    plt.show()