from test import Tester
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from models import MetaModelOnePlusOne, ChainMetaModelPowell, CMA, Cobyla, MetaModel, MetaModelFmin2
from training import Training

from pathlib import Path
import numpy as np
import settings


def run_test(dimensions, budget, model_name, directory, incumbent):
    trainings_function = Training(Path(*directory.parts[1:]), dimensions=dimensions, budget=budget)

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

    model_output = directory / model.name

    instances = [f"{problem}_{instance}" for problem in settings.problems for instance in settings.instances]
    index_dict = {instance: [i] for i, instance in enumerate(instances)}

    scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=10, 
                        output_directory=model_output, 
                        instances=instances,
                        instance_features=index_dict)
    
    smac = ACFacade(
        scenario,
        model.train, 
        overwrite=False, 
    )

    smac.optimize()

    test_configs = model.configspace.sample_configuration(size = settings.test_size)
    default_config = model.configspace.get_default_configuration()

    if model.name == "MetaModel" or model.name == "CMA" or model.name == "ChainMetaModelPowell":
        default_config["popsize"] = 4 + int(3 * np.log(dimensions[-1]))
    
    print(f"Running test for {model_name}, D: {dimensions}, B: {budget}")
    Tester(smac).test(incumbent, test_configs, default_config, model_output)


if __name__ == "__main__":
    configs = [
        {
            "model": "MetaModel",
            "budget": 999,
            "dimension": 15,
            "incumbent": """{
                "algorithm": "quad",
                "diagonal": True,
                "elitist": True,
                "fcmaes": False,
                "frequency_ratio": 0.08940571629592835,
                "high_speed": False,
                "popsize": 9,
                "popsize_factor": 9.20420081118838,
                "random_init": False,
                "scale": 9.237330289172416
            }""",
            "path": "Output/Final_D1015_B1510/B_999__D_15"
        },
        {
            "model": "MetaModelFmin2",
            "budget": 999,
            "dimension": 15,
            "incumbent": """{
                "algorithm": "quad",
                "frequency_ratio": 4.2172047025565396e-05,
                "random_restart": False
            }""",
            "path": "Output/Final_D1015_B1510/B_999__D_15"
        },
        {
            "model": "MetaModel",
            "budget": 1500,
            "dimension": 10,
            "incumbent": """{
                "algorithm": "quad",
                "diagonal": False,
                "elitist": True,
                "fcmaes": False,
                "frequency_ratio": 0.7540887058548547,
                "high_speed": True,
                "popsize": 23,
                "popsize_factor": 6.420191132969411,
                "random_init": True,
                "scale": 8.921031196089585
                }""",
            "path": "Output/Final_D1015_B1510/B_1500__D_10"
        },
    ]

    for config in configs:
        incumbent = eval(config["incumbent"])
        run_test([config["dimension"]], config["budget"], config["model"], Path(config["path"]), incumbent)


    
