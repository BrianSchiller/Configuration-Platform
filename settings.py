metaModelOnePlusOne = "MetaModelOnePlusOne"
chainMetaModelPowell = "ChainMetaModelPowell"
cma = "CMA"
cobyla = "Cobyla"
metaModel = "MetaModel"
metaModelFmin2 = "MetaModelFmin2"
models = [cma, metaModelOnePlusOne, chainMetaModelPowell, metaModel, metaModelFmin2]

partition = "Kathleen"
time = "168:00:00"

# Settings
problems = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
dimension_sets = [[2, 3, 5, 10, 15]]
instances = [1,2,3,4,5,6,7,8]
budgets = [200, 300, 500]
repetitions = 1
trials = 8000

store_problem_results = False
problem_result_dir = None

#Testing (test_size <= val_size)
test_size = 1
test_iterations = 3

#Logging
log_folder = "/storage/work/schiller/logs/configs"
