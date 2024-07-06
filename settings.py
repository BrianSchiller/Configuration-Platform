metaModelOnePlusOne = "MetaModelOnePlusOne"
chainMetaModelPowell = "ChainMetaModelPowell"
cma = "CMA"
cobyla = "Cobyla"
metaModel = "MetaModel"
metaModelFmin2 = "MetaModelFmin2"
# models = [cma, metaModelOnePlusOne, chainMetaModelPowell, metaModel, metaModelFmin2]
models = [metaModelOnePlusOne]

# Settings
problems = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
dimension_sets = [[3]]
instances = [1,2,3,4,5,6,7,8]
budgets = [200]
repetitions = 1
trials = 500

store_problem_results = False
problem_result_dir = None

#Validation
val_size = 5
val_iterations = 3

#Testing (test_size <= val_size)
test_size = 3
test_iterations = 5

#Logging
log_folder = "/storage/work/schiller/logs/configs"
