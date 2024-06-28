class Settings():
    #yabbob functions
    # functions = [
    #                 "hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek",
    #                 "deceptivemultimodal", "bucherastrigin", "multipeak",
    #                 "sphere", "doublelinearslope", "stepdoublelinearslope",
    #                 "cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid",
    #                 "discus", "bentcigar", "deceptiveillcond", "deceptivemultimodal", "deceptivepath"
    #             ]
    problems = [1,2]
    # problems = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    # dimension_sets = [[2], [3], [5]]
    dimension_sets = [[2]]
    instances = [1,2,3]
    budgets = [10]
    repetitions = 1
    trials = 5

    #Validation
    val_size = 5
    val_iterations = 3

    #Testing (test_size <= val_size)
    test_size = 3
    test_iterations = 5
