class Settings():
    #yabbob functions
    # functions = [
    #                 "hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek",
    #                 "deceptivemultimodal", "bucherastrigin", "multipeak",
    #                 "sphere", "doublelinearslope", "stepdoublelinearslope",
    #                 "cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid",
    #                 "discus", "bentcigar", "deceptiveillcond", "deceptivemultimodal", "deceptivepath"
    #             ]
    problems = [1,2,3]
    dimensions = [5]
    budget = 6
    repetitions = 2
    trials = 10

    #Validation
    val_size = 5
    val_iterations = 1

    #Testing (test_size <= val_size)
    test_size = 3
    test_iterations = 2
