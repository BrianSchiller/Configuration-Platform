class Settings():
    #yabbob functions
    # functions = [
    #                 "hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek",
    #                 "deceptivemultimodal", "bucherastrigin", "multipeak",
    #                 "sphere", "doublelinearslope", "stepdoublelinearslope",
    #                 "cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid",
    #                 "discus", "bentcigar", "deceptiveillcond", "deceptivemultimodal", "deceptivepath"
    #             ]
    problems = [1,2,3,4,5,6]
    dimensions = [5]
    instances = [1,2,3]
    #Why tf is causing budget > 10 errors?
    budget = 50
    repetitions = 1
    trials = 100

    #Validation
    val_size = 1
    val_iterations = 1

    #Testing (test_size <= val_size)
    test_size = 1
    test_iterations = 1
