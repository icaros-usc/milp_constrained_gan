from algos.milp.zelda.program import Program


def get_variable_names(model):
    """
    gets list of variable names in order of their index,
    can then perform following lookups:
    variable_index -> variable_name
    var_names[variable_index]

    variable_name -> variable_index
    reverse_lookup = dict(map(reversed, enumerate(var_names)))
    reverse_lookup[variable_name]

    :param model:
    :return:
    """
    stats = model.get_statistics()
    return [model.get_var_by_index[i] for i in range(stats.number_of_variables)]


def load_solution(model, solution):
    """
    Loads a given solution represented as a list of values into cplex
    :param model: docplex model
    :param solution: vector solution to load into cplex
    """

    # get number of variables from model stats
    stats = model.get_statistics()

    # stores upper and lower bounds for variables so they can be reset afterwards
    old_lbs = {}
    old_ubs = {}

    # read in old lb and ub to reset later
    # set lb and ub to the solution value
    for var_ind in range(stats.number_of_variables):
        var = model.get_var_by_index(var_ind)
        old_lbs[var] = var.lb
        old_ubs[var] = var.ub
        model.set_var_lb(var, solution[var_ind])
        model.set_var_ub(var, solution[var_ind])

    # solve instantaneously to set variables
    model.solve()

    # reset the upper and lower bounds so we don't mess anything up
    for var_ind in range(stats.number_of_variables):
        var = model.get_var_by_index(var_ind)
        model.set_var_lb(var, old_lbs[var])
        model.set_var_ub(var, old_ubs[var])


# example of usage
def main():
    # get example solution vector using model.solve but throw this away just to be sure
    random_program = Program()
    random_model = random_program._model
    random_model.solve()
    example_solution = random_model.solution.get_all_values()

    # here you have some program
    program = Program()
    model = program._model
    print("Model has solution before loading: ", model.solution is not None)
    load_solution(model, example_solution)
    print("Model has solution after loading: ", model.solution is not None)


if __name__ == '__main__':
    main()
