import sys

from docplex.mp.model import Model


# Adds constraints that ensure exactly one object is present in each cell
#
# mdl:                the mip model
# all_objects:        a list of all object variables [[W_i], [P_i], ...]
def add_object_placement(mdl, all_objects):
    # Transpose the given matrix and ensure exactly one object per graph node
    for cur_node in zip(*all_objects):
        mdl.add_constraint(sum(cur_node) == 1)


# Adds reachability constraints to MIP
#
# mdl:                the mip model
# graph:              an adjacency list
# source_objects:     objects that must reach the sink objects [[P_i], ...]
# sink_objects:       objects that must be reached by the source objects [[K_i], [D_i], ...]
# blocking_objects:   a list of object types that impede movement [[W_i], ...]
#
# post condition: these constraints ensure that a path exists from some source
#                 object to all sink objects
def add_reachability(mdl, graph, source_objects, sink_objects, blocking_objects):
    # Transpose the blocking objects matrix so all blocking objects for
    # a given node are easily accessible.
    blocking = list(zip(*blocking_objects))

    # Setup a flow network for each edge in the graph
    n_nodes = len(graph)
    # Add a flow variable for each edge in the graph
    # flow: the flow leaving node i
    # rev: flow edges entering node i
    flow = [[] for i in range(n_nodes)]
    rev = [[] for i in range(n_nodes)]
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            f = mdl.integer_var(name='p_{}_{}'.format(i, j), lb=0, ub=n_nodes)
            flow[i].append(f)
            rev[j].append(f)

    # Add supply and demand variables for the source and sink
    supplies = []
    demands = []
    for i in range(n_nodes):
        f = mdl.integer_var(name='p_s_{}'.format(i), lb=0, ub=n_nodes)
        supplies.append(f)
        f = mdl.integer_var(name='p_{}_t'.format(i), lb=0, ub=1)
        demands.append(f)
    # Add a flow conservation constraint for each node (outflow == inflow)
    for i in range(n_nodes):
        mdl.add_constraint(supplies[i] + sum(rev[i]) == demands[i] + sum(flow[i]))

    # Add capacity constraints for each edge ensuring that no flow passes through a blocking object
    for i, neighbors in enumerate(flow):
        blocking_limits = [n_nodes * b for b in blocking[i]]
        for f in neighbors:
            mdl.add_constraint(f + sum(blocking_limits) <= n_nodes)

    # Place a demand at this object location if it contains a sink type object.
    sinks = list(zip(*sink_objects))
    for i in range(n_nodes):
        mdl.add_constraint(sum(sinks[i]) == demands[i])

    # Allow this node to have supply if it contains a source object
    sources = list(zip(*source_objects))
    for i in range(n_nodes):
        capacity = sum(n_nodes * x for x in sources[i])
        mdl.add_constraint(supplies[i] <= capacity)

    # For each target object, it should have at least two edges


# Adds edit distance cost function and constraints for fixing the level with minimal edits.
#
# graph:              an adjacency list denoting allowed movement
# objects:            a list [([(T_i, O_i)], Cm, Cc), ...] representing the cost of moving each
#                     object by one edge (Cm) and the cost of an add or delete (Cc).
#                     T_i represents the object variable at node i
#                     O_i is a boolean value denoting whether node i originally contained T_i.
def add_edit_distance(mdl, graph, objects, add_movement=True):
    costs = []
    if not add_movement:
        for objects_in_graph, cost_move, cost_change in objects:
            for cur_var, did_contain in objects_in_graph:
                if did_contain:
                    costs.append(cost_change * (1 - cur_var))
                else:
                    costs.append(cost_change * cur_var)

    else:
        for obj_id, (objects_in_graph, cost_move, cost_change) in enumerate(objects):

            # Setup a flow network for each edge in the graph
            n_nodes = len(graph)
            # Add a flow variable for each edge in the graph
            # flow: the flow leaving node i
            # rev: flow edges entering node i
            flow = [[] for i in range(n_nodes)]
            rev = [[] for i in range(n_nodes)]
            for i, neighbors in enumerate(graph):
                for j in neighbors:
                    f = mdl.integer_var(name='edit({})_{}_{}'.format(obj_id, i, j), lb=0, ub=n_nodes)
                    costs.append(cost_move * f)
                    flow[i].append(f)
                    rev[j].append(f)

            # Add a supply if the object was in the current location.
            # Demands go everywhere.
            demands = []
            waste = []
            num_supply = 0
            for i, (cur_var, did_contain) in enumerate(objects_in_graph):
                f = mdl.integer_var(name='edit({})_{}_t'.format(obj_id, i), lb=0, ub=1)
                demands.append(f)

                # Add a second sink that eats any flow that doesn't find a home.
                # The cost of this flow is deleting the object.
                f = mdl.integer_var(name='edit({})_{}_t2'.format(obj_id, i), lb=0, ub=n_nodes)
                costs.append(cost_change * f)
                waste.append(f)

                # Flow conservation constraint (inflow == outflow)
                if did_contain:
                    # If we had a piece of this type in the current node, match it to the outflow
                    mdl.add_constraint(1 + sum(rev[i]) == demands[i] + sum(flow[i]) + waste[i])
                    num_supply += 1
                else:
                    mdl.add_constraint(sum(rev[i]) == demands[i] + sum(flow[i]) + waste[i])

            # Ensure we place a piece of this type to match it to the demand.
            for (cur_var, did_contain), node_demand in zip(objects_in_graph, demands):
                mdl.add_constraint(node_demand <= cur_var)

            # Ensure that the source and sink have the same flow.
            mdl.add_constraint(num_supply == sum(demands) + sum(waste))

    mdl.minimize(mdl.sum(costs))


def fix_pacman_level(level):
    n = len(level)
    m = len(level[0])

    deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # Build an adjacency list for the dynamics of Zelda
    n_nodes = n * m
    adj = [[] for _ in range(n_nodes)]

    hor_border_nodes = []
    left_border_nodes = []
    right_border_nodes = []
    for i in range(n_nodes):
        cur_row = i // m
        cur_col = i % m
        is_hor_border = False
        is_left_border = False
        is_right_border = False
        for dr, dc in deltas:
            nxt_row = cur_row + dr
            nxt_col = cur_col + dc
            if 0 <= nxt_row < n and 0 <= nxt_col < m:
                j = nxt_row * m + nxt_col
                adj[i].append(j)
            elif nxt_row < 0 or nxt_row >= n:
                is_hor_border = True
            elif nxt_col < 0:
                is_left_border = True
                # we should add the block on the other side as the adjacent block
                adj[i].append(nxt_row * m + m - 1)
            elif nxt_col >= m:
                is_right_border = True
                # we should add the block on the other side as the adjacent block
                adj[i].append(nxt_row * m + 0)
        if is_hor_border:
            hor_border_nodes.append(i)
        if is_left_border:
            left_border_nodes.append(i)
        if is_right_border:
            right_border_nodes.append(i)

    new_level = []

    with Model() as mdl:
        # Binary variables for each object type
        F = [mdl.integer_var(name='F_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # floor
        P = [mdl.integer_var(name='P_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # power pellet
        Pe = [mdl.integer_var(name='Pe_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # normal pellet
        A = [mdl.integer_var(name='A_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # avator
        E1 = [mdl.integer_var(name='E1_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # An enemy type
        E2 = [mdl.integer_var(name='E2_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # An enemy type
        E3 = [mdl.integer_var(name='E3_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # An enemy type
        E4 = [mdl.integer_var(name='E4_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # An enemy type
        Fr = [mdl.integer_var(name='Fr_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # The fruit pellet
        W = [mdl.integer_var(name='W_{}'.format(i), lb=0, ub=1) for i in range(n_nodes)]  # The wall
        all_objects = [F, P, Pe, A, E1, E2, E3, E4, Fr, W]
        all_characters = ['+', '0', '.', 'A', '1', '2', '3', '4', 'f', 'w']

        # Helper function that maps node ids to object characters
        def get_char_from_variables(solution, node_id):
            for object_var, label in zip(all_objects, all_characters):
                if solution.get_value(object_var[node_id]) == 1:
                    return label
            return '?'

        # Ensure that exactly one object is present in each cell
        add_object_placement(mdl, all_objects)

        # PacMan specific constraints
        #    ----
        # Ensure that all cells on the horizontal boundary are walls
        hor_borders_as_walls = [W[i] for i in hor_border_nodes]
        mdl.add_constraint(sum(hor_borders_as_walls) == len(hor_border_nodes))
        # Ensure that the nodes on the left boundary are the same as the nodes on the right boundary
        for i, j in zip(left_border_nodes, right_border_nodes):
            mdl.add_constraint(F[i] == F[j])
            mdl.add_constraint(P[i] == P[j])
            mdl.add_constraint(Pe[i] == Pe[j])
            mdl.add_constraint(A[i] == A[j])
            mdl.add_constraint(E1[i] == E1[j])
            mdl.add_constraint(E2[i] == E2[j])
            mdl.add_constraint(E3[i] == E3[j])
            mdl.add_constraint(E4[i] == E4[j])
            mdl.add_constraint(Fr[i] == Fr[j])
            mdl.add_constraint(W[i] == W[j])
        # Ensure that there are exactly one: enemy1, enemy2, enemy3, enemy4, player
        mdl.add_constraint(sum(A) == 1)
        mdl.add_constraint(sum(E1) == 1)
        mdl.add_constraint(sum(E2) == 1)
        mdl.add_constraint(sum(E3) == 1)
        mdl.add_constraint(sum(E4) == 1)

        add_reachability(mdl, adj, [A], [F, P, Pe, Fr, E1, E2, E3, E4], [W])

        # Examine the level and determine the edit distance costs
        objects = []
        cost_move = 1
        cost_change = 10
        for cur_object, cur_label in zip(all_objects, all_characters):
            objects_in_graph = []
            for r in range(n):
                for c in range(m):
                    i = r * m + c
                    objects_in_graph.append((cur_object[i], cur_label == level[r][c]))
            if cur_label in ['+', '0', '.', 'f', 'w']:
                cost_move = 1
                cost_change = 3
            else:
                cost_move = 10
                cost_change = 10
            objects.append((objects_in_graph, cost_move, cost_change))

        # Add edit distance constraints
        add_edit_distance(mdl, adj, objects)

        # for every node it surrounding nodes will have at most len(surrounding nodes) - 2 walls
        for i in range(n_nodes):
            sur_walls = [W[j] for j in adj[i]]
            sur_pes = [Pe[j] for j in adj[i]]
            sur_frs = [Fr[j] for j in adj[i]]
            sur_pos = [P[j] for j in adj[i]]
            mdl.add_constraint(sum(sur_walls) <= len(adj[i]) - 2 * (F[i] + P[i] + Pe[i] + Fr[i]))
            mdl.add_constraint(sum(sur_pes) + sum(sur_frs) + sum(sur_pos) >= Pe[i] * 2)

        # there will be at most 4 fruits and 4 power pellets
        mdl.add_constraint(sum(P) <= 4)
        mdl.add_constraint(sum(Fr) <= 4)

        # there shouldn't be a reachable nodes block
        for i in range(n_nodes):
            if i / m >= n - 1:
                continue
            if i % m == m - 1:
                mdl.add_constraint(P[i] + Pe[i] + Fr[i] + P[i - m + 1] + Pe[i -m + 1] + Fr[i - m + 1]
                                   + P[i + m] + Pe[i + m] + Fr[i + m] + P[i + 1] + Pe[
                                       i + 1] + Fr[i + 1] <= 3)
                continue
            mdl.add_constraint(P[i] + Pe[i] + Fr[i] + P[i + 1] + Pe[i + 1] + Fr[i + 1]
                               + P[i + m] + Pe[i + m] + Fr[i + m] + P[i + m + 1] + Pe[i + m + 1] + Fr[i + m + 1] <= 3)

        # print(len(list(mdl.iter_variables())))
        # print(len(list(mdl.iter_constraints())))

        solution = mdl.solve()

        # print(solution)
        # from IPython import embed
        # embed()
        # Extract the new level from the MIP
        for r in range(n):
            line = []
            for c in range(m):
                i = r * m + c
                line.append(get_char_from_variables(solution, i))
            new_level.append(''.join(line))

    return new_level


def main():
    with open(sys.argv[1], 'r') as f:
        level = [line.strip() for line in f]
        # from IPython import embed
        # embed()
        new_level = fix_pacman_level(level)
        for line in new_level:
            print(line)


if __name__ == "__main__":
    main()
