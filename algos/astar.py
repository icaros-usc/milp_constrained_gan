def search(grid, init, goal, cost, delta, heuristic):
    path = []
    val = 1

    visited = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    visited[init[0]][init[1]] = 1

    expand = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    expand[init[0]][init[1]] = 0

    x = init[0]
    y = init[1]
    g = 0
    f = g + heuristic[x][y]

    minList = [f, g, x, y]

    while minList[2:] != goal:
        for i in range(len(delta)):
            x2 = x + delta[i][0]
            y2 = y + delta[i][1]
            if 0 <= x2 < len(grid) and 0 <= y2 < len(grid[0]):
                if visited[x2][y2] == 0 and grid[x2][y2] == 0:
                    g2 = g + cost
                    f2 = g2 + heuristic[x2][y2]
                    path.append([f2, g2, x2, y2])
                    visited[x2][y2] = 1

        if not path:
            return 'fail', expand

        del minList[:]
        minList = min(path)
        path.remove(minList)
        x = minList[2]
        y = minList[3]
        g = minList[1]
        expand[x][y] = val
        val += 1

    return minList, expand


if __name__ == '__main__':
    grid = [[0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]

    heuristic = [[9, 8, 7, 6, 5, 4],
                 [8, 7, 6, 5, 4, 3],
                 [7, 6, 5, 4, 3, 2],
                 [6, 5, 4, 3, 2, 1],
                 [5, 4, 3, 2, 1, 0]]

    init = [0, 0]
    goal = [len(grid) - 1, len(grid[0]) - 1]
    cost = 1

    delta = [[-1, 0],
             [0, -1],
             [1, 0],
             [0, 1]]

    delta_name = ['^', '<', 'V', '>']
    path, expand = search(grid, init, goal, cost, delta, heuristic)

    print(path)

    print(expand)
