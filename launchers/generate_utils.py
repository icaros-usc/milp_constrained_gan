def zelda_gan_output_to_txt(input):
    grid = [[]]
    for i in range(9):
        for j in range(13):
            if input[i][j] == 0:
                grid[i].append('w')
            elif input[i][j] == 1:
                grid[i].append('.')
            elif input[i][j] == 2:
                grid[i].append('+')
            elif input[i][j] == 3:
                grid[i].append('g')
            elif input[i][j] == 4:
                grid[i].append('1')
            elif input[i][j] == 5:
                grid[i].append('2')
            elif input[i][j] == 6:
                grid[i].append('3')
            elif input[i][j] == 7:
                grid[i].append('A')
            else:
                raise NotImplementedError('Can not recognize the index type.')
        if i < 8:
            grid.append([])

    return grid


def zelda_milp_vars2grid(si, program):
    grid = [[1 for _ in range(13)] for _ in range(9)]
    for i in range(9):
        for j in range(13):
            ind = i * 13 + j
            if si.get_value(program.W[ind]) == 1:
                grid[i][j] = 0
            elif si.get_value(program.K[ind]) == 1:
                grid[i][j] = 2
            elif si.get_value(program.G[ind]) == 1:
                grid[i][j] = 3
            elif si.get_value(program.E1[ind]) == 1:
                grid[i][j] = 4
            elif si.get_value(program.E2[ind]) == 1:
                grid[i][j] = 5
            elif si.get_value(program.E3[ind]) == 1:
                grid[i][j] = 6
            elif si.get_value(program.P[ind]) == 1:
                grid[i][j] = 7
    return grid


def pacman_gan_output_to_txt(input):
    grid = [[]]
    for i in range(31):
        for j in range(28):
            if input[i][j] == 0:
                grid[i].append('+')
            elif input[i][j] == 1:
                grid[i].append('0')
            elif input[i][j] == 2:
                grid[i].append('.')
            elif input[i][j] == 3:
                grid[i].append('A')
            elif input[i][j] == 4:
                grid[i].append('1')
            elif input[i][j] == 5:
                grid[i].append('2')
            elif input[i][j] == 6:
                grid[i].append('3')
            elif input[i][j] == 7:
                grid[i].append('4')
            elif input[i][j] == 8:
                grid[i].append('f')
            elif input[i][j] == 9:
                grid[i].append('w')
            else:
                raise NotImplementedError('Can not recognize the index type.')
        if i < 30:
            grid.append([])

    return grid
