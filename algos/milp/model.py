from docplex.mp.model import Model


class Program(object):
    def __init__(self):
        """This class is a class for storing all the information about the milp we will use."""
        self._model = Model(name='zelda_heuristic_constraints')
        self._grid_width = 13
        self._grid_height = 9
        self.build_program()
        self._model.print_information()

    def set_objective(self, parameters):
        # self._model.maximize()
        pass

    def solve(self):
        return self._model.solve()

    def build_program(self):
        """Use this function to build the model."""
        N = self._grid_width * self._grid_height
        # First define variables
        # wall indicator variables, w_i in {0, 1}
        W = []
        for i in range(N):
            W.append(self._model.integer_var(name='w_{}'.format(i), ub=1))
        # door indicator variables, g_i in {0, 1}
        G = []
        for i in range(N):
            G.append(self._model.integer_var(name='g_{}'.format(i), ub=1))
        # key indicator variables, k_i in {0, 1}
        K = []
        for i in range(N):
            K.append(self._model.integer_var(name='k_{}'.format(i), ub=1))
        # player indicator variables, p_i in {0, 1}
        P = []
        for i in range(N):
            P.append(self._model.integer_var(name='p_{}'.format(i), ub=1))
        # enemy 1 indicator variables, e1_i in {0, 1}
        E1 = []
        for i in range(N):
            E1.append(self._model.integer_var(name='e1_{}'.format(i), ub=1))
        # enemy 2 indicator variables, e2_i in {0, 1}
        E2 = []
        for i in range(N):
            E2.append(self._model.integer_var(name='e2_{}'.format(i), ub=1))
        # enemy 3 indicator variables, e3_i in {0, 1}
        E3 = []
        for i in range(N):
            E3.append(self._model.integer_var(name='e3_{}'.format(i), ub=1))
        # X graph try to encode that the palyer should be able to reach the key
        # 1. super source node to every node inside
        Xs = []
        for i in range(N):
            Xs.append(self._model.integer_var(name='xs{}'.format(i), ub=1))
        # 2. super sind node from every node insied
        Xt = []
        for i in range(N):
            Xt.append(self._model.integer_var(name='x{}t'.format(i), ub=1))
        # 3. internal node how much in how much out

        def add_new_var(arry: list, id: str):
            if len(self._model.find_matching_vars(id)) == 0:
                arry.append(self._model.integer_var(name=id, ub=1))
            else:
                arry.append(self._model.find_matching_vars(id)[0])

        Xin = []
        Xout = []
        for i in range(N):
            xin = [Xs[i]]
            xout = [Xt[i]]
            if i == 0:
                add_new_var(xin, 'x{}{}'.format(13, i))
                add_new_var(xin, 'x{}{}'.format(1, i))
                add_new_var(xout, 'x{}{}'.format(i, 13))
                add_new_var(xout, 'x{}{}'.format(i, 1))
                # if len(self._model.find_matching_vars('x{}{}'.format(13, i))) == 0:
                #     xin.append(self._model.integer_var(name='x{}{}'.format(13, i), ub=1))
                # else:
                #     xin.append(self._model.find_matching_vars('x{}{}'.format(13, i))[0])
                # if len(self._model.find_matching_vars()) == 0:
                #     xout.append(self._model.integer_var(name='x{}{}'.format(i, 1), ub=1))
                # else:
                #     xout.append(self._model.find_matching_vars('x{}{}'.format(i, 1))[0])
                # if len(self._model.find_matching_vars('x{}{}'.format(i, 13))) == 0:
                #     xout.append(self._model.integer_var(name='x{}{}'.format(i, 13), ub=1))
                # else:
                #     xout.append(self._model.find_matching_vars('x{}{}'.format(i, 13))[0])
            elif i == 12:
                add_new_var(xin, 'x{}{}'.format(11, i))
                add_new_var(xin, 'x{}{}'.format(25, i))
                add_new_var(xout, 'x{}{}'.format(i, 11))
                add_new_var(xout, 'x{}{}'.format(i, 25))
                # if len(self._model.find_matching_vars('x{}{}'.format(11, i))) == 0:
                #     xin.append(self._model.integer_var(name='x{}{}'.format(11, i), ub=1))
                # else:
                #     xin.append(self._model.find_matching_vars('x{}{}'.format(11, i))[0])
                #
                # xin.append(self._model.integer_var(name='x{}{}'.format(25, i), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, 11), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, 25), ub=1))
            elif i == 116:
                add_new_var(xin, 'x{}{}'.format(115, i))
                add_new_var(xin, 'x{}{}'.format(103, i))
                add_new_var(xout, 'x{}{}'.format(i, 115))
                add_new_var(xout, 'x{}{}'.format(i, 103))
                # xin.append(self._model.integer_var(name='x{}{}'.format(115, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(103, i), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, 115), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, 103), ub=1))
            elif i == 104:
                add_new_var(xin, 'x{}{}'.format(105, i))
                add_new_var(xin, 'x{}{}'.format(91, i))
                add_new_var(xout, 'x{}{}'.format(i, 105))
                add_new_var(xout, 'x{}{}'.format(i, 91))
                # xin.append(self._model.integer_var(name='x{}{}'.format(105, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(91, i), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, 105), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, 91), ub=1))
            elif 0 < i < 12:
                add_new_var(xin, 'x{}{}'.format(i - 1, i))
                add_new_var(xin, 'x{}{}'.format(i + 1, i))
                add_new_var(xin, 'x{}{}'.format(i + 13, i))
                add_new_var(xout, 'x{}{}'.format(i, i - 1))
                add_new_var(xout, 'x{}{}'.format(i, i + 1))
                add_new_var(xout, 'x{}{}'.format(i, i + 13))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i - 1, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i + 1, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i + 13, i), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i - 1), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i + 1), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i + 13), ub=1))
            elif 104 < i < 116:
                add_new_var(xin, 'x{}{}'.format(i - 1, i))
                add_new_var(xin, 'x{}{}'.format(i + 1, i))
                add_new_var(xin, 'x{}{}'.format(i - 13, i))
                add_new_var(xout, 'x{}{}'.format(i, i - 1))
                add_new_var(xout, 'x{}{}'.format(i, i + 1))
                add_new_var(xout, 'x{}{}'.format(i, i - 13))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i - 1, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i + 1, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i - 13, i), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i - 1), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i + 1), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i - 13), ub=1))
            elif i % 13 == 0:
                add_new_var(xin, 'x{}{}'.format(i - 13, i))
                add_new_var(xin, 'x{}{}'.format(i + 1, i))
                add_new_var(xin, 'x{}{}'.format(i + 13, i))
                add_new_var(xout, 'x{}{}'.format(i, i - 13))
                add_new_var(xout, 'x{}{}'.format(i, i + 1))
                add_new_var(xout, 'x{}{}'.format(i, i + 13))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i - 13, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i + 1, i), ub=1))
                # xin.append(self._model.integer_var(name='x{}{}'.format(i + 13, i), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i + 13), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i + 1), ub=1))
                # xout.append(self._model.integer_var(name='x{}{}'.format(i, i - 13), ub=1))
            elif i % 13 == 12:
                add_new_var(xin, 'x{}{}'.format(i - 13, i))
                add_new_var(xin, 'x{}{}'.format(i - 1, i))
                add_new_var(xin, 'x{}{}'.format(i + 13, i))
                add_new_var(xout, 'x{}{}'.format(i, i - 13))
                add_new_var(xout, 'x{}{}'.format(i, i + 13))
                add_new_var(xout, 'x{}{}'.format(i, i - 1))
            else: # for internal nodes of internal nodes
                add_new_var(xin, 'x{}{}'.format(i - 13, i))
                add_new_var(xin, 'x{}{}'.format(i + 13, i))
                add_new_var(xin, 'x{}{}'.format(i - 1, i))
                add_new_var(xin, 'x{}{}'.format(i + 1, i))
                add_new_var(xout, 'x{}{}'.format(i, i - 13))
                add_new_var(xout, 'x{}{}'.format(i, i + 13))
                add_new_var(xout, 'x{}{}'.format(i, i - 1))
                add_new_var(xout, 'x{}{}'.format(i, i + 1))
            Xin.append(xin)
            Xout.append(xout)
        # Y graph try to encode that the palyer should be able to reach the door
        # 1. super source node to every node inside
        Ys = []
        for i in range(N):
            Ys.append(self._model.integer_var(name='ys{}'.format(i), ub=1))
        # 2. super sind node from every node insied
        Yt = []
        for i in range(N):
            Yt.append(self._model.integer_var(name='y{}t'.format(i), ub=1))
        # 3. internal node how much in how much out
        Yin = []
        Yout = []
        for i in range(N):
            yin = [Ys[i]]
            yout = [Yt[i]]
            if i == 0:
                add_new_var(yin, 'y{}{}'.format(13, i))
                add_new_var(yin, 'y{}{}'.format(1, i))
                add_new_var(yout, 'y{}{}'.format(i, 13))
                add_new_var(yout, 'y{}{}'.format(i, 1))
                # if len(self._model.find_matching_vars('y{}{}'.format(13, i))) == 0:
                #     yin.append(self._model.integer_var(name='y{}{}'.format(13, i), ub=1))
                # else:
                #     yin.append(self._model.find_matching_vars('y{}{}'.format(13, i))[0])
                # if len(self._model.find_matching_vars()) == 0:
                #     yout.append(self._model.integer_var(name='y{}{}'.format(i, 1), ub=1))
                # else:
                #     yout.append(self._model.find_matching_vars('y{}{}'.format(i, 1))[0])
                # if len(self._model.find_matching_vars('y{}{}'.format(i, 13))) == 0:
                #     yout.append(self._model.integer_var(name='y{}{}'.format(i, 13), ub=1))
                # else:
                #     yout.append(self._model.find_matching_vars('y{}{}'.format(i, 13))[0])
            elif i == 12:
                add_new_var(yin, 'y{}{}'.format(11, i))
                add_new_var(yin, 'y{}{}'.format(25, i))
                add_new_var(yout, 'y{}{}'.format(i, 11))
                add_new_var(yout, 'y{}{}'.format(i, 25))
                # if len(self._model.find_matching_vars('y{}{}'.format(11, i))) == 0:
                #     yin.append(self._model.integer_var(name='y{}{}'.format(11, i), ub=1))
                # else:
                #     yin.append(self._model.find_matching_vars('y{}{}'.format(11, i))[0])
                #
                # yin.append(self._model.integer_var(name='y{}{}'.format(25, i), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, 11), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, 25), ub=1))
            elif i == 116:
                add_new_var(yin, 'y{}{}'.format(115, i))
                add_new_var(yin, 'y{}{}'.format(103, i))
                add_new_var(yout, 'y{}{}'.format(i, 115))
                add_new_var(yout, 'y{}{}'.format(i, 103))
                # yin.append(self._model.integer_var(name='y{}{}'.format(115, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(103, i), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, 115), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, 103), ub=1))
            elif i == 104:
                add_new_var(yin, 'y{}{}'.format(105, i))
                add_new_var(yin, 'y{}{}'.format(91, i))
                add_new_var(yout, 'y{}{}'.format(i, 105))
                add_new_var(yout, 'y{}{}'.format(i, 91))
                # yin.append(self._model.integer_var(name='y{}{}'.format(105, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(91, i), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, 105), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, 91), ub=1))
            elif 0 < i < 12:
                add_new_var(yin, 'y{}{}'.format(i - 1, i))
                add_new_var(yin, 'y{}{}'.format(i + 1, i))
                add_new_var(yin, 'y{}{}'.format(i + 13, i))
                add_new_var(yout, 'y{}{}'.format(i, i - 1))
                add_new_var(yout, 'y{}{}'.format(i, i + 1))
                add_new_var(yout, 'y{}{}'.format(i, i + 13))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i - 1, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i + 1, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i + 13, i), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i - 1), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i + 1), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i + 13), ub=1))
            elif 104 < i < 116:
                add_new_var(yin, 'y{}{}'.format(i - 1, i))
                add_new_var(yin, 'y{}{}'.format(i + 1, i))
                add_new_var(yin, 'y{}{}'.format(i - 13, i))
                add_new_var(yout, 'y{}{}'.format(i, i - 1))
                add_new_var(yout, 'y{}{}'.format(i, i + 1))
                add_new_var(yout, 'y{}{}'.format(i, i - 13))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i - 1, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i + 1, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i - 13, i), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i - 1), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i + 1), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i - 13), ub=1))
            elif i % 13 == 0:
                add_new_var(yin, 'y{}{}'.format(i - 13, i))
                add_new_var(yin, 'y{}{}'.format(i + 1, i))
                add_new_var(yin, 'y{}{}'.format(i + 13, i))
                add_new_var(yout, 'y{}{}'.format(i, i - 13))
                add_new_var(yout, 'y{}{}'.format(i, i + 1))
                add_new_var(yout, 'y{}{}'.format(i, i + 13))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i - 13, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i + 1, i), ub=1))
                # yin.append(self._model.integer_var(name='y{}{}'.format(i + 13, i), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i + 13), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i + 1), ub=1))
                # yout.append(self._model.integer_var(name='y{}{}'.format(i, i - 13), ub=1))
            elif i % 13 == 12:
                add_new_var(yin, 'y{}{}'.format(i - 13, i))
                add_new_var(yin, 'y{}{}'.format(i - 1, i))
                add_new_var(yin, 'y{}{}'.format(i + 13, i))
                add_new_var(yout, 'y{}{}'.format(i, i - 13))
                add_new_var(yout, 'y{}{}'.format(i, i + 13))
                add_new_var(yout, 'y{}{}'.format(i, i - 1))
            else: # for internal nodes of internal nodes
                add_new_var(yin, 'y{}{}'.format(i - 13, i))
                add_new_var(yin, 'y{}{}'.format(i + 13, i))
                add_new_var(yin, 'y{}{}'.format(i - 1, i))
                add_new_var(yin, 'y{}{}'.format(i + 1, i))
                add_new_var(yout, 'y{}{}'.format(i, i - 13))
                add_new_var(yout, 'y{}{}'.format(i, i + 13))
                add_new_var(yout, 'y{}{}'.format(i, i - 1))
                add_new_var(yout, 'y{}{}'.format(i, i + 1))
            Yin.append(yin)
            Yout.append(yout)

        # now define constraints
        # 1. for every tile there could be one type or empty
        for i in range(N):
            self._model.add_constraint(W[i] + G[i] + K[i] + P[i] + E1[i] + E2[i] + E3[i] <= 1)
        # 2. in one grid, there could be only one goal, player, key
        self._model.add_constraint(self._model.sum(G) == 1)
        self._model.add_constraint(self._model.sum(K) == 1)
        self._model.add_constraint(self._model.sum(P) == 1)
        # 3. The border of the grid should be walls
        self._model.add_constraint(self._model.sum(W[0:13]) == 13)
        self._model.add_constraint(self._model.sum(W[104:117]) == 13)
        self._model.add_constraint(self._model.sum([W[i * 13] for i in range(9)]) == 9)
        self._model.add_constraint(self._model.sum([W[i * 13 + 12] for i in range(9)]) == 9)
        # 4. The number of enemies should be less than 0.6 percent of empty space
        self._model.add_constraint((N - self._model.sum(W) - self._model.sum(G) - self._model.sum(K) - self._model.sum(P) - self._model.sum(E1) -
                                    self._model.sum(E2) - self._model.sum(E3)) * 0.6 >=
                                   self._model.sum(E1) + self._model.sum(E2) + self._model.sum(E3))
        # 5. the player should be able to reach the key, X graph
        # 5.1 super source will only go into player node
        self._model.add_constraint(self._model.sum(Xs) == 1)
        for i in range(N):
            self._model.add_constraint(Xs[i] - P[i] == 0)
        # 5.2 the flow will only be pushed into super sink from the key node
        self._model.add_constraint(self._model.sum(Xt) == 1)
        for i in range(N):
            self._model.add_constraint(Xt[i] - K[i] == 0)
        # 5.3 for every internal node there how much in how much out
        for i in range(N):
            self._model.add_constraint(self._model.sum(Xin[i]) - self._model.sum(Xout[i]) == 0)
        # 5.4 for every internal node the wall node will not be used
        for i in range(N):
            self._model.add_constraint(self._model.sum(Xin[i]) + W[i] <= 1)
            self._model.add_constraint(self._model.sum(Xout[i]) + W[i] <= 1)
        # 6. the player should be able to reach the door, Y graph
        # 6.1 super source will only go into player node
        self._model.add_constraint(self._model.sum(Ys) == 1)
        for i in range(N):
            self._model.add_constraint(Ys[i] - P[i] == 0)
        # 6.2 the flow will only be pushed into super sink from the door node
        self._model.add_constraint(self._model.sum(Yt) == 1)
        for i in range(N):
            self._model.add_constraint(Yt[i] - G[i] == 0)
        # 6.3 for every internal node there how much in how much out
        for i in range(N):
            self._model.add_constraint(self._model.sum(Yin[i]) - self._model.sum(Yout[i]) == 0)
        # 6.4 for every internal node the wall node will not be used
        for i in range(N):
            self._model.add_constraint(self._model.sum(Yin[i]) + W[i] <= 1)
            self._model.add_constraint(self._model.sum(Yout[i]) + W[i] <= 1)


if __name__ == '__main__':
    program = Program()

    program.solve()
