import os

import numpy as np


def get_lvls(folder):
    lvls = []
    files = sorted([f for f in os.listdir(folder)], key=lambda x: int(x.split('.')[0]))
    for f in files:
        f = open(os.path.join(folder, f))
        lvl = f.readlines()
        clean_level = []
        for l in lvl:
            if len(l.strip()) > 0:
                clean_level.append(l.strip())
        lvls.append(clean_level)
    return lvls


def get_integer_lvl(lvl, str2index):
    numpyLvl = np.zeros((len(lvl), len(lvl[0])))
    for y in range(len(lvl)):
        for x in range(len(lvl[y])):
            c = lvl[y][x]
            numpyLvl[y][x] = str2index[c]
    return numpyLvl.astype(np.uint8)
