"""Use this file to translate txt grid to json grid"""

import json
import os

from utils.TrainLevelHelper import get_lvls, get_integer_lvl


def run(data_path,
        output_path):
    index2strJson = json.load(open('zelda_index2str.json', 'r'))
    str2index = {}
    for key in index2strJson:
        str2index[index2strJson[key]] = key

    os.makedirs(output_path, exist_ok=True)

    # get all levels and store them in one numpy array
    np_lvls = []
    lvls = get_lvls(data_path)
    for lvl in lvls:
        numpyLvl = get_integer_lvl(lvl, str2index)
        np_lvls.append(numpyLvl)

    for i, np_lvl in enumerate(np_lvls):
        with open(os.path.join(output_path, '{}.json'.format(i)), 'w') as f:
            f.write(json.dumps(np_lvl.tolist()))


if __name__ == '__main__':
    dataroot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human')
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'Human_Json')
    run(dataroot,
        output_path)
