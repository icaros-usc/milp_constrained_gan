import json
import os

from tqdm import tqdm

from launchers.visualize_utils import integerLvl2Img


def run(dataroot,
        output_path):
    if not os.path.exists(output_path):
        """If there is no such a folder."""
        os.makedirs(output_path, exist_ok=True)

    sprites_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GVGAI', 'sprites', 'oryx')
    sprites_path = {0: 'wall3.png', 1: 'floor3.png', 2: 'key3.png', 3: 'doorclosed1.png',
                    4: 'bat1.png', 5: 'bear1.png', 6: 'bee1.png', 7: 'swordman1_0.png'}

    target_img_size = (24, 24)

    for lvl in tqdm(os.listdir(dataroot)):
        with open(os.path.join(dataroot, lvl), 'r') as f:
            lvlJson = json.load(f)
            img = integerLvl2Img(lvlJson, sprites_root, sprites_path, target_img_size)
            img.save(os.path.join(output_path, lvl.split('.')[0] + '.png'))


if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvl_path', help='the name of the folder that contains generated levels.')
    parser.add_argument('--output_folder', help='the name of the folder where we output generated levels')
    opt = parser.parse_args()

    lvl_folder = os.path.join(data_root, opt.lvl_path)
    output_folder = os.path.join(data_root, opt.output_folder)

    run(lvl_folder,
        output_folder)
