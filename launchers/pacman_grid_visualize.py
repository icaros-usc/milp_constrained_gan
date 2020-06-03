"""Use this file to visualize generated pacman levels"""

import json
import os

from tqdm import tqdm

from launchers.visualize_utils import integerLvl2Img


def run(dataroot,
        output_path):
    if not os.path.exists(output_path):
        """If there is no such a folder."""
        os.makedirs(output_path, exist_ok=True)

    sprites_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GVGAI', 'sprites')
    sprites_path = {0: 'oryx/floor3.png', 1: 'oryx/orb2.png', 2: 'oryx/gold2.png', 3: 'newset/pacman.png',
                    4: 'oryx/ghost3.png', 5: 'oryx/ghost4.png', 6: 'oryx/ghost5.png', 7: 'oryx/ghost6.png',
                    8: 'newset/cherries2.png', 9: 'oryx/wall3.png'}

    target_img_size = (24, 24)

    for lvl in tqdm(os.listdir(dataroot)):
        with open(os.path.join(dataroot, lvl), 'r') as f:
            lvlJson = json.load(f)
            img = integerLvl2Img(lvlJson, sprites_root, sprites_path, target_img_size)
            img.save(os.path.join(output_path, lvl.split('.')[0] + '.png'))


if __name__ == '__main__':
    dataroot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pacman', 'pacman_gan')
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pacman', 'pacman_gan_visual')
    run(dataroot,
        output_path)
