import json
import os

from PIL import Image
from tqdm import tqdm


dataroot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'fake_milp')
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'zelda', 'visual_milp')

if not os.path.exists(output_path):
    """If there is no such a folder."""
    os.makedirs(output_path, exist_ok=True)


sprites_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GVGAI', 'sprites', 'oryx')
sprites_path = {0: 'wall3.png', 1: 'floor3.png', 2: 'key3.png', 3: 'doorclosed1.png',
                4: 'bat1.png', 5: 'bear1.png', 6: 'bee1.png', 7: 'swordman1_0.png'}

target_img_size = (24, 24)


def contcat_h(img1, img2):
    """Use this function to concatenate two images in horizontal direction."""
    if img2.height != img2.height:
        raise Exception
    dst = Image.new('RGB', (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    return dst


def contcat_v(img1, img2):
    if img1.width != img2.width:
        raise Exception
    dst = Image.new('RGB', (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def integerLvl2Img(lvl):
    """Use this function to translatte integer level to a image."""
    whole_img = None
    for row in range(len(lvl)):
        row_img = None
        for col in range(len(lvl[row])):
            # load image
            new_img = Image.open(os.path.join(sprites_root, sprites_path[lvl[row][col]]))
            # have to first resize the new image to the desired size
            new_img = new_img.resize(target_img_size)
            if row_img is None:
                row_img = new_img
            else:
                row_img = contcat_h(row_img, new_img)
        if whole_img is None:
            whole_img = row_img
        else:
            whole_img = contcat_v(whole_img, row_img)
    return whole_img


for lvl in tqdm(os.listdir(dataroot)):
    with open(os.path.join(dataroot, lvl), 'r') as f:
        lvlJson = json.load(f)
        img = integerLvl2Img(lvlJson)
        img.save(os.path.join(output_path, lvl.split('.')[0] + '.png'))
