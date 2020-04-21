import os
import sys

sys.path.append(os.getcwd())
from utils.TrainLevelHelper import *
import toml


def generate_training_level(level_config: str):
    parsed_toml = toml.load(level_config)

    level_path = parsed_toml["LevelPath"]
    level_width = parsed_toml["LevelWidth"]
    compressed = parsed_toml["Compressed"]
    print("Generating training levels, each in width " + str(level_width))
    X, z_dims, index2str = get_windows_from_folder(level_path, level_width, compressed)
    print("Training Level Generation Finished")
    return X, z_dims, index2str
