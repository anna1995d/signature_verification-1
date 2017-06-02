import os

from utils.config import CONFIG


def prepare_output_directory():
    if not os.path.exists(CONFIG.out_dir):
        os.mkdir(CONFIG.out_dir)
