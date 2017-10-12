import os
import shutil

from utils.config import CONFIG


def prepare_directories():
    if not os.path.exists(CONFIG.out_dir):
        os.mkdir(CONFIG.out_dir)

    if not os.path.exists(CONFIG.tmp_dir):
        os.mkdir(CONFIG.tmp_dir)

    shutil.copy(os.path.join(CONFIG.path, 'configuration.yaml'), CONFIG.out_dir)


def clean_directories():
    shutil.rmtree(CONFIG.tmp_dir)
