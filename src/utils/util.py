import os

def setup_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)