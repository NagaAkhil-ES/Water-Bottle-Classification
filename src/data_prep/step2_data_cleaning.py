from glob import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
import shutil

from utils.util import setup_save_dir


if __name__ == "__main__":
    # parameters
    raw_dir = "data/raw"
    metadata_path = "data/csvs/metadata.csv"
    save_dir = "data/clean"

    # read all png and jpeg image paths
    l_image_paths = glob(f"{raw_dir}/**/*.jpeg", recursive=True)
    l_image_paths.extend(glob(f"{raw_dir}/**/*.png", recursive=True))
    l_image_paths = pd.Series(l_image_paths)

    # copy all images to single folder and rename it
    meta_df = pd.read_csv(metadata_path)
    setup_save_dir(save_dir)
    for image_path in tqdm(l_image_paths):
        image_name = Path(image_path).stem
        image_ext = Path(image_path).suffix
        new_image_name = meta_df.new_name[meta_df.old_name == image_name].item()
        new_path = os.path.join(save_dir, new_image_name+image_ext)
        shutil.copyfile(image_path, new_path)


        