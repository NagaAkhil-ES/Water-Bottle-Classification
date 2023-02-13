import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from utils.util import setup_save_dir

if __name__ == "__main__":
    # parameters
    raw_dir = "data/raw"
    save_dir = "data/csvs"

    # read all png and jpeg image paths
    l_image_paths = glob(f"{raw_dir}/**/*.jpeg", recursive=True)
    l_image_paths.extend(glob(f"{raw_dir}/**/*.png", recursive=True)) 
    l_image_paths = pd.Series(l_image_paths)


    # create a metadata dataframe by parsing collected image paths
    meta_df = pd.DataFrame()
    meta_df["old_name"] = l_image_paths.apply(lambda i: Path(i).stem)
    meta_df["new_name"] = [f"img_{i+1:04}" for i in range(len(l_image_paths))]
    meta_df["ext"] = l_image_paths.apply(lambda i: Path(i).suffix)
    meta_df["txt_label"] = l_image_paths.apply(lambda i: str(Path(i).parent).split("/")[-1].lower().replace(" ","_"))
    label_encoder = LabelEncoder()
    meta_df["num_label"] = label_encoder.fit_transform(meta_df["txt_label"])

    # show metadata and save it to csv
    print(meta_df)
    print(meta_df.txt_label.value_counts())
    print(meta_df.num_label.value_counts())
    setup_save_dir(save_dir)
    meta_df.to_csv(f"{save_dir}/metadata.csv", index=False)
