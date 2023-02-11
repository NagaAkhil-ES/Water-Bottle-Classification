from torch.utils.data import DataLoader
import pandas as pd

from data.dataset import WaterBottleDataset
from data.transforms import get_transforms
from data.stats import show_class_distribution

def get_data_loader(params, f_train, f_norm=True):
    if f_train:
        csv_path = params.train_csv_path
        title = "train dataset"
    else:
        csv_path = params.val_csv_path
        title = "validation dataset"
    transforms = get_transforms(params, f_train, f_norm)
    meta_df = pd.read_csv(csv_path)
    show_class_distribution(meta_df.txt_label, title)
    d_set = WaterBottleDataset(params.images_dir, meta_df, transforms)
    d_loader = DataLoader(d_set, batch_size=params.batch_size, shuffle=False,
                          num_workers=params.num_workers)
    return d_loader

# unit testing
if __name__ == "__main__":
    from utils.config import load_config
    params = load_config("src/train/params.toml")
    dl = get_data_loader(params, f_train=True)
    features, labels = dl.dataset[0]
    print("dataset item:")
    print(features.shape)
    print(labels.shape)
    features, labels = next(iter(dl))
    print("dataloader item:")
    print(features.shape)
    print(labels.shape)
    print(f"Features Min: {features.min()}, Max: {features.max()}")