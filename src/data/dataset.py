from torch.utils.data import Dataset
import os
from PIL import Image
from torch import tensor as torch_tensor
from torchvision.transforms.functional import to_tensor

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

class WaterBottleDataset(Dataset):
    def __init__(self, images_dir, meta_df, transforms=None):
        self.images_dir = images_dir
        self.meta_df = meta_df
        self.transforms = transforms

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        sample = self.meta_df.iloc[idx]

        image_name = sample.new_name + sample.ext 
        image_path = os.path.join(self.images_dir, image_name)

        t_image = Image.open(image_path).convert("RGB")
        t_image = to_tensor(t_image)
        if self.transforms is not None:
            t_image = self.transforms(t_image)

        t_label = torch_tensor(sample.num_label)

        return t_image, t_label
   
        



# unit test
if __name__ == "__main__":
    import pandas as pd
    from torchvision.utils import save_image
    images_dir = "data/clean"
    meta_df = pd.read_csv("data/metadata.csv") #.sample(n=50, random_state=137)

    ds = WaterBottleDataset(images_dir, meta_df)
    print(len(ds))
    t1, t2 = ds[0]
    print(t1.shape, t2)
    save_image(t1, "temp.png")