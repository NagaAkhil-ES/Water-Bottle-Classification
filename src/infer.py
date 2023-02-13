from pathlib import Path
import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import softmax
import argparse

from utils.config import load_config
from utils.device import setup_device, setup_deterministic_env
from data.transforms import get_transforms
from model.loader import get_model

import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

class Predictor:
    def __init__(self, ptm_path, device):
        self.device = device
        self.idx_to_class = {0: "full_water_level", 1:"half_water_level", 2:"overflowing"}

        ptm_dir = str(Path(ptm_path).parent)
        train_params = load_config(os.path.join(ptm_dir, "params.toml"))

        setup_deterministic_env(train_params.seed)
        self.transforms = get_transforms(train_params, f_train=False)

        self.model = get_model(train_params.model_arch, device, ptm_path)
        self.model.eval()

    def predict(self, image_path, f_print=True):
        t_image = Image.open(image_path).convert("RGB")
        t_image = to_tensor(t_image).unsqueeze(dim=0)
        t_image = self.transforms(t_image).to(self.device)

        with torch.no_grad():
            logits = self.model(t_image)
            if not hasattr(self.model, "softmax"):
                logits = softmax(logits, dim=1) 

        out = logits.argmax(dim=1).squeeze().tolist()

        if f_print:
            print(f"Model prediction: {self.idx_to_class[out]}")

        return out

# main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--ptm_path", required=True, type=str)
    parser.add_argument("--device_type", default="cpu", type=str, help="cpu/gpu")
    args = parser.parse_args()

    device = setup_device(args.device_type)
    ptr = Predictor(args.ptm_path, device)
    ptr.predict(args.image_path)

# python src/infer.py --image_path "data/clean/img_0001.jpeg" --ptm_path "trained_models/code_test/ep5_model.pth" --device_type gpu