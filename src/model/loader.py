import torch
from torch.nn import DataParallel, Linear
from torchvision import models as tv_models
import os

from model.custom_cnn import CustomCNN1, CustomCNN2
from utils.util import listdir

def get_model(model_arch, device, ptm_path=""):
    if model_arch == "custom1":
        model = CustomCNN1(num_classes=3).to(device)
    elif model_arch == "custom2":
        model = CustomCNN2(num_classes=3).to(device)
    elif model_arch == "resnet18":
        model = tv_models.resnet18(True)
        model.fc = Linear(512, 3)
        model.to(device)

    if len(ptm_path)>0 :
        model_state_dict = torch.load(ptm_path, map_location=device)
        model.load_state_dict(model_state_dict)
        print(f'{ptm_path} reloaded!')
    
    if torch.cuda.device_count()>1:
        model = DataParallel(model)
        print("Using",torch.cuda.device_count(),"GPUs!")
    else:
        print("##__NOTE__: Data Parallelism is not activated\n")
    
    return model

def get_ptm_path(ptm_dir, ptm_ep=-1):
    if ptm_ep == -1: # default loads the lastest model
        pth_files = listdir(dir=ptm_dir, file_ext=".pth")
        return os.path.join(ptm_dir, pth_files[-1])  
    else:
        return os.path.join(ptm_dir, f"ep{ptm_ep}_model.pth")