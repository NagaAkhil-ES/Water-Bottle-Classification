import torch
from torch.nn import DataParallel

from model.custom_cnn import CustomCNN

def get_model(device, ptm_path=None):
    model = CustomCNN(num_classes=3).to(device)

    if ptm_path is not None:
        model_state_dict = torch.load(ptm_path, map_location=device)
        model.load_state_dict(model_state_dict)
        print(f'{ptm_path} reloaded!')
    
    if torch.cuda.device_count()>1:
        model = DataParallel(model)
        print("Using",torch.cuda.device_count(),"GPUs!")
    else:
        print("##__NOTE__: Data Parallelism is not activated\n")
    
    return model