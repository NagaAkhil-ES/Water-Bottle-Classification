import os
import random
import numpy as np
import torch

def setup_deterministic_env(seed=137):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # this is under beta version uncomment it if training is not deteministic
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)

def setup_device(device_type, gpu_ids="0"):
    if device_type == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        device = "cuda"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" 
        device = "cpu"
    print(f"Device used: {device}")
    print(f"torch cuda device count: {torch.cuda.device_count()}\n")
    return device

