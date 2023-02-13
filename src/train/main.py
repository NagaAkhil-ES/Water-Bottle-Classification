import torch

from utils.device import setup_device, setup_deterministic_env
from train.trainer import Trainer
from utils.config import load_config, save_config
from data.loader import get_data_loader
from model.loader import get_model

if __name__ == "__main__":
    params = load_config("src/train/params.toml")
    device = setup_device(params.device_type, params.gpu_ids)
    setup_deterministic_env(params.seed)

    train_loader = get_data_loader(params, f_train=True)
    val_loader = get_data_loader(params, f_train=False)

    model = get_model(params.model_arch, device, params.ptm_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    tr = Trainer(model, optimizer, train_loader, val_loader, device, params.f_weighted_loss)
    tr.fit(num_epochs=params.num_epochs, run_name=params.run_name)
    save_config(params, tr.save_dir)