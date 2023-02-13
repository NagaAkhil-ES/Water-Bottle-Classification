import os
import toml

from utils.device import setup_device, setup_deterministic_env
from utils.config import load_config
from evaluate.tester import Tester, combine_train_test_params
from data.loader import get_data_loader
from model.loader import get_model, get_ptm_path

if __name__ == "__main__":
    # parameters
    test_params = load_config("src/evaluate/params.toml")
    train_params = load_config(os.path.join(test_params.ptm_dir, "params.toml"))
    params = combine_train_test_params(train_params, test_params)
    device = setup_device(params.device_type, params.gpu_ids)
    setup_deterministic_env(params.seed)

    test_loader = get_data_loader(params, f_train=False)

    ptm_path = get_ptm_path(params.ptm_dir, params.ptm_ep)
    model = get_model(params.model_arch, device, ptm_path)

    tstr = Tester(model, test_loader, device)
    tstr.run()