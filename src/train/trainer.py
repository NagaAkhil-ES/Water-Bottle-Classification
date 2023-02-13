import torch
import torch.nn as nn
import numpy as np
import os
import random
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, f1_score

from utils.metrics import classwise_accuracy_score
from utils.util import setup_save_dir
from data.stats import get_class_weights

class Trainer:
    def __init__(self, model, optimizer, train_dl, test_dl, device, f_weighted_loss=True):
        self.model = model
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        if f_weighted_loss:
            self.criterion = nn.CrossEntropyLoss(weight=self._get_class_weights(), reduction="mean")
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.best_val_f1 = 0
    
    def _get_class_weights(self):
        l_labels = self.train_dl.dataset.meta_df.num_label
        class_weights = get_class_weights(l_labels)["method_1"].tolist()
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        return class_weights
    
    def _run_one_epoch(self, f_train):
        model = self.model
        optimizer = self.optimizer

        # set model mode
        model.train(f_train)
        loader = self.train_dl if f_train else self.test_dl

        losses = []
        y_pred = []
        y_true = []

        for b_image, b_label in loader:
            b_image = b_image.to(self.device)
            b_label = b_label.to(self.device)

            with torch.set_grad_enabled(f_train):
                logits = model(b_image)
                if not hasattr(model, "softmax"):
                    logits = softmax(logits, dim=1) 

                out = logits.argmax(dim=1)
                y_pred.extend(out.tolist())
                y_true.extend(b_label.tolist())

                loss = self.criterion(logits, b_label)
                loss = loss.mean() # aggregate losses if they are scattered on multiple gpus
                losses.append(loss.item())
            
            if f_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                optimizer.step()

        # show report
        acc = accuracy_score(y_true, y_pred)
        if f_train:
            train_loss = np.mean(losses[-10:])
            print(f"--train loss: {train_loss:.4f}, acc:{acc:.2f}", end="  ")
        else:
            val_loss = np.mean(losses)
            cw_acc = classwise_accuracy_score(y_true, y_pred)
            val_f1 = f1_score(y_true, y_pred, average='macro')
            print(f"--val loss: {val_loss:.5f}, acc:{acc:.2f}, cw_acc:{cw_acc}, f1:{val_f1:.2f}")
            return val_f1

    def _save_model(self, epoch, val_f1):
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
        
            if hasattr(self.model, 'module'):
                # module dict if there is a dataparallel wrap
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()

            model_name = os.path.join(self.save_dir, f"ep{epoch}_model.pth")
            torch.save(model_state_dict, model_name)
            print(f"Epoch {epoch} model saved!") 

    def fit(self, num_epochs, run_name):
        print("Model training started")

        self.save_dir = setup_save_dir(os.path.join("trained_models", run_name))
        
        for epoch in range(1, num_epochs+1):
            print(f"Epoch {epoch}/{num_epochs}", end="  ")
            
            self._run_one_epoch(f_train=True)
            val_f1 = self._run_one_epoch(f_train=False)
            self._save_model(epoch, val_f1)

        print(f"Model training completed!")

def setup_deterministic_training(seed=137):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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


