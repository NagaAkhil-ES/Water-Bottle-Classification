from tqdm import tqdm
from torch.nn.functional import softmax
import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from utils.metrics import classwise_accuracy_score
from utils.dotdict import DotDict

class Tester:
    def __init__(self, model, test_dl, device):
        self.model = model
        self.loader = test_dl
        self.device = device
        self.criterion = CrossEntropyLoss()
        
    def run(self):
        model = self.model
        loader = self.loader
        device = self.device

        model.eval()
        losses, y_pred, y_true = [], [], []

        for b_image, b_label in tqdm(loader, desc="Testing"):
            b_image = b_image.to(device)
            b_label = b_label.to(device)

            with torch.no_grad():
                logits = model(b_image)
                if not hasattr(model, "softmax"):
                    logits = softmax(logits, dim=1) 

                out = logits.argmax(dim=1)
                y_pred.extend(out.tolist())
                y_true.extend(b_label.tolist())

                loss = self.criterion(logits, b_label)
                loss = loss.mean() # aggregate losses if they are scattered on multiple gpus
                losses.append(loss.item())
        
        # show report
        acc = accuracy_score(y_true, y_pred)
        test_loss = np.mean(losses)
        cw_acc = classwise_accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Test loss: {test_loss:.4f}")
        print(f"Accuracy: {acc: .2f}, Classwise Accuracy: {cw_acc}")
        print(f"F1-score: {test_f1:.2f}")


def combine_train_test_params(train_params, test_params):
    params = train_params.copy()
    for key, value in test_params.items():
        params[key] = value
    return DotDict(params)
