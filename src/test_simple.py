import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from infer import Predictor
from utils.device import setup_device
from utils.metrics import classwise_accuracy_score

if __name__ == "__main__":
    # parameters
    images_dir = "data/clean"
    csv_path = "data/csvs/test_1.csv"
    ptm_path = "trained_models/code_test/ep5_model.pth"
    device_type = "gpu"

    df = pd.read_csv(csv_path)
    device = setup_device(device_type)
    ptr = Predictor(ptm_path, device)

    y_true = df.num_label.tolist()
    y_pred = []
    
    for _, sample in tqdm(df.iterrows(), desc="Testings", total=len(df)):
        image_name = sample.new_name + sample.ext
        image_path = os.path.join(images_dir, image_name)
        y_pred.append(ptr.predict(image_path, f_print=False))
 
    acc = accuracy_score(y_true, y_pred)
    cw_acc = classwise_accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc: .2f}, Classwise Accuracy: {cw_acc}")
    print(f"F1-score: {test_f1:.2f}")    
