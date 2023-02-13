# Water bottle classification
This repositary contains pytorch implementation to classify water bottels based on water level


## Requirements
- Linux machine
- CPU or NVIDIA GPU + CUDA 11.1+
- Python 3.8.13
- python environment with all dependencies


## Models
Model scripts of following networks were created and used
1. [CustomCNN1](https://www.kaggle.com/code/basu369victor/pytorch-tutorial-the-classification/notebook#The-Neural-Network)
2. [CustomCNN2](https://www.kaggle.com/code/ashmalvayani/96-67-accuracy-with-cnn-s)
3. [resnet18](https://pytorch.org/vision/0.8/models.html)

## Getting started

### 1. set current working directory
All paths mentioned in following sections are relative paths w.r.t "Water-Bottle-Classification" dir

```cd Water-Bottle-Classification```


### 2. setup virtualenv
```
conda activate p_3_8 (alternatively activate any conda env with python 3.8.13)
virtualenv env -p `which python`
conda deactivate
source env/bin/activate
echo "$(pwd)/src" > env/lib/python3.8/site-packages/src.pth
pip install -r requirements.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. setup data
1. Download [Water bottle classification dataset](https://www.kaggle.com/datasets/chethuhn/water-bottle-dataset) in `data` folder
2. ```unzip data/downloaded.zip -d data/raw/```
3. remove/retain single space between every Full & Water in `data/raw/Full  Water level/Full  Water level` folder names
4. run data preparation scripts one after other as shown below
```
python src/data_prep/step1_generate_meta_csv.py  -->  to get the metadata.csv
python src/data_prep/step2_data_cleaning.py  --> to combine images to single folder with new names
python src/data_prep/step3_train_test_split.py --> to get train test csvs
```

### 4. Training
1. update parameters like run_name in `src/train/params.toml`
2. Launch tmux session (optional)
3. `python src/train/main.py`
4. After completion of training save logs to `src/trained_models/<run_name>/log.txt` 

Note: best models and its config file will be saved in `trained_models/<run_name>`

### 5. Testing
To test model with metadata csv, images dir and pretrained model path (ptm_path)
1. for simple testing use `src/test_simple.py` after updating required parameters in it
2. for batch wise testing and to check output loss use `src/evaluate/main.py` after updating required parameters in `src/evaluate/params.toml`

### 6. Inference
To find prediction on a image run command in below syntax:
``` 
python src/infer.py --image_path "" --ptm_path "" --device_type "" 
```
Example:
```
python src/infer.py --image_path "data/clean/img_0001.jpeg" --ptm_path "trained_models/exp1/ep5_model.pth" --device_type gpu
```

## Pre trained models
