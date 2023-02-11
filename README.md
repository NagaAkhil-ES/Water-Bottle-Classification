# Water bottle classification
- This repositary contains pytorch implementation to classify water bottels based on water level


## Requirements
- Linux machine
- CPU or NVIDIA GPU + CUDA 11.1+
- Python 3.8.13
- python environment with all dependencies


## Getting started

### set cwd
cd Water-Bottle-Classification


### setup virtualenv

conda activate p_3_8 (alternatively activate any conda env with python 3.8.13)
virtualenv env -p `which python`
conda deactivate
source env/bin/activate
echo "$(pwd)/src" > env/lib/python3.8/site-packages/src.pth
pip install -r requirements.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

### setup data
- Download [Water bottle classification dataset](https://www.kaggle.com/datasets/chethuhn/water-bottle-dataset) to data folder
- unzip data/downloaded.zip -d data/raw/
- remove one space after every Full in folder names of `data/raw/Full  Water level/Full  Water level`
- python src/data_prep/step1_generate_meta_csv.py  -->  to get the metadata
- python src/data_prep/step2_data_cleaning.py  --> to combine images to single folder with new names
- python src/data_prep/step3_train_test_split.py