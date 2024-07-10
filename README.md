# Breast Cancer Project

This project aims to analyze and predict breast cancer using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Breast cancer is a common type of cancer that affects millions of women worldwide. This project focuses on developing a machine learning model to predict breast cancer based on various features and factors.

## Installation

To run this project locally, follow these steps:

1. Install [Pytorch](https://pytorch.org/get-started/locally/)
1. Install the required dependencies: `pip install -r requirements.txt`
1. Login to WanDB: `wandb login` if you want to sync experiment to WanDB

## Data Setup
### Datasets
| Dataset     | num_patients | num_samples | num_pos_samples | 
|-------------|---------------|--------------|------------------|
| [BMCD](https://zenodo.org/record/5036062)        | 82            | 328          | 22 (6.71 %)      | [1, 2]    |
| [CDD-CESM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611)    | 326           | 1003         | 331 (33 %)       | 
| [CMMD](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508) (or from [Kaggle](https://www.kaggle.com/datasets/tommyngx/cmmd2022))        | 1775          | 5202         | 2632 (50.6%)     |
| [MiniDDSM](https://www.kaggle.com/datasets/cheddad/miniddsm2)   | 1952          | 7808         | 1480 (18.95 %)   |
| [RSNA](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) | 11913          | 54706        | 1158 (2.12 %)     |
| [VinDr-Mammo](https://physionet.org/content/vindr-mammo/1.0.0/) | 5000          | 20000        | 226 (1.13 %)     | 
| All         | 21048          | 89047        | 5849 (6.57 %)   |

### Prepare datasets
To prepare the breast cancer datasets, follow these steps:
1. Download the dataset from the original site.
1. For `CMMD` and `BMCD`, copy the `<name_db>_raw_label.csv` file in the folder assets/cleaned_data, which we got from [\[1\]](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)
1. Run this command to convert dicom images and make `cleaned_label.csv`:
    ```bash
    PYTHONPATH=$(pwd):$PYTHONPATH python src/dataset/prepare_classification_dataset.py --dataset <name_dataset> --root-dir <path_to_downloaded_directory> --stage <stage>
    ```
1. Run this command to split dataset to 4 folds:
    ```bash
    PYTHONPATH=$(pwd):$PYTHONPATH python src/dataset/fold_split.py --dataset <name_dataset>
    ```

Or download from here: [BMCD](https://drive.google.com/file/d/1PMIHXB4OyjAmmtSV7dkAu9n_EVqUs00p/view), [CDD-CESM](https://drive.google.com/file/d/1azV9RyN0tlNIVSg7AbCi72wl-P5HMlse/view), [CMMD](https://drive.google.com/file/d/1F9wdsijc2EWCASXyta0W_8Abp_vzVHf9/view), [MiniDDSM](https://drive.google.com/file/d/1EiTK3N6SG1NXO5pxuIQknXtQRLoyMMxE/view?usp=sharing) [RSNA](https://drive.google.com/file/d/1AI-rNC_Ti51_q0wzBYtb4wfxVmy0fKhB/view), [VinDR-Mammo](hhttps://drive.google.com/file/d/1rwk2lF4mS25scuveoSoSFovhiQkyg7CE/view)

### Structure of datasets
The structure of folder datasets should be look like this:
    
```bash
$ tree -L 3 datasets

datasets
└── classification
    ├── bmcd
    │   ├── cleaned_images
    │   ├── cleaned_label.csv
    │   └── fold
    ├── cddcesm
    │   ├── cleaned_images
    │   ├── cleaned_label.csv
    │   └── fold
    ├── cmmd
    │   ├── cleaned_images
    │   ├── cleaned_label.csv
    │   └── fold
    ├── miniddsm
    │   ├── cleaned_images
    │   ├── cleaned_label.csv
    │   └── fold
    ├── rsna
    │   ├── cleaned_images
    │   ├── cleaned_label.csv
    │   └── fold
    └── vindr
        ├── cleaned_images
        ├── cleaned_label.csv
        └── fold
```
***Note***: the vindr already split training and testing images. So all of its folds will be same.

### Get datasets information
1. Run this command to get information of all datasets and their folds:
    ```bash
    PYTHONPATH=$(pwd):$PYTHONPATH python src/dataset/dataset_info.py
    ```

## Usage
1. Run this command to train model:
    ```bash
    PYTHONPATH=$(pwd):$PYTHONPATH python src/exp/trainval.py -f src/exp/trainer.py \
        --dataset <dataset_name> --experiment <experiment_name>  \
        --exp-kwargs fold_idx=<fold_index> \
        --model <model_name> --pretrained --num-classes 1 \
        --batch-size <train_batch_size> --validation-batch-size <validation_batch_size> --input-size <image_size> \
        --opt sgd --lr 3e-3 --min-lr 5e-5 --sched cosine --warmup-lr 3e-5 \
        --epochs 35 --warmup-epoch 4 --cooldown-epochs 1 \
        --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 \
        --workers 24 --eval-metric single_pfbeta \
        --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
        --save-images --model-ema --model-ema-decay 0.9998 --gp max --log-interval 100
    ```
1. Run this command to validate model:
    ```bash
    PYTHONPATH=$(pwd):$PYTHONPATH python src/exp/validate.py -f src/exp/trainer.py \
    --dataset <dataset_name> --exp-kwargs fold_idx=<fold_index> \
    --model <model_name>  --checkpoint $checkpoint --num-classes 1 \
    --batch-size 8 --input-size 3 1024 512 \
    --crop-pct 1.0 --workers 24 --amp --amp-impl native \
    --use-ema --gp max --log-interval 100
    ```
Notes:
- `name_dataset` should be `rsna`, `vindr`, `miniddsm`, `cmmd`, `cddcesm`, `bmcd`, or `all`.
- For small datasets, `train_batch_size` should be `8`. You can increase it for large datasets.
- Input image size should be `3 2048 1024` or `3 1024 512`.
- List of timm models can be found [here](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv)
- You can enable sync to WanDb by adding `--log-wandb` to the training command.
## Contributing

Contributions are welcome! If you have any ideas or suggestions, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.