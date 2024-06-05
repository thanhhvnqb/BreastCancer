# Breast Cancer Project

This project aims to analyze and predict breast cancer using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Breast cancer is a common type of cancer that affects millions of women worldwide. This project focuses on developing a machine learning model to predict breast cancer based on various features and factors.

## Dataset

The dataset used in this project is sourced from [source_name]. It contains [number_of_samples] samples with [number_of_features] features. The dataset is preprocessed and ready for analysis.

## Installation

To run this project locally, follow these steps:

1. Install [Pytorch](https://pytorch.org/get-started/locally/)
1. Install the required dependencies: `pip install -r requirements.txt`
1. Login to WanDB: `wandb login`

## Prepare dataset
### Datasets
| Dataset     | num_patients* | num_samples* | num_pos_samples* | 
|-------------|---------------|--------------|------------------|
| [VinDr-Mammo](https://physionet.org/content/vindr-mammo/1.0.0/) | 5000          | 20000        | 226 (1.13 %)     | 
| [MiniDDSM](https://www.kaggle.com/datasets/cheddad/miniddsm2)   | 1952          | 7808         | 1480 (18.95 %)   |
| [CMMD](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508)        | 1775          | 5202         | 2632 (50.6%)     |
| [CDD-CESM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611)    | 326           | 1003         | 331 (33 %)       | 
| [BMCD](https://zenodo.org/record/5036062)        | 82            | 328          | 22 (6.71 %)      | [1, 2]    |
| All         | 9135          | 34341        | 4691 (13.66 %)   |

### Prepare dataset
To prepare the breast cancer dataset, follow these steps:

1. Run the command: `PYTHONPATH=$(pwd):$PYTHONPATH python src/tools/data/prepare_classification_dataset.py --dataset <name_dataset> --root-dir <path_to_downloaded_directory> --stage <stage>`

Or download from here: [RSNA](https://drive.google.com/file/d/1AI-rNC_Ti51_q0wzBYtb4wfxVmy0fKhB/view), [VinDR-Mammo](https://drive.google.com/file/d/1yJLgy5kUoHrj79YbQ9o6-y_SYojUFVH8/view)

### Structure of datasets
The structure of folder datasets should be look like this:
```
$ tree -L 3 datasets

datasets
└── classification
    ├── rsna
    │   ├── cleaned_images
    │   ├── cleaned_label.csv
    │   └── fold
    └── vindr
        ├── cleaned_images
        └── cleaned_label.csv
```

## Contributing

Contributions are welcome! If you have any ideas or suggestions, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.