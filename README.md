# SemiSegECG: A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation
## Environments
### Requirements
- python 3.9
- einops 0.6.0
- mergedeep 1.3.4
- numpy 1.21.6
- pandas 1.4.2
- pytorch 1.11.0
- PyYAML 6.0
- scipy 1.8.1
- tensorboard
- torchmetrics 1.5.2
- tqdm
- wfdb

### Installation
```bash
semi-seg-ecg$ conda create -n semi_seg_ecg python=3.9
semi-seg-ecg$ conda activate semi_seg_ecg
semi-seg-ecg$ pip install -r requirements.txt
```

## Benchmark Data
### Installation
We recommend using [`gdown`](https://github.com/wkentaro/gdown) command for the installation.

```bash
semi-seg-ecg$ gdown $FILE_ID -O $OUT_PATH
```
- `$FILE_ID`: __Google Drive__ file ID, which can be extracted from the shareable links.<br>
  (e.g. `https://drive.google.com/file/d/$FILE_ID`)
- `$OUT_PATH`: the desired output path or directory.

- Example:
  ```bash
  semi-seg-ecg$ gdown 1qPAEmilpbSfCArhfDDKl1Vrqn4j89ZWK -O data/
  semi-seg-ecg$ gdown 1vWSola1ySAt5XI8jMoG6ZPAwcFn8OAjP -O index/
  semi-seg-ecg$ unzip -q data/ludb.zip -d data/
  semi-seg-ecg$ unzip -q index/ludb.zip -d index/
  semi-seg-ecg$ rm data/ludb.zip
  semi-seg-ecg$ rm index/ludb.zip
  ```

### Data
| **Database** | **File ID**                              | **Download** |
|--------------|-------------------------------------------|--------------|
| *LUDB*       | 1qPAEmilpbSfCArhfDDKl1Vrqn4j89ZWK         | [link](https://drive.google.com/file/d/1qPAEmilpbSfCArhfDDKl1Vrqn4j89ZWK) |
| *QTDB*       | 1PjoRMw7ZpzmwPWAZq8e7KXpAOVBwucrZ         | [link](https://drive.google.com/file/d/1PjoRMw7ZpzmwPWAZq8e7KXpAOVBwucrZ) |
| *ISP*        | 1NUPQ5rhGvkFPIYNvXz5nWBk8_wF0EIFE         | [link](https://drive.google.com/file/d/1NUPQ5rhGvkFPIYNvXz5nWBk8_wF0EIFE) |
| *Zhejiang*   | 1EYTOrK5GhskO8ulwJyea6UKCFG4-UZ21         | [link](https://drive.google.com/file/d/1EYTOrK5GhskO8ulwJyea6UKCFG4-UZ21) |
| *PTB-XL*     | 1zs8g5ivGImTctPYyfBx14b1JZhe0u53w         | [link](https://drive.google.com/file/d/1zs8g5ivGImTctPYyfBx14b1JZhe0u53w) |

### Index
| **Dataset** | **File ID** | **Download** |
|-------------|-------------|--------------|
| **In-domain setting** |||
| *LUDB*      | 1vWSola1ySAt5XI8jMoG6ZPAwcFn8OAjP | [link](https://drive.google.com/file/d/1vWSola1ySAt5XI8jMoG6ZPAwcFn8OAjP) |
| *QTDB*      | 1VCpOrIC0B5V57Jc7b-kR26-T22Bq__nQ | [link](https://drive.google.com/file/d/1VCpOrIC0B5V57Jc7b-kR26-T22Bq__nQ) |
| *ISP*       | 1V2_YoolEnK8I8jRd0W8rixv1xHDBdvyd | [link](https://drive.google.com/file/d/1V2_YoolEnK8I8jRd0W8rixv1xHDBdvyd) |
| *Zhejiang*  | 1Q8zTQcZDZToaN6L-BLOBoNEJol89xgNW | [link](https://drive.google.com/file/d/1Q8zTQcZDZToaN6L-BLOBoNEJol89xgNW) |
| **Cross-domain setting** |||
| *Merged*    | 1B_uCeuVS-eyUjWZ85AswGLn6wwFnfFvd | [link](https://drive.google.com/file/d/1B_uCeuVS-eyUjWZ85AswGLn6wwFnfFvd) |

**Note**: For detailed data preprocessing and generation of dataset split indices (i.e., train/validation/test), please refer to the code repository [here](https://github.com/jeonghwaimnida/Delineation_dataset).

## Usage
```bash
semi-seg-ecg$ bash scripts/train.sh --help
Usage: bash scripts/train.sh [options]
Options:
  --master_port PORT               Master port (default=12345)
  --gpus GPUS                      GPU indices (default=0)
  -f, --config_path PATH           Path of config file (required)
  -o, --override_config_path PATH  Path of override config file (optional)
  --output_dir PATH                Output directory (optional)
  --exp_name NAME                  Experiment name (optional)
  --resume PATH                    Path of checkpoint to resume (optional)
  --start_epoch EPOCH              Start epoch (optional)
  -h, --help                       Print help
```
Please check the YAML files in the `configs` directory. These are the configurations used for the benchmarking experiments.

- Example:
  ```bash
  semi-seg-ecg$ bash scripts/train.sh \
  > -f ../configs/base/fixmatch.yaml \
  > -o ../configs/bench/ludb/1over16.yaml
  ```

## License
© VUNO Inc. All rights reserved.
 
This repository contains code developed at VUNO Inc. by its employees as part of their official duties.
Do not distribute, modify, or use this code outside the scope permitted by the license without explicit permission from VUNO.

## Citation

If you find this work or code is helpful in your research, please cite:
```
@article{park2025multi,
  title={SemiSegECG: A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation},
  author={Park, Minje and Lim, Jeonghwa and Yu, Taehyung and Joo, Sunghoon},
  journal={arXiv preprint arXiv:2507.18323},
  year={2025}
}
```
