# AGVBench

AGVBench is a benchmark codebase for studying data augmentation in palm-vein and finger-vein identification. It provides training configurations, augmentation implementations, biometric evaluation utilities, and analysis scripts for comparing augmentation strategies across CNN, Transformer, and vein-specific backbones.

## Highlights

- Unified benchmark for sample-level, label-level, and vein-specific augmentation methods.
- Classification and biometric verification evaluation, including EER and FPR@TPR.
- Robustness and analysis tools for occlusion, corruption, adversarial attack, calibration, t-SNE, CAM, ERF, Fourier, and loss-landscape visualization.
- Ready-to-run 600-epoch classification configs for multiple vein datasets and backbones.
- Supports standard CNN/ViT families and vein-oriented architectures such as StarLKNet, FVRASNet, AMPVNet, WTxGRN, FVCNN, PVCNN, and RSNet.

## News

- 2026-06-25: WeReleased the core codebase `v0.1.0`.

## Installation

AGVBench is based on PyTorch, MMCV, and OpenMMLab-style configuration files. The current runtime requirements include `mmcv-full`, `timm`, `opencv-python`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn`, and `tensorboard`.

```shell
conda create -n agvbench python=3.8 pytorch=1.12 torchvision cudatoolkit=11.3 -c pytorch -y
conda activate agvbench

pip install openmim
mim install mmcv-full

git clone https://github.com/Advance-VeinTech-Innovators/AGVBench.git
cd AGVBench
pip install -r requirements/runtime.txt
pip install -r requirements/tests.txt
python setup.py develop
```

For a full development environment, install all requirement groups:

```shell
pip install -r requirements.txt
```

## Repository Structure

```text
agvbench/                 Core datasets, models, augmentations, hooks, losses, and APIs
configs/classification/   Training configs grouped by dataset, backbone, and augmentation type
tools/train.py            Training entry point
tools/test.py             Testing entry point
tools/analysis_tools/     EER, calibration, attack, corruption, occlusion, FLOPs, and log tools
tools/visualizations/     Augmentation, CAM, EER, ERF, Fourier, LR, and loss-landscape visualization
scipts/                   Shell launch scripts for training, testing, extraction, and evaluation
demo/                     Example augmentation images
requirements/             Runtime, test, docs, and optional dependencies
```

Note: the script directory is currently named `scipts` in this repository.

## Supported Datasets

The current configuration tree provides classification configs for:

| Dataset | Config path |
| --- | --- |
| CASIA200 | `configs/classification/casia200` |
| FV-USM | `configs/classification/fv_usm` |
| HKPU500 | `configs/classification/hkpu500` |
| SCUT1100 | `configs/classification/scut1100` |
| SDUMLA-HMT | `configs/classification/sdumla_hmt` |
| TJU600 | `configs/classification/tju600` |
| VERA220 | `configs/classification/vera220` |

The benchmark assumes an image classification style data layout and uses the dataset definitions in `agvbench/datasets/`.


## Augmentation Methods

AGVBench organizes augmentation methods into sample-level, label-level, and vein-specific categories.

### Basic and Policy Augmentations

Implemented under `agvbench/models/augments/basic/` and classification pipelines:

- Blur
- Cutout
- GridMask
- KeepAugment
- Noise
- Randomized Quantization
- RICAP
- SMDWT-PCA
- SoftAugment
- YOCO
- Flip
- Rotation
- Translation
- Random Erasing
- AutoAugment
- RandAugment
- TeachAugment
- TrivialAugment

### Mix-Based Augmentations

Implemented under `agvbench/models/augments/mixups/`:

- AlignMix
- AttentiveMix
- AugMix
- CutMix
- FMix
- GridMix
- GuidedMix
- MixPro
- Mixup
- PuzzleMix
- ResizeMix
- SaliencyMix
- SMMix
- SmoothMix
- SnapMix
- StarMix
- TLA
- TokenMix
- TransMix

### Label-Level Methods

Current label-level configs and losses support:

- Label Smoothing
- Online Label Smoothing
- Dirichlet Label Smoothing
- Confidence Penalty
- Bootstrapping

Related losses are implemented in `agvbench/models/losses/`, including cross entropy, focal loss, ArcFace loss, label smoothing loss, label enhancement loss, and distillation loss.

## Configuration Coverage

Most benchmark configs follow this layout:

```text
configs/classification/{dataset}/{backbone}/600ep/
```

Common config groups include:

- `basic/`: basic and policy augmentation configs.
- `mixups/`: mix-based augmentation configs.
- `label/`: label-level augmentation configs.
- Top-level `*_vanilla_*`, `*_autoaug_*`, `*_randaug_*`, `*_madaug_*`, and related files for common training settings.

Examples:

```text
configs/classification/scut1100/starlknet/600ep/starlknet_s_vanilla_sz224_bs32.py
configs/classification/scut1100/starlknet/600ep/basic/starlknet_s_cutout.py
configs/classification/scut1100/starlknet/600ep/mixups/starlknet_s_starmix.py
configs/classification/scut1100/starlknet/600ep/label/starlknet_s_labelsmooth.py
```

## Started

### Training and Evaluation Scripts

Here, we provide scripts for starting a quick end-to-end training with multiple `GPUs` and the specified `CONFIG_FILE`. 
```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
For example, you can run the script below to train a ResNet-50 classifier on ImageNet with 4 GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/classification/scut1100/starlknet/600ep/starlknet_s_vanilla_sz224_bs32.py 4
```
After training, you can test the trained models with the corresponding evaluation script:
```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${GPUS} ${PATH_TO_MODEL} [optional arguments]
```

## Evaluation and Analysis Tools

The current version includes:

- `tools/analysis_tools/compute_eer.py`: biometric EER and related verification metrics.
- `tools/analysis_tools/calibration_fgsm.py`: calibration analysis and FGSM/PGD adversarial attack evaluation.
- `tools/analysis_tools/corruption.py`: corruption robustness evaluation.
- `tools/analysis_tools/occlusion_robustness.py`: occlusion robustness analysis.
- `tools/analysis_tools/count_parameters.py`: parameter counting.
- `tools/analysis_tools/get_flops.py`: FLOPs computation.
- `tools/analysis_tools/tsne_clustering_visualization.py`: t-SNE feature visualization.
- `tools/analysis_tools/analyze_logs.py` and `merge_logs.py`: log analysis utilities.
- `tools/analysis_tools/analyze_sparse.py` and `save_purning_model.py`: sparsity and pruning utilities.

Visualization tools:

- `tools/visualizations/vis_aug.py`
- `tools/visualizations/vis_cam.py`
- `tools/visualizations/vis_eer.py`
- `tools/visualizations/vis_erf.py`
- `tools/visualizations/vis_fourier.py`
- `tools/visualizations/vis_loss_landscape.py`
- `tools/visualizations/vis_lr.py`

## Citation

If AGVBench is useful for your research, please cite the related work:

```bibtex
@article{jin2024starlknet,
  title={StarLKNet: star Mixup with large kernel networks for palm vein identification},
  author={Jin, Xin and Zhu, Hongyu and Yacoubi, Moun{\^\i}m A El and Li, Haiyang and Liao, Hongchao and Qin, Huafeng and Jiang, Yun},
  journal={arXiv preprint arXiv:2405.12721},
  year={2024}
}

@inproceedings{jin2025starmixup,
  title={StarMixup: A More Suitable Mixup Method for Palm-Vein Identification},
  author={Jin, Xin and Zhu, Hongyu and Fong, Simon and Marques, Jo{\~a}o Alexandre Lobo and Qin, Huafeng and Jiang, Yun},
  booktitle={2025 7th International Symposium on Computational and Business Intelligence (ISCBI)},
  pages={83--87},
  year={2025},
  organization={IEEE}
}
```

## Contributors

Current contributors include Xin Jin ([@JinXins](https://github.com/JinXins)), Haiyang Li ([@OceanLee66](https://github.com/OceanLee66)). We thank all public contributors and contributors from MMPreTrain (MMSelfSup and MMClassification) and OpenMixup team!

## License

This project follows the Apache Software License metadata declared in `setup.py`.
