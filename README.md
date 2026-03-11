# AGVBench

### 🔭 A Comprehensive Benchmark and Analysis of Data Augmentation in Palm Vein Identification

## 📌 Research Background & Significance

### 🌟 Why this benchmark matters?
- 🔒 **Data Scarcity Challenge**  
  **Vein image datasets are typically limited due to storage constraints and privacy concerns. Insufficient training samples per class often lead to overfitting in deep learning models (*e.g.*, DNNs, CNNs, ViTs). Data augmentation is critical for expanding training datasets and improving model generalization.**  
- ⚠️ **Limitations of Current Approaches**  
  **Existing vein identification research relies heavily on empirical augmentation strategies:**
  - 🔄 **Majority of papers default to ``MixUp + CutMix`` combinations**
  - 🎨 **Some studies use basic methods like ``ColorJittering``**
- These practices lack systematic validation and may underutilize model potential due to suboptimal augmentation choices.
### 🚀 Our Contribution
- 📊 **Systematic evaluation of augmentation techniques for vein identification**  
- 🎯 **Optimal augmentation strategies tailored to different model architectures (CNN/ViT)**  
- 🧪 **Objective benchmarks to guide researchers away from trial-and-error approaches**
____

### 💥 News
- **2026.03.09** Support new bash for evaluations.
- **2026.01.19** Support **"RSNet"[[IEEE TIFS 2025]](https://ieeexplore.ieee.org/abstract/document/10896759)** and **"WTxGRN"[[IEEE TIFS 2025]](https://ieeexplore.ieee.org/abstract/document/11095785)**.
- **2025.12.12** Fixed bugs of EER and StarMix. Added config files of StarLKNet, AMPVNet and FVRASNet.
- **2025.07.15** We supported a vein-specific augmentation method, **"MAdAugment"[[IEEE TIM 2024]](https://xplorestaging.ieee.org/document/10530126)**.
- **2025.07.13** We supported compute FPR@TPR for biometric task, and supported **"AMPVNet"[[IEEE TIFS 2024]](https://ieeexplore.ieee.org/document/10474047)**.
- **2025.07.12** We supported some Randomized Quantization augmentation: **"Randomized Quantization"[[ICCV 2023]](https://arxiv.org/abs/2212.08663)**.
- **2025.07.11** We supported some policy-based augmentation: **"KeepAugment"[[CVPR 2021]](https://arxiv.org/abs/2011.117781)**, **"TrivialAugment"[[ICCV 2021]](https://arxiv.org/abs/2103.10158)**, **"TeachAugment"[[CVPR 2022]](https://arxiv.org/abs/2202.12513)**, **"SoftAugment"[[CVPR 2023]](https://arxiv.org/abs/2211.04625)**.
- **2025.07.08** Fixed some bugs and supported **PGD Adversarial Attack** in `calibration_fgsm.py`, We add the training config files for training. Now. you can training the models with supported augmentation methods.
- **2025.06.23** We update some analysis tools code: `compute_eer.py`, `analyze_sparse.py`, `save_purning_model.py`, `tsne_clustering_visualization.py`, and `draw_eer.py` files for your analysis study. Modify the `classification.py` file in `agvbench/datasets/` for computing the eer score.
- **2025.06.22** we support two mix augmentation method **"AugMix"[[ICLR 2020]](https://arxiv.org/abs/1912.02781)** and **"StarMixup"[[ICSBI 2025]](https://ieeexplore.ieee.org/abstract/document/11015373/)**. 
- **2025.06.21** We relase the core codebase files.  

___


### Installation

AGVBench is compatible with **Python 3.6/3.7/3.8/3.9** and **PyTorch >= 1.6**. Here are quick installation steps for development:

```shell
conda create -n agvbench python=3.8 pytorch=1.12 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate agvbench
pip install openmim
mim install mmcv-full
git clone https://github.com/Advance-VeinTech-Innovators/AGVBench.git
cd agvbench
pip install -r requirements.txt
python setup.py develop
```

___

### 🐳 1. Augmentation methods
We divided the augmentations into three types: ``Sample-level``, ``Label-level``, and ``Specific-level``.

- **Sample-level**
  - Single-sample
    1. Disruption-based
       1. [x] Flip
       2. [x] Rotate
       3. [x] Blur
       4. [x] Noisy
       5. [x] Translation, 
    2. Policy-based: 
       1. [x] AutoAugment
       2. [x] RandAugment
       3. [x] TeachAugment
       4. [x] KeepAugment
       5. [x] SoftAugment
       6. [x] TrivialAugment
    3. Cutting-based
       1. [x] YOCO
       2. [x] Cutout
       3. [x] GridMask
       4. [x] RandomErasing
       5. [x] Randomized Quantization
  - Multi-samples
    1. Static
       1. [x] Mixup
       2. [x] CutMix
       3. [x] Manifold Mixup
       4. [x] FMix
       5. [x] GridMix
       6. [x] ResizeMix
       7. [x] AugMix
       8. [x] StarMix
    2. Dynamic
       1. [x] SaliencyMix
       2. [x] PuzzleMix
       3. [x] GudiedMix
       4. [x] AutoMix
  - Generating (**We do not use generation-based augmentation since they can't cross domain.**)
    1. [ ] GANs
    2. [ ] Diffusion Model
- **Label-level**
  - [x] Label Smooth
  - [ ] Fuzzy C-Means
  - [ ] Label Propagation
  - [ ] Mainifold Learning
  - [ ] Label Distribution
  - [ ] Token Labeling
- **Specific-level**
  1. [x] StarMix
  2. [x] MixedAA
  3. [x] Explainable AI (**Somthing wrong with this method, they can't training normal, I don't konw why happend this.**)

### 📊 2. Experiments Settings

- **Datasets**

  | Dataset   | Total Img  | Classes | Train/Test set |   Link   |
  | --------- | ---------- | ------- | -------------- |--------- |
  | TJU600    | 12,000     | 600     | 6,000/6,000    | [TJU600](https://cslinzhang.github.io/ContactlessPalm/)         |
  | VERA220   | 2,200      | 220     | 1,100/1,100    | [VERA220](https://www.idiap.ch/en/scientific-research/data/vera-palmvein)         |
  | CASIA200  | 1,200      | 200     | 600/600        | [CASIA](http://www.cbsr.ia.ac.cn/english/MS_PalmprintDatabases.asp)         |
  | HKPU500   | 6,000      | 500     | 3,000/3,000    | [HKPU500](https://www4.comp.polyu.edu.hk/~cslzhang/paper/TIM_10_Feb.pdf)         |
  | SCUT834   | 8,340      | 834     | 4,170/4,170    | [SCUT834] |
  
  We build a ``Corrpution dataset/policy`` for the test set like ``CIFAR100-C/ImageNet-C``
- **Backbone**
  - CNNs
    - [x] ResNet18/ResNet50: Classic backbone
    - [x] MobileNet v2: Efficient backbone for mobile devices
    - [x] StarLKNet: Large kernel backbone
    - [x] FVRASNet/AMPVNet: Specific design backbone for vein
  - ViTs
    - [x] DeiT (tiny, small)
    - [x] ViT (small)
    - [x] Swin (tiny, small)
- **Settings**

  We resize the size of image to 3x224x224...
    
  | Optimizer | Batch Size | LR     | Scheduler     | Hyperparameters            |
  | --------- | ---------- | ------ | -------------- | ------------------------- |
  | SGD       | 32         | `0.01` | ✅ Cosine     | momentum=0.9, wd=1e-4      |
  | AdamW     | 32         | `1e-3` | ✅ Cosine     | betas=(0.9, 0.999), wd=1e-2 |

- **Experiments**
  - [x] 📉 Classification
  - [x] 🎯 EER
  - [ ] 💥 Corruption (image perturbations)
  - [x] ⚔️ Adversarial Attack: FGSM, PGD
  - [ ] 📊 Quality Assessment: PSNR (Peak Signal-to-Noise Ratio) / SSIM (Structural Similarity Index)

### 🧪 3. Analysis Studies

  - [x] ✂️ Occlusion (cutting-based & masking-based)
  - [x] 🎯 Calibration
  - [x] 🛡️ ROC Curves
  - [ ] ⏱️ Time-Cost (1. Single-img, 2. Total Dataset img for one epoch, 3. Few epochs's mean.)
  - [x] 🧬 t-SNE Visualization
  - [x] 🔍 CAM Visualization

---

Current contributors include: Xin Jin ([@JinXins](https://github.com/JinXins)).

### 😉 Citation
**🤗 If you feel that our work has contributed to your research, please cite it. Thanks.**  
```
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

___

| 模块 | 建议 |
| --- | ---- |
| ⚠️ **Label-level 增强模块** | 可以考虑明确每种标签增强的输入需求。例如 FCM/LP/Manifold 依赖 feature embedding，建议加一句：`requires extracted features from pretrained encoder.` |
| ✅ **Corruption Dataset**  | 可构建类 ImageNet-C 的测试扰动集，建议稍微细化一下该模块的结构。例如：`blur, brightness, jpeg, occlusion, salt&pepper, contrast` 这类扰动类型。 |
| ❌ **Manifold 方法说明** | 加入 manifold learning，在文档中补充一小段解释其原理与作用，例如：“We utilize manifold-preserving neighborhood label propagation to generate smooth label distributions from UMAP-embedded features.” |
| ✅ **更多分析指标** | 如 `Expected Calibration Error (ECE)` 或 `Brier Score` 等作为 calibration 的补充；|
| ✅ **多方法混合增益**|增加 Augmentation Interaction Study, e.g., `Mixup + GridMask works best for CNN`|


### Label Enhancement Methods
| 维度          |  FCM  |  LP  | Manifold |     LS     |
| ------------- | ---- | ---- | --------- | ---------- |
| 是否依赖标签   | ❌   | ✅  | 可选       | ✅        |
| 是否依赖邻接图 | ❌   | ✅  | ✅        | ❌        |
| 是否全局一致性 | ✅   | ✅  | ❌（局部） | ✅        |
| 是否适合半监督 | ✅   | ✅  | ✅        | ⛔（需GT） |

___


## Questions

1. 是否需要扩展到指静脉？
2. 是否需要分开Vein-sepcific的数据增强方法？
  
## Paper Plan
**Titles**
| Type | Examples |
| --- | ------- |
| Scholar     | **AGVBench: A Systematic Benchmark for Data Augmentation in Vein Biometrics**           |
| Project     | **Revisiting Data Augmentation in Palm Vein Recognition: A Comprehensive Benchmark**    |
| Experiments | **What Works in Vein Recognition? A Comparative Study of Data Augmentation Techniques** |


**Framework & Content**
1. **Abstract**
2. **Introduction**
3. **Related Work**
  - Data Augmentation in Computer Vision
  - Data Augmentation in Biometrics
  - Benchmarking Data Augmentation
4. **Overview of AGVBench**
  - Problem Formulation
  - Augmentation Taxonomy
    1. Single Image Augmentation
    2. Multi Image Augmentation
    3. Label-Level Augmentation
  - Evaluation Protocol
5. **Experimental Setup**
     1. Datasets
     2. Model Architectures
     3. Implementations
6. **Results of Augmentation Strategies**
  - Recognition Performance
     1. Classification
     2. EER
  - Robustness
     3. Calibration
     2. Corruption
     4. Adversarial Attack (FGSM, PGD)
     5. Image Quality Preservation (PSNR, SSIM)
  - Efficiency Analysis
     1. Training overhead & Memory usage & Inference speed
7. **Analysis and Insights**
  - Feature Space Visualization
     1. t-SNE & CAM
  - Orthogonality of Augmentations
     1. sample-level + label-level, sample-level + multi-sample
  - Discussions
8.  **Conclusion and Future Work**
