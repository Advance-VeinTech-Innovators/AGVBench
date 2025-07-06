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
 - **2025.06.23** We update some analysis tools code: `compute_eer.py`, `analyze_sparse.py`, `save_purning_model.py`, `tsne_clustering_visualization.py`, and `draw_eer.py` files for your analysis study. Modify the `classification.py` file in `agvbench/datasets/` for computing the eer score.
- **2025.06.22** we support two mix augmentation method **"AugMix"[[ICLR 2020]](https://arxiv.org/abs/1912.02781)** and **"StarMixup"[[ICSBI 2025]](https://ieeexplore.ieee.org/abstract/document/11015373/)**. 
- **2025.06.21** We relase the core codebase files.  

___

### 🐳 1. Augmentation methods
We divided the augmentations into three types: ``Sample-level``, ``Label-level``, and ``Specific-level``.

- **Sample-level**
  - Single-sample
    1. Disruption-based
       1. [x] Flip
       2. [x] Rotate
       3. [ ] Blur
       4. [ ] Noisy
       5. [x] Translation, 
    2. Policy-based: 
       1. [x] AutoAugment
       2. [x] RandAugment
       3. [ ] TeachAugment
       4. [ ] KeepAugment
       5. [ ] SoftAugment
       6. [ ] TrivialAugment
    3. Cutting-based
       1. [x] YOCO
       2. [x] Cutout
       3. [x] GridMask
       4. [x] RandomErasing
       5. [ ] Randomized Quantization
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
  - Generating (*This depends on the difficulty of coding, and if it is difficult to implement, we can choose not to include it.*)
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
  2. [ ] MixedAA
  3. [ ] Explainable AI
  4. [ ] MTPV

### 📊 2. Experiments Settings

- **Datasets**

  | Dataset   | Total Img  | Classes | Train/Test set |   Link   |
  | --------- | ---------- | ------- | -------------- |--------- |
  | TJU600    | 12,000     | 600     | 6,000/6,000    | [TJU600](https://cslinzhang.github.io/ContactlessPalm/)         |
  | VERA220   | 2,200      | 220     | 1,100/1,100    | [VERA220](https://www.idiap.ch/en/scientific-research/data/vera-palmvein)         |
  | CASIA200  | 1,200      | 200     | 600/600        | [CASIA](http://www.cbsr.ia.ac.cn/english/MS_PalmprintDatabases.asp)         |
  | HKPU500   | 6,000      | 500     | 3,000/3,000    | [HKPU500](https://www4.comp.polyu.edu.hk/~cslzhang/paper/TIM_10_Feb.pdf)         |
  
  We could build a ``Corrpution dataset/policy`` for the test set like ``CIFAR100-C/ImageNet-C``
- **Backbone**
  - CNNs
    - [x] ResNet18/ResNet50: Classic backbone
    - [x] MobileNet v2: Effeicent backbone for mobile devices
    - [x] StarLKNet: Large kernel backbone
    - [x] FVCNN/PCVNN/FVRASNet: Specific design backbone for vein
  - ViTs
    - [x] DeiT (tiny, small, base)
    - [x] ViT (small, base, large)
    - [x] Swin (tiny, small, base)
- **Settings**

  We resize the size of image to 3x224x224...
    
  | Optimizer | Batch Size | LR     | Scheduler     | Hyperparameters            |
  | --------- | ---------- | ------ | -------------- | ------------------------- |
  | SGD       | 32         | `0.01` | ✅ Cosine     | momentum=0.9, wd=1e-4      |
  | AdamW     | 32         | `3e-4` | ✅ Cosine     | betas=(0.9, 0.98), wd=1e-2 |

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
  - [ ] ⏱️ Time-Cost
  - [x] 🧬 t-SNE Visualization
  - [x] 🔍 CAM Visualization

---

Current contributors include: Xin Jin ([@JinXins](https://github.com/JinXins)), Haiyang Li ([@OceanLee66](https://github.com/OceanLee66)), Jing Chen, Yuming Fu, Hongyu Zhu and Hongchao Liao.

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


| 模块 | 建议 |
| --- | ---- |
| **Label-level 增强模块** | 可以考虑明确每种标签增强的输入需求。例如 FCM/LP/Manifold 依赖 feature embedding，建议加一句：`requires extracted features from pretrained encoder.` |
| **Corruption Dataset**  | 可构建类 ImageNet-C 的测试扰动集，建议稍微细化一下该模块的结构。例如：`blur, brightness, jpeg, occlusion, salt&pepper, contrast` 这类扰动类型。 |
| **Manifold 方法说明** | 加入 manifold learning，在文档中补充一小段解释其原理与作用，例如：“We utilize manifold-preserving neighborhood label propagation to generate smooth label distributions from UMAP-embedded features.” |
| **更多分析指标** | 如 `Expected Calibration Error (ECE)` 或 `Brier Score` 等作为 calibration 的补充； |

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
4. **Overview of AGVBench**
     1. **Problem Formulation**
     2. **Method Taxonomy: Sample-Level, Label-Level, Specific-Level**
5. **Experimental Setup**
     1. **Datasets and Preprocessing**
     2. **Model Architectures**
     3. **Training Protocols**
6. **Systematic Evaluation of Augmentation Strategies**
     1. **Classification**
     2. **EER**
     3. **Corruption**
     4. **FGSM, PGD**
     5. **PSNR, SSIM**
7. **Analysis and Visualization**
     1. **Occlusion Robustness**
     2. **Calibration**
     3. **ROC**
     4. **t-SNE & CAM**
8. **Ablation and Discussions**
9.  **Conclusion and Future Work**
