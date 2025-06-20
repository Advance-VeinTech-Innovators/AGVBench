# AGVBench

**A Comprehensive Benchmark and Analysis of Data Augmentation in Vein Identification**

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

### 1. Augmentation methods
We divided the augmentations into three types: ``Sample-level``, ``Label-level``, and ``Specific-level``.

- **Sample-level**
  - Single-sample
    - Disruption-based: Flip, Rotate, Blur, Noisy
    - Policy-based: AutoAug, RandAug, TeachAug, ColorEnhancement
    - Cutting-based: YOCO, Cutout, GridMask, RandomErasing
  - Multi-samples
    - Mixups
  - Generating (This depends on the difficulty of coding, and if it is difficult to implement, we can choose not to include it.)
    - GANs
    - Diffusion Model
- **Label-level**
  - Label Smooth
  - Token Labeling
  - Label Distribution
- **Specific-level**

### 2. Experiments Settings

- **Datasets**
  - TJU600
  - VERA220
  - CASIA200
  - HKPU500
  - We could build a ``Corrpution dataset/policy`` for the test set like ``CIFAR100-C/ImageNet-C``
- **Backbone**
  - CNNs
    - ResNet18/ResNet50: Classic backbone
    - MobileNet v2: Effeicent backbone for devices
    - StarLKNet: Large kernel backbone
    - FVCNN/PCVNN/FVRASNet: Specific design backbone
  - ViTs
    - DeiT (tiny, small, base)
    - ViT (small, base, large)
    - Swin (tiny, small, base)
  - **Settings**  
    We resize the size of image to 3x224x224...
    
    | Optimizer | Batch Size | LR     | Scheduler     | Hyperparameters            |
    | --------- | ---------- | ------ | -------------- | ------------------------- |
    | SGD       | 32         | `0.01` | ✅ Cosine     | momentum=0.9, wd=1e-4      |
    | AdamW     | 32         | `3e-4` | ✅ Cosine     | betas=(0.9, 0.98), wd=1e-2 |


### 3. Analysis Studies
