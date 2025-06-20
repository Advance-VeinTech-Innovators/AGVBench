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
