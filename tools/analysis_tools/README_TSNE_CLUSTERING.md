# TSNE聚类可视化工具

这个工具可以对测试集中的所有图片使用TSNE降维，然后进行K-means聚类，并生成详细的可视化分析图表。

## 功能特性

- 🎯 从预训练模型中提取深度特征
- 📊 使用TSNE将高维特征降维到2D
- 🔍 使用K-means对2D特征进行聚类（默认100个类）
- 📈 生成多种可视化图表：
  - 按真实标签着色的TSNE散点图
  - 按聚类结果着色的TSNE散点图
  - 显示聚类中心的TSNE图
  - 详细的聚类分析图表
- 📋 计算聚类评估指标（ARI、NMI、轮廓系数等）
- 💾 保存所有中间结果和最终数据

## 安装依赖

确保安装了以下Python包：

```bash
pip install scikit-learn matplotlib numpy
```

## 使用方法

### 方法1：直接运行Python脚本

```bash
python tools/analysis_tools/tsne_clustering_visualization.py \
    --config configs/resnet/resnet50_8xb32_in1k.py \
    --checkpoint checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth \
    --work_dir work_dirs/tsne_clustering_results \
    --n_clusters 100 \
    --max_samples 5000 \
    --perplexity 30 \
    --gpu-id 0
```

### 方法2：使用便捷脚本

1. 修改 `run_tsne_clustering.sh` 中的配置文件和检查点路径
2. 运行脚本：

```bash
bash tools/analysis_tools/run_tsne_clustering.sh
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--config` | str | 必需 | 模型配置文件路径 |
| `--checkpoint` | str | 必需 | 模型检查点文件路径 |
| `--work_dir` | str | `work_dirs/tsne_clustering` | 结果保存目录 |
| `--n_clusters` | int | 100 | K-means聚类数量 |
| `--max_samples` | int | 5000 | 最大处理样本数（避免内存溢出） |
| `--perplexity` | int | 30 | TSNE perplexity参数 |
| `--gpu-id` | int | 0 | 使用的GPU ID |
| `--cfg-options` | list | None | 配置文件覆盖选项 |

## 输出文件

运行完成后，在 `work_dir` 目录下会生成以下文件：

### 可视化图片
- `tsne_visualization_true_labels.png` - 按真实标签着色的TSNE散点图
- `tsne_visualization_cluster_labels.png` - 按聚类结果着色的TSNE散点图  
- `tsne_visualization_clusters_with_centers.png` - 显示聚类中心的TSNE图
- `clustering_analysis.png` - 聚类分析图表（包含6个子图）

### 数据文件
- `original_features.npy` - 原始高维特征
- `features_2d_tsne.npy` - TSNE降维后的2D特征
- `true_labels.npy` - 真实标签
- `cluster_labels.npy` - 聚类标签
- `statistics.json` - 详细统计信息和评估指标

### 日志文件
- `tsne_clustering_YYYYMMDD_HHMMSS.log` - 详细运行日志

## 输出图表说明

### 1. TSNE可视化图
- **真实标签图**：展示数据的真实类别分布
- **聚类结果图**：展示K-means聚类的结果
- **聚类中心图**：显示各聚类的中心位置

### 2. 聚类分析图（6个子图）
- **聚类大小分布**：各聚类包含的样本数量
- **真实类别分布**：数据中各真实类别的样本数量
- **聚类评估指标**：ARI和NMI分数
- **距离分布**：类内距离vs类间距离直方图
- **轮廓系数分布**：聚类质量评估
- **聚类质心分布**：各聚类中心在2D空间的分布

## 评估指标

- **ARI (调整兰德指数)**：衡量聚类结果与真实标签的一致性，值越大越好
- **NMI (标准化互信息)**：衡量聚类与真实分布的相似性，值越大越好
- **轮廓系数**：衡量聚类的紧密度和分离度，值越大越好（范围-1到1）

## 注意事项

1. **内存使用**：TSNE计算较为耗时且内存密集，建议适当调整 `max_samples` 参数
2. **Perplexity参数**：应设置为样本数的1/4左右，默认值30适用于大多数情况
3. **GPU内存**：特征提取阶段会使用GPU，确保GPU内存充足
4. **运行时间**：完整流程可能需要几分钟到十几分钟，取决于样本数量和硬件配置

## 示例使用场景

1. **模型分析**：了解预训练模型学到的特征表示质量
2. **数据探索**：可视化数据集的内在结构和类别分布
3. **聚类评估**：评估无监督聚类算法的效果
4. **特征质量**：检查不同层提取的特征的可分性

## 故障排除

### 常见问题

**Q: 出现内存不足错误**  
A: 减少 `max_samples` 参数值，或者增加系统内存

**Q: TSNE运行很慢**  
A: 降低 `perplexity` 值或减少样本数量

**Q: 聚类效果不好**  
A: 尝试调整 `n_clusters` 参数，或者检查特征提取是否正确

**Q: 图片显示不全**  
A: 调整matplotlib的图像大小设置，或者减少显示的类别数量

## 技术原理

1. **特征提取**：从预训练模型的backbone+neck提取深度特征
2. **TSNE降维**：将高维特征映射到2D空间，保持局部邻域结构
3. **K-means聚类**：在2D空间进行聚类，找到数据的聚集模式
4. **可视化**：生成多种角度的散点图和分析图表 