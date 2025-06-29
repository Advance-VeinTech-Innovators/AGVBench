import argparse
import os
import os.path as osp
import time

import mmcv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from agvbench.datasets import build_dataloader, build_dataset
from agvbench.models import build_model
from agvbench.utils import get_root_logger, print_log, setup_multi_processes, traverse_replace


class TSNEClusteringVisualizer:
    
    def __init__(self, model, n_clusters=100):
        self.model = model
        self.n_clusters = n_clusters
        self.tsne = None
        self.kmeans = None
        
    def extract_all_features_and_labels(self, data_loader, max_samples=100):
        """
        Encoding all test data and obtaining the features, labels
        """
        
        self.model.eval()
        features = []
        labels = []
        count = 0
        
        print("Staring encoding...")
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if count >= max_samples:
                    break
                    
                if i % 10 == 0:
                    print(f"Processing Batch {i}/{len(data_loader)}, Encoded samples: {count}")
                
                imgs = data['img'].cuda()
                gt_labels = data['gt_label'].numpy()
                

                feat = self.model.module.backbone(imgs)
                if isinstance(feat, (list, tuple)):
                    feat = feat[-1]
                
                if hasattr(self.model.module, 'neck') and self.model.module.neck is not None:
                    feat = self.model.module.neck(feat)

                    if isinstance(feat, (list, tuple)):
                        feat = feat[-1]
                
                # Doing Average pooling if the shape was 4-dim
                if len(feat.shape) > 2:
                    feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
                if len(feat.shape) == 1:
                    feat = feat.unsqueeze(0)
                
                features.append(feat.cpu().numpy())
                labels.append(gt_labels)
                count += len(gt_labels)
        
        if len(features) == 0:
            raise ValueError("The outputs are None.")
        
        features = np.vstack(features)
        labels = np.hstack(labels)
        
        print(f"Finished: {len(features)} samples, feature shape: {features.shape[1]}")
        return features, labels
    
    def perform_tsne_reduction(self, features, n_components=2, perplexity=30, random_state=42):

        print("Staring t-SNE...")

        # adjusting the perplexity for the different number of samples
        perplexity = min(perplexity, len(features) // 4)
        
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            n_jobs=-1
        )
        
        features_2d = self.tsne.fit_transform(features)
        print("Finished.")
        return features_2d
    
    def perform_clustering(self, features_2d, random_state=42):

        print(f"Staring K-means clsuting, and the cluster's number: {self.n_clusters}")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = self.kmeans.fit_predict(features_2d)
        print("Finish clusting.")
        return cluster_labels
    
    def plot_tsne_visualization(self, features_2d, true_labels, cluster_labels, 
                               class_names, save_path_prefix):
        
        # 1. TSNE plot colored by by real labels
        plt.figure(figsize=(10, 8))
        colors = cm.tab20(np.linspace(0, 1, 20))  # 20 colors
        extended_colors = []
        for i in range(self.n_clusters):
            extended_colors.append(colors[i % 20])
        # Draw each category
        for i in range(self.n_clusters):
            mask = true_labels == i
            if mask.any():
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[extended_colors[i]], s=20, alpha=0.8, 
                           label=f'{i}: {class_names[i] if i < len(class_names) else f"class_{i}"}')
        
        plt.title('t-SNE vis by labels', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_true_labels.svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. TSNE plot colored by clustering results
        plt.figure(figsize=(10, 8))
        cluster_colors = cm.tab20(np.linspace(0, 1, 20))
        extended_cluster_colors = []
        for i in range(self.n_clusters):
            extended_cluster_colors.append(cluster_colors[i % 20])
        
        for i in range(self.n_clusters):
            mask = cluster_labels == i
            if mask.any():
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[extended_cluster_colors[i]], s=20, alpha=0.8, 
                           label=f'cluster_{i}')
        
        plt.title('t-SNE vis by clusters', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_cluster_labels.svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Mixed view — Displays the center of the cluster
        plt.figure(figsize=(10, 8))
        # Plot all points (smaller, more transparent)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=cluster_labels, cmap='tab20', s=10, alpha=0.8)
        
        # Plot the center of the cluster
        centers = self.kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        
        plt.title('t-SNE vis - clustering center and cluster data points', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Clsutering Labels', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path_prefix}_clusters_with_centers.svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"The results saved already:")
        print(f"  - True Lables: {save_path_prefix}_true_labels.svg")
        print(f"  - Clustering Results: {save_path_prefix}_cluster_labels.svg")
        print(f"  - Clustering Center: {save_path_prefix}_clusters_with_centers.svg")
    
    def plot_clustering_analysis(self, features_2d, true_labels, cluster_labels, 
                                class_names, save_path):
        """绘制聚类分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 聚类大小分布
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        axes[0, 0].bar(unique_clusters[::5], cluster_counts[::5])  # 每5个显示一个以避免拥挤
        axes[0, 0].set_title('聚类大小分布(per 5 sample)')
        axes[0, 0].set_xlabel('聚类ID')
        axes[0, 0].set_ylabel('样本数量')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 真实类别分布
        unique_true, true_counts = np.unique(true_labels, return_counts=True)
        axes[0, 1].bar(unique_true[::5], true_counts[::5])  # 每5个显示一个
        axes[0, 1].set_title('真实类别分布(per 5 sample)')
        axes[0, 1].set_xlabel('真实类别ID')
        axes[0, 1].set_ylabel('样本数量')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 聚类评估指标
        ari_score = adjusted_rand_score(true_labels, cluster_labels)
        nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
        
        metrics = ['ARI', 'NMI']
        scores = [ari_score, nmi_score]
        bars = axes[0, 2].bar(metrics, scores, color=['skyblue', 'lightcoral'])
        axes[0, 2].set_title('聚类评估指标')
        axes[0, 2].set_ylabel('分数')
        axes[0, 2].set_ylim(0, 1)
        
        for bar, score in zip(bars, scores):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 4. 类内距离 vs 类间距离分析
        intra_cluster_distances = []
        inter_cluster_distances = []
        
        # 随机采样以减少计算量
        n_samples = min(1000, len(features_2d))
        sample_idx = np.random.choice(len(features_2d), n_samples, replace=False)
        sample_features = features_2d[sample_idx]
        sample_clusters = cluster_labels[sample_idx]
        
        for cluster_id in range(min(20, self.n_clusters)):  # 只分析前20个聚类
            cluster_mask = sample_clusters == cluster_id
            if cluster_mask.sum() < 2:
                continue
                
            cluster_points = sample_features[cluster_mask]
            
            # 计算类内距离
            for i in range(len(cluster_points)):
                for j in range(i+1, min(i+10, len(cluster_points))):  # 限制计算量
                    dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
                    intra_cluster_distances.append(dist)
            
            # 计算类间距离（与其他聚类的中心）
            other_centers = self.kmeans.cluster_centers_[self.kmeans.cluster_centers_ != self.kmeans.cluster_centers_[cluster_id]]
            current_center = self.kmeans.cluster_centers_[cluster_id]
            for other_center in other_centers[:10]:  # 只计算前10个
                dist = np.linalg.norm(current_center - other_center)
                inter_cluster_distances.append(dist)
        
        if intra_cluster_distances and inter_cluster_distances:
            axes[1, 0].hist(intra_cluster_distances, alpha=0.5, label='类内距离', bins=30, color='blue')
            axes[1, 0].hist(inter_cluster_distances, alpha=0.5, label='类间距离', bins=30, color='red')
            axes[1, 0].set_title('距离分布')
            axes[1, 0].set_xlabel('欧式距离')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].legend()
        
        # 5. 聚类紧密度分析
        silhouette_samples = []
        for i in range(min(500, len(features_2d))):  # 限制计算量
            point = features_2d[i]
            cluster_id = cluster_labels[i]
            
            # 计算到同聚类其他点的平均距离
            same_cluster_mask = (cluster_labels == cluster_id) & (np.arange(len(cluster_labels)) != i)
            if same_cluster_mask.any():
                same_cluster_points = features_2d[same_cluster_mask]
                a = np.mean([np.linalg.norm(point - p) for p in same_cluster_points[:10]])  # 限制计算量
                
                # 计算到最近其他聚类的平均距离
                other_clusters = np.unique(cluster_labels[cluster_labels != cluster_id])
                b_values = []
                for other_cluster in other_clusters[:5]:  # 只计算前5个其他聚类
                    other_cluster_mask = cluster_labels == other_cluster
                    other_cluster_points = features_2d[other_cluster_mask]
                    if len(other_cluster_points) > 0:
                        b_cluster = np.mean([np.linalg.norm(point - p) for p in other_cluster_points[:5]])
                        b_values.append(b_cluster)
                
                if b_values:
                    b = min(b_values)
                    silhouette = (b - a) / max(a, b)
                    silhouette_samples.append(silhouette)
        
        if silhouette_samples:
            axes[1, 1].hist(silhouette_samples, bins=30, alpha=0.7, color='green')
            axes[1, 1].set_title(f'轮廓系数分布\n平均值: {np.mean(silhouette_samples):.3f}')
            axes[1, 1].set_xlabel('轮廓系数')
            axes[1, 1].set_ylabel('频次')
        
        # 6. 聚类质心分布
        centers = self.kmeans.cluster_centers_
        axes[1, 2].scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
        axes[1, 2].set_title('聚类质心分布')
        axes[1, 2].set_xlabel('TSNE维度 1')
        axes[1, 2].set_ylabel('TSNE维度 2')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"聚类分析图已保存: {save_path}")
        
        return {
            'ari_score': ari_score,
            'nmi_score': nmi_score,
            'avg_silhouette': np.mean(silhouette_samples) if silhouette_samples else 0,
            'n_clusters': self.n_clusters,
            'intra_cluster_avg_dist': np.mean(intra_cluster_distances) if intra_cluster_distances else 0,
            'inter_cluster_avg_dist': np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        }


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visulization tool.')
    parser.add_argument('--config', type=str, required=True,
                        help='the path of confiuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='the path of checkpoint file')
    parser.add_argument('--work_dir', type=str, default='work_dirs/tsne_clustering',
                        help='the path of saved results')
    parser.add_argument('--n_clusters', type=int, default=100,
                        help='the number of clusters')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='the max samples of visulization')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='set the hypermeter of t-SNE perplexity')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='the configuration file`s options')
    
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_clusters <= 0:
        raise ValueError("Clusters number should be more than 0.")

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    setup_multi_processes(cfg)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.work_dir = args.work_dir
    cfg.gpu_ids = [args.gpu_id]
    cfg.model.pretrained = None
    
    if traverse_replace is not None:
        traverse_replace(cfg, 'memcached', False)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'tsne_clustering_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    print_log(f"Beging to t-SNE visulization, the number of clusters: {args.n_clusters}", logger=logger)
    
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=min(cfg.data.imgs_per_gpu, 32),
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    
    if hasattr(dataset.data_source, 'CLASSES'):
        class_names = dataset.data_source.CLASSES
    else:
        class_names = [f"class_{i}" for i in range(self.n_clusters)]
    
    print_log(f"Dataset constaints {len(class_names)} classes", logger=logger)
    
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.cuda()
    
    visualizer = TSNEClusteringVisualizer(model, n_clusters=args.n_clusters)
    print_log("Beging Feature Extractor...", logger=logger)
    features, labels = visualizer.extract_all_features_and_labels(
        data_loader, args.max_samples)
    
    print_log(f"Extracting {len(features)} samples, and the dimension was: {features.shape[1]}", logger=logger)
    
    # 统计每个类别的样本数
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print_log(f"Including {len(unique_labels)} classes", logger=logger)
    
    # TSNE降维
    print_log(f"Staring t-SNE. perplexity={args.perplexity}...", logger=logger)
    features_2d = visualizer.perform_tsne_reduction(features, perplexity=args.perplexity)
    
    # K-means聚类
    print_log(f"Staring K-means clustering. the number of clusters = {args.n_clusters}...", logger=logger)
    cluster_labels = visualizer.perform_clustering(features_2d)
    
    # 生成可视化图
    print_log("Generating the visulization figure...", logger=logger)
    visualization_path_prefix = osp.join(cfg.work_dir, 'tsne_visualization')
    visualizer.plot_tsne_visualization(features_2d, labels, cluster_labels, 
                                     class_names, visualization_path_prefix)

if __name__ == '__main__':
    main() 