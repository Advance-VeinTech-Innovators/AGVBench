# pip install umap-learn torch torchvision scikit-learn tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import umap
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# 1. CIFAR-10 数据
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# 2. 特征提取器：ResNet18
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = torchvision.models.resnet18(pretrained=True)
backbone.fc = nn.Identity()
backbone.to(device)
backbone.eval()

features = []
labels = []

print("Extracting features...")
with torch.no_grad():
    for images, targets in tqdm(train_loader):
        images = images.to(device)
        feats = backbone(images).cpu().numpy()
        features.append(feats)
        labels.extend(targets.numpy())

features = np.concatenate(features, axis=0)  # (N, D)
labels = np.array(labels)

# -----------------------------
# 3. Manifold Learning (UMAP)
# -----------------------------
print("Running UMAP for manifold embedding...")
umap_embedder = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric='euclidean')
embedded = umap_embedder.fit_transform(features)  # shape: (N, 10)

# -----------------------------
# 4. Manifold-Aware Soft Labeling (KNN on manifold)
# -----------------------------
print("Generating soft labels based on manifold neighbors...")

n_classes = 10
N = len(labels)
soft_labels = np.zeros((N, n_classes))
k = 10  # number of neighbors

knn = NearestNeighbors(n_neighbors=k + 1).fit(embedded)
distances, indices = knn.kneighbors(embedded)

for i in range(N):
    neighbor_labels = labels[indices[i][1:]]  # exclude self
    hist = np.bincount(neighbor_labels, minlength=n_classes)
    soft_labels[i] = hist / hist.sum()

soft_labels = torch.tensor(soft_labels, dtype=torch.float32)

# -----------------------------
# 5. 分类器定义
# -----------------------------
class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

model = Classifier().to(device)

# -----------------------------
# 6. 使用 Soft Label 训练模型
# -----------------------------
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_with_manifold_labels():
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            outputs = model(images)
            log_probs = torch.log_softmax(outputs, dim=1)

            soft_batch = soft_labels[i * 64: i * 64 + len(images)].to(device)
            loss = criterion(log_probs, soft_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

train_with_manifold_labels()

# -----------------------------
# 7. 测试模型准确率
# -----------------------------
def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate()
