# pip install scikit-fuzzy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import skfuzzy as fuzz
from tqdm import tqdm

# -----------------------
# 1. 数据准备
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------
# 2. 预训练特征提取（用ResNet18）
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = torchvision.models.resnet18(pretrained=True)
feature_extractor.fc = nn.Identity()  # 去掉最后分类层
feature_extractor.to(device)
feature_extractor.eval()

print("Extracting features for FCM clustering...")
features = []
labels = []
with torch.no_grad():
    for images, targets in tqdm(train_loader):
        images = images.to(device)
        feats = feature_extractor(images)
        features.append(feats.cpu().numpy())
        labels.extend(targets.numpy())
features = np.concatenate(features, axis=0)

# -----------------------
# 3. 使用 FCM 聚类增强标签
# -----------------------
n_classes = 10
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data=features.T, c=n_classes, m=2.0, error=0.005, maxiter=1000, init=None)

# u 是 (cluster_num, sample_num)，表示隶属度
fuzzy_labels = u.T  # shape: (N, 10)

# -----------------------
# 4. 定义分类模型
# -----------------------
class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)

model = Classifier(num_classes=n_classes).to(device)

# -----------------------
# 5. 模糊标签训练
# -----------------------
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_with_fuzzy_labels():
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            outputs = model(images)

            # 获取 batch 中的 fuzzy label
            fuzzy_batch = torch.tensor(fuzzy_labels[i * 64: i * 64 + len(images)], dtype=torch.float32).to(device)

            log_probs = torch.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, fuzzy_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}")

train_with_fuzzy_labels()

# -----------------------
# 6. 评估函数
# -----------------------
def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate()
