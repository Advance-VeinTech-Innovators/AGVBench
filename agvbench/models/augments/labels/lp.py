import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from tqdm import tqdm

# -----------------------------
# 1. 数据预处理与加载
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
# 2. 提取特征用于标签传播
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = torchvision.models.resnet18(pretrained=True)
backbone.fc = nn.Identity()
backbone.to(device)
backbone.eval()

features = []
labels = []

print("Extracting features for label propagation...")
with torch.no_grad():
    for images, targets in tqdm(train_loader):
        images = images.to(device)
        feats = backbone(images).cpu().numpy()
        features.append(feats)
        labels.extend(targets.numpy())

features = np.concatenate(features, axis=0)   # shape: (N, D)
labels = np.array(labels)

# -----------------------------
# 3. Label Propagation 增强标签
# -----------------------------
print("Running Label Propagation (fully labeled)...")
label_prop = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
label_prop.fit(features, labels)

# 得到 soft label 分布 (shape: N x 10)
soft_labels = torch.tensor(label_prop.label_distributions_, dtype=torch.float32)

# -----------------------------
# 4. 构建分类模型
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
# 5. 使用 soft label 训练模型
# -----------------------------
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_soft_label_model():
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            outputs = model(images)
            log_probs = torch.log_softmax(outputs, dim=1)

            # 获取soft label
            label_batch = soft_labels[i * 64: i * 64 + len(images)].to(device)

            loss = criterion(log_probs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

train_soft_label_model()

# -----------------------------
# 6. 测试模型性能
# -----------------------------
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate()
