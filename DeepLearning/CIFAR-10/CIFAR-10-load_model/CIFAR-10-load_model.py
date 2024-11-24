import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

# 定义网络模型
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = CIFAR10Classifier()
# 使用 weights_only=True 来加载模型权重
model.load_state_dict(torch.load('cifar10_classifier.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 加载CIFAR10数据集
cifar10_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 随机选择16张图像
indices = np.random.choice(len(cifar10_dataset), 16, replace=False)
images = [cifar10_dataset[i][0] for i in indices]
labels = [cifar10_dataset[i][1] for i in indices]

# 将图像转换为批量张量
images = torch.stack(images)

# 进行预测
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    predicted_classes = [classes[p] for p in predicted]

# 显示图像和预测结果
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.ravel()

for i in range(16):
    img = images[i].permute(1, 2, 0).numpy()  # 将图像张量转换为numpy数组
    img = (img * 0.5) + 0.5  # 反归一化
    axes[i].imshow(img)
    axes[i].set_title(f"Pred: {predicted_classes[i]}\nTrue: {classes[labels[i]]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()