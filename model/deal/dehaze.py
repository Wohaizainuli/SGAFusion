import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ---- 模型部分：预测透射率图和大气光 ----
class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()
        # 卷积层：用于提取特征
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=1, padding=0)  # 输出1个通道，用于透射率图

        # 全连接层：用于估计大气光值
        self.fc1 = nn.Linear(32 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 提取透射率图的特征
        x = nn.LeakyReLU()(self.conv2(x))
        x = nn.LeakyReLU()(self.conv3(x))
        x = nn.LeakyReLU()(self.conv4(x))
        transmission = torch.sigmoid(self.conv5(x))  # 透射率图输出范围为[0,1]

        # 用于估计大气光值的特征池化
        #(x.shape)
        pooled = F.adaptive_avg_pool2d(x, (32, 32))
        pooled = pooled.view(pooled.size(0), -1)  # flatten
        #print(pooled.shape)
        A = torch.sigmoid(self.fc1(pooled))
        A = torch.sigmoid(self.fc2(A))  # 大气光值输出范围为[0,1]（RGB）

        return transmission, A


# ---- 去雾过程：通过大气散射模型公式去雾 ----
def dehaze_image(hazy_image, transmission, A, t0=0.1):
    # 假设 hazy_image 形状为 (B, C, H, W)
    transmission = torch.clamp(transmission, min=t0)  # 避免除0情况，透射率下限 t0

    # 扩展 A 到 (B, C, H, W) 维度
    A = A.view(A.size(0), A.size(1), 1, 1)  # A shape: (B, 1, 1, 1)
    A = A.expand_as(hazy_image)

    # 去雾公式：J(x) = (I(x) - A) / t(x) + A
    dehazed_image = (hazy_image - A) / transmission + A
    #dehazed_image = torch.clamp(dehazed_image, 0, 1)  # 确保像素值在[0,1]范围内

    return dehazed_image

