import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# MoEAdapter：包含多个专家和门控网络，并在输出后添加激活函数
class MoEAdapter(nn.Module):
    def __init__(self, input_channels, num_experts=3, output_channels=128, negative_slope=0.2, mix_sense=None):
        super(MoEAdapter, self).__init__()
        self.num_experts = num_experts

        # 创建专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, int(input_channels //8), kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(int(input_channels //8), input_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            ) for _ in range(num_experts)
        ])

        # 创建门控网络
        self.gating_network = nn.Conv2d(input_channels, num_experts, kernel_size=1)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        # 获取每个专家的输出
        expert_outputs = [expert(x) for expert in self.experts]

        # 获取门控网络的权重
        gate_weights = F.softmax(self.gating_network(x), dim=1)  # 输出每个专家的权重

        # 加权求和各个专家的输出
        weighted_sum = sum(gate_weights[:, i:i+1, :, :] * expert_output for i, expert_output in enumerate(expert_outputs))

        # 添加激活函数（LeakyReLU）
        return self.leaky_relu(weighted_sum)

#------------用于resnet浅层的专家------------#
from model.deal.dehaze import DehazeNet, dehaze_image
from model.deal.low_enhance import Low_enhance_net, low_enhance_image
class MoEAdapter_shallow(nn.Module):
    def __init__(self):
        super(MoEAdapter_shallow, self).__init__()
        self.dehazenet = DehazeNet()
        self.lownet = Low_enhance_net()
        self.experts = nn.Conv2d(in_channels=16, out_channels=16, stride=1, kernel_size=1, padding=0)

        # 创建门控网络
        self.gating_network = nn.Conv2d(16, 3, kernel_size=1)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        tx, a = self.dehazenet(x)
        r = self.lownet(x)
        x1 = dehaze_image(x, tx, a)
        x2 = low_enhance_image(x, r)
        x3 = self.experts(x)
        expert_outputs = [x1, x2, x3]

        # 获取门控网络的权重
        gate_weights = F.softmax(self.gating_network(x), dim=1)  # 输出每个专家的权重

        # 加权求和各个专家的输出
        weighted_sum = sum(
            gate_weights[:, i:i + 1, :, :] * expert_output for i, expert_output in enumerate(expert_outputs))

        # 添加激活函数（LeakyReLU）
        return self.leaky_relu(weighted_sum)


# class ResNetWithMOE(nn.Module):
#     def __init__(self, moe_layer=MoEAdapter,moe_layer_scene=MoEAdapter_shallow, input_shape=(3, 224, 224)):
#         super(ResNetWithMOE, self).__init__()
#         self.resnet = models.resnet50(pretrained=True)
#         layers = []
#
#         # Initialize a sample tensor
#         x = torch.rand(1, *input_shape)  # Create a sample input tensor
#         self.scene = True
#         for layer in list(self.resnet.children())[:-2]:
#             # Forward pass to infer output channels
#             x = layer(x)
#             out_channels = x.size(1)  # Number of output channels
#             # Append the current layer
#             layers.append(layer)
#             # Append the MoE layer
#             if self.scene:
#                 layers.append(moe_layer_scene())
#                 #print('add', x.shape)
#                 self.scene = False
#             else:
#                 layers.append(moe_layer(input_channels=out_channels, output_channels=out_channels, num_experts=2))
#
#         self.resnet_with_moe = nn.Sequential(*layers)
#
#     def forward(self, x):
#         features = []
#         for layer in self.resnet_with_moe:
#             x = layer(x)
#             features.append(x)  # Save the output of each layer
#
#         return features  # Return all layer outputs as a list
#----------这一个融合网络，不使用ResNet------------------#
class feature_exbase(nn.Module):
    def __init__(self, in_channel, out_channel): #初始化
        super(feature_exbase, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=1, padding=1, kernel_size=3)

    def forward(self, x):   #操作
        act = nn.LeakyReLU()
        x = self.conv(x)
        x = act(x)
        return x



class ResNetWithMOE(nn.Module):
    def __init__(self):
        super(ResNetWithMOE, self).__init__()
        self.conv1 = feature_exbase(3, 16)
        self.conv2 = feature_exbase(16, 32)
        self.conv3 = feature_exbase(32, 64)
        self.conv4 = feature_exbase(64, 128)
        self.moe1 = MoEAdapter_shallow()
        self.moe2 = MoEAdapter(input_channels=32, output_channels=32, num_experts=2)
        self.moe3 = MoEAdapter(input_channels=64, output_channels=64, num_experts=2)
        self.moe4 = MoEAdapter(input_channels=128, output_channels=128, num_experts=2)

    def forward(self, x, vi=False):
        x = self.conv1(x)
        if vi:
            x = self.moe1(x)
        y1 = x
        x = self.conv2(x)
        x = self.moe2(x)
        y2 = x
        x = self.conv3(x)
        x = self.moe3(x)
        y3 = x
        x = self.conv4(x)
        x = self.moe4(x)
        return x, y3, y2, y1




#--------------交叉注意力融合-----------------------#
class CrossAttentionFusion(nn.Module):
    def __init__(self, spatial_dim=(640, 480), feature_dim1=128, feature_dim2=512, embed_dim=64, num_heads=2):
        super(CrossAttentionFusion, self).__init__()
        self.spatial_dim = spatial_dim
        self.feature_dim1 = feature_dim1
        self.feature_dim2 = feature_dim2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 使用 1x1 卷积将 x1 降维到 embed_dim
        self.fc1 = nn.Conv2d(feature_dim1, embed_dim, kernel_size=1)

        # 将 x2 降维到 embed_dim
        self.fc2 = nn.Linear(feature_dim2, embed_dim)

        # 使用多头注意力实现交叉注意力
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x1, x2):
        # 处理 x1: (1, 128, 640, 480) -> (seq_len, batch_size, embed_dim)
        b, c, w, h = x1.shape
        x1 = self.fc1(x1)  # (1, embed_dim, 640, 480)
        x1 = x1.view(b, self.embed_dim, -1).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)

        # 处理 x2: (1, 512) -> (1, embed_dim)
        x2 = self.fc2(x2)  # (1, embed_dim)
        x2 = x2.unsqueeze(0)  # 增加序列长度维度，变为 (1, batch_size, embed_dim)

        # 交叉注意力：x1 以 x2 作为键和值
        attn_output, _ = self.cross_attention(x1, x2, x2)

        # 将结果加回到 x1
        fused_output = attn_output + x1

        # 恢复到原始的 (b, embed_dim, w, h) 格式
        fused_output = fused_output.permute(1, 2, 0).view(b, self.embed_dim, w, h)

        return fused_output


# ResNetSegmentationModelWithMoE：集成MoE-Adapter的ResNet分割模型
class ResNetSegmentationModelWithMoE(nn.Module):
    def __init__(self, num_classes=9, num_experts=2):
        super(ResNetSegmentationModelWithMoE, self).__init__()

        # 加载预训练的ResNet50模型，并去掉最后的全连接层
        self.resnet = ResNetWithMOE()

        # 定义卷积块（包含卷积和激活函数）
        self.conv_block4 = self._conv_block(256, 128)
        self.conv_block5 = self._conv_block(128, 64)

        # 最终的分割输出层
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # 最终的融合输出层
        self.fusion_head = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # 交叉注意力场景信息嵌入
        self.cross = CrossAttentionFusion(embed_dim=128)

    def _conv_block(self, in_channels, out_channels):
        """卷积块，包含卷积、激活函数、MoE-Adapter"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            MoEAdapter(input_channels=out_channels, output_channels=out_channels, num_experts=2),
        )

    def forward(self, vi, ir, feature):
        vi, vi3, vi2, vi1 = self.resnet(vi, vi=True)
        ir, ir3, ir2, ir1 = self.resnet(ir)

        #print(vi1.shape, vi2.shape, vi3.shape, vi4.shape, vi_last.shape)
        x = vi + ir
        # x = self.fusion_module(x)
        x = torch.cat([self.cross(x, feature), x], dim=1)

        # 经过每个卷积块并添加MoE-Adapter
        x = self.conv_block4(x)
        x = self.conv_block5(x) + vi3 + ir3

        # 最后卷积输出分割掩码
        seg = self.seg_head(x)
        fusion = self.fusion_head(x)

        return seg, fusion

# # 示例输入（假设输入是3通道图像，尺寸为256x256）
# input_image = torch.randn(1, 3, 640, 480).cuda()
# feature = torch.rand(1,512).cuda()
#
# # 创建带MoE优化的ResNet分割模型并进行前向传播
# model = ResNetSegmentationModelWithMoE(num_classes=9).cuda()
# output = model(input_image, input_image, feature)

