from model.resnet import *
#------------这个是去除分割任务的-----------------------#
class ResNetSegmentationModelWithMoE_wo_seg(nn.Module):
    def __init__(self, num_classes=9, num_experts=2):
        super(ResNetSegmentationModelWithMoE_wo_seg, self).__init__()

        # 加载预训练的ResNet50模型，并去掉最后的全连接层
        self.resnet = ResNetWithMOE()

        # 定义卷积块（包含卷积和激活函数）
        self.conv_block4 = self._conv_block(256, 128)
        self.conv_block5 = self._conv_block(128, 64)
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

        fusion = self.fusion_head(x)
        return fusion

#-----------这个是去除LoraCLIP的----------------------#
class ResNetSegmentationModelWithMoE_wo_LoraCLIP(nn.Module):
    def __init__(self, num_classes=9, num_experts=2):
        super(ResNetSegmentationModelWithMoE_wo_LoraCLIP, self).__init__()

        # 加载预训练的ResNet50模型，并去掉最后的全连接层
        self.resnet = ResNetWithMOE()

        # 定义卷积块（包含卷积和激活函数）
        self.conv_block4 = self._conv_block(256, 128)
        self.conv_block5 = self._conv_block(128, 64)

        # 最终的分割输出层
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # 最终的融合输出层
        self.fusion_head = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)


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
        # x = vi + ir
        # x = self.fusion_module(x)
        x = torch.cat([vi, ir], dim=1)

        # 经过每个卷积块并添加MoE-Adapter
        x = self.conv_block4(x)
        x = self.conv_block5(x) + vi3 + ir3

        # 最后卷积输出分割掩码
        seg = self.seg_head(x)
        fusion = self.fusion_head(x)

        return seg, fusion


#---------------这个是去除Moe的-----------------------#
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = feature_exbase(3, 16)
        self.conv2 = feature_exbase(16, 32)
        self.conv3 = feature_exbase(32, 64)
        self.conv4 = feature_exbase(64, 128)

    def forward(self, x, vi=False):
        x = self.conv1(x)
        y1 = x
        x = self.conv2(x)
        y2 = x
        x = self.conv3(x)
        y3 = x
        x = self.conv4(x)
        return x, y3, y2, y1



class ResNetSegmentationModelWithMoE(nn.Module):
    def __init__(self, num_classes=9, num_experts=2):
        super(ResNetSegmentationModelWithMoE, self).__init__()

        # 加载预训练的ResNet50模型，并去掉最后的全连接层
        self.resnet = ResNet()

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
