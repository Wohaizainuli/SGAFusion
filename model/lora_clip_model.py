import torch
import torch.nn as nn
import clip

# 定义自定义 LoRALinear 类
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1):
        super(LoRALinear, self).__init__()
        self.r = r
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features)

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r) * (alpha / r))
            self.lora_B = nn.Parameter(torch.randn(r, out_features) * (alpha / r))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        if self.r > 0:
            lora_out = torch.matmul(x, self.lora_A).matmul(self.lora_B)
            return self.linear(x) + lora_out
        else:
            return self.linear(x)

# 定义 CLIPClassifier 类并使用 LoRALinear 进行微调
class CLIPClassifier(nn.Module):
    def __init__(self, num_classes, clip_model='ViT-B/32', r=4, alpha=1, pretrained=True):
        super(CLIPClassifier, self).__init__()

        # 加载预训练 CLIP 模型
        self.clip_model, _ = clip.load(clip_model, device="cuda")
        self.clip_model.float()

        # 仅解冻图像编码器的权重
        if pretrained:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # 遍历 visual 模块并用 LoRA 层替换线性层
        visual_modules = list(self.clip_model.visual.named_modules())
        for name, module in visual_modules:
            if isinstance(module, nn.Linear):
                in_features, out_features = module.in_features, module.out_features
                # 替换为 LoRALinear
                setattr(self.clip_model.visual, name, LoRALinear(in_features, out_features, r, alpha))

        # 获取图像编码器的输出维度
        image_encoding_dim = self.clip_model.visual.output_dim

        # 定义分类器的全连接层
        self.fc1 = nn.Linear(image_encoding_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 使用 CLIP 的图像编码器提取特征
        x = self.clip_model.encode_image(x)
        x = self.relu(self.fc1(x))
        feature = x
        x = self.fc2(x)
        return x, feature


if __name__ == '__main__':
    model = CLIPClassifier(num_classes=3).cuda()
    img = torch.randn(1,3,224,224).cuda()
    for i in model(img):
        print(i.shape)
