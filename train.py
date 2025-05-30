import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import torch.nn.functional as F
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloder.loder_fusion import ImageSegmentationDataset, ImageFusionDataset
from model.resnet import ResNetSegmentationModelWithMoE
from scripts.losses import fusion_loss
import torch.nn as nn
from model.lora_clip_model import CLIPClassifier
loss_base = fusion_loss()
# 定义分割损失函数
criterion = nn.CrossEntropyLoss()

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    init_seeds(2024)
    save_path = 'runs/'

    batch_size = 1
    num_works = 1
    lr = 0.001
    Epoch = 50

    visible_dir = "MSRS-main/train/vi"
    infrared_dir = "MSRS-main/train/ir"
    label_dir = "MSRS-main/train/Segmentation_labels"
    vis_gt_dir = "MSRS-main/train/vi"
    train_dataset = ImageSegmentationDataset(visible_dir, infrared_dir, label_dir, visible_gt_dir=vis_gt_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = ResNetSegmentationModelWithMoE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()

    clip_model = CLIPClassifier(num_classes=9).cuda()
    clip_model = torch.load('pretrained/best_cls.pth')
    clip_model.cuda()
    clip_model.eval()

    model.train()
    for epoch in range(Epoch):
        if epoch < Epoch // 2:
            lr = lr
        else:
            lr = lr * (Epoch - epoch) / (Epoch - Epoch // 2)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_tqdm = tqdm(train_loader, total=len(train_loader), ascii=True)
        for visible_image, infrared_image, visible_image_gt, visible_image_224, label in train_tqdm:
            optimizer.zero_grad()
            visible = visible_image.cuda()
            infrared = infrared_image.cuda()
            visible_image_gt = visible_image_gt.cuda()
            visible_image_224 = visible_image_224.cuda()
            label = label.cuda()
            _, feature = clip_model(visible_image_224)

            with torch.cuda.amp.autocast():
                seg, fusion = model(visible, torch.cat([infrared]*3, dim=1), feature)
                loss_seg = criterion(seg, label)  # seg: (b, 9, w, h), label: (b, h, w)
                loss_fusion = loss_base(visible_image_gt, infrared, fusion)
                loss = loss_seg + loss_fusion

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            # **梯度裁剪**
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()


            ##### Display loss and current learning rate
            train_tqdm.set_postfix(
                epoch=epoch,
                loss=loss.item(),
                loss_seg=loss_seg.item(),
                loss_fusion=loss_fusion.item(),
                lr=optimizer.param_groups[0]['lr']  # 显示当前学习率
            )

        #### Save the trained model
        torch.save(model.state_dict(), f'{save_path}/model_{epoch}.pth')
