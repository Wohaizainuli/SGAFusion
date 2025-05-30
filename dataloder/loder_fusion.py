import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# 定义transform
import torchvision.transforms as transforms
to_tensor = transforms.Compose([        # 调整图像大小
    transforms.ToTensor(),                 # 转换为Tensor
])

class ImageSegmentationDataset(Dataset):
    def __init__(self, visible_dir, infrared_dir, label_dir, visible_gt_dir=None, transform=to_tensor):
        self.visible_images = sorted(os.listdir(visible_dir))
        self.infrared_images = sorted(os.listdir(infrared_dir))
        self.label_images = sorted(os.listdir(label_dir))
        if visible_gt_dir != None:
            self.visible_gt_images = sorted(os.listdir(visible_gt_dir))
        self.visible_dir = visible_dir
        self.infrared_dir = infrared_dir
        self.visible_gt_dir = visible_gt_dir
        self.label_dir = label_dir
        self.transform = transform

        # 获取标签图像的唯一像素值，计算类别数
        self.class_map = self._get_class_map()

    def _get_class_map(self):
        """
        遍历标签图像，获取所有类的灰度值，计算类别数
        """
        class_map = set()  # 使用 set 去重
        for label_image_name in self.label_images:
            label_image = Image.open(os.path.join(self.label_dir, label_image_name))
            label_array = np.array(label_image)

            # 确保标签图像是单通道（灰度图）
            if label_array.ndim == 2:  # 单通道灰度图
                unique_values = np.unique(label_array)
                for value in unique_values:
                    class_map.add(value)
            else:
                raise ValueError(f"标签图像应为单通道灰度图，当前标签图像为: {label_array.ndim}维")

        return class_map

    def __len__(self):
        return len(self.visible_images)

    def __getitem__(self, idx):
        visible_image_name = self.visible_images[idx]
        infrared_image_name = self.infrared_images[idx]
        label_image_name = self.label_images[idx]
        visible_image_gt_name = self.visible_gt_images[idx]

        visible_image = Image.open(os.path.join(self.visible_dir, visible_image_name))
        visible_image_224 = visible_image.resize((224,224))
        infrared_image = Image.open(os.path.join(self.infrared_dir, infrared_image_name))
        label_image = Image.open(os.path.join(self.label_dir, label_image_name))
        visible_image_gt = Image.open(os.path.join(self.visible_gt_dir, visible_image_gt_name))

        # 将标签图像从灰度值映射为单通道类别标签
        label_array = np.array(label_image)
        label = self._map_labels_to_single_channel(label_array)

        if self.transform:
            visible_image = self.transform(visible_image)
            infrared_image = self.transform(infrared_image)
            visible_image_gt = self.transform(visible_image_gt)
            visible_image_224 = self.transform(visible_image_224)
            # 标签不进行 ToTensor 操作
            label = torch.tensor(label, dtype=torch.long)  # 将标签转换为 long 类型的 Tensor

        return visible_image, infrared_image, visible_image_gt, visible_image_224, label

    def _map_labels_to_single_channel(self, label_array):
        """
        将灰度标签图像映射为类别标签图像
        """
        h, w = label_array.shape
        label = np.zeros((h, w), dtype=np.uint8)

        # 对于灰度图标签图像，直接使用灰度值作为类索引
        for i in range(h):
            for j in range(w):
                value = label_array[i, j]
                if value in self.class_map:
                    label[i, j] = list(self.class_map).index(value)

        return label

    def get_num_classes(self):
        return len(self.class_map)


class ImageFusionDataset(Dataset):
    def __init__(self, visible_dir, infrared_dir, visible_gt_dir=None, transform=to_tensor):
        self.name = False
        self.visible_images = sorted(os.listdir(visible_dir))
        self.infrared_images = sorted(os.listdir(infrared_dir))
        if visible_gt_dir != None:
            self.visible_gt_images = sorted(os.listdir(visible_gt_dir))
            self.name = True
        self.visible_dir = visible_dir
        self.infrared_dir = infrared_dir
        self.visible_gt_dir = visible_gt_dir
        self.transform = transform


    def __len__(self):
        return len(self.visible_images)

    def __getitem__(self, idx):
        if self.name:
            visible_image_name = self.visible_images[idx]
            infrared_image_name = self.infrared_images[idx]
            visible_image_gt_name = self.visible_gt_images[idx]

            visible_image = Image.open(os.path.join(self.visible_dir, visible_image_name))
            visible_image_224 = visible_image.resize((224,224))
            infrared_image = Image.open(os.path.join(self.infrared_dir, infrared_image_name)).convert("L")
            visible_image_gt = Image.open(os.path.join(self.visible_gt_dir, visible_image_gt_name))


            if self.transform:
                visible_image = self.transform(visible_image)
                infrared_image = self.transform(infrared_image)
                visible_image_gt = self.transform(visible_image_gt)
                visible_image_224 = self.transform(visible_image_224)

            return visible_image, infrared_image, visible_image_gt, visible_image_224
        else:
            visible_image_name = self.visible_images[idx]
            infrared_image_name = self.infrared_images[idx]

            visible_image = Image.open(os.path.join(self.visible_dir, visible_image_name))
            visible_image_224 = visible_image.resize((224, 224))
            infrared_image = Image.open(os.path.join(self.infrared_dir, infrared_image_name)).convert("L")

            if self.transform:
                visible_image = self.transform(visible_image)
                infrared_image = self.transform(infrared_image)

                visible_image_224 = self.transform(visible_image_224)


            return visible_image, infrared_image, visible_image_224, self.visible_images[idx]





if __name__ == '__main__':

    # 用法示例
    visible_dir = r"G:\2024F_10_16\LoraMoe\MSRS-main\test\vi"
    infrared_dir = r"G:\2024F_10_16\LoraMoe\MSRS-main\test\ir"
    label_dir = r"G:\2024F_10_16\LoraMoe\MSRS-main\test\Segmentation_labels"

    # 创建数据集实例
    dataset = ImageSegmentationDataset(visible_dir, infrared_dir, label_dir)

    # 打印类别数
    print(f"Number of classes: {dataset.get_num_classes()}")

    # 使用 DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 遍历数据集
    for visible, infrared, label in dataloader:
        print(visible.shape, infrared.shape, label.shape)  # 输出每个批次的图像和标签形状
