"""测试融合网络"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloder.loder_fusion import ImageFusionDataset
from model.resnet import ResNetSegmentationModelWithMoE
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'





if __name__ == '__main__':


    batch_size = 1
    num_works = 1
    datastes = 'LLVIP'
    visible_dir = f"test/{datastes}/Vis"
    infrared_dir = f"test/{datastes}/Inf"
    save_path = f'test/{datastes}/ours'
    test_dataset = ImageFusionDataset(visible_dir, infrared_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = ResNetSegmentationModelWithMoE().cuda()
    model.load_state_dict(torch.load('runs/model_19_finetune.pth'))
    model.eval()


    #clip_model = CLIPClassifier(num_classes=9).cuda()
    clip_model = torch.load('pretrained/best_cls.pth')
    clip_model.cuda()
    clip_model.eval()



    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for visible_image, infrared_image, visible_image_224, name in test_tqdm:
            visible_image = visible_image.cuda()
            infrared_image = infrared_image.cuda()
            visible_image_224 = visible_image_224.cuda()
            _, feature = clip_model(visible_image_224)

            with torch.cuda.amp.autocast():
                seg, fused = model(visible_image, torch.cat([infrared_image]*3, dim=1), feature)
                fused = torch.clamp(fused, min=0.00001, max=1)

            #fused = YCrCb2RGB(fused[0], cr[0], cb[0])

            rgb_fused_image = transforms.ToPILImage()(fused[0])
            rgb_fused_image.save(f'{save_path}/{name[0]}')
