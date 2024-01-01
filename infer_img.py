import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archs import UNext
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='wrist', help='model name')
    parser.add_argument('--video', default='inputs/output.mp4', help='video name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.model, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    
    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    

    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.cuda()
    model.eval()
    
    infer_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    
    cap = cv2.VideoCapture(args.video)
    img_index = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        # preprocess
        mask = np.zeros((frame.shape[0], frame.shape[1], 1))
        augmented = infer_transform(image=frame, mask=mask)
        img = augmented['image']
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        # print(f"before inference:{img.shape}")
        
        
        # inference
        with torch.no_grad():
            input = img.cuda()
            model = model.cuda()
            output = model(input)
            
            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
                 
            # print(f"after inference:{output.shape}")

            for c in range(config['num_classes']):
                cv2.imwrite(os.path.join('outputs', config['name'], str(c), str(img_index) + '.jpg'),
                                (output[0, c] * 255).astype('uint8'))
            #     # mask image
            #     out_mask = (output[0, c] * 255).astype('uint8')
            #     print(f"mask shape:{out_mask.shape}")
            #     _, out_mask = cv2.threshold(out_mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
            #     green_mask = np.copy(img[0])
            #     green_mask = green_mask.transpose(1, 2, 0)
                
            #     # boolean indexing and assignment based on mask
            #     green_mask[(out_mask==255).all(-1)] = [0,255,0]
            #     print(f"green_mask shape:{green_mask.shape}")

            #     temp = (img[0]*255).numpy().transpose(1, 2, 0)
            #     print(f"image shape:{temp.shape}")

            #     result = cv2.addWeighted(green_mask, 0.2, temp, 0.8, 0, green_mask)
            #     cv2.imwrite(os.path.join('masks', config['name'], str(c), str(img_index) + '.jpg'), temp)
            
            
            img_index += 1
        # print(img_index, img.shape)
        
        
        #     if self.transform is not None:
        # augmented = self.transform(image=img, mask=mask)
        # img = augmented['image']
        # mask = augmented['mask']
        

        # mask = mask.astype('float32') / 255
        # mask = mask.transpose(2, 0, 1)
        
        # if success:
        #     # cv2.imwrite('inputs/%d.jpg' % img_index, frame)
        #     img_index += 1

    # val_transform = Compose([
    #     Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])
    
    

    # val_dataset = Dataset(
    #     img_ids=val_img_ids,
    #     img_dir=os.path.join('inputs', config['dataset'], 'images'),
    #     mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    #     img_ext=config['img_ext'],
    #     mask_ext=config['mask_ext'],
    #     num_classes=config['num_classes'],
    #     transform=val_transform)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     num_workers=config['num_workers'],
    #     drop_last=False)

    # iou_avg_meter = AverageMeter()
    # dice_avg_meter = AverageMeter()

    # count = 0
    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    # with torch.no_grad():
    #     for input, target, meta in tqdm(val_loader, total=len(val_loader)):
    #         input = input.cuda()
    #         target = target.cuda()
    #         model = model.cuda()
    #         # compute output
    #         output = model(input)


    #         iou,dice = iou_score(output, target)
    #         iou_avg_meter.update(iou, input.size(0))
    #         dice_avg_meter.update(dice, input.size(0))

    #         output = torch.sigmoid(output).cpu().numpy()
    #         output[output>=0.5]=1
    #         output[output<0.5]=0

    #         for i in range(len(output)):
    #             for c in range(config['num_classes']):
    #                 cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
    #                             (output[i, c] * 255).astype('uint8'))

    # print('IoU: %.4f' % iou_avg_meter.avg)
    # print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
