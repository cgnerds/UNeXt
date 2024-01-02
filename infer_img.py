import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import albumentations as A
import torch.backends.cudnn as cudnn
from albumentations.core.composition import Compose
# project 
import archs

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
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])
    
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
        os.makedirs(os.path.join('masks', config['name'], str(c)), exist_ok=True)
    
    cap = cv2.VideoCapture(args.video)
    img_index = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        
        # preprocess img-[1,3,512,512]
        img = infer_transform(image=frame)['image']
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        # inference
        with torch.no_grad():
            input = img.cuda()
            model = model.cuda()
            output = model(input)
            
            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
            
            # TODO multi-class
            # save mask (0) - (512, 512)
            mask_name = os.path.join('masks', config['name'], str(0), str(img_index) + '.jpg')
            mask_img = (output[0, c] * 255).astype('uint8')
            cv2.imwrite(mask_name, mask_img)
            # save image with mask
            ret_name = os.path.join('outputs', config['name'], str(0), str(img_index) + '.jpg')
            ret_img = cv2.resize(frame, dsize=(config['input_h'], config['input_w']))
            color = np.array([0,255,0], dtype='uint8')
            masked_img = np.where(mask_img[...,None], color, ret_img)
            ret_img = cv2.addWeighted(ret_img, 0.7, masked_img, 0.3, 0)
            cv2.imwrite(ret_name, ret_img)
            
            img_index += 1

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
