import os
import cv2
import yaml
import time
import torch
import datetime
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
    parser.add_argument('--camid', default=0, help='camera id')
    parser.add_argument('--video', default='', help='video name')
    parser.add_argument('--output', default='outputs', help='output dir')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.model, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # print('-'*20)
    # for key in config.keys():
    #     print('%s: %s' % (key, str(config[key])))
    # print('-'*20)
    
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
    
    # open camera or video file 
    cap = cv2.VideoCapture()
    if args.video != '':
        flag = cap.open(args.video)
    else:
        flag = cap.open(args.camid)
    if flag == False:
        print('open video failed')
        return
    # set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # create folder to save original images
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    out_folder = os.path.join(args.output, time_now)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    img_index = 0
    starttime = datetime.datetime.now()
    while (True):
        success, frame = cap.read()
        if not success:
            break
        
        # save original images
        img_file = os.path.join(out_folder, '{:06d}.jpg'.format(img_index))
        cv2.imwrite(img_file, frame)
        
        # image shape
        img_h, img_w, _ = frame.shape
        
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
            ret_img = frame.copy()
            mask_img = cv2.resize(mask_img, dsize=(img_w, img_h))
            color = np.array([0,255,0], dtype='uint8')
            masked_img = np.where(mask_img[...,None], color, ret_img)
            ret_img = cv2.addWeighted(ret_img, 0.7, masked_img, 0.3, 0)
            # cv2.imwrite(ret_name, ret_img)
            
            img_index += 1
            
            # display the resulting frame 
            cv2.imshow('ret_img', ret_img) 
        
            # the 'q' button is set as the quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    torch.cuda.empty_cache()
    # After the loop release the cap object 
    cap.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
    
    endtime = datetime.datetime.now()
    print(f'infer {img_index} images in {(endtime-starttime).seconds} seconds.')

if __name__ == '__main__':
    main()
