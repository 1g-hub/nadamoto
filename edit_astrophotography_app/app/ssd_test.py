from app.ssd_model import build_ssd
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from math import sqrt
from glob import glob
from PIL import Image

import torch
from app.utils.augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans
from app.utils.dataloader import RareplanesDataset, DataTransform, xml_to_list, od_collate_fn, get_color_mean
from app.data import BaseTransform, VOC_CLASSES as labelmap
from app.edit import *

def get_device(gpu_id=-1):
    torch.device("cpu")

def detect(frame, net, transform, check):
    height, width = frame.shape[0], frame.shape[1]
    grayscale = True
    img = frame.copy()
    frame_t = transform(frame)[0]
    frame_t = np.array(frame_t)                                                                                 
    x = torch.from_numpy(frame_t).permute(2, 0, 1)                                                                  
    x = x.unsqueeze(0)                                                                                               
    y = net(x)                                                                                                                                                                                             
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    conf = 0
    count = 0
    for i in range(detections.size(1)):                                                                             
        j = 0
        max_j = 0
        while detections[0, i, j, 0] >= 0.6:
            if max_j < detections[0, i, j, 0]:                                                                        
                pt = (detections[0, i, j, 1:] * scale).numpy()
                max_j = detections[0, i, j, 0]                                                          

            # 信頼度が最も高い検出領域の情報の保存
            if detections[0, i, j, 0] > conf:
                pt_max = pt
                conf = detections[0, i, j, 0]
                label = labelmap[i-1]
            j += 1
            count += 1

        """
        SSD の検出結果を表示する (flame を保存することで確認できる)
        """
        # if max_j > 0:
        #     cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 0, 255), 2)                
        #     print((int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])))
        #     cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[3])-40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)  
        #     cv2.putText(frame, str(float(max_j)), (int(pt[0]), int(pt[3])), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

    if count > 0:
        # 信頼度が最も高い領域のみを切り出す
        mask = np.zeros((height, width, 3), np.uint8)
        mask = cv2.rectangle(mask, (int(pt_max[0]), int(pt_max[1])), (int(pt_max[2]), int(pt_max[3])), (255, 255, 255), -1)
        img_AND = cv2.bitwise_and(img, mask)
        img_AND = cv2.cvtColor(img_AND,cv2.COLOR_BGR2GRAY)
        # 星座の正確な位置検出
        edit_img, coordinate, size, color = edit(img_AND, label, check)

    return edit_img, coordinate, size, color

def test(image, check):
    device = get_device(gpu_id=0)

    input_size = 300

    # 検出星座 (現時点では 4 星座)
    my_classes = ["Gemini", "Canis Major", "Orion", "Taurus"]

    color_mean = (0.28077395684049855, 0.28077395684049855, 0.28077395684049855)

    # SSD300 の設定
    ssd_cfg = {
        'num_classes': 5, # 背景を含む
        'input_size': 300,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
    }

    # データの前処理
    transform = Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
             ])


    net = build_ssd('test', ssd_cfg)
    size = 300
    grayscale = True

    # 学習済みの SSD の重み
    net_weights = torch.load('app/static/model/SSD300_gray_500.pth', torch.device('cpu'))

    net.load_state_dict(net_weights)

    """
    ここからは輪郭抽出をしやすくする前処理となっているが, ここはまだ調整の必要がある
    """
    kernel = np.ones((3,3),np.uint8)
    # 膨張処理
    image = cv2.dilate(image, kernel, iterations=1)
    # 閾値処理
    ret, image = cv2.threshold(image, 20, 255, cv2.THRESH_TOZERO)

    # 輪郭抽出
    contours_check, hierarchy_check = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count_check = 0
    for c in range(len(contours_check)):
        M_check = cv2.moments(contours_check[c])
        if M_check['m00'] != 0.0:
            count_check += 1

    # 抽出できた星の数が 500 より多い場合は再び閾値処理
    if count_check > 500:
        ret, image = cv2.threshold(image, 120, 255, cv2.THRESH_TOZERO)

    # 膨張処理
    image = cv2.dilate(image, kernel, iterations = 1)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ssd, coordinate, size, color = detect(image, net.eval(), transform, check)
    ssd = np.array(ssd)
    ssd = cv2.cvtColor(ssd, cv2.COLOR_BGR2RGB)
    ssd = Image.fromarray(ssd)
    ssd.save('app/static/input/output.png')
    
    return coordinate, size, color

if __name__ == '__main__':
    test()

