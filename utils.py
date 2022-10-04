import numpy as np
import torch.nn as nn


def get_label_file(image_file, dataset):
    if dataset in ['land', 'road', 'building']:
        label_file = image_file.replace('_sat.png', '_mask.png')
    elif dataset == 'chaos':
        label_file = image_file.replace('/DICOM_anon/', '/Ground/')
        if 'T1DUAL' in image_file:
            if 'InPhase/' in image_file:
                label_file = label_file.replace('InPhase/', '')
            elif 'OutPhase/' in image_file:
                label_file = label_file.replace('OutPhase/', '')
            image_number = int(image_file.split('/')[-1].split('.')[0].split('-')[-1])
            if image_number % 2:
                image_number_string = "%05d" % image_number
                label_number_string = "%05d" % ((image_number // 2 + 1) * 2)
                label_file = label_file.replace(image_number_string, label_number_string)
    elif dataset == 'promise':
        label_file = image_file.replace('_image.png', '_mask.png')
    return label_file


def encode_label(label_bgr, class_rgb_values):
    label = np.zeros((label_bgr.shape[0], label_bgr.shape[1]), dtype=np.uint8)
    for i, rgb in enumerate(class_rgb_values):
        bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
        loc = (label_bgr == bgr).all(axis=-1)
        label[loc] = i
    return label


def initialize(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.running_mean is not None:
                nn.init.constant_(m.running_mean, 0)