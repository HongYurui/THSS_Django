#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 17:42
# @File     : pre_process.py

"""
import torchvision


def normal_transform():
    normal = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return normal

def data_augment_transform(random_crop=None, horizontal_flip=False, verticle_flip=False):
    composed_transforms: list = []
    if random_crop is tuple or random_crop is int and random_crop > 0:
        composed_transforms.append(torchvision.transforms.RandomCrop(random_crop))
    if horizontal_flip:
        composed_transforms.append(torchvision.transforms.RandomHorizontalFlip())
    if verticle_flip:
        composed_transforms.append(torchvision.transforms.RandomVerticalFlip())
    composed_transforms.append(torchvision.transforms.ToTensor())
    data_augment = torchvision.transforms.Compose(composed_transforms)
    return data_augment
