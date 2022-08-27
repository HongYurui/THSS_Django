#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 11:22
# @File     : visualizer.py

"""
import cv2
import torch


def display_image(img, target_size=None):
    if target_size:
        ratio = target_size / max(img.shape[0], img.shape[1])
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=interp)
    cv2.imshow("", img)
    cv2.waitKey(0)


def demo_display_single_image(images, labels):
    image_to_display = images[0]
    image_to_display = image_to_display.numpy()
    image_to_display = image_to_display.transpose(1, 2, 0)
    display_image(image_to_display, target_size=240)
    label = labels[0]
    cv2.imwrite(f"{label.item}.jpg", image_to_display)


def demo_display_specific_digit_combination(images, labels):
    assert images.shape[0] == labels.shape[0]

    # this is an example for retrieve digits image
    digit_image_dict = {}
    for i in range(0, labels.shape[0]):
        digit = int(labels[i].item())
        if digit not in digit_image_dict:
            digit_image_dict[digit] = images[i]

        if len(digit_image_dict.keys()) >= 10:
            break

    # build image according to requirement
    digit_2 = digit_image_dict[2]
    digit_0 = digit_image_dict[0]
    target_img = torch.cat((digit_2, digit_0, digit_2, digit_0), 2)
    target_img = target_img.numpy()
    target_img = target_img.transpose(1, 2, 0)
    display_image(target_img, target_size=240)

def my_display_specific_digit_combination(images, labels):
    assert images.shape[0] == labels.shape[0]

    # retrieve digits image
    digit_image_dict = {}
    for i in range(0, labels.shape[0]):
        digit = int(labels[i].item())
        if digit not in digit_image_dict:
            digit_image_dict[digit] = images[i]

        if len(digit_image_dict.keys()) >= 10:
            break

    # build image according to requirement
    id_digits = [2, 0, 2, 0, 0, 1, 3, 2, 7, 5]
    id_images = [digit_image_dict[digit] for digit in id_digits]
    target_img = torch.cat(id_images, 2)
    target_img = target_img.numpy()
    target_img = target_img.transpose(1, 2, 0)
    display_image(target_img, target_size=240)
