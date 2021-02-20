#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


class Image:
    def __init__(self):
        pass

    @staticmethod
    def rgb2gray(img: np.ndarray) -> np.ndarray:
        return np.dot(img, [0.299, 0.587, 0.114]).astype(np.uint8)

    @staticmethod
    def rgb2yuv(rgb_img: np.ndarray) -> np.ndarray:
        yuv = np.zeros(shape=rgb_img.shape, dtype=float)
        rgb_img = rgb_img.astype(float)
        yuv[:, :, 0] = np.dot(rgb_img, [0.299, 0.587, 0.114])
        yuv[:, :, 1] = np.dot(rgb_img, [-0.168736, -0.331264, 0.5]) + 128
        yuv[:, :, 2] = np.dot(rgb_img, [0.5, -0.418688, -0.081312]) + 128
        yuv = np.clip(yuv, 0, 255).astype(np.uint8)
        return yuv

    @staticmethod
    def yuv2rgb(yuv_img: np.ndarray) -> np.ndarray:
        rgbTemp = np.zeros(shape=yuv_img.shape, dtype=float)
        yuv_img = yuv_img.astype(float)
        rgbTemp[:, :, 0] = yuv_img[:, :, 0] + 1.4075 * (yuv_img[:, :, 2] - 128)
        rgbTemp[:, :, 1] = yuv_img[:, :, 0] - 0.3455 * (yuv_img[:, :, 1] - 128) - (0.7169 * (yuv_img[:, :, 2] - 128))
        rgbTemp[:, :, 2] = yuv_img[:, :, 0] + 1.7790 * (yuv_img[:, :, 1] - 128)
        rgb = np.clip(rgbTemp, 0, 255).astype(np.uint8)
        return rgb

    @staticmethod
    def negative(img: np.ndarray) -> np.ndarray:
        return 255 - img

    @staticmethod
    def gaussBlur(img: np.ndarray, k: float, s: float, core_shape: list = None) -> np.ndarray:
        h, w = img.shape[0:2]
        if not core_shape:
            cw = ch = math.ceil(s * 6)
        else:
            if not isinstance(core_shape, list) or len(core_shape) != 2:
                raise TypeError(
                    "core need to be a list contians of core's odd height and odd width. ie, [height, width]")
            else:
                ch, cw = core_shape[0:2]
                ch, cw = int(ch), int(cw)
        if not cw & 1:
            cw += 1
        if not ch & 1:
            ch += 1

        # compute core's distances
        core = np.zeros([cw, ch], dtype=float)
        center_cw = cw >> 1
        center_ch = ch >> 1
        for i in range(ch):
            for j in range(cw):
                core[i][j] = pow(i - center_ch, 2) + pow(j - center_cw, 2)

        # compute core's coefficients
        co_tem = pow(s, 2) * 2
        compute_co = lambda r: k / pow(math.e, r / co_tem)
        core = compute_co(core)
        total_weight = np.sum(core)
        core = core / total_weight

        # blur image by guass core
        blurred_img = np.zeros(img.shape, dtype=float)
        padding_img = np.pad(img, ((ch >> 1, ch >> 1), (cw >> 1, cw >> 1)),
                             mode='edge')  # padding matrix with edge values
        h, w = blurred_img.shape[0:2]
        for i in range(h):
            for j in range(w):
                blurred_img[i][j] = np.sum(core * padding_img[i:i + ch, j:j + cw])
        return blurred_img.astype(np.uint8)

    @staticmethod
    def outline(img_gray: np.ndarray, k: float, s: float, core_shape: list = None) -> np.ndarray:
        blured_img = Image.gaussBlur(img_gray, k, s, core_shape)
        outline_img = np.clip(img_gray - blured_img, 0, 255)
        return outline_img.astype(np.uint8)

    @staticmethod
    def blend(img_gray: np.ndarray, img_blured: np.ndarray) -> np.ndarray:
        blend_img = cv2.divide(img_gray, img_blured, scale=255)
        return blend_img.astype(np.uint8)


def sketch(img: np.ndarray, k: float = 1, s: float = 6.5, core_shape: list = None, opt: str = 'gray') -> np.ndarray:
    if opt == 'gray':
        if img.shape[2] == 3:
            img = Image.rgb2gray(img)
        blured_img = Image.gaussBlur(img, k, s, core_shape)
        outline_img = Image.outline(img, 1, 2)
        blured_img = np.clip(blured_img - outline_img, 0, 255)
        res = Image.blend(img, blured_img)
    elif opt == 'color':
        if img.shape[2] != 3:
            raise KeyError("do color sketch need inputing a rgb image")
        yuv_img = Image.rgb2yuv(img)
        img_y = yuv_img[:, :, 0]
        blured_img = Image.gaussBlur(img_y, k, s, core_shape)
        outline_img = Image.outline(img_y, 1, 2)
        blured_img = np.clip(blured_img - outline_img, 0, 255)
        blend_img = Image.blend(img_y, blured_img)
        yuv_img[:, :, 0] = blend_img
        res = Image.yuv2rgb(yuv_img)
    else:
        raise KeyError("opt only supports gray and color, default is gray")
    return res


# test sketching algorithm
if __name__ == '__main__':
    import os

    for pic in os.listdir(os.path.join('.', 'pictures')):
        print(pic)
        name = pic.split('.')[0]
        img = plt.imread(os.path.join('pictures', pic))
        sketch_gray = sketch(img, k=2, s=7.5)
        sketch_color = sketch(img, k=2, s=7.5, opt='color')
        fig = plt.figure(dpi=400, linewidth=0.5, tight_layout=True)

        ax = fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis('off')
        ax.set_title('origin image', fontsize='small')

        ax = fig.add_subplot(1, 3, 2)
        plt.imshow(sketch_gray, cmap='gray')
        plt.axis('off')
        ax.set_title('gray sketch', fontsize='small')

        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(sketch_color)
        plt.axis('off')
        ax.set_title('color sketch', fontsize='small')

        save_img = os.path.join('results3', name + '.png')
        plt.savefig(save_img, dpi=400)
        plt.close(fig)
        del fig
