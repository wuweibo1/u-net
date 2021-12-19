# coding = utf-8
import cv2
from PIL import Image
from PIL import ImageEnhance
from numpy.ma import array
import numpy as np
import os
# 批量处理代码
rootdir = 'data/membrane/train/image/' # 指明被遍历的文件夹

def operate(currentPath, filename, targetPath):
    # 读取图像
    image = Image.open(currentPath)
    image_cv = cv2.imread(currentPath)
    # image.show()
    # 增强亮度 bh_
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.07
    image_brightened_h = enh_bri.enhance(brightness)
    # image_brightened_h.show()
    image_brightened_h.save(targetPath + 'bh_' + filename)  # 保存

    # 降低亮度 bl_
    enh_bri_low = ImageEnhance.Brightness(image)
    brightness = 0.87
    image_brightened_low = enh_bri_low.enhance(brightness)
    # image_brightened_low.show()
    image_brightened_low.save(targetPath + 'bl_' + filename)

    # 改变色度 co_
    enh_col = ImageEnhance.Color(image)
    color = 0.8
    image_colored = enh_col.enhance(color)
    # image_colored.show()
    image_colored.save(targetPath + 'co_' + filename)

    # 改变对比度 cont_
    enh_con = ImageEnhance.Contrast(image)
    contrast = 0.8
    image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()
    image_contrasted.save(targetPath + 'cont_' + filename)

    # 改变锐度 sha_
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 3.0
    image_sharp = enh_sha.enhance(sharpness)
    # image_sharp.show()
    image_sharp.save(targetPath + 'sha_' + filename)


for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        # print('parent is:' + parent)
        print('filename is: ' + filename)
        # 把文件名添加到一起后输出
        currentPath = os.path.join(parent, filename)
        # print('the full name of file is :' + currentPath)
        # 保存处理后的图片的目标文件夹
        targetPath = 'data/membrane/train/aug/'
        # 进行处理
        operate(currentPath, filename, targetPath)
