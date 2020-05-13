import cv2
import os
import random
from deeplab.utils.picture_utils import *



# png_file_path = r'E:\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_01\stuttgart_01_000000_000001_leftImg8bit.png'
output_video_path = '/media/xzq/DA18EBFA09C1B27D/exp/train_on_train_set/video/stuttgart_01.mp4'
files = os.listdir('/media/xzq/DA18EBFA09C1B27D/exp/train_on_train_set/test')
out_num = len(files)
png_file_path = '/media/xzq/DA18EBFA09C1B27D/exp/train_on_train_set/test/0.png'
img = cv2.imread(png_file_path)  # 读取第一张图片
# # print(img)
fps = 25
imgInfo = img.shape
size = (imgInfo[1], imgInfo[0])  # 获取图片宽高度信息
# print(size)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWrite = cv2.VideoWriter(output_video_path, fourcc, fps, size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
#videoWrite = cv2.VideoWriter('0.mp4',fourcc,fps,(1920,1080))
#

print(out_num)
fileDir = '/media/xzq/DA18EBFA09C1B27D/exp/train_on_train_set/test'
for i in range(0, out_num):
    fileName = fileDir + '/%d.png' % i   #循环读取所有的图片,假设以数字顺序命名
    print(fileName)
    # print(i)
    img = cv2.imread(fileName)
    videoWrite.write(img)# 将图片写入所创建的视频对象

(parent_path, file_name) = os.path.split(output_video_path)

output_video_path = parent_path + "\\" + "segment_" + file_name
print(output_video_path)
