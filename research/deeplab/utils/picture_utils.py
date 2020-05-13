import time

import cv2
import os
import re

from progressbar import ProgressBar


def read_directory(curdir, directory_name):
    '''
    获取某一目录下的所有图片   图片命名格式：  curdir_000000_编号_leftImg8bit.png
    Args:
        curdir: 当前文件夹
        directory_name: 当前文件夹路径

    Returns:

    '''
    num_picture = len(os.listdir(directory_name))
    fileNames = sorted(os.listdir(directory_name))
    firstName = fileNames[0]
    fileName_list = []
    # print(firstName)
    pattern = "_\d*_left"
    print("正在读取图片.........")
    start_index = int(re.findall(pattern, firstName)[0][1:7])   # 获取第一张图片的编号
    progress = ProgressBar()    # 可视化进度条
    for i in progress(range(num_picture)):
        file_index = start_index + i
        fileName = directory_name + '\\' + fileNames[i]
        fileName_list.append(fileName)
        time.sleep(0.01)
    return fileName_list, num_picture

def png2Video(output_video_path, processed_picture_dir, fps):
    '''
    将得到的语义分割图片序列转化成视频
    Args:
        output_video_path: 输出的视频路径
        processed_picture_dir: 语义分割得到的图片的目录
        fps: 视频fps

    Returns:

    '''

    files = sorted(os.listdir(processed_picture_dir))
    out_num = len(files)
    # print(out_num)
    first_png_file_path = processed_picture_dir + '\\' + files[0]
    # print(first_png_file_path)
    img = cv2.imread(first_png_file_path)  # 读取第一张图片
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])  # 获取图片宽高度信息
    # print(size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWrite = cv2.VideoWriter(output_video_path, fourcc, fps, size)  # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
    print("正在生成视频.........")
    progress = ProgressBar()  # 可视化进度条
    for i in progress(range(0, out_num)):
        fileName = processed_picture_dir + '\\stuttgart_00_000000_%.6d_leftImg8bit.png' % (i+1)  # 循环读取所有的图片,假设以数字顺序命名
        # print(fileName)
        img = cv2.imread(fileName)
        # print(img.shape)
        videoWrite.write(img)  # 将图片写入所创建的视频对象
        time.sleep(0.01)

def video2Png(video_dir, videoName, store_dir):
    '''
    将视频分割成帧图片
    Args:
        video_dir: 视频目录
        videoName: 视频名称
        store_dir: 图片存放目录

    Returns:

    '''
    video_path = video_dir + videoName +'.mp4'
    vc = cv2.VideoCapture(video_path)
    c = 0
    rval = vc.isOpened()
    while rval:
        c = c + 1
        rval, frame = vc.read()
        resize_frame = cv2.resize(frame, (2048, 1024), interpolation=cv2.INTER_AREA)
        store_picture_path = store_dir + 'street/street_000000_%.6d_leftImg8bit' % (c) + '.png'
        # print(store_picture_path)
        if rval:
            cv2.imwrite(store_picture_path, resize_frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()
    print('save_success')
    print(store_picture_path)

# output_video_path = r'D:\Cityscapes\data\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00.mp4'
# processed_picture_dir = r'D:\Cityscapes\data\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00'
# png2Video(output_video_path, processed_picture_dir, 25)
original_directory = r'D:\Cityscapes\data\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00'
curdir = 'stuttgart_00'
fileNameList = read_directory(curdir, original_directory)
print(fileNameList)
