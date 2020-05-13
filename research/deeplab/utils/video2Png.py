import cv2
import os
def save_img():
    video_dir = '/media/xzq/DA18EBFA09C1B27D/video/'
    video_path = video_dir + 'street.mp4'

    vc = cv2.VideoCapture(video_path)
    c = 0
    rval = vc.isOpened()

    while rval:
        c = c + 1

        rval, frame = vc.read()
        resize_frame = cv2.resize(frame, (2048, 1024), interpolation=cv2.INTER_AREA)
        store_picture_path = video_dir + 'street/street_000000_%.6d_leftImg8bit' %(c) + '.png'
        print(store_picture_path)
        if rval:
            cv2.imwrite(store_picture_path, resize_frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()
    print('save_success')
    print(store_picture_path)

# save_img()
vc = cv2.VideoCapture('G:\Desktop\Video\cityscapes.mp4')
number = vc.get(7)
print(number)
