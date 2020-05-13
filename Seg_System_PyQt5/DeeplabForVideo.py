import os
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.Qt import *
import time
from progressbar import ProgressBar
import tensorflow as tf

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 2048
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    with tf.gfile.FastGFile(tarball_path, 'rb') as model_file:
      graph_def = tf.GraphDef.FromString(model_file.read())


    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    # print(image.size)
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})  # ////
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

# cityscapes ，每一个标签对应的颜色
def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  # colormap = create_pascal_label_colormap()
  colormap = create_cityscapes_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_TARBALL_NAME = 'deeplab_model.tar.gz'


def get_mixed_video(pd_file_path, video_path):


  LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'ignore'
  ])

  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
 # 加载deeplab模型
  MODEL = DeepLabModel(pd_file_path)
  print('model loaded successfully!')
  vc = cv2.VideoCapture(video_path)
  progress = QProgressDialog()
  progress.setWindowTitle("语义分割")
  progress.setMinimumDuration(5)
  progress.setLabelText("正在进行视频分割，可能需要一定时间，请等待...")
  progress.setCancelButtonText("取消")
  progress.setWindowModality(Qt.WindowModal)
  num = int(vc.get(7))
  progress.setRange(0, num)
  # for i in range(num):
  #   progress.setValue(i)
  #   if progress.wasCanceled():
  #     QMessageBox.warning("提示", "操作失败")
  #     break
  #   time.sleep(0.01)
    # else:
    #     progress.setValue(i)
    #     QMessageBox.information(self, "提示", "操作成功")
  success = vc.isOpened()
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  (parent_path, file_name) = os.path.split(video_path)
  fps = 25
  output_video_path = parent_path + "\\" + "segment_"+file_name
  videoWrite = cv2.VideoWriter(output_video_path, fourcc, fps, (2048, 1024))  # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
  for i in range(int(num)):
    progress.setValue(i)
    if progress.wasCanceled():
      QMessageBox.warning("提示", "操作失败")
      break
    success, frame = vc.read()
    original_image = cv2.resize(frame, (2048, 1024), interpolation=cv2.INTER_AREA)
    original_image = Image.fromarray(cv2.cvtColor(original_image,cv2.COLOR_RGB2BGR))
    if success:
        resized_image, seg_map = MODEL.run(original_image)
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        seg_image = Image.fromarray(np.uint8(seg_image))
        final_image = Image.blend(resized_image, seg_image, 0.5)
        final_image = cv2.cvtColor(np.asarray(final_image), cv2.COLOR_RGB2BGR)
        videoWrite.write(final_image)
        time.sleep(0.01)
    else:
        break
  vc.release()
  return videoWrite, output_video_path
  # final_image.show()
  # parant_path = os.path.dirname(image_path)
  #
  #
  # save_path = mixed_picture_saveDir + '\\' + str(i) + '.png'
  # final_image.save(save_path)






if __name__ == "__main__":
    pd_file_path = r"D:\pb文件\xception\frozen_inference_graph.pb"
    # original_directory = r'D:\Cityscapes\data\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00'
    # curdir = 'stuttgart_00'
    # mixed_picture_saveDir = r'D:\Cityscapes\data\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00_segmentation_xception'
    video_path = r'G:\Desktop\Video\cityscapes.mp4'
    # parant_path = os.path.dirname(image_path)
    get_mixed_video(pd_file_path, video_path)