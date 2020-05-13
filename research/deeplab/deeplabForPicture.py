import os

from deeplab.utils.picture_utils import read_directory
import numpy as np
from PIL import Image
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

  colormap = create_pascal_label_colormap()
  # colormap = create_cityscapes_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_TARBALL_NAME = 'deeplab_model.tar.gz'


def get_mixed_picture(pd_file_path, curdir, original_directory, mixed_picture_saveDir):


  LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])

  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

  MODEL = DeepLabModel(pd_file_path)
  print('model loaded successfully!')

  png_fileName_list, num_file = read_directory(curdir, original_directory)
  # num_file = len(png_fileName_list)
  print(num_file)
  if not os.path.isdir(mixed_picture_saveDir):
    os.mkdir(mixed_picture_saveDir)

  progress = ProgressBar()
  for i in progress(range(num_file)):
    png_file_path = png_fileName_list[i]
    image_path = png_file_path
    original_im = Image.open(image_path)
    resized_im, seg_map = MODEL.run(original_im)

    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image = Image.fromarray(np.uint8(seg_image))

    final_image = Image.blend(resized_im, seg_image, 0.5)
    save_path = mixed_picture_saveDir + '\\' + str(i) + '.png'
    final_image.save(save_path)
    time.sleep(0.01)


pd_file_path = r"D:\pb文件\xception\frozen_inference_graph.pb"
original_directory = r'D:\Cityscapes\data\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00'
curdir = 'stuttgart_00'
mixed_picture_saveDir = r'D:\Cityscapes\data\leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_00_segmentation_xception'


if __name__ == "__main__":
  get_mixed_picture(pd_file_path, curdir, original_directory, mixed_picture_saveDir)