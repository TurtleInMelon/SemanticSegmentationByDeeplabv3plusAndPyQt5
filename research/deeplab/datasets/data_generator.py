# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for providing semantic segmentaion data.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

3. ADE20K dataset (http://groups.csail.mit.edu/vision/datasets/ADE20K)

The ADE20K dataset contains 150 semantic labels both urban street scenes and
indoor scenes.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""

import collections
import os
import tensorflow as tf
from deeplab import common
from deeplab import input_preprocess

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple( # 各个数据集描述器类
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
        'ignore_label',  # Ignore label value.
    ])

_CITYSCAPES_INFORMATION = DatasetDescriptor(    # cityscapes数据集
    splits_to_sizes={'train_fine': 2975,
                     'train_coarse': 22973,
                     'trainval_fine': 3475,
                     'trainval_coarse': 23473,
                     'val_fine': 500,
                     'test_fine': 1525},
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'train_aug': 10582,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)

_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,  # num of samples in images/training
        'val': 2000,  # num of samples in images/validation
    },
    num_classes=151,
    ignore_label=0,
)

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_cityscapes_dataset_name():
  return 'cityscapes'


class Dataset(object):
  """Represents input dataset for deeplab model."""

  def __init__(self,
               dataset_name,
               split_name,
               dataset_dir,
               batch_size,
               crop_size,
               min_resize_value=None,
               max_resize_value=None,
               resize_factor=None,
               min_scale_factor=1.,
               max_scale_factor=1.,
               scale_factor_step_size=0,
               model_variant=None,
               num_readers=1,
               is_training=False,
               should_shuffle=False,
               should_repeat=False):
    """Initializes the dataset.

    Args:
      dataset_name: Dataset name.
      split_name: A train/val Split name.
      dataset_dir: The directory of the dataset sources.
      batch_size: Batch size.
      crop_size: The size used to crop the image and label.
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      model_variant: Model variant (string) for choosing how to mean-subtract
        the images. See feature_extractor.network_map for supported model
        variants.
      num_readers: Number of readers for data provider.
      is_training: Boolean, if dataset is for training or not.
      should_shuffle: Boolean, if should shuffle the input data.
      should_repeat: Boolean, if should repeat the input data.

    Raises:
      ValueError: Dataset name and split name are not supported.
    """
    if dataset_name not in _DATASETS_INFORMATION:
      # 判断输入的数据集名称 是否在 _DATASETS_INFORMATION中 cityscapes  pascal_voc_seg  ade20k
      raise ValueError('The specified dataset is not supported yet.')
    self.dataset_name = dataset_name

    # 对数据集进行划分 train val
    splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

    # 数据集分割的是否具有split_name 默认为"train"
    if split_name not in splits_to_sizes:
      raise ValueError('data split name %s not recognized' % split_name)

    if model_variant is None:    # 模型名称 默认为mobilenet_v2  本次训练为xception_65
      tf.logging.warning('Please specify a model_variant. See '
                         'feature_extractor.network_map for supported model '
                         'variants.')

    self.split_name = split_name    # 默认为"train"
    self.dataset_dir = dataset_dir  # 数据集目录
    self.batch_size = batch_size    # 指定batch_size大小  默认为8   当设定微调BN层的时候 应大于12
    self.crop_size = crop_size  # 裁剪大小 513，513
    self.min_resize_value = min_resize_value    # None
    self.max_resize_value = max_resize_value    # None
    self.resize_factor = resize_factor   # None
    self.min_scale_factor = min_scale_factor    # 0.5
    self.max_scale_factor = max_scale_factor    # 2
    self.scale_factor_step_size = scale_factor_step_size    # 变换尺度每次增加0.25
    self.model_variant = model_variant  # 默认为mobilenet_v2  本次训练为xception_65
    self.num_readers = num_readers   # 4
    self.is_training = is_training   # True
    self.should_shuffle = should_shuffle    # True 是否将输入的数据随机打乱
    self.should_repeat = should_repeat  # True 训练是否重复下去

    self.num_of_classes = _DATASETS_INFORMATION[self.dataset_name].num_classes  # 数据集中类别数量 19
    self.ignore_label = _DATASETS_INFORMATION[self.dataset_name].ignore_label   # 255

  def _parse_function(self, example_proto):
    """Function to parse the example proto.

    Args:
      example_proto: Proto in the format of tf.Example.

    Returns:
      A dictionary with parsed image, label, height, width and image name.

    Raises:
      ValueError: Label is of wrong shape.
    """
    '''
            返回字典形式 {图像名：tensor，
                            label：tensor，
                            高度：tensor，
                            宽度：tensor，
                            image：tensor}
    '''

    # Currently only supports jpeg and png.
    # Need to use this logic because the shape is not known for
    # tf.image.decode_image and we rely on this info to
    # extend label if necessary.
    def _decode_image(content, channels):   # 根据输入的图像类型进行解码得到像素值
      return tf.cond(   # tf.cond()类似于c语言中的if...else...，用来控制数据流向
          tf.image.is_jpeg(content),
          lambda: tf.image.decode_jpeg(content, channels),
          lambda: tf.image.decode_png(content, channels))

    features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    '''
            tf.parse_single_example(
                serialized,
                features,
                name=None,
                example_names=None
            )
            erialized：标量字符串Tensor，单个序列化Example。
            features：字典映射到FixedLenFeature或者VarLenFeature的值。
            name：此操作的名称（可选）。
            example_names：（可选）标量字符串Tensor，关联名称。
        '''
    parsed_features = tf.parse_single_example(example_proto, features)

    image = _decode_image(parsed_features['image/encoded'], channels=3)

    label = None
    if self.split_name != common.TEST_SET:  # TEST_SET 为 ’test‘
      label = _decode_image(
          parsed_features['image/segmentation/class/encoded'], channels=1)

    image_name = parsed_features['image/filename']
    if image_name is None:
      image_name = tf.constant('')

    sample = {
        common.IMAGE: image,
        common.IMAGE_NAME: image_name,
        common.HEIGHT: parsed_features['image/height'],
        common.WIDTH: parsed_features['image/width'],
    }

    if label is not None:
      if label.get_shape().ndims == 2:
        label = tf.expand_dims(label, 2)
      elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
        pass
      else:
        raise ValueError('Input label shape must be [height, width], or '
                         '[height, width, 1].')

      label.set_shape([None, None, 1])

      sample[common.LABELS_CLASS] = label

    return sample

  def _preprocess_image(self, sample):
    """Preprocesses the image and label.

    Args:
      sample: A sample containing image and label.

    Returns:
      sample: Sample with preprocessed image and label.

    Raises:
      ValueError: Ground truth label not provided during training.
    """
    image = sample[common.IMAGE]
    label = sample[common.LABELS_CLASS]

    original_image, image, label = input_preprocess.preprocess_image_and_label(  # 预处理图像和标签  经过了尺度缩放和左右翻转进行数据增强
        image=image,
        label=label,
        crop_height=self.crop_size[0],
        crop_width=self.crop_size[1],
        min_resize_value=self.min_resize_value,
        max_resize_value=self.max_resize_value,
        resize_factor=self.resize_factor,
        min_scale_factor=self.min_scale_factor,
        max_scale_factor=self.max_scale_factor,
        scale_factor_step_size=self.scale_factor_step_size,
        ignore_label=self.ignore_label,
        is_training=self.is_training,
        model_variant=self.model_variant)

    sample[common.IMAGE] = image    # 处理过的图像

    if not self.is_training:
      # Original image is only used during visualization.
      # 原图片在可视化中被用到
      sample[common.ORIGINAL_IMAGE] = original_image

    if label is not None:
      sample[common.LABEL] = label

    # Remove common.LABEL_CLASS key in the sample since it is only used to
    # derive label and not used in training and evaluation.
    sample.pop(common.LABELS_CLASS, None)

    return sample

  def get_one_shot_iterator(self):
    """Gets an iterator that iterates across the dataset once.

    Returns:
      An iterator of type tf.data.Iterator.
    """

    files = self._get_all_files()   # 获取数据集生成的tfrecorder文件中所有训练集的recorder文件  名称中带有'train'或者‘val’可选
                                    # 查找匹配pattern的文件并以列表的形式返回
    dataset = (
        tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
        .map(self._parse_function, num_parallel_calls=self.num_readers)     # 定义解析tfrecorder数据的方式  解析图像和label
        .map(self._preprocess_image, num_parallel_calls=self.num_readers))   # 定义解析tfrecorder数据的方式 处理图像和label

    if self.should_shuffle:
      dataset = dataset.shuffle(buffer_size=100)

    '''
               dataset.repeat(num)
               num为空表示无限重复下去
               不设置则表示只重复一次
    '''
    if self.should_repeat:
      dataset = dataset.repeat()  # Repeat forever for training.
    else:
      dataset = dataset.repeat(1)

    dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)  #设置batch_size
    return dataset.make_one_shot_iterator()

  def _get_all_files(self):
    """Gets all the files to read data from.

    Returns:
      A list of input files.
    """
    '''
            获取数据集的tfrecorder文件
    '''
    file_pattern = _FILE_PATTERN    # _FILE_PATTERN = '%s-*'
    file_pattern = os.path.join(self.dataset_dir,
                                file_pattern % self.split_name)  # split_name为train
    return tf.gfile.Glob(file_pattern)  # tf.gfile.Glob 查找匹配pattern的文件并以列表的形式返回，filename可以是一个具体的文件名，也可以是包含通配符的正则表达式。
