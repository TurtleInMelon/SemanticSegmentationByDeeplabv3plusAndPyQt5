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
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import tfprof as contrib_tfprof
from tensorflow.contrib import training as contrib_training
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.
# 模型评估文件存放目录
flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

# ckpt文件路径
flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.
# 一次性评估大小
flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')
# 评估裁剪的大小 一般设置为图像大小
flags.DEFINE_list('eval_crop_size', '513,513',
                  'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
# 空洞率设置 跟训练时一样
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')
# 输出步长
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
# 评估时模型缩放尺度
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
# 是否左右翻转图片进行评估
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.
# 数据集名称
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')
# 验证机名称
flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')
# 数据集tfrecord存放路径
flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  dataset = data_generator.Dataset(  # 获取验证集图片数据
      dataset_name=FLAGS.dataset,    #　数据集名称　cityscapes  默认为　pascal_voc_seg
      split_name=FLAGS.eval_split,  # 指定带有val的tfrecorder数据集 默认为“val”
      dataset_dir=FLAGS.dataset_dir,    # 数据集目录　tfrecoder文件的数据集目录
      batch_size=FLAGS.eval_batch_size,  # 每个batch包含的image数量 默认为1
      crop_size=[int(sz) for sz in FLAGS.eval_crop_size],   # 评估时crop_size 默认为513,513
      min_resize_value=FLAGS.min_resize_value,  #　默认为None
      max_resize_value=FLAGS.max_resize_value,  #　默认为None
      resize_factor=FLAGS.resize_factor,    #　默认为None
      model_variant=FLAGS.model_variant,     # 模型的变体　 本次训练为 xception_65
      num_readers=2,     # 并行读取图片的数量
      is_training=False,    # 不训练
      should_shuffle=False,     # 不将输入的数据随机打乱
      should_repeat=False)      # 不一直重复

  tf.gfile.MakeDirs(FLAGS.eval_logdir)  # 创建评估目录
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()    # 获取一次迭代的验证集数据
    '''
        samples:
            {'image_name': <tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=string>, 
             'width': <tf.Tensor 'IteratorGetNext:5' shape=(?,) dtype=int64>, 
             'image': <tf.Tensor 'IteratorGetNext:1' shape=(?, 1024, 2048, 3) dtype=float32>, 
             'height': <tf.Tensor 'IteratorGetNext:0' shape=(?,) dtype=int64>, 
             'label': <tf.Tensor 'IteratorGetNext:3' shape=(?, 1024, 2048, 1) dtype=int32>, 
             'original_image': <tf.Tensor 'IteratorGetNext:4' shape=(?, ?, ?, 3) dtype=uint8>}
        '''
    model_options = common.ModelOptions(    # 模型参数
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},    # {semantic: 19}
        crop_size=[int(sz) for sz in FLAGS.eval_crop_size],     # 1024,2048
        atrous_rates=FLAGS.atrous_rates,     # 6,12,18
        output_stride=FLAGS.output_stride)  # 16

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
    samples[common.IMAGE].set_shape(    # 设置形状
        [FLAGS.eval_batch_size,      # 默认为1
         int(FLAGS.eval_crop_size[0]),
         int(FLAGS.eval_crop_size[1]),
         3])
    if tuple(FLAGS.eval_scales) == (1.0,):  # 默认 评估尺度为1
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(samples[common.IMAGE], model_options,  # 进行每个像素点预测
                                         image_pyramid=FLAGS.image_pyramid)
      '''
              predictions:
                  {'semantic': <tf.Tensor 'ArgMax:0' shape=(1, 1024, 2048) dtype=int64>, 
                   'semantic_prob': <tf.Tensor 'Softmax:0' shape=(1, 1024, 2048, 19) dtype=float32>}
            '''
    else:
      tf.logging.info('Performing multi-scale test.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')

      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]
    predictions = tf.reshape(predictions, shape=[-1])   # 预测标签
    labels = tf.reshape(samples[common.LABEL], shape=[-1])  # 真实标签
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))   # 各标签权重

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'    # MIoU
    predictions_tag1 = 'accuracy_pixel'     # 像素精度
    for eval_scale in FLAGS.eval_scales:    # 默认为单尺度[1.0]
      predictions_tag += '_' + str(eval_scale)
      predictions_tag1 += '_' + str(eval_scale)
    if FLAGS.add_flipped_images:    # 默认为False 不设置左右翻转来评估模型
      predictions_tag += '_flipped'
      predictions_tag1 += '_flipped'

    # Define the evaluation metric.
    metric_map = {}
    num_classes = dataset.num_of_classes    # 19
    metric_map['eval/%s_overall' % predictions_tag] = tf.metrics.mean_iou(
        labels=labels, predictions=predictions, num_classes=num_classes,
        weights=weights)
    '''
          metric_map:
            {'eval/miou_1.0_overall': (<tf.Tensor 'mean_iou/Select_1:0' shape=() dtype=float32>, 
                                       <tf.Tensor 'mean_iou/AssignAdd:0' shape=(19, 19) dtype=float64_ref>)}
    '''
    metric_map['eval/%s_overall_accuracy_' % predictions_tag] = tf.metrics.accuracy(
        labels=labels, predictions=predictions, weights=weights)
    # IoU for each class.
    '''
        tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
            Returns a one-hot tensor.
        ndices表示输入的多个数值，通常是矩阵形式；depth表示输出的尺寸。
    '''
    one_hot_predictions = tf.one_hot(predictions, num_classes)
    one_hot_predictions = tf.reshape(one_hot_predictions, [-1, num_classes])    # 预测输出的one_hot
    one_hot_labels = tf.one_hot(labels, num_classes)
    one_hot_labels = tf.reshape(one_hot_labels, [-1, num_classes])  # 真实label的one_hot
    for c in range(num_classes):
      predictions_tag_c = '%s_class_%d' % (predictions_tag, c)  # miou_1.0_class_c
      predictions_tag_c1 = '%s_class_%d' % (predictions_tag1, c)
      tp, tp_op = tf.metrics.true_positives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      fp, fp_op = tf.metrics.false_positives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      fn, fn_op = tf.metrics.false_negatives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      tn, tn_op = tf.metrics.true_negatives(
          labels=one_hot_labels[:, c], predictions=one_hot_predictions[:, c],
          weights=weights)
      tp_fp_fn_op = tf.group(tp_op, fp_op, fn_op)
      iou = tf.where(tf.greater(tp + fn, 0.0),
                     tp / (tp + fn + fp),
                     tf.constant(np.NaN))
      ap = tf.where(tf.greater(tp + fn, 0.0),
                    (tp + tn) / (tp + tn + fn + fp),
                    tf.constant(np.NaN))
      metric_map['eval/%s' % predictions_tag_c] = (iou, tp_fp_fn_op)
      metric_map['eval/%s' % predictions_tag_c1] = (ap, tp_fp_fn_op)

    (metrics_to_values,
     metrics_to_updates) = contrib_metrics.aggregate_metric_map(metric_map)
    '''
        (metrics_to_values, metrics_to_updates):
            ({'eval/miou_1.0_class_5': <tf.Tensor 'Select_6:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_18': <tf.Tensor 'Select_19:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_13': <tf.Tensor 'Select_14:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_1': <tf.Tensor 'Select_2:0' shape=() dtype=float32>, 
             'eval/miou_1.0_overall': <tf.Tensor 'mean_iou/Select_1:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_17': <tf.Tensor 'Select_18:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_8': <tf.Tensor 'Select_9:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_2': <tf.Tensor 'Select_3:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_0': <tf.Tensor 'Select_1:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_3': <tf.Tensor 'Select_4:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_14': <tf.Tensor 'Select_15:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_11': <tf.Tensor 'Select_12:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_6': <tf.Tensor 'Select_7:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_15': <tf.Tensor 'Select_16:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_4': <tf.Tensor 'Select_5:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_9': <tf.Tensor 'Select_10:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_16': <tf.Tensor 'Select_17:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_7': <tf.Tensor 'Select_8:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_10': <tf.Tensor 'Select_11:0' shape=() dtype=float32>, 
             'eval/miou_1.0_class_12': <tf.Tensor 'Select_13:0' shape=() dtype=float32>}, 

            {'eval/miou_1.0_class_5': <tf.Operation 'group_deps_5' type=NoOp>, 
              'eval/miou_1.0_class_18': <tf.Operation 'group_deps_18' type=NoOp>, 
              'eval/miou_1.0_class_13': <tf.Operation 'group_deps_13' type=NoOp>, 
              'eval/miou_1.0_class_1': <tf.Operation 'group_deps_1' type=NoOp>, 
              'eval/miou_1.0_overall': <tf.Tensor 'mean_iou/AssignAdd:0' shape=(19, 19) dtype=float64_ref>, 
              'eval/miou_1.0_class_17': <tf.Operation 'group_deps_17' type=NoOp>, 
              'eval/miou_1.0_class_8': <tf.Operation 'group_deps_8' type=NoOp>, 
              'eval/miou_1.0_class_2': <tf.Operation 'group_deps_2' type=NoOp>, 
              'eval/miou_1.0_class_0': <tf.Operation 'group_deps' type=NoOp>, 
              'eval/miou_1.0_class_3': <tf.Operation 'group_deps_3' type=NoOp>, 
              'eval/miou_1.0_class_14': <tf.Operation 'group_deps_14' type=NoOp>, 
              'eval/miou_1.0_class_11': <tf.Operation 'group_deps_11' type=NoOp>, 
              'eval/miou_1.0_class_6': <tf.Operation 'group_deps_6' type=NoOp>, 
              'eval/miou_1.0_class_15': <tf.Operation 'group_deps_15' type=NoOp>, 
              'eval/miou_1.0_class_4': <tf.Operation 'group_deps_4' type=NoOp>, 
              'eval/miou_1.0_class_9': <tf.Operation 'group_deps_9' type=NoOp>, 
              'eval/miou_1.0_class_16': <tf.Operation 'group_deps_16' type=NoOp>, 
              'eval/miou_1.0_class_7': <tf.Operation 'group_deps_7' type=NoOp>, 
              'eval/miou_1.0_class_10': <tf.Operation 'group_deps_10' type=NoOp>, 
              'eval/miou_1.0_class_12': <tf.Operation 'group_deps_12' type=NoOp>})

        '''
    '''
    tf.Print(input, data, message=None, first_n=None, summarize=None, name=None)
        最低要求两个输入，input和data，input是需要打印的变量的名字，data要求是一个list，里面包含要打印的内容。
    '''
    summary_ops = []
    for metric_name, metric_value in six.iteritems(metrics_to_values):
      op = tf.summary.scalar(metric_name, metric_value)     # 显示标量信息
      op = tf.Print(op, [metric_value], metric_name)
      summary_ops.append(op)

    summary_op = tf.summary.merge(summary_ops)
    summary_hook = contrib_training.SummaryAtEndHook(
        log_dir=FLAGS.eval_logdir, summary_op=summary_op)
    hooks = [summary_hook]

    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:  # 为0  暂不考虑
      num_eval_iters = FLAGS.max_number_of_evaluations

    if FLAGS.quantize_delay_step >= 0:  # -1 暂不考虑
      contrib_quantize.create_eval_graph()

    contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer
        .TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
    contrib_training.evaluate_repeatedly(
        checkpoint_dir=FLAGS.checkpoint_dir,
        master=FLAGS.master,
        eval_ops=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        hooks=hooks,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
