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
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import tfprof as contrib_tfprof
from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from deeplab.utils import train_utils
from deployment import model_deploy

slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.
# 设置多GPU训练，多少个GPU就设置多少
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then '
    'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.
# 训练日志输出目录
flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')
# 每训练多少步就输出训练日志loss
flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

# 多少秒保存一次模型的checkpoint文件到logdir
flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')
# 多少秒保存一次summaries。 600
flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean(
    'save_summaries_images', False,
    'Save sample inputs, labels, and semantic predictions as '
    'images to summary.')

# Settings for profiling.

flags.DEFINE_string('profile_logdir', None,
                    'Where the profile files are stored.')

# Settings for training strategy.
# 训练策略  训练优化器 默认为momentum
flags.DEFINE_enum('optimizer', 'momentum', ['momentum', 'adam'],
                  'Which optimizer to use.')


# Momentum optimizer flags
# # 学习策略 默认为poly
flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
# 学习率
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('decay_steps', 0.0,
                   'Decay steps for polynomial learning rate schedule.')

flags.DEFINE_float('end_learning_rate', 0.0,
                   'End learning rate for polynomial learning rate schedule.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')
# 学习率衰减
flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')
# poly学习率上的power
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
# 训练次数
flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')
# 动量下降值
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# Adam optimizer flags
# Adam学习率
flags.DEFINE_float('adam_learning_rate', 0.001,
                   'Learning rate for the adam optimizer.')
# adam优化器参数epsilon
flags.DEFINE_float('adam_epsilon', 1e-08, 'Adam optimizer epsilon.')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
'''
    当设置微调 BN 为True时，batch_size至少为12 否则将fine_tune_batch_norm设为False
'''
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
'''
    权重衰减策略，减小模型复杂度，防止梯度爆炸：
        MobileNet-V2或者Xception中  weight_decay= 0.00004
        ResNet中  0.0001
'''
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')
# 裁剪大小 默认513x513
flags.DEFINE_list('train_crop_size', '513,513',
                  'Image crop size [height, width] during training.')

flags.DEFINE_float(
    'last_layer_gradient_multiplier', 1.0,
    'The gradient multiplier for last layers, which is used to '
    'boost the gradient of last layers if the value > 1.')

# 是否进行上采样
flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Hyper-parameters for NAS training strategy.

flags.DEFINE_float(
    'drop_path_keep_prob', 1.0,
    'Probability to keep each path in the NAS cell when training.')

# Settings for fine-tuning the network.
# 使用预训练权重进行微调 ckpt文件路径
flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
'''
    可以在utils/train_util.py中更改exclude_list中添加不想加载的模型部件，例如logit, globel_step
'''
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
# 微调BN层参数，默认为True  若想节省GPU内存。则设置为False
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')
# 尺度缩放最小因子
flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')
# 尺度缩放最大因子
flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')
# 尺度缩放增加量
flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
'''
    Xception_65   OS=8    空洞率为 12,24,36
    Xception_65   OS=16   空洞率为 6,12,18
    mobilenet_V2  空洞率为None
'''
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')
# 输出步长
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Hard example mining related flags.
flags.DEFINE_integer(
    'hard_example_mining_step', 0,
    'The training step in which exact hard example mining kicks off. Note we '
    'gradually reduce the mining percent to the specified '
    'top_k_percent_pixels. For example, if hard_example_mining_step=100K and '
    'top_k_percent_pixels=0.25, then mining percent will gradually reduce from '
    '100% to 25% until 100K steps after which we only mine top 25% pixels.')

flags.DEFINE_float(
    'top_k_percent_pixels', 1.0,
    'The top k percent pixels (in terms of the loss values) used to compute '
    'loss during training. This is useful for hard pixel mining.')

# Quantization setting.
flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.
# 数据集名称
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')
# 指定带有train的tfrecorder数据集
flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')
# 数据集目录 tfrecord类型
flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


def _build_deeplab(iterator, outputs_to_num_classes, ignore_label):
  """Builds a clone of DeepLab.

  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    outputs_to_num_classes: A map from output type to the number of classes. For
      example, for the task of semantic segmentation with 21 semantic classes,
      we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.
  """
  samples = iterator.get_next()


  # Add name to input and label nodes so we can add to summary.
  samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
  samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)
  '''
      samples形式：
            {'width': <tf.Tensor 'IteratorGetNext:4' shape=(?,) dtype=int64>, 
             'height': <tf.Tensor 'IteratorGetNext:0' shape=(?,) dtype=int64>, 
             'image_name': <tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=string>, 
             'image': <tf.Tensor 'image:0' shape=(?, 513, 513, 3) dtype=float32>, 
             'label': <tf.Tensor 'label:0' shape=(?, 513, 513, 1) dtype=int32>}
  '''
  model_options = common.ModelOptions(  # 模型设置
      outputs_to_num_classes=outputs_to_num_classes,    # 输出类别的数量
      crop_size=[int(sz) for sz in FLAGS.train_crop_size],  # 训练中一次裁剪图片大小 [513,513]
      atrous_rates=FLAGS.atrous_rates,  # 空洞率 6,12,18
      output_stride=FLAGS.output_stride)     # 输出步长 16

  outputs_to_scales_to_logits = model.multi_scale_logits(   # 多尺度输入得到的特征图
      samples[common.IMAGE],     # 一个tensor （batch, 高度， 宽度， 通道数） ?, 513, 513, 3
      model_options=model_options,  # 模型基本设置
      image_pyramid=FLAGS.image_pyramid,     # 图像金字塔输入的尺度 默认为 None， 也就是默认尺度为1.0
      weight_decay=FLAGS.weight_decay,  # 权重衰减，也叫L2正则化  默认为 0.00004
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,  # 微调BN层参数 默认为True
      nas_training_hyper_parameters={    # NAS-Net网络训练的超参数，暂时忽略
          'drop_path_keep_prob': FLAGS.drop_path_keep_prob,
          'total_training_steps': FLAGS.training_number_of_steps,
      })
  # outputs_to_scales_to_logits = {'semantic': <tf.Tensor 'logits/semantic/BiasAdd:0' shape=(?, 129, 129, 19) dtype=float32>}

  # Add name to graph node so we can add to summary.
  output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]    # OUTPUT_TYPE = 'semantic'
  output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(    # MERGED_LOGITS_SCOPE = 'merged_logits'
      output_type_dict[model.MERGED_LOGITS_SCOPE], name=common.OUTPUT_TYPE)
  # outputs_to_num_classes: {'semantic': 19 }
  for output, num_classes in six.iteritems(outputs_to_num_classes):
    train_utils.add_softmax_cross_entropy_loss_for_each_scale(  # softmax交叉熵损失函数
        outputs_to_scales_to_logits[output],    #   <tf.Tensor 'logits/semantic/BiasAdd:0' shape=(?, 129, 129, 19) dtype=float32>
        samples[common.LABEL],  # 'label': <tf.Tensor 'label:0' shape=(?, 513, 513, 1) dtype=int32>
        num_classes,    # 19
        ignore_label,   # 255
        loss_weight=model_options.label_weights,    # 每个标签的权重 默认为None
        upsample_logits=FLAGS.upsample_logits,  # 进行上采样 True
        hard_example_mining_step=FLAGS.hard_example_mining_step,    # 0
        top_k_percent_pixels=FLAGS.top_k_percent_pixels,    # 1
        scope=output)   # 'semantic'


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  # 设置多gpu训练的相关参数
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,  # gpu数量
      clone_on_cpu=FLAGS.clone_on_cpu,  # 默认为False
      replica_id=FLAGS.task,    # taskId
      num_replicas=FLAGS.num_replicas,  # 默认为1
      num_ps_tasks=FLAGS.num_ps_tasks)  # 默认为0

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = FLAGS.train_batch_size // config.num_clones    # 各个gpu均分batch_size

  tf.gfile.MakeDirs(FLAGS.train_logdir)     # 创建存放训练日志的文件
  tf.logging.info('Training on %s set', FLAGS.train_split)

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):
      dataset = data_generator.Dataset(     # 定义数据集参数
          dataset_name=FLAGS.dataset,   #　数据集名称　cityscapes
          split_name=FLAGS.train_split,  # 指定带有train的tfrecorder数据集 默认为“train”
          dataset_dir=FLAGS.dataset_dir,   # 数据集目录　tfrecoder文件的数据集目录
          batch_size=clone_batch_size,  # 均分后各个gpu训练中指定batch_size 的大小
          crop_size=[int(sz) for sz in FLAGS.train_crop_size],  # 训练中裁剪的图像大小　513,513
          min_resize_value=FLAGS.min_resize_value,  #　默认为 None
          max_resize_value=FLAGS.max_resize_value,  #　默认为None
          resize_factor=FLAGS.resize_factor,    # 默认为None
          min_scale_factor=FLAGS.min_scale_factor,   # 训练中，图像变换尺度，用于数据增强　默认最小为0.5
          max_scale_factor=FLAGS.max_scale_factor,   # 训练中，图像变换尺度，用于数据增强　默认最大为２
          scale_factor_step_size=FLAGS.scale_factor_step_size,      # 训练中，图像变换尺度增加的步长,默认为0.25　　从0.5到2
          model_variant=FLAGS.model_variant,    # 指定模型 xception_65
          num_readers=4,    # 读取数据个数 若多gpu可增大加快训练速度
          is_training=True,
          should_shuffle=True,
          should_repeat=True)

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      # 计数作用，每训练一个batch, global加１
      global_step = tf.train.get_or_create_global_step()

      # Define the model and create clones.
      model_fn = _build_deeplab  # 定义deeplab模型
      model_args = (dataset.get_one_shot_iterator(), {
          common.OUTPUT_TYPE: dataset.num_of_classes
      }, dataset.ignore_label) #模型参数
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for model variables.
    for model_var in tf.model_variables():
      summaries.add(tf.summary.histogram(model_var.op.name, model_var))

    # Add summaries for images, labels, semantic predictions
    if FLAGS.save_summaries_images:      # 默认为False
      summary_image = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))
      summaries.add(
          tf.summary.image('samples/%s' % common.IMAGE, summary_image))

      first_clone_label = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
      # Scale up summary image pixel values for better visualization.
      pixel_scaling = max(1, 255 // dataset.num_of_classes)
      summary_label = tf.cast(first_clone_label * pixel_scaling, tf.uint8)
      summaries.add(
          tf.summary.image('samples/%s' % common.LABEL, summary_label))

      first_clone_output = graph.get_tensor_by_name(
          ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/'))
      predictions = tf.expand_dims(tf.argmax(first_clone_output, 3), -1)

      summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
      summaries.add(
          tf.summary.image(
              'samples/%s' % common.OUTPUT_TYPE, summary_predictions))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Build the optimizer based on the device specification.
    with tf.device(config.optimizer_device()):
      learning_rate = train_utils.get_model_learning_rate(  # 获取模型学习率
          FLAGS.learning_policy,    # poly学习策略
          FLAGS.base_learning_rate,     # 0.0001
          FLAGS.learning_rate_decay_step,   # 固定2000次进行一次学习率衰退
          FLAGS.learning_rate_decay_factor,     # 0.1
          FLAGS.training_number_of_steps,   # 训练次数 20000
          FLAGS.learning_power,     # poly power 0.9
          FLAGS.slow_start_step,    # 0
          FLAGS.slow_start_learning_rate,   # 1e-4 缓慢开始的学习率
          decay_steps=FLAGS.decay_steps,    # 0.0
          end_learning_rate=FLAGS.end_learning_rate)     # 0.0

      summaries.add(tf.summary.scalar('learning_rate', learning_rate))
      # 模型训练优化器
      if FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
      elif FLAGS.optimizer == 'adam':   # adam优化器 寻找全局最优点的优化算法，引入了二次方梯度校正
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.adam_learning_rate, epsilon=FLAGS.adam_epsilon)
      else:
        raise ValueError('Unknown optimizer')

    if FLAGS.quantize_delay_step >= 0:  #　默认为-1　忽略
      if FLAGS.num_clones > 1:
        raise ValueError('Quantization doesn\'t support multi-clone yet.')
      contrib_quantize.create_training_graph(
          quant_delay=FLAGS.quantize_delay_step)

    startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps    # FLAGS.startup_delay_steps 默认为15

    with tf.device(config.variables_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)    # 计算total_loss
      total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
      summaries.add(tf.summary.scalar('total_loss', total_loss))

      # Modify the gradients for biases and last layer variables.
      last_layers = model.get_extra_layer_scopes(
          FLAGS.last_layers_contain_logits_only)
      # 获取梯度乘子
      grad_mult = train_utils.get_model_gradient_multipliers(
          last_layers, FLAGS.last_layer_gradient_multiplier)
      # grad_mult : {'logits/semantic/biases': 2.0, 'logits/semantic/weights': 1.0}
      if grad_mult:
        grads_and_vars = slim.learning.multiply_gradients(
            grads_and_vars, grad_mult)

      # Create gradient update op.
      grad_updates = optimizer.apply_gradients(     # 将计算的梯度用于变量上，返回一个应用指定的梯度的操作 opration
          grads_and_vars, global_step=global_step)  # 对global_step进行自增
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries))

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)

    # Start the training.
    profile_dir = FLAGS.profile_logdir   # 默认为None
    if profile_dir is not None:
      tf.gfile.MakeDirs(profile_dir)

    with contrib_tfprof.ProfileContext(
        enabled=profile_dir is not None, profile_dir=profile_dir):
      init_fn = None
      if FLAGS.tf_initial_checkpoint:   # 获取预训练权重
        init_fn = train_utils.get_model_init_fn(
            FLAGS.train_logdir,
            FLAGS.tf_initial_checkpoint,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True)

      slim.learning.train(
          train_tensor,
          logdir=FLAGS.train_logdir,
          log_every_n_steps=FLAGS.log_steps,
          master=FLAGS.master,
          number_of_steps=FLAGS.training_number_of_steps,
          is_chief=(FLAGS.task == 0),
          session_config=session_config,
          startup_delay_steps=startup_delay_steps,
          init_fn=init_fn,
          summary_op=summary_op,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('train_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
