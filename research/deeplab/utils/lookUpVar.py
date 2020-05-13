import tensorflow as tf
#从ckpt文件中获取variable变量的名字
def get_trainable_variables_name_from_ckpt(meta_graph_path, ckpt_path):
    #定义一个新的graph
    graph = tf.Graph()
    #将其设置为默认图:
    with graph.as_default():
        with tf.Session() as session:
            #加载计算图
            saver = tf.train.import_meta_graph(meta_graph_path)
            #加载模型到session中关联的graph中，即将模型文件中的计算图加载到这里的graph中
            saver.restore(session, ckpt_path)
            v_names = []
            #获取session所关联的图中可被训练的variable
            #使用tf.trainable_variables()获取variable时，只有在该函数前面定义的variable才会被获取到
            #在其后面定义不会被获取到，
            for v in tf.trainable_variables():
                v_names.append(v)
            return v_names

#利用pywrap_tensorflow获取ckpt文件中的所有变量，得到的是variable名字与shape的一个map
from tensorflow.python import pywrap_tensorflow
def get_all_variables_name_from_ckpt(ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    all_var = reader.get_variable_to_shape_map()
    #reader.get_variable_to_dtype_map()
    return all_var

meta_graph_path = '/media/xzq/DA18EBFA09C1B27D/exp/train_on_train_set/train/model.ckpt-200000.meta'
ckpt_path = '/media/xzq/DA18EBFA09C1B27D/exp/train_on_train_set/train/model.ckpt-200000'
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file('/media/xzq/DA18EBFA09C1B27D/exp/train_on_train_set/train/model.ckpt-200000', None, True)
v_names = get_trainable_variables_name_from_ckpt(meta_graph_path, ckpt_path)
print(v_names)