from tensorflow.python.framework import graph_util
import tensorflow as tf
from tensorflow.contrib.layers import flatten

'''
计算模型所需的运算量
'''
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    time_and_memory = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.time_and_memory())
    print('GFLOPs: {};    Trainable params: {}'.format(flops.total_float_ops / 1000000000.0, params.total_parameters))


def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


# ***** (3) Load frozen graph *****
with tf.Graph().as_default() as graph:
    graph = load_pb(r'D:\pb文件\xception\frozen_inference_graph.pb')
    print('stats after freezing')
    stats_graph(graph)
