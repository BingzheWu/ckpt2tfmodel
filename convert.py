import tensorflow as tf
import numpy as np
import config
from tensorflow.python.framework import graph_util
from utils import show_graph 
import os
from read_ckpt import load_weights, import_graph_def
FLAGS = tf.app.flags.FLAGS
def write_proto(file_path, graph_def):
    with open(file_path,'wb')as f:
        f.write(graph_def.SerializeToString())
def del_moving_average_op():
    #ans = []
    #for op in graph.get_operations():
     #   if 'ExponentialMovingAverage' not in op.name:
      #      ans.append(op.name)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = import_graph_def()
            load_weights(sess = sess, saver = saver)
            saver.save(sess,'/home/ceca/bingzhe/data/model/model', global_step = 0)
    return 0

def convert(save_path = FLAGS.save_path, model_name = 'facenet.tfmodel', RESULTS_NODE_NAME = 'embeddings'):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = import_graph_def()
            load_weights(sess = sess, saver = saver)
            show_graph(sess.graph)
            #white_list = del_moving_average_op(sess.graph)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [RESULTS_NODE_NAME])
            #write_proto(os.path.join(save_path, model_name), constant_graph)
            print type(constant_graph)
            tf.train.write_graph(constant_graph, save_path, model_name, as_text = False)
            #tf.train.write_graph(sess.graph_def, './', 'nn4.pbtxt', as_text = False)
def convert1(save_path = FLAGS.save_path, model_name = 'facenet.tfmodel', RESULTS_NODE_NAME = 'embeddings'):
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open('nn4.pbtxt', 'rb') as graph_file:
            graph_def.ParseFromString(graph_file.read())
            tf.import_graph_def(graph_def)
        with tf.Session() as sess:
            #saver = import_graph_def()
            saver = tf.train.Saver(tf.trainable_variables())
            load_weights(sess = sess, saver = saver)
            show_graph(sess.graph)
            #white_list = del_moving_average_op(sess.graph)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [RESULTS_NODE_NAME])
            #write_proto(os.path.join(save_path, model_name), constant_graph)
            print type(constant_graph)
            tf.train.write_graph(constant_graph, save_path, model_name, as_text = False)
            #tf.train.write_graph(sess.graph_def, save_path, model_name, as_text = False)
def main(argv = None):
    #del_moving_average_op()
    convert()
    #convert1()
if __name__ =='__main__':
    tf.app.run()




