import tensorflow as tf
import numpy as np
import config
from tensorflow.python.framework import graph_util
from utils import show_graph 
from read_ckpt import load_weights, import_graph_def
FLAGS = tf.app.flags.FLAGS
def convert(save_path = FLAGS.save_path, model_name = 'facenet.tfmodel', RESULTS_NODE_NAME = 'embeddings'):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = import_graph_def()
            load_weights(sess = sess, saver = saver)
            show_graph(sess.graph)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [RESULTS_NODE_NAME])
            tf.train.write_graph(constant_graph, save_path, model_name, as_text = False)
def main(argv = None):
    convert()
if __name__ =='__main__':
    tf.app.run()




