import tensorflow as tf
#from tensorflow.core.framework import meta_graph_pb2
from tensorflow.python.training.saver import read_meta_graph_file
import config
FLAGS = tf.app.flags.FLAGS

def import_graph_def(meta_graph_file = FLAGS.graph_path):
    '''
    import graph define by metagraph file.
    return a saver which contain the define of a graph
    '''
    #meta_graph_def = read_meta_graph_file(meta_graph_file)
    #saver = tf.train.Saver(tf.trainable_variables(), saver_def = meta_graph_def)
    saver = tf.train.import_meta_graph(meta_graph_file)
    return saver
def load_weights(sess, saver, weight_file = FLAGS.ckpt_path):
    saver.restore(sess, weight_file)
    print "load model complete"
