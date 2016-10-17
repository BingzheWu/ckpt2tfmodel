import tensorflow as tf
import config
FLAGS = tf.app.flags.FLAGS

def import_graph_def(meta_graph_file = FLAGS.graph_path):
    '''
    import graph define by metagraph file.
    return a saver which contain the define of a graph
    '''
    saver = tf.train.import_meta_graph(meta_graph_file)
    return saver
def load_weights(sess, saver, weight_file = FLAGS.ckpt_path):
    saver.restore(sess, weight_file)
    print "load model complete"
