import tensorflow as tf
import config
import numpy as np
from scipy import misc
FLAGS = tf.app.flags.FLAGS

def show_graph(graph):
    print '===========\n graph description'
    for op in graph.get_operations(): 
        print op.name
    print '===========\n variables'
    for v in graph.get_collection('constant'):
        print v.name
    print '========'    
def load_images(image_path):
    image = np.array(misc.imread(image_path))
    image = misc.imresize(image,(224,224))
    print image.shape
    image = np.expand_dims(image,axis = 0)
    print image.shape
    return image
def replace_variables_name(sess):
    var = {}
    for v in tf.trainable_variables():
        var[v.value().name] = sess.run(v)
    return var
def forward_by_tfmodel(input_image, result_node_name = 'embedding', path_to_model = FLAGS.tfmodel_path):
    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open(path_to_model, 'rb') as graph_file:
            graph_def.ParseFromString(graph_file.read())
            tf.import_graph_def(graph_def)
        show_graph(tf.get_default_graph())
        with tf.Session() as sess:
            graph = tf.get_default_graph()
            input_tensor = graph.get_tensor_by_name("import/input:0")
            print input_tensor.get_shape()
            #phase_train = graph.get_tensor_by_name("import/phase_train:0")
            embedding = graph.get_tensor_by_name("import/output2:0")
            embedding_array = sess.run(embedding, feed_dict = {input_tensor:input_image})
    return embedding_array

def calculate_distance(images1, images2):
    embedding1 = forward_by_tfmodel(input_image = images1)
    embedding2 = forward_by_tfmodel(input_image = images2)
    print embedding1.shape
    ans = np.linalg.norm(embedding1-embedding2)
    print ans

if __name__ == '__main__':
    image1 = load_images('data/test1.jpg')
    image2 = load_images('data/test2.jpg')
    calculate_distance(image1, image2)




