import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("ckpt_path", '~/', "path to ckpt model which used for transforming")
tf.app.flags.DEFINE_string("graph_path", '~/', "path to meta graph which used for transforming")
tf.app.flags.DEFINE_string("save_path","~/","path to save tfmodel")
tf.app.flags.DEFINE_string("tfmodel_path","/home/ceca/bingzhe/data/model/inception/v5.pb", "path to your tfmodel")

