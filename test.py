import tensorflow as tf 
from utils import getsess
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./tmp/model.ckpt-0.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
  print(getsess())