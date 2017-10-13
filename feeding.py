import tensorflow as tf
import numpy as np

def main(_):
  x = tf.placeholder(dtype=tf.float32, shape=[2, 2])
  y = tf.placeholder(dtype=tf.float32, shape=[2, 2])
  z = tf.add(x, y)

  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    feed_dict = {x: np.array([[1, 2], [3, 4]]), y: np.array([[5, 6], [7, 8]])}
    result = sess.run(z, feed_dict=feed_dict)
    print result

if __name__ == '__main__':
  tf.app.run()