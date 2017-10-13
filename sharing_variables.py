import tensorflow as tf

def main(_):
  with tf.variable_scope("scope"):
    x = tf.get_variable("x", shape=[2, 2], initializer=tf.constant_initializer([1, 2, 3, 4]))
    y = tf.get_variable("y", shape=[2, 2], initializer=tf.constant_initializer([5, 6, 7, 8]))
  with tf.variable_scope("scope", reuse=True):
    x1 = tf.get_variable("x")
  z = tf.add(x1, y)

  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(z)
    print result

if __name__ == '__main__':
  tf.app.run()