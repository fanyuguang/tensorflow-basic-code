import tensorflow as tf

def main(_):
  x = tf.Variable([1, 2, 3, 4])
  y = tf.Variable([5, 6, 7, 8])
  with tf.device('/gpu:0'):
    z = tf.add(x, y)

  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(z)
    print result

if __name__ == '__main__':
  tf.app.run()