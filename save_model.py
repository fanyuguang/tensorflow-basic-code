import tensorflow as tf

def main(_):
  x = tf.Variable([1, 2, 3, 4])
  y = tf.Variable([5, 6, 7, 8])
  z = tf.add(x, y)

  init_op = tf.global_variables_initializer()
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(z)
    print result
    saver.save(sess, '/tmp/checkpoint/model.ckpt')

if __name__ == '__main__':
  tf.app.run()