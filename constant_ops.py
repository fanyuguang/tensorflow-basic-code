import tensorflow as tf

def main(_):
  x = tf.constant([1, 2, 3, 4])
  y = tf.constant([5, 6, 7, 8])
  z = tf.add(x, y)

  with tf.Session() as sess:
    result = sess.run(z)
    print result

if __name__ == '__main__':
  tf.app.run()