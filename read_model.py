import tensorflow as tf

def main(_):
  x = tf.Variable([1, 2, 3, 4])
  y = tf.Variable([5, 6, 7, 8])
  z = tf.add(x, y)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/tmp/checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    result = sess.run(z)
    print result

if __name__ == '__main__':
  tf.app.run()