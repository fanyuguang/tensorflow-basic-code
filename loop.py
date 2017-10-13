import tensorflow as tf

i = tf.constant(0, dtype=tf.int64)
cond = lambda i: i < 10
body = lambda i: tf.add(i, tf.constant(1, dtype=tf.int64))
result = tf.while_loop(cond=cond, body=body, loop_vars=[i])

with tf.Session() as sess:
  print sess.run(result)