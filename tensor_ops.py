import tensorflow as tf

# replace untrusted prop that less than prop_limit
prop_limit = 0.999
predict_scores = tf.constant([[1.0, 0.999, 0.99, 0.9, 0.5], [1.0, 0.998, 0.98, 0.8, 0.5]], dtype=tf.float32)
predict_labels = tf.constant([[5, 6, 5, 5, 0], [7, 5, 6, 0, 0]], dtype=tf.int64)
replaced_labels = tf.zeros_like(predict_labels, dtype=tf.int64)
trusted_prop_flag = tf.greater_equal(predict_scores, tf.constant(prop_limit, dtype=tf.float32))
new_predict_labels = tf.where(trusted_prop_flag, predict_labels, replaced_labels)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  print sess.run(predict_labels)
  print sess.run(new_predict_labels)