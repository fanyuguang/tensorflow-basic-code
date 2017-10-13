import tensorflow as tf


def main():
  num_steps = 5
  words_len = tf.constant([2, 3], dtype=tf.int64)
  batch_size = words_len.shape[0].value
  slice_indices = tf.constant([], dtype=tf.int64)
  for index in range(batch_size):
    sub_slice_indices = tf.range(words_len[index])
    sub_slice_indices = tf.add(tf.constant(index * num_steps, dtype=tf.int64), sub_slice_indices)
    slice_indices = tf.concat([slice_indices, sub_slice_indices], axis=0)

  x = tf.get_variable('x', shape=[10, 7], initializer=tf.random_uniform_initializer())
  y = tf.get_variable('y', shape=[10], initializer=tf.random_uniform_initializer())
  x_slice = tf.gather(x, indices=slice_indices)
  y_slice = tf.gather(y, indices=slice_indices)

  words = tf.constant([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]])
  slice_words_indices = tf.constant([[0, 0], [0, 1], [1, 0], [2, 2]])
  slice_words = tf.gather_nd(words, slice_words_indices)

  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(slice_indices)
    x_value, x_slice_value, y_value, y_slice_value = sess.run([x, x_slice, y, y_slice])
    print 'x : ', x_value.shape
    print x_value
    print 'x_slice : ', x_slice_value.shape
    print x_slice_value
    print 'y : ', y_value.shape
    print y_value
    print 'y_slice : ', y_slice_value.shape
    print y_slice_value

    print sess.run(words)
    print sess.run(slice_words)


if __name__ == '__main__':
  main()
