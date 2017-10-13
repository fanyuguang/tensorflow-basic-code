import tensorflow as tf
import numpy as np

indices = tf.constant([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [2, 3]], dtype=tf.int64)
values = tf.constant(["a", "b", "c", "d", "e", "f", "g", "h", "i"], dtype=tf.string)
dense_shape = tf.constant([3, 4], dtype=tf.int64)
split_input_dense1 = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

input_dense2 = tf.constant(["a b c", "d e", "f g h i"], dtype=tf.string)
split_input_dense2 = tf.string_split(input_dense2)

# print split_input_dense1.get_shape().as_list()
# print split_input_dense2.get_shape().as_list()


input_dense = tf.placeholder(dtype=tf.string, shape=[None], name='input_sentences')
split_input_dense = tf.string_split(input_dense)

# slice_input_list = tf.sparse_split(sp_input=split_input_dense, num_split=split_input_dense.get_shape()[0], axis=0)
# slice_input_list = tf.sparse_split(sp_input=split_input_dense, num_split=split_input_dense.dense_shape[0], axis=0)
slice_input_list = tf.sparse_split(sp_input=split_input_dense, num_split=3, axis=0)

join_input_list = []
for slice_input in slice_input_list:
  slice_input_value = slice_input.values
  join_input = tf.reduce_join(slice_input_value, reduction_indices=0, separator=' ')
  join_input_list.append(join_input)
output_dense = tf.stack(join_input_list)

with tf.Session() as sess:
  feed_dict = {input_dense: np.array(["a b c", "d e", "f g h i"])}
  print sess.run(output_dense, feed_dict=feed_dict)
