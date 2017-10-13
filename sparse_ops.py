import tensorflow as tf

str = tf.constant(["how are you going today", "", "hello world"], dtype=tf.string)

# split 1-D String dense Tensor to words SparseTensor
sparse_words = tf.string_split(str, " ")

# sparse to dense
words = tf.sparse_to_dense(sparse_indices=sparse_words.indices,
                           output_shape=[sparse_words.dense_shape[0], 6],
                           sparse_values=sparse_words.values,
                           default_value='_PAD')

# replace values of SparseTensor
replace_sparse_words = tf.SparseTensor(indices=sparse_words.indices,
                                       values=tf.fill(tf.shape(sparse_words.values), "o"),
                                       dense_shape=sparse_words.dense_shape)

# slice SparseTensor
slice1_dim1_indices = tf.less(sparse_words.indices, tf.constant([4], dtype=tf.int64))
slice1_dim1_indices = tf.reshape(tf.split(slice1_dim1_indices, [1, 1], axis=1)[1], [-1])
slice1_sparse_words = tf.sparse_retain(sparse_words, slice1_dim1_indices)

slice2_dim1_indices = tf.greater_equal(sparse_words.indices, tf.constant([4], dtype=tf.int64))
slice2_dim1_indices = tf.reshape(tf.split(slice2_dim1_indices, [1, 1], axis=1)[1], [-1])
slice2_sparse_words = tf.sparse_retain(sparse_words, slice2_dim1_indices)

# concat SparseTensor
concat_sparse_words = tf.SparseTensor(indices=tf.concat(axis=0, values=[slice1_sparse_words.indices, slice2_sparse_words.indices]),
                                      values=tf.concat(axis=0, values=[slice1_sparse_words.values, slice2_sparse_words.values]),
                                      dense_shape=slice1_sparse_words.dense_shape)
concat_sparse_words = tf.sparse_reorder(concat_sparse_words)

# join SparseTensor to 1-D String dense Tensor
join_words_list = []
slice_words_list = tf.sparse_split(sp_input=sparse_words, num_split=3, axis=0)
# slice_words_list = tf.sparse_split(sp_input=sparse_words, num_split=sparse_words.get_shape()[0], axis=0)
for slice_words in slice_words_list:
  slice_words = slice_words.values
  join_words = tf.reduce_join(slice_words, reduction_indices=0, separator=" ")
  join_words_list.append(join_words)
join_str = tf.stack(join_words_list)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  print sess.run(join_str)