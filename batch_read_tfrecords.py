import tensorflow as tf

def main(_):
  filename_queue = tf.train.string_input_producer(['/tmp/train.tfrecords'], num_epochs=None)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  datas = tf.parse_single_example(serialized_example, features={
    'features': tf.VarLenFeature(tf.int64),
    'labels': tf.FixedLenFeature([], tf.int64),
  })
  num_steps = 10
  features = datas['features']
  features = tf.sparse_to_dense(sparse_indices=features.indices[:num_steps], output_shape=[num_steps],
                                sparse_values=features.values[:num_steps], default_value=0)
  labels = datas['labels']
  features_batch, labels_batch = tf.train.shuffle_batch([features, labels], batch_size=5, capacity=(10 + 3 * 5),
                                                        min_after_dequeue=10, num_threads=1)
  with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        batch_features_r, batch_labels_r = sess.run([features_batch, labels_batch])
        print 'batch_words_r : ', batch_features_r.shape
        print batch_features_r
        print 'batch_labels_r : ', batch_labels_r.shape
        print batch_labels_r
    except tf.errors.OutOfRangeError:
      print 'Done reading'
    finally:
      coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()