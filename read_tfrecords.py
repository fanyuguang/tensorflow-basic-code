import tensorflow as tf

def main(_):
  for serialized_example in tf.python_io.tf_record_iterator('/tmp/train.tfrecords'):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    features = example.features.feature['features'].int64_list.value
    labels = example.features.feature['labels'].int64_list.value
    feature_list = [feature for feature in features]
    label_list = [label for label in labels]
    print('labels: {}, features: {}'.format(label_list, feature_list))

if __name__ == '__main__':
  tf.app.run()