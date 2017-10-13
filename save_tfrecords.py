import tensorflow as tf

def main(_):
  writer = tf.python_io.TFRecordWriter('/tmp/train.tfrecords')
  for label in range(2):
    for i in range(5):
      example = tf.train.Example(features=tf.train.Features(feature={
        'features': tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3, 4, 5, 6])),
        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
      }))
      writer.write(example.SerializeToString())
  writer.close()

if __name__ == '__main__':
  tf.app.run()