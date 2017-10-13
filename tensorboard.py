import tensorflow as tf

def main(_):
  step = tf.Variable(-1, name='step')
  one = tf.constant(1)
  update_step = tf.assign_add(step, one)

  x = tf.Variable(2, name='x')
  new_x = tf.multiply(x, tf.constant(2))
  update_x = tf.assign(x, new_x)

  tf.summary.scalar('x', x)
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/tensorboard/', sess.graph)
    sess.run(init_op)
    for _ in range(10):
      sess.run([update_x, update_step])
      x_value, step_value, summary_value = sess.run([x, step, summary_op])
      writer.add_summary(summary_value, global_step=step_value)

if __name__ == '__main__':
  tf.app.run()