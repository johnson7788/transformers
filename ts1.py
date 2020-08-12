import tensorflow as tf
data = tf.constant([2, 0, 4], dtype=tf.float32)
result_1 = tf.norm(data)
with tf.Session() as sess:
    print(sess.run(result_1))

tf.keras.regularizers.l1()