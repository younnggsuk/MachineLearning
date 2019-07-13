import tensorflow as tf


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2) # node3 = node1 + node2

print("Node1:", node1)
print("Node2:", node2)
print("Node3:", node3)

sess = tf.Session()

print(sess.run([node1, node2]))
print(sess.run(node3))
