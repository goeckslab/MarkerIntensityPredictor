import tensorflow as tf
import tensorflow_probability as tfp

a = tf.Variable(5.)
b = tf.Variable(7.)

c = a * b
print(c)

list_a = [1, 2, 3, 4, 5]
list_b = [1, 2, 3, 4, 5]

zipped_lists = zip(list_a, list_b)


print(tuple(zipped_lists))
print(tuple(zipped_lists))

