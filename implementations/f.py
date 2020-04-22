import tensorflow as tf
def map(fn, arrays, dtype=tf.float32):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out

# example: batch affine tranformation
x = tf.random_normal([4,5,6])
M = tf.random_normal([4,6,10])
b = tf.random_normal([4,10])

f = lambda x0,M0,b0: tf.matmul(x0,M0) + b0
batch_y = map(f, [x,M,b])
print batch_y.shape