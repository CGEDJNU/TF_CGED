import tensorflow as tf
import numpy as np
data = [[[3,4,5,4],[1,2,3,5],[1,1,2,2]],[[3,4,5,5],[0,1,2,4],[1,2,3,4]],[[5,6,79,5],[5,6,7,6],[1,2,3,4]],[[1,3,4,5],[4,5,6,5],[1,2,3,4]]]

with tf.Session() as sess:
    data0 = tf.constant(data)
    data1 = tf.constant(data)
    print('Done!')
