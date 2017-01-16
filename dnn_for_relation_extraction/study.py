import math
import numpy as np
import tensorflow as tf

def main():
    print "Learn by doing."

    R = 3
    S = 4
    B = 2

    att_w = tf.Variable(tf.constant(value=1, shape=[R, S], dtype=tf.float32), trainable=True)
    output_w = tf.Variable(tf.constant(value=1, shape=[S, R], dtype=tf.float32), trainable=True)
    output_b = tf.Variable(tf.constant(value=1, shape=[R], dtype=tf.float32), trainable=True)

    semb = tf.placeholder(dtype=tf.float32, shape=[None, S])
    sent_num_per_pair = tf.placeholder(dtype=tf.int32, shape=[None])
    labels = tf.placeholder(dtype=tf.int32, shape=[B])
    keep_prob = tf.placeholder(dtype=tf.float32)

    so = tf.nn.softmax(semb, dim=0)

    #partition batch into examples
    semb_partitions = tf.dynamic_partition(semb, sent_num_per_pair, B)
    label_partitions = tf.unstack(labels)

    #calculate quadratic form
    weights = [tf.reduce_sum(tf.multiply(eg_semb, tf.multiply(tf.slice(att_w, begin=[eg_lb,0], size=[1,-1])[0], tf.reshape(tf.slice(output_w, begin=[0, eg_lb], size=[-1,1]), [S]))), axis=1) for eg_semb, eg_lb in zip(semb_partitions, label_partitions)]
    normalized_weights = [eg_weights / tf.reduce_sum(eg_weights) for eg_weights in weights]
    features = tf.stack([tf.reduce_sum(tf.multiply(eg_semb, tf.expand_dims(eg_weights, 1)), axis=0) for eg_semb, eg_weights in zip(semb_partitions, normalized_weights)])

    #drop out
    features_drop = tf.nn.dropout(features, keep_prob, name="dropped_att_cnn_features")

    #feed output layer
    logits = tf.matmul(features_drop, output_w) + output_b

    # calculate the multi-class cross-entropy loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")



    init = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init)

    feed_dict = {semb:[[1,0,1,0],[0, 0, 0, 1],[1,1,1,1]], sent_num_per_pair:[0,0,1], labels:[1,2], keep_prob:1}

    value = sess.run(so, feed_dict=feed_dict)
    print value

if __name__=="__main__":
    main()