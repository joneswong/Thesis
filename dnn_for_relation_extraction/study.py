import math
import numpy as np
import tensorflow as tf

def main():
    with tf.device("/cpu:0"), tf.name_scope("kb_embedding_layer"), tf.variable_scope("kb_embedding_layer"):
        entity_embeddings = tf.get_variable(name="entity_embeddings", shape=[3, 2], dtype=tf.float32, initializer=tf.constant_initializer(value=1), trainable=True)

    eids = tf.placeholder(dtype=tf.int32, shape=[None])

    eemb = tf.nn.embedding_lookup(entity_embeddings, eids, max_norm=1.0)

    loss = 0.5 * tf.reduce_sum(tf.multiply(eemb, eemb))

    optimizer = tf.train.GradientDescentOptimizer(0.1)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)



    init = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init)

    _, loss_value = sess.run([train_op, loss], feed_dict={eids:[0,1,2]})
    print loss_value
    emb_value = sess.run(entity_embeddings)
    print emb_value

if __name__=="__main__":
    main()