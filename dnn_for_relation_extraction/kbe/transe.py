from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

def set_embeddings(entity_vocabulary_size, structural_relation_embedding_size, dimension):
    with tf.device("/cpu:0"), tf.name_scope("kb_embedding_layer"), tf.variable_scope("kb_embedding_layer"):
        entity_embeddings = tf.get_variable(name="entity_embeddings", shape=[entity_vocabulary_size, dimension], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(6.0/(entity_vocabulary_size+dimension)), math.sqrt(6.0/(entity_vocabulary_size+dimension))), trainable=True)
        structural_relation_embeddings = tf.get_variable(name="structural_relation_embeddings", shape=[structural_relation_embedding_size, dimension], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-math.sqrt(6.0/(structural_relation_embedding_size+dimension)), maxval=math.sqrt(6.0/(structural_relation_embedding_size+dimension))), trainable=True)

        entity_embedding_l2norms = tf.sqrt(tf.reduce_sum(tf.multiply(entity_embeddings, entity_embeddings), axis=1), name="entity_embedding_l2norms")
        normalized_entity_embeddings = tf.div(entity_embeddings, tf.expand_dims(entity_embedding_l2norms,axis=1))
        normalize_entity_embeddings_op = entity_embeddings.assign(normalized_entity_embeddings)

        structural_relation_embedding_l2norms = tf.sqrt(
            tf.reduce_sum(tf.multiply(structural_relation_embeddings, structural_relation_embeddings), axis=1),
            name="structural_relation_embedding_l2norms")
        normalized_structural_relation_embeddings = tf.div(structural_relation_embeddings,
                                                           tf.expand_dims(structural_relation_embedding_l2norms, axis=1))
        normalize_structural_relation_embeddings_op = structural_relation_embeddings.assign(
            normalized_structural_relation_embeddings)

    return normalize_entity_embeddings_op, normalize_structural_relation_embeddings_op

def get_embeddings(entity_ids, relation_ids):
    with tf.name_scope("kb_embedding_layer"), tf.variable_scope("kb_embedding_layer", reuse=True):
        entity_embeddings = tf.get_variable(name="entity_embeddings")
        structural_relation_embeddings = tf.get_variable(name="structural_relation_embeddings")

        eemb = tf.nn.embedding_lookup(params=entity_embeddings, ids=entity_ids)
        remb = tf.nn.embedding_lookup(params=structural_relation_embeddings, ids=relation_ids)

    return eemb, remb

def get_loss(eemb, remb, batch_size, margin, norm_flag):
    ph = eemb[:batch_size]
    pt = eemb[batch_size:2*batch_size]
    nh = eemb[2*batch_size:3*batch_size]
    nt = eemb[3*batch_size:]

    pos_residuals = tf.subtract(tf.add(ph, remb), pt)
    neg_residuals = tf.subtract(tf.add(nh, remb), nt)

    if norm_flag == 2:
        pos_scores = tf.reduce_sum(tf.multiply(pos_residuals, pos_residuals), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(neg_residuals, neg_residuals), axis=1)
    else:
        pos_scores = tf.reduce_sum(tf.abs(pos_residuals), axis=1)
        neg_scores = tf.reduce_sum(tf.abs(neg_residuals), axis=1)

    seperations = pos_scores + margin - neg_scores

    hingeloss = tf.reduce_mean(tf.maximum(seperations, 0))

    return hingeloss

def training(loss, learning_rate):
    tf.summary.scalar("loss", loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # training operation
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def entity_linking(entity_ids, eemb, remb, batch_size, norm_flag, topK=10):
    with tf.name_scope("kb_embedding_layer"), tf.variable_scope("kb_embedding_layer", reuse=True):
        entity_embeddings = tf.get_variable(name="entity_embeddings")
        #structural_relation_embeddings = tf.get_variable(name="structural_relation_embeddings")

    head_ids = entity_ids[:batch_size]
    tail_ids = entity_ids[batch_size:]

    h = eemb[:batch_size]
    t = eemb[batch_size:2 * batch_size]

    tail_residuals = tf.subtract(tf.expand_dims(tf.add(h, remb), axis=1), entity_embeddings)
    head_residuals = tf.add(tf.expand_dims(tf.subtract(remb, t), axis=1), entity_embeddings)

    if norm_flag == 2:
        tail_predictions = -tf.reduce_sum(tf.multiply(tail_residuals, tail_residuals), axis=2)
        head_predictions = -tf.reduce_sum(tf.multiply(head_residuals, head_residuals), axis=2)
    else:
        tail_predictions = -tf.reduce_sum(tf.abs(tail_residuals), axis=2)
        head_predictions = -tf.reduce_sum(tf.abs(head_residuals), axis=2)

    tail_topk_accuracy = tf.nn.in_top_k(tail_predictions, tail_ids, topK)
    head_topk_accuracy = tf.nn.in_top_k(head_predictions, head_ids, topK)

    num_of_correct_tails = tf.reduce_sum(tf.cast(tail_topk_accuracy, tf.float32))
    num_of_correct_heads = tf.reduce_sum(tf.cast(head_topk_accuracy, tf.float32))

    return num_of_correct_tails, num_of_correct_heads