from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

def set_embeddings(vocabulary_size, position_size, word_embedding_dimension, position_embedding_dimension, initial_value):
    with tf.device("/cpu:0"), tf.name_scope("embedding_layer"), tf.variable_scope("embedding_layer"):
        word_embeddings = tf.get_variable(name="word_embeddings", shape=[vocabulary_size, word_embedding_dimension], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=True)
        #set the word embeddings as given pre-trained word embeddings
        pretrain_initialization = word_embeddings.assign(initial_value)

        #learn position embeddings for relation extraction
        head_position_embeddings = tf.get_variable(name="head_position_embeddings", shape=[position_size, position_embedding_dimension], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(6.0/(position_size+position_embedding_dimension)), math.sqrt(6.0/(position_size+position_embedding_dimension))))
        tail_position_embeddings = tf.get_variable(name="tail_position_embeddings",
                                                   shape=[position_size, position_embedding_dimension],
                                                   dtype=tf.float32, initializer=tf.random_uniform_initializer(
                -math.sqrt(6.0 / (position_size + position_embedding_dimension)),
                math.sqrt(6.0 / (position_size + position_embedding_dimension))))

    return pretrain_initialization

def get_embeddings(mentions, head_positions, tail_positions):
    with tf.variable_scope("embedding_layer", reuse=True):
        word_embeddings = tf.get_variable(name="word_embeddings")
        wemb = tf.nn.embedding_lookup(word_embeddings, mentions)

        head_position_embeddings = tf.get_variable(name="head_position_embeddings")
        tail_position_embeddings = tf.get_variable(name="tail_position_embeddings")
        hpemb = tf.nn.embedding_lookup(head_position_embeddings, head_positions)
        tpemb = tf.nn.embedding_lookup(tail_position_embeddings, head_positions)

        #append word and position embeddings as features to feed cnn
        emb = tf.concat(2, [wemb, hpemb, tpemb], name="input_features")

    return emb

def conv(emb, filter_width, word_embedding_dimension, position_embedding_dimension, out_channels):
    '''
        given an input tensor wemb of shape [batch_size, input_width, in_channels] (default data_format)
        and a filter/kernel tensor of shape [filter_width, in_channels, out_channels]
    '''
    feature_dimension = word_embedding_dimension + 2 * position_embedding_dimension
    with tf.name_scope("convolutional_layer"), tf.variable_scope("convolutional_layer"):
        #Xavier initialization
        conv_filter = tf.get_variable(name="conv_filter_w", shape=[filter_width, feature_dimension, out_channels], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(6.0/(feature_dimension*filter_width + out_channels)), math.sqrt(6.0/(feature_dimension*filter_width + out_channels))))

        conv_representation = tf.nn.conv1d(emb, conv_filter, stride = 1, padding="VALID", name="convolutional_representation")

        #conv_representation = tf.nn.tanh(tf.nn.conv1d(wemb, conv_filter, stride=1, padding="SAME") + conv_bias)
    return conv_representation

def pooling(conv_representation, masks, out_width, out_channels):
    with tf.name_scope("pooling_layer"):
        #handle variable lengths of sentences
        variable_length_conv_representation = conv_representation + tf.expand_dims(masks, -1)

        #max pooling
        conv_2d_representation = tf.reshape(variable_length_conv_representation, [-1, 1, out_width, out_channels], name="var_len_conv_representation_2d")
        pooled_representation = tf.nn.max_pool(conv_2d_representation, ksize=[1,1,out_width,1], strides=[1,1,out_width,1], padding="VALID", name="pooled_representation_2d")
        semb = tf.reshape(pooled_representation, [-1, out_channels], name="pooled_representation")

        conv_bias = tf.Variable(tf.constant(0.1, shape=[out_channels]), name="conv_filter_b")
        nolineared_semb = tf.nn.tanh(semb + conv_bias, name="cnn_features")

        #semb = tf.reduce_max(conv_representation, axis=1)
        #semb = tf.reduce_max(variable_length_conv_representation, axis=1)
    return nolineared_semb

def set_output_layer(out_channels, category_size):
    with tf.name_scope("output_layer"), tf.variable_scope("output_layer"):
        output_w = tf.get_variable(name="softmax_w", shape=[out_channels, category_size], dtype=tf.float32,
                                   initializer=tf.random_uniform_initializer(
                                       minval=-math.sqrt(6.0 / (category_size + out_channels)),
                                       maxval=math.sqrt(6.0 / (category_size + out_channels))))
        output_b = tf.get_variable(name="softmax_b", shape=[category_size], dtype=tf.float32,
                                   initializer=tf.constant_initializer(value=0.1))
        return

def classifier(semb, keep_prob):
    with tf.name_scope("output_layer"), tf.variable_scope("output_layer", reuse=True):
        #drop out
        semb_drop = tf.nn.dropout(semb, keep_prob, name="dropped_cnn_features")

        output_w = tf.get_variable(name="softmax_w")
        output_b = tf.get_variable(name="softmax_b")

        #calculate logits of the softmax output layer
        logits = tf.matmul(semb_drop, output_w) + output_b

    return logits

# lb_masks = tf.placeholder(dtype=tf.float32, shape=[None, category_size])
# sent_num_per_pair = tf.placeholder(dtype=tf.int32, shape=[None]) for dynamic partition
def get_one_loss(logits, lb_masks, sent_num_per_pair, labels, batch_size, beta):
    #select the sentence whose proability (a normalized distribution) over the label class is max
    probabilities = tf.nn.softmax(logits, name="probabilities_wrt_sentence_relation")
    lb_probabilities = tf.reduce_sum(tf.multiply(probabilities, lb_masks), axis=1)

    partitions = tf.dynamic_partition(logits, sent_num_per_pair, batch_size)
    lb_prob_partitions = tf.dynamic_partition(lb_probabilities, sent_num_per_pair, batch_size)

    max_lb_prob_idxs = [tf.cast(tf.argmax(lb_probs, axis=0), tf.int32) for lb_probs in lb_prob_partitions]
    eg_logits = tf.concat(0, [tf.slice(p, begin=[idx, 0], size=[1, -1]) for p, idx in
                                     zip(partitions, max_lb_prob_idxs)])

    #calculate the multi-class cross-entropy loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(eg_logits, labels, name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

    with tf.variable_scope("convolutional_layer", reuse=True):
        conv_filter = tf.get_variable(name="conv_filter_w")
        conv_filter_norm = tf.reduce_sum(tf.multiply(conv_filter, conv_filter, name="conv_filter_w_square"), name="conv_filter_l2_norm_square")

    #classification objective (loss) combined with regularization
    objective = tf.add(loss, beta * conv_filter_norm, name="objective")

    return objective

def set_att_layer(out_channels, category_size):
    with tf.name_scope("attention_layer"), tf.variable_scope("attention_layer"):
        att_w = tf.get_variable(name="att_w", shape=[category_size, out_channels], dtype=tf.float32, initializer=tf.constant_initializer(1), trainable=True)
        att_w_l2_norm = tf.sqrt(tf.reduce_sum(tf.multiply(att_w, att_w)), name="att_w_l2_norm")
        tf.summary.scalar("att_w_l2_norm", att_w_l2_norm)

def get_att_loss(semb, out_channels, sent_num_per_pair, labels, batch_size, keep_prob, beta):
    with tf.name_scope("attention_layer"), tf.variable_scope("attention_layer", reuse=True):
        att_w = tf.get_variable(name="att_w")
    with tf.name_scope("output_layer"), tf.variable_scope("output_layer", reuse=True):
        output_w = tf.get_variable(name="softmax_w")
        output_b = tf.get_variable(name="softmax_b")

    #partition batch into examples
    semb_partitions = tf.dynamic_partition(semb, sent_num_per_pair, batch_size)
    label_partitions = tf.unstack(labels)

    #calculate quadratic form
    weights = [tf.reduce_sum(tf.multiply(eg_semb, tf.multiply(tf.slice(att_w, begin=[eg_lb,0], size=[1,-1])[0], tf.reshape(tf.slice(output_w, begin=[0, eg_lb], size=[-1,1]), [out_channels]))), axis=1) for eg_semb, eg_lb in zip(semb_partitions, label_partitions)]
    normalized_weights = [tf.nn.softmax(eg_weights) for eg_weights in weights]
    features = tf.stack([tf.reduce_sum(tf.multiply(eg_semb, tf.expand_dims(eg_weights, 1)), axis=0) for eg_semb, eg_weights in zip(semb_partitions, normalized_weights)])

    #drop out
    features_drop = tf.nn.dropout(features, keep_prob, name="dropped_att_cnn_features")

    #feed output layer
    logits = tf.matmul(features_drop, output_w) + output_b

    # calculate the multi-class cross-entropy loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

    with tf.variable_scope("convolutional_layer", reuse=True):
        conv_filter = tf.get_variable(name="conv_filter_w")
        conv_filter_norm = tf.reduce_sum(tf.multiply(conv_filter, conv_filter, name="conv_filter_w_square"), name="conv_filter_l2_norm_square")

    #classification objective (loss) combined with regularization
    objective = tf.add(loss, beta * conv_filter_norm, name="objective")

    return objective

def training(loss, learning_rate):
    tf.summary.scalar("loss", loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    #training operation
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def robust_training(loss, learning_rate):
    tf.summary.scalar("loss", loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    tvars = tf.trainable_variables()

    grads = [tf.clip_by_norm(t=g, clip_norm=200) for g in tf.gradients(loss, tvars)]

    train_op = optimizer.apply_gradients(zip(grads, tvars))

    return train_op

def get_one_prediction(logits, sent_num_per_pair, batch_size):
    all_sentence_probabilities = tf.nn.softmax(logits, name="probabilities_wrt_sentence_relation_eval")
    example_partitions = tf.dynamic_partition(all_sentence_probabilities, sent_num_per_pair, batch_size)
    predictions = tf.stack([tf.reduce_max(eg_par, axis=0) for eg_par in example_partitions])

    return predictions

def get_att_prediction(semb, out_channels, category_size, sent_num_per_pair, batch_size, keep_prob):
    with tf.name_scope("attention_layer"), tf.variable_scope("attention_layer", reuse=True):
        att_w = tf.get_variable(name="att_w")
    with tf.name_scope("output_layer"), tf.variable_scope("output_layer", reuse=True):
        output_w = tf.get_variable(name="softmax_w")
        output_b = tf.get_variable(name="softmax_b")

    #[S, R]
    part_of_quadratic_pred = tf.multiply(tf.transpose(att_w), output_w)

    # partition batch into examples
    semb_partitions_pred = tf.dynamic_partition(semb, sent_num_per_pair, batch_size)
    # calculate quadratic form
    weights_pred = [tf.matmul(eg_semb, part_of_quadratic_pred) for eg_semb in semb_partitions_pred]
    normalized_weights_pred = [tf.nn.softmax(eg_weights, dim=0) for eg_weights in weights_pred]
    features_pred = [tf.reduce_sum(tf.multiply(tf.expand_dims(eg_semb, axis=1), tf.expand_dims(eg_normalized_weights, axis=2)), axis=0) for
                     eg_semb, eg_normalized_weights in zip(semb_partitions_pred, normalized_weights_pred)]
    # drop out
    features_drop_pred = [tf.nn.dropout(ftd, keep_prob) for ftd in features_pred]

    # feed output layer
    predictions = tf.stack([tf.reduce_max(tf.matmul(ftdp, output_w) + output_b, axis=0) for ftdp in features_drop_pred])

    return predictions

def get_one_prediction_se(logits):
    all_sentence_of_one_example_probabilities = tf.nn.softmax(logits, name="probabilities_wrt_sentence_relation_of_se_eval")
    predictions_se = tf.reduce_max(all_sentence_of_one_example_probabilities, axis=0)

    return predictions_se

def get_att_prediction_se(semb, keep_prob):
    with tf.name_scope("attention_layer"), tf.variable_scope("attention_layer", reuse=True):
        att_w = tf.get_variable(name="att_w")
    with tf.name_scope("output_layer"), tf.variable_scope("output_layer", reuse=True):
        output_w = tf.get_variable(name="softmax_w")
        output_b = tf.get_variable(name="softmax_b")

    #calculate quadratic form
    part_of_quadratic_pred_se = tf.multiply(tf.transpose(att_w), output_w)
    weights_pred_se = tf.matmul(semb, part_of_quadratic_pred_se)
    normalized_weights_pred_se = tf.nn.softmax(weights_pred_se, dim=0)
    features_pred_se = tf.reduce_sum(tf.multiply(tf.expand_dims(semb, axis=1), tf.expand_dims(normalized_weights_pred_se, axis=2)), axis=0)

    #drop out
    features_drop_pred_se = tf.nn.dropout(features_pred_se, keep_prob)

    #feed output layer
    predictions_se = tf.reduce_max(tf.matmul(features_drop_pred_se, output_w) + output_b, axis=0)

    return predictions_se