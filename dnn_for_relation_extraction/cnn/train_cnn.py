import time
import os.path

import numpy as np
import tensorflow as tf

from cnn import *

from test_cnn import *

def preprocessing(dataset, st_len, pos_len, ws):
    print("Preprocessing training data.")

    num_of_categories = len(dataset.relation2id)

    #triple level
    begins = []
    lens = []
    one_hot_Ys = []
    Ys = []

    #sentence_level
    Xs = []
    HPs = []
    TPs = []
    As = []

    for k, v in dataset.train_triple2mentions.items():
        begins.append(len(Xs))
        lens.append(len(v))

        for j in range(len(v)):
            idx = v[j]

            if j == 0:
                Y = dataset.train_labels[idx]
                Ys.append(Y)
                one_hot_Y = [0] * num_of_categories
                one_hot_Y[Y] = 1
                one_hot_Ys.append(one_hot_Y)

            ms = [0] * st_len
            hps = [0] * st_len
            tps = [0] * st_len
            ls = [0] * (st_len - ws + 1)

            if st_len >= dataset.train_mention_lengths[idx]:
                for i in range(dataset.train_mention_lengths[idx]):
                    ms[i] = dataset.train_mentions[idx][i]
                    hps[i] = dataset.train_head_positions[idx] - i
                    tps[i] = dataset.train_tail_positions[idx] - i
                for i in range(dataset.train_mention_lengths[idx] - ws + 1, st_len - ws + 1):
                    ls[i] = -999999
            else:
                left = min(dataset.train_head_positions[idx], dataset.train_tail_positions[idx]) - 15
                right = max(dataset.train_head_positions[idx], dataset.train_tail_positions[idx]) + 15
                if left < 0:
                    left = 0
                if right >= dataset.train_mention_lengths[idx]:
                    right = dataset.train_mention_lengths[idx] - 1
                i = 0
                if st_len > right - left:
                    for si in range(left, right+1):
                        ms[i] = dataset.train_mentions[idx][si]
                        hps[i] = dataset.train_head_positions[idx] - si
                        tps[i] = dataset.train_tail_positions[idx] - si
                        i += 1
                    for i in range(right-left+1-ws+1, st_len - ws + 1):
                        ls[i] = -999999
                else:
                    for si in range(left, left + (st_len / 2)):
                        ms[i] = dataset.train_mentions[idx][si]
                        hps[i] = dataset.train_head_positions[idx] - si
                        tps[i] = dataset.train_tail_positions[idx] - si
                        i += 1
                    for si in range(right - (st_len / 2) + 1, right+1):
                        ms[i] = dataset.train_mentions[idx][si]
                        hps[i] = dataset.train_head_positions[idx] - si
                        tps[i] = dataset.train_tail_positions[idx] - si
                        i += 1

            for i in range(st_len):
                if hps[i] > pos_len:
                    hps[i] = pos_len
                if hps[i] < -pos_len:
                    hps[i] = -pos_len
                if tps[i] > pos_len:
                    tps[i] = pos_len
                if tps[i] < -pos_len:
                    tps[i] = -pos_len
                hps[i] += pos_len
                tps[i] += pos_len

            Xs.append(ms)
            HPs.append(hps)
            TPs.append(tps)
            As.append(ls)

    return begins, lens, one_hot_Ys, Ys, Xs, HPs, TPs, As

def train(dataset, we, config):
    print("Train an %s_cnn model for relation extraction."%(config.mi))

    with tf.Graph().as_default():
        #place holders to be fed with inputs or initial values
        initial_wemb_place_holder = tf.placeholder(tf.float32, shape=[len(we.words), we.dimension])

        tokenid_place_holder = tf.placeholder(tf.int32, shape=[None, config.max_length])
        headposition_place_holder = tf.placeholder(tf.int32, shape=[None, config.max_length])
        tailposition_place_holder = tf.placeholder(tf.int32, shape=[None, config.max_length])

        #for "one":
        mask_place_holder = tf.placeholder(tf.float32, shape=[None, config.max_length - config.conv_filter_width + 1])

        lb_masks_place_holder = tf.placeholder(tf.float32, shape=[None, len(dataset.relation2id)])
        sent_num_per_pair_place_holder = tf.placeholder(dtype=tf.int32, shape=[None])
        labels_place_holder = tf.placeholder(tf.int32, shape=[config.batch_size])

        keep_prob = tf.placeholder(tf.float32)



        #define the tensors and operations of considered model
        wemb_init_op = set_embeddings(len(we.words), 1+2*config.max_position, we.dimension, config.position_embedding_dim, initial_wemb_place_holder)

        emb = get_embeddings(tokenid_place_holder, headposition_place_holder, tailposition_place_holder)

        conv_representation = conv(emb, config.conv_filter_width, we.dimension, config.position_embedding_dim, config.sent_embedding_dim)

        semb = pooling(conv_representation, mask_place_holder, config.max_length - config.conv_filter_width + 1, config.sent_embedding_dim)

        set_output_layer(config.sent_embedding_dim, len(dataset.relation2id))

        if config.mi == "att":
            set_att_layer(config.sent_embedding_dim, len(dataset.relation2id))
            loss = get_att_loss(semb, config.sent_embedding_dim, sent_num_per_pair_place_holder, labels_place_holder, config.batch_size, keep_prob, config.beta)
            prediction = get_att_prediction(semb, config.sent_embedding_dim, len(dataset.relation2id), sent_num_per_pair_place_holder, config.batch_size, keep_prob)
            train_op = training(loss, config.learning_rate)#robust_training(loss, config.learning_rate)
            prediction_se = get_att_prediction_se(semb, keep_prob)
        else:
            logits = classifier(semb, keep_prob)
            loss = get_one_loss(logits, lb_masks_place_holder, sent_num_per_pair_place_holder, labels_place_holder, config.batch_size, config.beta)
            prediction = get_one_prediction(logits, sent_num_per_pair_place_holder, config.batch_size)
            train_op = training(loss, config.learning_rate)
            prediction_se = get_one_prediction_se(logits)



        #set up the execution
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        sess.run(init)
        sess.run(wemb_init_op, feed_dict={initial_wemb_place_holder: we.embeddings})

        #prepare training data
        begins, lens, one_hot_Ys, Ys, Xs, HPs, TPs, As = preprocessing(dataset, config.max_length, config.max_position, config.conv_filter_width)
        #prepare test data
        begins_test, lens_test, Ys_test, Xs_test, HPs_test, TPs_test, As_test = preprocessing_for_test(dataset, config.max_length, config.max_position, config.conv_filter_width)
        eg_ids = np.array(range(len(begins))).astype("int32")
        batch_count = 0
        monitor_segment = config.monitor_density

        #begin traversing training data
        for epoch in xrange(config.max_epochs):
            print("%d epoch begins."%(epoch))
            start_time = time.time()

            np.random.shuffle(eg_ids)

            for i in range(0,len(eg_ids),config.batch_size):
                ''' prepare values for previously defined place holders:
                tokenid_place_holder
                headposition_place_holder
                tailposition_place_holder

                mask_place_holder

                lb_masks_place_holder
                sent_num_per_pair_place_holder
                labels_place_holder
                batch_size_place_holder
                '''
                tk_xs = []
                hp_xs = []
                tp_xs = []
                mask_xs = []
                mask_ys = []
                partition_ids = []
                lb_ys = []

                cur_batch_size = min(config.batch_size, len(eg_ids)-i)

                if cur_batch_size != config.batch_size:
                    continue

                for j in range(config.batch_size):
                    eg_id = eg_ids[i+j]
                    for k in range(lens[eg_id]):
                        tk_xs.append(Xs[begins[eg_id]+k])
                        hp_xs.append(HPs[begins[eg_id]+k])
                        tp_xs.append(TPs[begins[eg_id]+k])
                        mask_xs.append(As[begins[eg_id]+k])
                        mask_ys.append(one_hot_Ys[eg_id])
                        partition_ids.append(j)
                    lb_ys.append(Ys[eg_id])

                feed_dict = {tokenid_place_holder: tk_xs, headposition_place_holder:hp_xs, tailposition_place_holder:tp_xs, mask_place_holder: mask_xs,
                             lb_masks_place_holder:mask_ys, sent_num_per_pair_place_holder:partition_ids, labels_place_holder: lb_ys, keep_prob: config.dropout_rate}

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                batch_count += 1

                if batch_count % monitor_segment == 0:
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, batch_count)
                    summary_writer.flush()

            duration = time.time() - start_time



            #evaluate current model on training set
            print("Training data eval:")
            num_of_correct = 0

            for i in range(0,len(eg_ids),config.batch_size):
                tk_xs = []
                hp_xs = []
                tp_xs = []
                mask_xs = []
                mask_ys = []
                partition_ids = []
                lb_ys = []

                cur_batch_size = min(config.batch_size, len(eg_ids)-i)
                if cur_batch_size != config.batch_size:
                    continue

                for j in range(config.batch_size):
                    eg_id = eg_ids[i+j]
                    for k in range(lens[eg_id]):
                        tk_xs.append(Xs[begins[eg_id]+k])
                        hp_xs.append(HPs[begins[eg_id]+k])
                        tp_xs.append(TPs[begins[eg_id]+k])
                        mask_xs.append(As[begins[eg_id]+k])
                        mask_ys.append(one_hot_Ys[eg_id])
                        partition_ids.append(j)
                    lb_ys.append(Ys[eg_id])

                feed_dict = {tokenid_place_holder: tk_xs, headposition_place_holder:hp_xs, tailposition_place_holder:tp_xs, mask_place_holder: mask_xs,
                             lb_masks_place_holder:mask_ys, sent_num_per_pair_place_holder:partition_ids, labels_place_holder: lb_ys, keep_prob: 1}

                prediction_value = sess.run(prediction, feed_dict=feed_dict)
                predicted = np.argmax(prediction_value, axis=1)

                for j in range(cur_batch_size):
                    if predicted[j] == lb_ys[j]:
                        num_of_correct += 1

            print("%d epoch: accuracy = %.4f (%.3f sec)."%(epoch, float(num_of_correct)/((len(Ys) / config.batch_size) * config.batch_size), duration))
            checkpoint_file = os.path.join(config.log_dir, "model.ckpt")
            saver.save(sess, checkpoint_file, global_step=epoch)

            generate_pr(sess, tokenid_place_holder, headposition_place_holder, tailposition_place_holder, mask_place_holder, keep_prob, prediction_se, dataset, begins_test, lens_test, Ys_test, Xs_test, HPs_test, TPs_test, As_test, config, str(epoch))