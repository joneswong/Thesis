import time
import os.path

import numpy as np
import tensorflow as tf

from cnn import *

def preprocessing_for_test(dataset, st_len, pos_len, ws):
    print("Preprocessing test data.")

    num_of_categories = len(dataset.relation2id)

    #triple level
    begins = []
    lens = []
    Ys = []

    #sentence_level
    Xs = []
    HPs = []
    TPs = []
    As = []

    for k, v in dataset.test_triple2mentions.items():
        begins.append(len(Xs))
        lens.append(len(v))
        lbs = set()

        for j in range(len(v)):
            idx = v[j]

            lbs.add(dataset.test_labels[idx])

            ms = [0] * st_len
            hps = [0] * st_len
            tps = [0] * st_len
            ls = [0] * (st_len - ws + 1)

            if st_len >= dataset.test_mention_lengths[idx]:
                for i in range(dataset.test_mention_lengths[idx]):
                    ms[i] = dataset.test_mentions[idx][i]
                    hps[i] = dataset.test_head_positions[idx] - i
                    tps[i] = dataset.test_tail_positions[idx] - i
                for i in range(dataset.test_mention_lengths[idx] - ws + 1, st_len - ws + 1):
                    ls[i] = -999999
            else:
                left = min(dataset.test_head_positions[idx], dataset.test_tail_positions[idx]) - 15
                right = max(dataset.test_head_positions[idx], dataset.test_tail_positions[idx]) + 15
                if left < 0:
                    left = 0
                if right >= dataset.test_mention_lengths[idx]:
                    right = dataset.test_mention_lengths[idx] - 1
                i = 0
                if st_len > right - left:
                    for si in range(left, right+1):
                        ms[i] = dataset.test_mentions[idx][si]
                        hps[i] = dataset.test_head_positions[idx] - si
                        tps[i] = dataset.test_tail_positions[idx] - si
                        i += 1
                    for i in range(right-left+1-ws+1, st_len - ws + 1):
                        ls[i] = -999999
                else:
                    for si in range(left, left + (st_len / 2)):
                        ms[i] = dataset.test_mentions[idx][si]
                        hps[i] = dataset.test_head_positions[idx] - si
                        tps[i] = dataset.test_tail_positions[idx] - si
                        i += 1
                    for si in range(right - (st_len / 2) + 1, right+1):
                        ms[i] = dataset.test_mentions[idx][si]
                        hps[i] = dataset.test_head_positions[idx] - si
                        tps[i] = dataset.test_tail_positions[idx] - si
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

        Ys.append(lbs)

    return begins, lens, Ys, Xs, HPs, TPs, As

def generate_pr(sess, tokenid_place_holder, headposition_place_holder, tailposition_place_holder, mask_place_holder, keep_prob, pred_op, dataset, begins, lens, Ys, Xs, HPs, TPs, As, config, epoch_tag=""):
    predictions = []
    tot = 0

    for i in range(len(Ys)):
        '''
            prepare values for previously defined place holders:
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

        for k in range(lens[i]):
            tk_xs.append(Xs[begins[i] + k])
            hp_xs.append(HPs[begins[i] + k])
            tp_xs.append(TPs[begins[i] + k])
            mask_xs.append(As[begins[i] + k])

        feed_dict = {tokenid_place_holder: tk_xs, headposition_place_holder: hp_xs, tailposition_place_holder: tp_xs,
                     mask_place_holder: mask_xs, keep_prob: 1}

        # _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        prediction_value = sess.run(pred_op, feed_dict=feed_dict)

        for j in range(1, len(dataset.relation2id)):
            if j in Ys[i]:
                tot += 1
                predictions.append((True, prediction_value[j]))
            else:
                predictions.append((False, prediction_value[j]))

    ranks = sorted(predictions, key=lambda x: x[1], reverse=True)

    num_of_correct = 0
    pr_path = os.path.join(config.log_dir, "pr_"+epoch_tag+".txt")
    f = open(pr_path, 'w')
    for i in range(min(2000, len(ranks))):
        if ranks[i][0] == True:
            num_of_correct += 1
        p = float(num_of_correct) / (i + 1)
        r = float(num_of_correct) / tot
        f.write(str(p) + '\t' + str(r) + '\n')
    f.close()

def test(dataset, we, config):
    print("Test a cnn model for relation extraction.")

    with tf.Graph().as_default():
        #place holders to be fed with inputs or initial values
        initial_wemb_place_holder = tf.placeholder(tf.float32, shape=[len(we.words), we.dimension])

        tokenid_place_holder = tf.placeholder(tf.int32, shape=[None, config.max_length])
        headposition_place_holder = tf.placeholder(tf.int32, shape=[None, config.max_length])
        tailposition_place_holder = tf.placeholder(tf.int32, shape=[None, config.max_length])

        mask_place_holder = tf.placeholder(tf.float32, shape=[None, config.max_length - config.conv_filter_width + 1])

        #lb_masks_place_holder = tf.placeholder(tf.float32, shape=[None, len(dataset.relation2id)])
        #sent_num_per_pair_place_holder = tf.placeholder(dtype=tf.int32, shape=[None])
        #labels_place_holder = tf.placeholder(tf.int32, shape=[None])

        keep_prob = tf.placeholder(tf.float32)



        #define the tensors and operations of considered model
        wemb_init_op = set_embeddings(len(we.words), 1+2*config.max_position, we.dimension, config.position_embedding_dim, initial_wemb_place_holder)

        emb = get_embeddings(tokenid_place_holder, headposition_place_holder, tailposition_place_holder)

        conv_representation = conv(emb, config.conv_filter_width, we.dimension, config.position_embedding_dim, config.sent_embedding_dim)

        semb = pooling(conv_representation, mask_place_holder, config.max_length - config.conv_filter_width + 1, config.sent_embedding_dim)

        set_output_layer(config.sent_embedding_dim, len(dataset.relation2id))

        if config.mi == "att":
            set_att_layer(config.sent_embedding_dim, len(dataset.relation2id))
            prediction = get_att_prediction_se(semb, keep_prob)
        else:
            logits = classifier(semb, keep_prob)
            prediction = get_one_prediction_se(logits)

        #train_op = training(loss, config.learning_rate, config.beta)



        #set up the execution
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        sess.run(init)
        #_ = sess.run([wemb_init_op], feed_dict={initial_wemb_place_holder: we.embeddings})

        checkpoint_file = os.path.join(config.log_dir, "model.ckpt-" + str(config.max_epochs-1))
        saver.restore(sess, checkpoint_file)



        #prepare test data
        begins, lens, Ys, Xs, HPs, TPs, As = preprocessing_for_test(dataset, config.max_length, config.max_position, config.conv_filter_width)

        predictions = []
        tot = 0

        for i in range(len(Ys)):
            '''
                prepare values for previously defined place holders:
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

            for k in range(lens[i]):
                tk_xs.append(Xs[begins[i]+k])
                hp_xs.append(HPs[begins[i]+k])
                tp_xs.append(TPs[begins[i]+k])
                mask_xs.append(As[begins[i]+k])

            feed_dict = {tokenid_place_holder: tk_xs, headposition_place_holder:hp_xs, tailposition_place_holder:tp_xs, mask_place_holder: mask_xs, keep_prob: 1}

            # _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            prediction_value = sess.run(prediction, feed_dict=feed_dict)

            for j in range(1,len(dataset.relation2id)):
                if j in Ys[i]:
                    tot += 1
                    predictions.append((True, prediction_value[j]))
                else:
                    predictions.append((False, prediction_value[j]))

        ranks = sorted(predictions, key=lambda x : x[1], reverse=True)

        num_of_correct = 0
        pr_path = os.path.join(config.log_dir, "pr.txt")
        f = open(pr_path, 'w')
        for i in range(min(2000,len(ranks))):
            if ranks[i][0] == True:
                num_of_correct += 1
            p = float(num_of_correct) / (i + 1)
            r = float(num_of_correct) / tot
            f.write(str(p) + '\t' + str(r) + '\n')
        f.close()
