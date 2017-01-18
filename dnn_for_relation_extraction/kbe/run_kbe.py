import time
import os.path

import numpy as np
import tensorflow as tf

from transe import *

def preprocess_for_run(dataset):
    train = []
    valid = []
    test = []

    for tri in dataset.train:
        train.append((dataset.entity2id[tri[0]], dataset.relation2id[tri[1]], dataset.entity2id[tri[2]]))
    for tri in dataset.valid:
        valid.append((dataset.entity2id[tri[0]], dataset.relation2id[tri[1]], dataset.entity2id[tri[2]]))
    for tri in dataset.test:
        test.append((dataset.entity2id[tri[0]], dataset.relation2id[tri[1]], dataset.entity2id[tri[2]]))

    assert len(train) == len(dataset.train)
    assert len(valid) == len(dataset.valid)
    assert len(test) == len(dataset.test)

    return train, valid, test

def run_kbe(dataset, config):
    print("Train an TransE model for KBC.")

    with tf.Graph().as_default():
        entid_place_holder = tf.placeholder(tf.int32, shape=[None])
        relid_place_holder = tf.placeholder(tf.int32, shape=[None])

        batch_size = tf.placeholder(tf.int32)

        # define the tensors and operations of considered model
        normalize_eemb_op, normalize_remb_op = set_embeddings(len(dataset.entity2id), len(dataset.relation2id), config.entity_embedding_dim)

        eemb, remb = get_embeddings(entid_place_holder, relid_place_holder)

        loss = get_loss(eemb, remb, batch_size, config.margin, config.norm_flag)

        train_op = training(loss, config.learning_rate)

        count_correct_tails, count_correct_heads = entity_linking(entid_place_holder, eemb, remb, batch_size, config.norm_flag)

        # set up the execution
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)

        sess.run(init)
        sess.run(normalize_remb_op)

        train, valid, test = preprocess_for_run(dataset)

        eg_ids = np.array(range(len(train))).astype("int32")
        batch_count = 0
        bs = config.batch_size
        eids = [0] * (4*bs)
        rids = [0] * bs
        cur_batch_size = 0
        sample_space = len(dataset.entity2id)
        monitor_segment = config.monitor_density
        best_hits = 0.0

        for epoch in range(config.max_epochs):
            np.random.shuffle(eg_ids)

            start_time = time.time()

            for i in range(len(eg_ids)):
                eg_idx = eg_ids[i]

                eids[cur_batch_size] = train[eg_idx][0]
                rids[cur_batch_size] = train[eg_idx][1]
                eids[bs+cur_batch_size] = train[eg_idx][2]

                if i % 2 == 0:
                    eids[bs+bs+cur_batch_size] = train[eg_idx][0]
                    eids[bs + bs + bs + cur_batch_size] = np.random.randint(0, sample_space)
                else:
                    eids[bs + bs + cur_batch_size] = np.random.randint(0, sample_space)
                    eids[bs+bs+bs+cur_batch_size] = train[eg_idx][2]

                cur_batch_size += 1
                if cur_batch_size == bs:
                    cur_batch_size = 0
                    feed_dict = {entid_place_holder:eids, relid_place_holder:rids, batch_size:bs}
                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                    batch_count += 1

                    if batch_count % monitor_segment == 0:
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, batch_count)
                        summary_writer.flush()

            duration = time.time() - start_time
            print("%d epoch finished (%.2f sec)"%(epoch, duration))

            if epoch % 5 == 0:
                print("Training data eval:")
                num_of_correct_tails = 0
                num_of_correct_heads = 0
                cur_batch_size = 0
                valid_batch_count = 0
                valid_eids = [0] * (2*bs)
                valid_rids = [0] * bs
                for i in range(len(valid)):
                    valid_eids[cur_batch_size] = valid[i][0]
                    valid_rids[cur_batch_size] = valid[i][1]
                    valid_eids[bs + cur_batch_size] = valid[i][2]

                    cur_batch_size += 1
                    if cur_batch_size == bs:
                        feed_dict = {entid_place_holder:valid_eids, relid_place_holder:valid_rids, batch_size:bs}
                        batch_correct_tails, batch_correct_heads = sess.run([count_correct_tails, count_correct_heads], feed_dict=feed_dict)
                        num_of_correct_tails += batch_correct_tails
                        num_of_correct_heads += batch_correct_heads
                        cur_batch_size = 0
                        valid_batch_count += 1
                tail_hits = float(num_of_correct_tails)/(bs*valid_batch_count)
                head_hits = float(num_of_correct_heads)/(bs*valid_batch_count)
                hits = 0.5 * (tail_hits + head_hits)
                print("Hits@10 on predicting tails, heads, both_sides are %.3f, %.3f, %.3f"%(tail_hits, head_hits, hits))
                if hits >= best_hits:
                    best_hits = hits
                    checkpoint_file = os.path.join(config.log_dir, "model.ckpt")
                    saver.save(sess, checkpoint_file, global_step=epoch)
                    print("Saved model at %d epoch"%(epoch))

        print("Test data eval:")
        num_of_correct_tails = 0
        num_of_correct_heads = 0
        cur_batch_size = 0
        test_batch_count = 0
        test_eids = [0] * (2 * bs)
        test_rids = [0] * bs
        for i in range(len(test)):
            test_eids[cur_batch_size] = test[i][0]
            test_rids[cur_batch_size] = test[i][1]
            test_eids[bs + cur_batch_size] = test[i][2]

            cur_batch_size += 1
            if cur_batch_size == bs:
                feed_dict = {entid_place_holder: test_eids, relid_place_holder: test_rids, batch_size: bs}
                batch_correct_tails, batch_correct_heads = sess.run([count_correct_tails, count_correct_heads],
                                                                    feed_dict=feed_dict)
                num_of_correct_tails += batch_correct_tails
                num_of_correct_heads += batch_correct_heads
                cur_batch_size = 0
                test_batch_count += 1
        tail_hits = float(num_of_correct_tails) / (bs * test_batch_count)
        head_hits = float(num_of_correct_heads) / (bs * test_batch_count)
        hits = 0.5 * (tail_hits + head_hits)
        print("Hits@10 on predicting tails, heads, both_sides are %.3f, %.3f, %.3f" % (tail_hits, head_hits, hits))