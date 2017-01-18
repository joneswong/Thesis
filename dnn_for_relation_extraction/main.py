from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from utility.word_embeddings import EmbeddingModel
from utility.nyt_dataset import NYTDataset
from utility.fb_dataset import FBDataset
import cnn.train_cnn as train_cnn
import cnn.test_cnn as test_cnn
from kbe.run_kbe import run_kbe

FLAGS = None

def main():
    '''we = EmbeddingModel("./thu_baselines/NRE/data/vec.bin", "./thu_baselines/NRE/data/vector2.txt")
    dataset = NYTDataset("./thu_baselines/NRE/data/RE", we.word2id)

    train_cnn.train(dataset, we, FLAGS)
    test_cnn.test(dataset, we, FLAGS)'''

    dataset = FBDataset(train_path="./FB15k/freebase_mtr100_mte100-train.txt", valid_path="./FB15k/freebase_mtr100_mte100-valid.txt", test_path="./FB15k/freebase_mtr100_mte100-test.txt")
    run_kbe(dataset, FLAGS)

    print("Completed.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mi",
        type=str,
        default="att",
        help="Startegy of handling multi-instance learning with options \"one\", \"att\""
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Max number of tokens considered for each sentence"
    )
    parser.add_argument(
        "--max_position",
        type=int,
        default=30,
        help="Max distance from a token to a entity mention"
    )
    parser.add_argument(
        "--position_embedding_dim",
        type=int,
        default=5,
        help="Dimension of position embeddings"
    )
    parser.add_argument(
        "--entity_embedding_dim",
        type=int,
        default=50,
        help="Dimension of entity embeddings"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,#160,
        help="Number of examples in each batch"
    )
    parser.add_argument(
        "--conv_filter_width",
        type=int,
        default=3,
        help="Width of convolutional filter"
    )
    parser.add_argument(
        "--sent_embedding_dim",
        type=int,
        default=230,
        help="Dimension of sentence embeddings"
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Rate of drop out"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,#0.2,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0001,
        help="Weight of regularization"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./kbe_out",#"./log",
        help="Directory to put the log data"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,#50,
        help="Number of epochs to traverse training examples"
    )
    parser.add_argument(
        "--monitor_density",
        type=int,
        default=64,
        help="Number of batches each summarization takes"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="The margin to seperate positive triples from negative ones"
    )
    parser.add_argument(
        "--norm_flag",
        type=int,
        default=1,
        help="Which norm to use"
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
    #tf.app.run(main=train_cnn.train, argv=[sys.argv[0]] + unparsed)