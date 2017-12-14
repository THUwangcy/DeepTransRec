# -*- coding: UTF-8 -*-

import os
import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
from sklearn.externals import joblib

from src.corpus import Corpus
from src.options import Options
from src.model.trans_rec import TransRec
from src.model.pop_rec import PopRec
from src.common import utils

tf.flags.DEFINE_string("data_path", "", "Click tuples path.")
tf.flags.DEFINE_integer("user_min", 5, "Minimal items a user has to has interaction with for she to be included.")
tf.flags.DEFINE_integer("item_min", 5, "Minimal users a item has to has interaction with for it to be included.")
tf.flags.DEFINE_integer("epoch_max", 3000, "Maximal epochs to train.")
tf.flags.DEFINE_integer("k", 10, "Embedding dimension.")
tf.flags.DEFINE_integer("reg", 0.01, "L2 regularizer for item embedding and all user embedding.")
tf.flags.DEFINE_integer("bias_reg", 0.01, "L2 regularizer for item bias.")
tf.flags.DEFINE_integer("user_reg", 0.01, "L2 regularizer for user embedding.")

FLAGS = tf.flags.FLAGS


def go_transrec(opts, corp, session):
    model = TransRec(opts, corp, session)
    model.train()
    model.save_model()
    print("  TEST: AUC = {:<.4f}".format(model.best_test_auc))


def go_poprec(opts, corp, session):
    model = PopRec(opts, corp, session)
    model.eval(sample=False)


def main(_):
    np.random.seed(0)
    opts = Options(FLAGS)
    print("{")
    print("  \"corpus\": \"{}\"".format(opts.data_path))
    corpus_save_path = "../data/corpus.npz"
    if os.path.exists(corpus_save_path):
        corp = joblib.load(corpus_save_path)
    else:
        corp = Corpus()
        corp.load_data(opts.data_path, opts.user_min, opts.item_min)
        joblib.dump(corp, corpus_save_path)
    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(0)
        # go_poprec(opts, corp, session)
        go_transrec(opts, corp, session)

    print("}")


if __name__ == "__main__":
    # Avoid showing tensorflow warning for instructions supporting
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
