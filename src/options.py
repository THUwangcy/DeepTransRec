# -*- coding: UTF-8 -*-

import os
import time


class Options:
    def __init__(self, FLAGS):
        # Where is the clicked file used for training and testing
        self.data_path = FLAGS.data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        # Minimal items a user has to has interaction with for she to be included
        self.user_min = FLAGS.user_min

        # Minimal users a item has to has interaction with for it to be included
        self.item_min = FLAGS.item_min

        # Maximal epochs to train
        self.epoch_max = FLAGS.epoch_max

        # Embedding dimension
        self.emb_dim = FLAGS.k

        # L2 regularizer for item embedding and all user embedding
        self.reg = FLAGS.reg

        # L2 regularizer for item bias
        self.bias_reg = FLAGS.bias_reg

        # L2 regularizer for user embedding
        self.user_reg = FLAGS.user_reg

        # Learning rate
        self.lr = 0.05

        # HR topN
        self.top_n = 50

        # Train user batch size
        self.train_batch_size = 16

        # Eval user batch size
        self.eval_batch_size = 1000

        # Eval sample item num
        self.eval_sample_item = 5000

        # Eval sample user num
        self.eval_sample_user = 50000

        # Timestamp
        self.timestamp = int(time.time())

        # Log save path
        self.save_path = '../logs/'

        # Checkpoints save path
        self.checkpoints_path = '../checkpoints/'
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        # Write summary
        self.write_summary = True
