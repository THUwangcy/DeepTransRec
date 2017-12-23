# -*- coding: UTF-8 -*-

import os
import sys
import time

import numpy as np
import tensorflow as tf

from src.model.model import Model
from src.common import utils


class TransRec(Model):
    def __init__(self, options, corpus, session):
        Model.__init__(self, options, corpus, session)
        self.last_save_epoch = -1

        # Model parameter
        self._all_user_emb = None
        self._user_emb = None
        self._item_emb = None
        self._item_bias = None

        # Norm parameter
        self._all_user_emb_norm = None
        self._item_emb_norm = None

        # Best parameter
        self._all_user_emb_best = None
        self._user_emb_best = None
        self._item_emb_best = None
        self._item_bias_best = None

        # Nodes in the graph which are used to run/feed/fetch.
        # Train
        self._cur_user = None
        self._prev_item = None
        self._pos_item = None
        self._neg_item = None

        # Key Nodes
        # Train
        self._loss = None
        self._train = None
        self.global_step = None
        self._debug = None
        self._train_writer = None
        # Op
        self._norm_all_user_emb = None
        self._norm_item_emb = None
        self._norm_param_op = None
        self._save_best_param_op = None
        self._restore_best_param_op = None
        # Summary
        self._loss_summary = None

        self.build_graph()
        self.build_eval_graph()

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        if len(self._options.checkpoint_name) > 0:
            ckpt_path = self._options.checkpoints_path + self._options.checkpoint_name
            self.load_model(ckpt_path)

        if self._options.write_summary:
            self._train_writer = tf.summary.FileWriter("{}{}_{}/train".format(
                self._options.save_path, self._options.timestamp, self.to_string()), self._session.graph)
            self._test_writer = tf.summary.FileWriter("{}{}_{}/test".format(
                self._options.save_path, self._options.timestamp, self.to_string()))

    def load_params(self, filename):
        opts = self._options
        corp = self._corpus
        with open("../data/" + filename, 'r') as f:
            params = f.read().split()
            all_user_emb = np.zeros([opts.emb_dim], dtype=np.float32)
            user_emb = np.zeros([corp.n_users, opts.emb_dim], dtype=np.float32)
            item_bias = np.zeros([corp.n_items], dtype=np.float32)
            item_emb = np.zeros([corp.n_items, opts.emb_dim], dtype=np.float32)
            index = 0
            for i in range(opts.emb_dim):
                all_user_emb[i] = params[index + i]
            index += opts.emb_dim
            for i in range(corp.n_users):
                for j in range(opts.emb_dim):
                    user_emb[i][j] = params[index + j]
                index += opts.emb_dim
            for i in range(corp.n_items):
                item_bias[i] = params[index + i]
            index += corp.n_items
            for i in range(corp.n_items):
                for j in range(opts.emb_dim):
                    item_emb[i][j] = params[index + j]
                index += opts.emb_dim
            assert(index == len(params))
            self._session.run([
                tf.assign(self._all_user_emb, tf.convert_to_tensor(all_user_emb)),
                tf.assign(self._user_emb, tf.convert_to_tensor(user_emb)),
                tf.assign(self._item_bias, tf.convert_to_tensor(item_bias)),
                tf.assign(self._item_emb, tf.convert_to_tensor(item_emb))
            ])
        self._session.run(self._norm_param_op)

    def predict(self, cur_user, prev_item, target_item):
        with tf.name_scope('prob'):
            user_num = tf.size(cur_user)
            item_num = tf.shape(target_item)[1]
            target_item_bias = tf.gather(self._item_bias, target_item)
            target_item_emb = tf.gather(self._item_emb, target_item)
            prev_item_emb = tf.gather(self._item_emb, prev_item)
            cur_user_emb = tf.gather(self._user_emb, cur_user)
            pred_item_emb = tf.tile(tf.expand_dims(self._all_user_emb, 0), [user_num, 1]) + cur_user_emb + prev_item_emb
            pred_item_emb = tf.tile(tf.expand_dims(pred_item_emb, 1), [1, item_num, 1])
            return -target_item_bias - self.dist(pred_item_emb, target_item_emb)

    def predict_eval(self, cur_user, prev_item, target_item):
        with tf.name_scope('prob'):
            user_num = tf.size(cur_user)
            item_num = tf.shape(target_item)[1]
            target_item_bias = tf.gather(self._item_bias, target_item)
            target_item_emb = tf.gather(self._item_emb_norm, target_item)
            prev_item_emb = tf.gather(self._item_emb_norm, prev_item)
            cur_user_emb = tf.gather(self._user_emb, cur_user)
            pred_item_emb = tf.tile(tf.expand_dims(self._all_user_emb, 0), [user_num, 1]) \
                + cur_user_emb + prev_item_emb
            pred_item_emb = tf.tile(tf.expand_dims(pred_item_emb, 1), [1, item_num, 1])
            return -target_item_bias - self.dist(pred_item_emb, target_item_emb)

    def dist(self, emb1, emb2):
        return tf.reduce_sum(tf.square(emb1 - emb2), tf.rank(emb1) - 1)

    def forward(self, cur_user, prev_item, pos_item, neg_item):
        """Build the graph for the forward pass."""
        corp = self._corpus
        opts = self._options

        # Declare all variables we need.
        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # All user embedding: [emb_dim]
        init_width = 6.0 / np.sqrt(opts.emb_dim)
        all_user_emb = tf.get_variable(
            initializer=tf.random_uniform([opts.emb_dim], -init_width, init_width),
            name="all_user_emb")
        # self._norm_all_user_emb = tf.assign(all_user_emb, utils.norm_to_unit_ball(all_user_emb, opts.emb_dim))
        self._norm_all_user_emb = tf.assign(all_user_emb, tf.nn.l2_normalize(all_user_emb, dim=0, epsilon=1.0))
        self._all_user_emb = all_user_emb

        # User embedding: [n_users, emb_dim]
        user_emb = tf.get_variable(
            initializer=tf.zeros([corp.n_users, opts.emb_dim]), name="user_emb")
        self._user_emb = user_emb

        # Item bias: [n_items]
        item_bias = tf.get_variable(
            initializer=tf.zeros([corp.n_items]), name="item_bias")
        self._item_bias = item_bias
        # Item embedding: [n_items, emb_dim]
        init_width = 6.0 / np.sqrt(opts.emb_dim)
        item_emb = tf.get_variable(
            initializer=tf.random_uniform([corp.n_items, opts.emb_dim], -init_width, init_width),
            name="item_emb")
        # self._norm_item_emb = tf.assign(item_emb, utils.norm_to_unit_ball(item_emb, opts.emb_dim))
        self._norm_item_emb = tf.assign(item_emb, tf.nn.l2_normalize(item_emb, dim=1, epsilon=1.0))
        self._item_emb = item_emb
        pos_item = tf.reshape(pos_item, [-1, 1])
        neg_item = tf.reshape(neg_item, [-1, 1])
        eval_item = tf.concat([pos_item, neg_item], 1)
        return self.predict(cur_user, prev_item, eval_item)

    def loss(self, item_pred, cur_user, prev_item, pos_item, neg_item):
        """Build the graph for the loss."""
        opts = self._options
        with tf.name_scope("loss"):
            item_num = tf.size(prev_item)
            pos_item_pred, neg_item_pred = tf.gather(item_pred, 0, axis=1), tf.gather(item_pred, 1, axis=1)
            p = tf.reduce_sum(tf.log(tf.sigmoid(pos_item_pred - neg_item_pred)))
            eval_item = tf.concat([pos_item, neg_item], 0)
            eval_item_bias = tf.gather(self._item_bias, eval_item)
            eval_item_emb = tf.gather(self._item_emb, eval_item)
            prev_item_emb = tf.gather(self._item_emb, prev_item)
            cur_user_emb = tf.gather(self._user_emb, cur_user)
            related_params = tf.concat([eval_item_emb, prev_item_emb,
                                        tf.tile(tf.expand_dims(self._all_user_emb, 0), [item_num, 1])], 0)
            reg_losses = tf.nn.l2_loss(related_params) * opts.reg + \
                         tf.nn.l2_loss(eval_item_bias) * opts.bias_reg + \
                         tf.nn.l2_loss(cur_user_emb) * opts.user_reg
            return -p, tf.reduce_sum(reg_losses)

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""
        opts = self._options
        # optimizer = tf.train.GradientDescentOptimizer(opts.lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=opts.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        train = optimizer.minimize(loss, global_step=self.global_step)
        self._train = train

    def build_save_params_graph(self):
        opts = self._options
        corp = self._corpus
        self._all_user_emb_best = tf.get_variable(
            shape=[opts.emb_dim], trainable=False, name="all_user_emb_best")
        self._user_emb_best = tf.get_variable(
            shape=[corp.n_users, opts.emb_dim], trainable=False, name="user_emb_best")
        self._item_bias_best = tf.get_variable(
            shape=[corp.n_items], trainable=False, name="item_bias_best")
        self._item_emb_best = tf.get_variable(
            shape=[corp.n_items, opts.emb_dim], trainable=False, name="item_emb_best")
        self._save_best_param_op = [
            tf.assign(self._all_user_emb_best, self._all_user_emb),
            tf.assign(self._user_emb_best, self._user_emb),
            tf.assign(self._item_emb_best, self._item_emb),
            tf.assign(self._item_bias_best, self._item_bias)
        ]
        self._restore_best_param_op = [
            tf.assign(self._all_user_emb, self._all_user_emb_best),
            tf.assign(self._user_emb, self._user_emb_best),
            tf.assign(self._item_emb, self._item_emb_best),
            tf.assign(self._item_bias, self._item_bias_best)
        ]

    def build_norm_params_graph(self):
        opts = self._options
        corp = self._corpus
        self._all_user_emb_norm = tf.get_variable(
            shape=[opts.emb_dim], trainable=False, name="all_user_emb_norm")
        self._item_emb_norm = tf.get_variable(
            shape=[corp.n_items, opts.emb_dim], trainable=False, name="item_emb_norm")
        self._norm_param_op = [
            tf.assign(self._item_emb_norm, tf.nn.l2_normalize(self._item_emb, dim=1, epsilon=1.0)),
            # tf.assign(self._all_user_emb_norm, tf.nn.l2_normalize(self._all_user_emb, dim=0, epsilon=1.0))
        ]

    def build_graph(self):
        # Declace input placeholder
        # shape [batch_size]
        cur_user = tf.placeholder(dtype=tf.int32, name="cur_user")
        prev_item = tf.placeholder(dtype=tf.int32, name="prev_item")
        pos_item = tf.placeholder(dtype=tf.int32, name="pos_item")
        neg_item = tf.placeholder(dtype=tf.int32, name="neg_item")
        self._cur_user, self._prev_item, self._pos_item, self._neg_item = cur_user, prev_item, pos_item, neg_item

        # get predict valud
        item_pred = self.forward(cur_user, prev_item, pos_item, neg_item)

        # calculate loss
        l1, l2 = self.loss(item_pred, cur_user, prev_item, pos_item, neg_item)
        self._loss = l1 + l2
        self._loss1, self._loss2 = l1, l2

        # optimize
        self.optimize(self._loss)

        self.build_save_params_graph()
        self.build_norm_params_graph()

        # summary
        loss_summary = tf.summary.scalar("Loss", self._loss)
        self._loss_summary = tf.summary.merge([loss_summary])

    def get_train_batch(self, batch_no, total_range):
        opts = self._options
        corp = self._corpus
        batch_size = opts.train_batch_size
        users, prev_items, pos_items, neg_items = list(), list(), list(), list()
        for ind in total_range[batch_no * batch_size:(batch_no + 1) * batch_size]:
            if ind % 10000 == 0:
                print('.', end='')
                sys.stdout.flush()
            # Sample a user
            while True:
                u = np.random.randint(0, corp.n_users)
                if len(self._clicked_per_user[u]) >= 2:
                    break
            users.append(u)
            # Sample a positive item
            prev_ind = np.random.randint(0, len(corp.pos_per_user[u]) - 1)
            prev_i = corp.pos_per_user[u][prev_ind]['item']
            pos_i = corp.pos_per_user[u][prev_ind + 1]['item']
            prev_items.append(prev_i)
            pos_items.append(pos_i)
            # Sample a negative item
            while True:
                neg_i = np.random.randint(0, corp.n_items)
                if neg_i not in self._clicked_per_user[u]:
                    break
            neg_items.append(neg_i)
        return users, prev_items, pos_items, neg_items

    def one_iter(self):
        corp = self._corpus
        opts = self._options

        batch_size = opts.train_batch_size
        total_num = corp.n_clicks
        batch_num = total_num // batch_size + (1 if total_num % batch_size != 0 else 0)
        total_range = np.arange(total_num)
        prepare_time, session_time, avg_l, avg_l1, avg_l2 = 0, 0, 0.0, 0.0, 0.0
        batch_start = time.time()
        for b_no in range(batch_num):
            # Batch Begin!
            prepare_start = time.time()
            users, prev_items, pos_items, neg_items = self.get_train_batch(b_no, total_range)
            feed_dict = {
                self._cur_user: users,
                self._prev_item: prev_items,
                self._pos_item: pos_items,
                self._neg_item: neg_items
            }
            # self.show_params(u, prev_i, pos_i, neg_i)
            prepare_time += time.time() - prepare_start

            session_start = time.time()
            [summary, _, l1, l2, step] = self._session.run(
                [self._loss_summary, self._train, self._loss1, self._loss2, self.global_step],
                feed_dict=feed_dict
            )
            if opts.write_summary:
                self._train_writer.add_summary(summary, step)
            avg_l += l1 + l2
            avg_l1 += l1
            avg_l2 += l2
            # TODO: fast norm item_emb and all_user_emb
            session_time += (time.time() - session_start)
            # self.show_params(u, prev_i, pos_i, neg_i)
        self._session.run([self._norm_item_emb])
        avg_l /= batch_num * batch_size
        avg_l1 /= batch_num * batch_size
        avg_l2 /= batch_num * batch_size
        print("\n  Time: np: {:<.3f}s, tf: {:<.3f}s, total:[{:<.3f}s]".format(
            prepare_time, session_time, time.time() - batch_start))
        print("  l1: {:<.5f}, l2: {:<.5f}, Loss: {:<.5f}\n".format(avg_l1, avg_l2, avg_l))

    def train(self):
        opts = self._options
        self._session.run([self._norm_all_user_emb, self._norm_item_emb])
        for i in range(self.last_save_epoch + 1, opts.epoch_max):
            print("  Inter {}".format(i), end=' ')
            self.one_iter()
            self.last_save_epoch += 1
            if (i + 1) % 50 == 0:
                self.show_params()
                # self._session.run(self._norm_param_op)
                best_auc, best_iter = self.best_valid_auc.eval(), self.best_iter.eval()
                print("  Best valid AUC: {:<.6f}({})".format(best_auc, best_iter))
                valid_auc, valid_hr, test_auc, test_hr = self.eval(sample=True, epoch=i)
                # record best result
                if valid_auc >= best_auc:
                    self._session.run([tf.assign(self.best_valid_auc, valid_auc), tf.assign(self.best_iter, i)])
                    self._session.run(self._save_best_param_op)
                elif i - best_iter > 400:
                    # TODO: test overfit stop
                    print('  Overfitted. Exiting... ', end='\n\n')
                    break
            if (i + 1) % 50 == 0:
                self.save_model()
        self._session.run(self._restore_best_param_op)
        self._session.run(self._norm_param_op)
        if opts.write_summary:
            self._train_writer.close()
            self._test_writer.close()

    def show_params(self, u=0, prev_i=0, pos_i=0, neg_i=0):
        corp = self._corpus
        print("  @all_user_emb\n  ", end='')
        print(self._all_user_emb.eval())
        # print("  @user_emb[{}]\n  ".format(u), end='')
        # print(self._user_emb[u].eval())
        # print("  @prev_item({})\n  ".format(prev_i), end='')
        # print(self._item_emb[prev_i].eval())
        # print("  @pos_item({})\n  ".format(pos_i), end='')
        # print(self._item_emb[pos_i].eval())
        # print("  @neg_item({})\n  ".format(neg_i), end='')
        # print(self._item_emb[neg_i].eval())
        print()

    def save_model(self):
        opts = self._options
        path = self.saver.save(self._session, "{}{}_{}.ckpt".format(
            opts.checkpoints_path, opts.timestamp, self.last_save_epoch))
        print("  Saved to {}".format(path), end="\n\n")

    def load_model(self, file_path):
        file_name = file_path[file_path.rfind('/') + 1:file_path.rfind('.')]
        self._options.timestamp = file_name.split('_')[0]
        self.last_save_epoch = int(file_name.split('_')[1])
        self.saver.restore(self._session, file_path)
        print("  Load model from {}".format(file_path), end="\n\n")

    def to_string(self):
        opts = self._options
        return "TransRec__K_{}_lambda_{}_relationReg_{}_biasReg_{}".format(
            opts.emb_dim, opts.reg, opts.user_reg, opts.bias_reg)
