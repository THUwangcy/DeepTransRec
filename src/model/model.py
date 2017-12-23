# -*- coding: UTF-8 -*-

import sys
import time

import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, options, corpus, session):
        self._options = options
        self._corpus = corpus
        self._session = session

        # -------------------------------
        #          Numpy Record
        # -------------------------------
        # Dataset
        self._valid_prev_item = list()   # item before valid item
        self._valid_pos_item = list()    # valid item (also item before test item)
        self._test_pos_item = list()     # test item
        self._clicked_per_user = list()  # distinct items clicked per user
        self._len_per_user = list()      # sequence length per user
        self._train_legal_user = None  # users can be sampled to train step

        # -------------------------------
        #        Tensorflow Node
        # -------------------------------
        # Input Node
        self._eval_user = None
        self._eval_prev_item = None
        self._eval_valid_item = None
        self._eval_test_item = None
        self._eval_item = None
        self._eval_mask_indices = None

        self._total_valid_auc = None
        self._total_test_auc = None
        self._total_max = None
        self._total_valid_hr = None
        self._total_test_hr = None
        self._total_users = None
        # Key Node
        self._valid_auc = None
        self._test_auc = None
        self._valid_hr = None
        self._test_hr = None
        self._max_cnt = None

        self._avg_valid_auc = None
        self._avg_test_auc = None
        self._avg_valid_hr = None
        self._avg_test_hr = None

        self.best_valid_auc = None
        self.best_iter = None
        # Summary
        self._eval_summary = None
        self._test_writer = None

        self._gen_valid_test()

    def _gen_valid_test(self):
        corp = self._corpus
        for u in range(corp.n_users):
            if len(corp.pos_per_user[u]) >= 3:
                self._test_pos_item.append(corp.pos_per_user[u][-1]['item'])
                del corp.pos_per_user[u][-1]
                self._valid_pos_item.append(corp.pos_per_user[u][-1]['item'])
                self._valid_prev_item.append(corp.pos_per_user[u][-2]['item'])
                del corp.pos_per_user[u][-1]
            else:
                self._test_pos_item.append(-1)
                self._valid_pos_item.append(-1)
                self._valid_prev_item.append(-1)
            self._clicked_per_user.append(set([x['item'] for x in corp.pos_per_user[u]]))
            self._len_per_user.append(len(corp.pos_per_user[u]))
        train_legal_indice = np.array([len(x) for x in self._clicked_per_user]) >= 2
        self._train_legal_user = np.arange(corp.n_users)[train_legal_indice]
        self._len_per_user = np.array(self._len_per_user)

    @abstractmethod
    def predict(self, cur_user, prev_item, target_item):
        """
        predict the value of @target_item given @cur_user and @prev_item
        :param cur_user: shape [batch_size], every row is a user id
        :param prev_item: shape [batch_size], every row is the previous item bought by each user
        :param target_item: shape [batch_size, item_num], every row is the items to be predicted for each user
        :return: shape [batch_size, item_num], values for each target_item
        """
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def to_string(self):
        return "BaseModel"

    def build_eval_graph(self):
        opts = self._options
        # Declace input placeholder
        # shape [batch_size]
        self._eval_user = tf.placeholder(dtype=tf.int32, name="eval_user")
        self._eval_prev_item = tf.placeholder(dtype=tf.int32, name="eval_prev_item")
        self._eval_valid_item = tf.placeholder(dtype=tf.int32, name="eval_valid_item")
        self._eval_test_item = tf.placeholder(dtype=tf.int32, name="eval_test_item")
        # shape [item_num], item_num can change, predict these item's value for every user
        self._eval_item = tf.placeholder(dtype=tf.int32, name="eval_item")
        # list of invalid value indices in matrix [eval_user * eval_item]
        # invalid means the item is in the user's history purchase sequence
        # e.g. [[1, 2]] means eval_item[2] is in eval_user[1]'s purchase sequence
        self._eval_mask_indices = tf.placeholder(dtype=tf.int32, name="eval_mask_indices")

        # Cauculate
        item_num = tf.size(self._eval_item)
        batch_size = tf.size(self._eval_user)
        invalid_num = tf.shape(self._eval_mask_indices)[0]

        # reshape eval_item to [batch_size, item_num], every user has the same items to predict
        # (differents are controlled by eval_mask_indices)
        eval_item = tf.tile(tf.expand_dims(self._eval_item, 0), [batch_size, 1])

        # predict these negtive candidate items' values
        valid_neg_pred = self.predict(self._eval_user, self._eval_prev_item, eval_item)
        test_neg_pred = self.predict(self._eval_user, self._eval_valid_item, eval_item)

        # reshape valid_pred and test_pred to [batch_size, 1]
        eval_valid_item = tf.reshape(self._eval_valid_item, [-1, 1])
        eval_test_item = tf.reshape(self._eval_test_item, [-1, 1])

        # predict valid(test) items' values, result shape [batch_size, 1]
        valid_pred = self.predict(self._eval_user, self._eval_prev_item, eval_valid_item)
        test_pred = self.predict(self._eval_user, self._eval_valid_item, eval_test_item)

        # repeat every row item_num times to match valid(test)_neg_pred, shape [batch_size, item_num]
        valid_pred = tf.tile(valid_pred, [1, item_num])
        test_pred = tf.tile(test_pred, [1, item_num])

        # construct mask matrix, invalid indices are filled with 0, otherwise 1
        mask = tf.sparse_to_dense(sparse_indices=self._eval_mask_indices,
                                  sparse_values=tf.zeros([invalid_num]),
                                  output_shape=[batch_size, item_num],
                                  default_value=1,
                                  validate_indices=False)

        # calculate AUC (just sum up, average in the last)
        valid_res = tf.multiply(tf.cast(valid_pred > valid_neg_pred, tf.float32), mask)
        test_res = tf.multiply(tf.cast(test_pred > test_neg_pred, tf.float32), mask)
        valid_res_per_user = tf.reduce_sum(valid_res, 1)
        test_res_per_user = tf.reduce_sum(test_res, 1)
        self._valid_auc = tf.reduce_sum(valid_res_per_user)
        self._test_auc = tf.reduce_sum(test_res_per_user)

        # calculate HR (just sum up, average in the last)
        max_per_user = tf.reduce_sum(mask, 1)
        valid_rank = max_per_user - valid_res_per_user + 1
        test_rank = max_per_user - test_res_per_user + 1
        self._valid_hr = tf.reduce_sum(tf.cast(valid_rank <= opts.top_n, tf.int32))
        self._test_hr = tf.reduce_sum(tf.cast(test_rank <= opts.top_n, tf.int32))

        # for final average
        self._max_cnt = tf.cast(batch_size * item_num - invalid_num, tf.float32)

        # Average for all users
        self._build_avg_metric_graph()

    def _build_avg_metric_graph(self):
        # for record summary of AUC, build a small graph to calculate average AUC
        self._total_valid_auc = tf.placeholder(dtype=tf.int32, name="total_valid_auc")
        self._total_test_auc = tf.placeholder(dtype=tf.int32, name="total_test_auc")
        self._total_max = tf.placeholder(dtype=tf.int32, name="total_max")
        self._total_valid_hr = tf.placeholder(dtype=tf.int32, name="total_valid_hr")
        self._total_test_hr = tf.placeholder(dtype=tf.int32, name="total_test_hr")
        self._total_users = tf.placeholder(dtype=tf.int32, name="total_users")

        self.best_valid_auc = tf.Variable(-1.0, dtype=tf.float32, trainable=False, name="best_valid_auc")
        self.best_iter = tf.Variable(0, trainable=False, name="best_iter")

        self._avg_valid_auc = self._total_valid_auc / self._total_max
        self._avg_test_auc = self._total_test_auc / self._total_max
        self._avg_valid_hr = self._total_valid_hr / self._total_users * 100
        self._avg_test_hr = self._total_test_hr / self._total_users * 100

        summary_valid = tf.summary.scalar("Valid AUC", self._avg_valid_auc)
        summary_test = tf.summary.scalar("Test AUC", self._avg_test_auc)
        self._eval_summary = tf.summary.merge([summary_valid, summary_test])

    def get_eval_batch(self, batch_no, sample):
        opts = self._options
        corp = self._corpus
        users, prev_items, valid_items, test_items, mask_indices = list(), list(), list(), list(), list()

        batch_size = opts.eval_batch_size
        eval_users = np.arange(corp.n_users)
        sample_item = min(corp.n_items, opts.eval_sample_item)
        # if sample, sample opts.eval_sample_item negtive items
        if sample:
            eval_item_pool = np.random.choice(np.arange(corp.n_items), sample_item, replace=False)
            # record item index in current eval_item_pool, only useful when sampling items
            item2ind = dict()
            for ind, item in enumerate(eval_item_pool):
                item2ind[item] = ind
        else:
            eval_item_pool = np.arange(corp.n_items)

        user_indice = -1
        for u in eval_users[batch_no * batch_size:(batch_no + 1) * batch_size]:
            if u % 1000 == 0:
                print('.', end='')
                sys.stdout.flush()
            # context
            pos_item = self._valid_pos_item[u]
            prev = self._valid_prev_item[u]
            valid_pos = self._valid_pos_item[u]
            test_pos = self._test_pos_item[u]
            # user who has too short history purchase sequence
            if pos_item == -1:
                continue
            # if users num is large, sample around opts.eval_sample_user users
            sample_user = opts.eval_sample_user
            if sample and len(eval_users) > sample_user and np.random.randint(0, len(eval_users)) > sample_user:
                continue
            # legal user
            user_indice += 1
            users.append(u)
            prev_items.append(prev)
            valid_items.append(valid_pos)
            test_items.append(test_pos)

            # construct mask_indices, every item in a user's click history/valid_pos/test_pos should be masked
            valid_pos_in, test_pos_in = False, False

            def get_item_indice(x):
                return item2ind[x] if sample else x

            for item in self._clicked_per_user[u]:
                if item in eval_item_pool:
                    item_indice = get_item_indice(item)
                    mask_indices.append([user_indice, item_indice])
                    if not valid_pos_in and item == self._valid_pos_item[u]:
                        valid_pos_in = True
                    if not test_pos_in and item == self._test_pos_item[u]:
                        test_pos_in = True
            if not valid_pos_in and self._valid_pos_item[u] in eval_item_pool:
                item_indice = get_item_indice(self._valid_pos_item[u])
                mask_indices.append([user_indice, item_indice])
            if not test_pos_in and self._test_pos_item[u] in eval_item_pool:
                item_indice = get_item_indice(self._test_pos_item[u])
                mask_indices.append([user_indice, item_indice])

        return user_indice, users, prev_items, valid_items, test_items, mask_indices, eval_item_pool

    def eval(self, sample=True, epoch=0):
        corp = self._corpus
        opts = self._options

        # final metric value
        valid_auc, test_auc = 0.0, 0.0
        valid_hr, test_hr = 0.0, 0.0

        # eval batch config
        batch_size = opts.eval_batch_size
        eval_users = np.arange(corp.n_users)
        batch_num = len(eval_users) // batch_size + (1 if len(eval_users) % batch_size > 0 else 0)

        # other variable
        total_max = 0.0
        n_users = 0.0

        # timing
        eval_start = time.time()
        prepare_time, session_time = 0, 0

        print('  ---------------------------------------------------------------------------------')
        print("  Eval", end=' ')
        for b_no in range(batch_num):
            # Batch begin!
            prepare_start = time.time()
            user_indice, users, prev_items, valid_items, test_items, mask_indices, eval_item_pool = \
                self.get_eval_batch(b_no, sample)
            n_users += (user_indice + 1)
            feed_dict = {
                self._eval_user: users,
                self._eval_prev_item: prev_items,
                self._eval_valid_item: valid_items,
                self._eval_test_item: test_items,
                self._eval_item: eval_item_pool,
                self._eval_mask_indices: mask_indices
            }
            prepare_time += time.time() - prepare_start

            # run session to get AUC and HR
            session_start = time.time()
            [cur_valid_auc, cur_test_auc, max_cnt, cur_valid_hr, cur_test_hr] = self._session.run(
                [self._valid_auc, self._test_auc, self._max_cnt, self._valid_hr, self._test_hr],
                feed_dict=feed_dict)
            session_time += time.time() - session_start

            valid_auc += cur_valid_auc
            test_auc += cur_test_auc
            total_max += max_cnt
            valid_hr += cur_valid_hr
            test_hr += cur_test_hr

        # average and print out
        feed_dict = {
            self._total_valid_auc: valid_auc,
            self._total_test_auc: test_auc,
            self._total_max: total_max,
            self._total_valid_hr: valid_hr,
            self._total_test_hr: test_hr,
            self._total_users: n_users
        }
        [summary, valid_auc, test_auc, valid_hr, test_hr] = self._session.run(
            [self._eval_summary, self._avg_valid_auc, self._avg_test_auc, self._avg_valid_hr, self._avg_test_hr],
            feed_dict=feed_dict)
        if opts.write_summary:
            self._test_writer.add_summary(summary, epoch + 1)

        print("\n  Iter {0:>3}: [Valid: AUC = {1:<.6f}, HR@{2} = {3:<.4f}%], Test: AUC = {4:<.6f}, HR@{2} = {5:<.4f}%  "
              "\n  Time  np: {6:<.3f}s tf: {7:<.3f}s total:[{8:<.3f}s]".format(
                epoch, valid_auc, opts.top_n, valid_hr, test_auc, test_hr,
                prepare_time, session_time, time.time() - eval_start))
        print('  ---------------------------------------------------------------------------------', end="\n\n")
        sys.stdout.flush()
        return valid_auc, valid_hr, test_auc, test_hr

    def cal_in_np(self, u, eval_item_pool):
        """
        Deperated!! Too slow but clear, calculate AUC for a single user
        :param u:
        :param eval_item_pool:
        :return:
        """
        prev = self._valid_prev_item[u]
        valid_pos = self._valid_pos_item[u]
        test_pos = self._test_pos_item[u]
        eval_item = list()
        for i in eval_item_pool:
            if i not in self._clicked_per_user[u] and i != valid_pos and i != test_pos:
                eval_item.append(i)
        eval_item.append(valid_pos)
        valid_pred_res = self.predict(u, prev, eval_item).eval()
        eval_item[-1] = test_pos
        test_pred_res = self.predict(u, valid_pos, eval_item).eval()
        valid_cnt, test_cnt = 0.0, 0.0
        x_u_val = valid_pred_res[-1]
        x_u_test = test_pred_res[-1]
        for ind, j in enumerate(eval_item[:-1]):
            x_uj = valid_pred_res[ind]
            if x_u_val > x_uj:
                valid_cnt += 1
            x_uj = test_pred_res[ind]
            if x_u_test > x_uj:
                test_cnt += 1
        valid_auc = valid_cnt / (len(eval_item) - 1)
        test_auc = test_cnt / (len(eval_item) - 1)
        print(valid_cnt, test_cnt)
        return valid_auc, test_auc
