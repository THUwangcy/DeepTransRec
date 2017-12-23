# -*- coding: UTF-8 -*-

import tensorflow as tf

from src.model.model import Model


class PopRec(Model):
    def __init__(self, options, corpus, session):
        Model.__init__(self, options, corpus, session)
        # Key node
        self._pop_per_item = list()

        self.train()
        self.build_eval_graph()

    def train(self):
        corp = self._corpus
        self._pop_per_item = [0.0 for _ in range(corp.n_items)]
        for u in range(corp.n_users):
            for record in corp.pos_per_user[u]:
                self._pop_per_item[record['item']] += 1

    def predict(self, cur_user, prev_item, target_item, norm):
        pop_val = tf.gather(tf.convert_to_tensor(self._pop_per_item), target_item)
        return pop_val

    def save_best_parameters(self):
        pass

    def save_model(self):
        pass

    def to_string(self):
        opts = self._options
        return "PopRec"
