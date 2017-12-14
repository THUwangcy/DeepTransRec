# -*- coding: UTF-8 -*-

import tensorflow as tf


train_return = {
    'FINE': 1,
    'OVERFIT': -1
}


def norm_to_unit_ball(tensor, emb_dim):
    """Norm a 1-D or 2-D tensor to a unit ball, shape [emb_dim] or [None, emb_dim]."""
    with tf.name_scope("norm"):
        one_dim = tf.equal(tf.rank(tensor), 1)
        tensor = tf.cond(one_dim, lambda: tf.expand_dims(tensor, 0), lambda: tf.identity(tensor))
        norm = tf.sqrt(tf.reduce_sum(tf.square(tensor), 1, keep_dims=True))
        norm = tf.maximum(norm, tf.ones_like(norm))
        res = tf.div(tensor, tf.tile(norm, [1, emb_dim]))
        res = tf.cond(one_dim, lambda: tf.squeeze(res), lambda: tf.identity(res))
        return res
