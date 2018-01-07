import tensorflow as tf
from util import *
import cPickle
import numpy as np


class GEN():
    def __init__(self, user_num, item_num, embedding_size, param=None, learning_rate=0.05):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size

        with tf.variable_scope('generator'):
            if param == None:
                self.user_embed = tf.Variable(tf.random_uniform([self.user_num, self.embedding_size], -0.05, 0.05))
                self.item_embed = tf.Variable(tf.random_uniform([self.item_num, self.embedding_size], -0.05, 0.05))
                self.item_bias = tf.Variable(tf.zeros([self.item_num]))
            else:
                self.user_embed = tf.Variable(param[0])
                self.item_embed = tf.Variable(param[1])
                self.item_bias = tf.Variable(param[2])

        self.g_params = [self.user_embed, self.item_embed, self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embed, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embed, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embed), 1) + self.item_bias
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)
        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + 0.1 * (
            tf.nn.l2_loss(self.u_embedding)
            + tf.nn.l2_loss(self.i_embedding)
            + tf.nn.l2_loss(self.i_bias))
        g_opt = tf.train.GradientDescentOptimizer(learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)

        self.all_rating = tf.matmul(self.u_embedding, self.item_embed, transpose_b=True) + self.item_bias


def main():
    param = cPickle.load(open(WORK_DIR + "model_dns_ori.pkl"))
    generator = GEN(USER_NUM, ITEM_NUM, EMB_DIM, param=param, learning_rate=0.001)
    #generator = GEN(USER_NUM, ITEM_NUM, EMB_DIM, learning_rate=0.001)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "gen ", simple_test(sess, generator)

        for g_epoch in range(30):
            sum_loss = 0
            total_train = 0
            for u in user_pos_train:
                pos = user_pos_train[u]
                all_rating = sess.run(generator.all_rating, {generator.u: [u]})[0]
                exp_rating = np.exp(all_rating)
                prob = exp_rating / np.sum(exp_rating)
                pn = 0.8 * prob
                pn[pos] += 0.2 / len(pos)

                samples = np.random.choice(np.arange(ITEM_NUM), len(pos) * 2, p=pn)
                reward = []
                pos_samples, neg_samples = [], []
                alpha = (len(pos) + 8.0) / 8.0
                for s in samples:
                    if s in pos:
                        pos_samples.append(s)
                    else:
                        neg_samples.append(s)
                for s in samples:
                    if s in pos:
                        p = s
                        neg_pairs = 0
                        for n in neg_samples:
                            if n not in pos and all_rating[n] > all_rating[p]:
                                neg_pairs += 1.0
                        reward.append((alpha + neg_pairs) / (len(neg_samples) + alpha) * prob[p] / pn[p])
                    else:
                        n = s
                        pos_pairs = 1.0
                        for p in pos_samples:
                            if all_rating[p] > all_rating[n]:
                                pos_pairs += 1.0
                        reward.append((-(pos_pairs + 1.0) / (len(pos) + 1.0)) * prob[p] / pn[p])

                gan_loss, _ = sess.run(
                    [generator.gan_loss, generator.gan_updates],
                    {generator.u: u, generator.i: samples, generator.reward: reward})

                #if u == 0:
                #    print reward
                #    print gan_loss

                sum_loss += gan_loss
                total_train += len(pos)
            sum_loss /= total_train

            result = simple_test_train(sess, generator)
            print "train:", g_epoch, "loss:", sum_loss, "gen:", result
            result = simple_test(sess, generator)
            print "test:", g_epoch, "loss:", sum_loss, "gen:", result
            print


if __name__ == '__main__':
    main()
