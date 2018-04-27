import tensorflow as tf
import random

from irgan.util import *

class PRFM():
    def __init__(self, item_num, user_num, emb_dim, alpha=0.5, initdelta=0.05, learning_rate=0.01):
        self.item_num = item_num
        self.user_num = user_num
        self.emb_dim = emb_dim
        self.initdelta = initdelta
        self.learning_rate = learning_rate

        with tf.variable_scope('discriminator'):
            self.user_embeddings_raw = tf.Variable(tf.random_uniform(
                [self.user_num, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta, dtype=tf.float32))
            self.item_embeddings_raw = tf.Variable(tf.random_uniform(
                [self.item_num, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta, dtype=tf.float32))
            self.user_embeddings = tf.nn.l2_normalize(self.user_embeddings_raw, dim=1)
            self.item_embeddings = tf.nn.l2_normalize(self.item_embeddings_raw, dim=1)

        self.u = tf.placeholder(dtype=tf.int32)
        self.positive = tf.placeholder(dtype=tf.int32)
        self.negative = tf.placeholder(dtype=tf.int32)

        self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.p_embed = tf.nn.embedding_lookup(self.item_embeddings, self.positive)
        self.n_embed = tf.nn.embedding_lookup(self.item_embeddings, self.negative)

        self.positive_mf = tf.sigmoid(tf.reduce_sum(tf.multiply(self.u_embed, self.p_embed),1))
        self.negative_mf = tf.sigmoid(tf.reduce_sum(tf.multiply(self.u_embed, self.n_embed),1))
        self.mf_loss = self.negative_mf - self.positive_mf

        # self.x_positive = tf.concat([self.u_embed, self.p_embed], 1)
        # self.x_negative = tf.concat([self.u_embed, self.n_embed], 1)
        # with tf.variable_scope("fm"):
        #     self.bias = tf.get_variable("w0", shape=(1), dtype=tf.float32)
        #     self.w = tf.get_variable("w", shape=(self.emb_dim*2, 1), dtype=tf.float32)
        #     self.fm = tf.get_variable("fm", shape=(self.emb_dim, self.emb_dim), dtype=tf.float32)
        #
        # self.positive_fm = self.bias + tf.reduce_sum(tf.multiply(self.x_positive, self.w), 1)\
        #                    + tf.reduce_sum(tf.multiply(tf.matmul(self.x_positive, tf.transpose(self.x_positive)), self.fm))
        # self.negative_fm = self.bias + tf.reduce_sum(tf.multiply(self.x_negative, self.w), 1)\
        #                    + tf.reduce_sum(tf.multiply(tf.matmul(self.x_negative, tf.transpose(self.x_negative)), self.fm))
        # self.fm_loss = self.negative_fm - self.positive_fm

        self.loss = self.mf_loss
        # + self.fm_loss * (1 - alpha)
        # + tf.nn.l2_loss(self.bias) + tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.fm)

        self.d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = self.d_opt.minimize(self.loss)

        self.all_rating = tf.matmul(self.u_embed, self.item_embeddings, transpose_a=False, transpose_b=True)
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embed, self.item_embeddings), 1)


def generate_dns(sess, model):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]
        all_rating = sess.run(model.dns_rating, {model.u: u})
        all_rating = np.array(all_rating)
        neg = []
        candidates = list(ALL_ITEMS - set(pos))

        for i in range(len(pos)):
            choice = np.random.choice(candidates, 5)
            choice_score = all_rating[choice]
            neg.append(choice[np.argmax(choice_score)])
            data.append((u, pos[i], neg[i]))
    return data


def main():
    prfm = PRFM(ITEM_NUM, USER_NUM, EMB_DIM, initdelta=0.05, learning_rate=0.05)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_p5 = 0.

        for epoch in range(100):
            data = generate_dns(sess, prfm)
            loss = 0
            for v in data:
                u, i, j = v[0], v[1], v[2]
                mf_loss, _ = sess.run(
                    [prfm.mf_loss, prfm.d_updates],
                    feed_dict={prfm.u: [u], prfm.positive: [i], prfm.negative: [j]})
                loss += mf_loss

            print "epoch {0}, len {1} loss: {2}".format(epoch, len(data), loss)
            for i in range(5):
                u = random.randrange(USER_NUM)
                pos = user_pos_train[u]
                p = pos[random.randrange(len(pos))]
                candidates = list(ALL_ITEMS - set(pos))
                n = candidates[random.randrange(len(candidates))]
                ue, pe, ne = sess.run([prfm.u_embed, prfm.p_embed, prfm.n_embed],
                                  feed_dict={prfm.u: [u], prfm.positive: [p], prfm.negative: [n]})
                ue = ue[0]
                pe = pe[0]
                ne = ne[0]
                # print 'random ue', ue
                # print 'random pe', pe
                # print 'random ne', ne
                print 'sum', np.sum(np.multiply(ue, pe)), np.sum(np.multiply(ue, ne))

            result = simple_test_train(sess, prfm)
            print "train: ", epoch, "dis: ", result
            result = simple_test(sess, prfm)
            print "test ", epoch, "dis: ", result
            best_p5 = result[1] if result[1] > best_p5 else best_p5

        print "best P@5: ", best_p5


if __name__ == '__main__':
    main()