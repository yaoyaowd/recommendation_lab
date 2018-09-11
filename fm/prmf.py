# train:  53 dis:  [0.62166666666666637, 0.58760000000000012, 0.55269999999999975, 0.62893321502716015, 0.60350459487159391, 0.57385066493517545]
# test  53 dis:  [0.26833333333333381, 0.25699999999999923, 0.24669999999999925, 0.27345814789721612, 0.26413650273981193, 0.25474495548878373]
# train:  54 dis:  [0.61399999999999944, 0.59020000000000095, 0.54710000000000003, 0.62452537884909198, 0.60514636023459412, 0.57019252450323066]
# test  54 dis:  [0.26833333333333392, 0.25739999999999941, 0.25069999999999926, 0.26991050739051048, 0.26204302464307444, 0.25591393965618625]
# train:  59 dis:  [0.62166666666666637, 0.59620000000000017, 0.55679999999999996, 0.6293299203764221, 0.6094628492467058, 0.5773215041497376]
# test  59 dis:  [0.27866666666666756, 0.25999999999999895, 0.2482999999999991, 0.27998886917119337, 0.26677201057518596, 0.25631087709205064]
# train:  64 dis:  [0.61800000000000055, 0.59560000000000057, 0.55539999999999989, 0.6314416746435243, 0.61226165987549097, 0.57840719141574215]
# test  64 dis:  [0.28133333333333371, 0.26219999999999916, 0.25219999999999915, 0.28150845961636384, 0.2682638742019855, 0.25951038362983886]

import tensorflow as tf
import random

from irgan.util import *
from tqdm import trange, tqdm
from multiprocessing.dummy import Pool as ThreadPool

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

        self.positive_mf = tf.reduce_sum(tf.multiply(self.u_embed, self.p_embed),1)
        self.negative_mf = tf.reduce_sum(tf.multiply(self.u_embed, self.n_embed),1)
        self.logits = tf.sigmoid(self.positive_mf - self.negative_mf)
        self.loss = -self.logits

        self.d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = self.d_opt.minimize(self.loss)

        self.all_rating = tf.matmul(self.u_embed, self.item_embeddings, transpose_a=False, transpose_b=True)
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embed, self.item_embeddings), 1)


def generate_dns(sess, model):
    data = []
    for u in user_pos_train: # (user_pos_train, desc="generate_dns"):
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

def generate_dns2(sess, model):

    train_samples = []

    def gen_train_samples(uidx_ratings_tup):
        uidx, ratings = uidx_ratings_tup
        all_neg_candidates = list(ALL_ITEMS - set(user_pos_train[uidx]))

        for purchased_idx in user_pos_train[uidx]:
            neg_cands = random.sample(all_neg_candidates, 5)
            train_samples.append((uidx, purchased_idx, neg_cands[np.argmax(
                ratings[neg_cands])]))

    pool = ThreadPool(16)
    train_uidxs = user_pos_train.keys()
    batch_size = 512
    for start in trange(
            0, USER_NUM, batch_size, desc="Generating train candidates"):
        u_batch = train_uidxs[start:start + batch_size]
        ratings = sess.run(model.all_rating, {model.u: u_batch})
        pool.map(gen_train_samples, zip(u_batch, ratings))

    return train_samples


def main():
    prfm = PRFM(ITEM_NUM, USER_NUM, EMB_DIM, initdelta=0.05, learning_rate=0.05)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_p5 = 0.

        for epoch in range(100): # trange(100, desc="Epoch"):
            data = generate_dns2(sess, prfm)
            loss = 0
            for v in data: # tqdm(data, desc="Training"):
                u, i, j = v[0], v[1], v[2]
                mf_loss, _ = sess.run(
                    [prfm.loss, prfm.d_updates],
                    feed_dict={prfm.u: [u], prfm.positive: [i], prfm.negative: [j]})
                loss += mf_loss

            print "epoch {0}, len {1} loss: {2}".format(epoch, len(data), loss)
            for i in trange(5):
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