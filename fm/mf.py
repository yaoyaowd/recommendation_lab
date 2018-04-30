# train:  73 dis:  [0.5996666666666659, 0.57700000000000129, 0.53720000000000023, 0.60979341069852355, 0.5913328258787317, 0.55870955965814328]
# test  73 dis:  [0.25766666666666715, 0.25459999999999888, 0.25459999999999933, 0.25749154038363592, 0.25538460876063535, 0.25507453728210944]
# train:  74 dis:  [0.60066666666666679, 0.57380000000000031, 0.53589999999999971, 0.61135218203403563, 0.59018944395580442, 0.55834039933147039]
# test  74 dis:  [0.2573333333333338, 0.25039999999999907, 0.25049999999999933, 0.25573196815056892, 0.25129559506076832, 0.25146942276128936]
# train:  75 dis:  [0.59966666666666657, 0.57780000000000042, 0.53649999999999998, 0.60765360636988586, 0.59041519564652634, 0.55724244753071928]
# test  75 dis:  [0.25466666666666732, 0.25599999999999912, 0.24809999999999913, 0.25463134471227317, 0.25540188731384489, 0.25027257252218899]

import tensorflow as tf
import random
from tqdm import trange, tqdm
from irgan.util import *
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

        self.positive_mf = tf.sigmoid(tf.reduce_sum(tf.multiply(self.u_embed, self.p_embed),1))
        self.negative_mf = tf.sigmoid(tf.reduce_sum(tf.multiply(self.u_embed, self.n_embed),1))
        self.loss = self.negative_mf - self.positive_mf

        self.d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = self.d_opt.minimize(self.loss)

        self.all_rating = tf.matmul(self.u_embed, self.item_embeddings, transpose_a=False, transpose_b=True)
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embed, self.item_embeddings), 1)


def generate_dns(sess, model):
    data = []
    for u in tqdm(user_pos_train, desc="generate_dns"):
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

        for epoch in trange(100, desc="Epoch"):
            data = generate_dns2(sess, prfm)
            loss = 0
            for v in tqdm(data, desc="Training"):
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