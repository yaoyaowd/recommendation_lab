import tensorflow as tf
import random

from irgan.util import *
from tqdm import trange, tqdm
from multiprocessing.dummy import Pool as ThreadPool


class PRFM():
    def __init__(self, item_num, user_num, emb_dim, k=5, initdelta=0.05, learning_rate=0.01):
        self.item_num = item_num
        self.user_num = user_num
        self.emb_dim = emb_dim
        self.initdelta = initdelta
        self.learning_rate = learning_rate

        with tf.variable_scope('mf'):
            self.user_embeddings_raw = tf.Variable(tf.random_uniform(
                [self.user_num, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta, dtype=tf.float32))
            self.item_embeddings_raw = tf.Variable(tf.random_uniform(
                [self.item_num, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta, dtype=tf.float32))
            self.user_embeddings = tf.nn.l2_normalize(self.user_embeddings_raw, dim=1)
            self.item_embeddings = tf.nn.l2_normalize(self.item_embeddings_raw, dim=1)
        self.mf_params = [self.user_embeddings_raw, self.item_embeddings_raw]

        self.u = tf.placeholder(dtype=tf.int32)
        self.positive = tf.placeholder(dtype=tf.int32)
        self.negative = tf.placeholder(dtype=tf.int32)

        self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.p_embed = tf.nn.embedding_lookup(self.item_embeddings, self.positive)
        self.n_embed = tf.nn.embedding_lookup(self.item_embeddings, self.negative)

        self.positive_mf = tf.reduce_sum(tf.multiply(self.u_embed, self.p_embed), 1)
        self.negative_mf = tf.reduce_sum(tf.multiply(self.u_embed, self.n_embed), 1)
        self.mf_loss = -tf.sigmoid(self.positive_mf - self.negative_mf)

        self.mf_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.mf_updates = self.mf_opt.minimize(self.mf_loss, var_list=self.mf_params)

        self.all_rating = tf.matmul(self.u_embed, self.item_embeddings, transpose_a=False, transpose_b=True)
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embed, self.item_embeddings), 1)

        self.x_positive = tf.concat([self.u_embed, self.p_embed], 1)
        self.x_negative = tf.concat([self.u_embed, self.n_embed], 1)

        with tf.variable_scope("fm"):
            self.w = tf.get_variable("w", shape=(self.emb_dim*2, 1), dtype=tf.float32)
            self.fm = tf.get_variable("fm", shape=(k, self.emb_dim*2), dtype=tf.float32)
        # self.fm_params = [self.user_embeddings_raw, self.item_embeddings_raw, self.w, self.fm]
        self.fm_params = [self.w, self.fm]

        self.positive_fm = tf.reduce_sum(tf.multiply(self.x_positive, self.w)) \
                           + 0.5 * (tf.square(tf.reduce_sum(tf.multiply(self.fm, self.x_positive)))
                                    - tf.reduce_sum(tf.square(tf.multiply(self.fm, self.x_positive))))
        self.negative_fm = tf.reduce_sum(tf.multiply(self.x_negative, self.w)) \
                           + 0.5 * (tf.square(tf.reduce_sum(tf.multiply(self.fm, self.x_negative)))
                                    - tf.reduce_sum(tf.square(tf.multiply(self.fm, self.x_negative))))
        self.fm_loss = -tf.sigmoid(self.positive_fm - self.negative_fm) \
                       + 0.02 * tf.reduce_sum(tf.nn.l2_loss(self.w) + tf.nn.l2_loss(self.fm))

        self.fm_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.fm_updates = self.fm_opt.minimize(self.fm_loss, var_list=self.fm_params)

        self.uu = tf.stack([self.u_embed for _ in range(self.item_num)])
        self.uuii = tf.concat([self.uu, self.item_embeddings], 1)
        self.fm_rating = tf.matmul(self.uuii, self.w) + \
                         0.5 * tf.transpose(tf.expand_dims((
                             tf.square(tf.reduce_sum(tf.matmul(self.uuii, self.fm, transpose_b=True),1))
                             - tf.reduce_sum(tf.square(tf.matmul(self.uuii, self.fm, transpose_b=True)),1)), 0))


def simple_test_fm(sess, model):
    result = np.array([0.] * 6)
    test_users = user_pos_test.keys()
    test_user_num = len(test_users)
    for index in range(0, test_user_num):
        user_batch_rating = sess.run(model.fm_rating, {model.u: test_users[index]})
        re = simple_test_one_user((user_batch_rating, test_users[index]))
        result += re
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_dns(sess, model):
    data = []
    for u in user_pos_train:#tqdm(user_pos_train, desc="generate_dns"):
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

        for epoch in trange(100, desc="Epoch"):
            data = generate_dns2(sess, prfm)

            # loss = 0
            # for v in data:
            #     u, i, j = v[0], v[1], v[2]
            #     fm_loss, _ = sess.run(
            #         [prfm.fm_loss, prfm.fm_updates],
            #         feed_dict={prfm.u: [u], prfm.positive: [i], prfm.negative: [j]})
            #     loss += fm_loss
            #
            # print 'fm epoch {0}, loss {1}'.format(epoch, loss)
            # result = simple_test_fm(sess, prfm)
            # print "fm: ", result

            loss = 0
            for v in data:
                u, i, j = v[0], v[1], v[2]
                mf_loss, _ = sess.run(
                    [prfm.mf_loss, prfm.mf_updates],
                    feed_dict={prfm.u: [u], prfm.positive: [i], prfm.negative: [j]})
                loss += mf_loss

            print 'epoch {0}, loss {1}'.format(epoch, loss)
            result = simple_test_train(sess, prfm)
            print "train: ", epoch, "dis: ", result
            result = simple_test(sess, prfm)
            print "test ", epoch, "dis: ", result

            if epoch % 10 == 9:
                for fm_epoch in range(5):
                    loss = 0
                    for v in data:
                        u, i, j = v[0], v[1], v[2]
                        fm_loss, _ = sess.run(
                            [prfm.fm_loss, prfm.fm_updates],
                            feed_dict={prfm.u: [u], prfm.positive: [i], prfm.negative: [j]})
                        loss += fm_loss

                    print 'fm epoch {0}, loss {1}'.format(epoch, loss)
                    result = simple_test_fm(sess, prfm)
                    print "fm test", fm_epoch, "fm: ", result


if __name__ == '__main__':
    main()