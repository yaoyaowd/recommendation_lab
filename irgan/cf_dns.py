import tensorflow as tf
import numpy as np
from util import *


def generate_dns(sess, model, filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]
        all_rating = sess.run(model.dns_rating, {model.u: u})
        all_rating = np.array(all_rating)
        neg = []
        candidates = list(ALL_ITEMS - set(pos))

        for i in range(len(pos)):
            # choice = np.random.choice(candidates, 5)
            # choice_score = all_rating[choice]
            # neg.append(choice[np.argmax(choice_score)])
            # data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))
            choice = np.random.choice(candidates, 3)
            for c in choice:
                data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(c))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):
            self.user_embeddings_raw = tf.Variable(tf.random_uniform(
                [self.user_num, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta, dtype=tf.float32))
            self.item_embeddings_raw = tf.Variable(tf.random_uniform(
                [self.item_num, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta, dtype=tf.float32))
            self.user_embeddings = tf.nn.l2_normalize(self.user_embeddings_raw, dim=1)
            self.item_embeddings = tf.nn.l2_normalize(self.item_embeddings_raw, dim=1)
            self.item_bias = tf.Variable(tf.zeros([self.itemNum]))

        self.d_params = [self.user_embeddings_raw, self.item_embeddings_raw, self.item_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.pos = tf.placeholder(tf.int32)
        self.neg = tf.placeholder(tf.int32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.pos_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos)
        self.pos_bias = tf.gather(self.item_bias, self.pos)
        self.neg_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg)
        self.neg_bias = tf.gather(self.item_bias, self.neg)

        self.pre_logits = tf.sigmoid(
            tf.reduce_sum(tf.multiply(self.u_embedding, self.pos_embedding - self.neg_embedding),
                          1) + self.pos_bias - self.neg_bias)
        self.pre_loss = -tf.reduce_mean(tf.log(self.pre_logits)) + self.lamda * (
            tf.nn.l2_loss(self.pos_bias) + tf.nn.l2_loss(self.neg_bias))

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias


def main():
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1, initdelta=0.05, learning_rate=0.05)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "dis ", simple_test(sess, discriminator)
        best_p5 = 0.

        for epoch in range(80):
            generate_dns(sess, discriminator, DIS_TRAIN_FILE)  # dynamic negative sample
            with open(DIS_TRAIN_FILE)as fin:
                for line in fin:
                    line = line.split()
                    u = int(line[0])
                    i = int(line[1])
                    j = int(line[2])
                    _ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.u: [u], discriminator.pos: [i],
                                            discriminator.neg: [j]})

            result = simple_test_train(sess, discriminator)
            print "train: ", epoch, "dis: ", result
            result = simple_test(sess, discriminator)
            print "epoch ", epoch, "dis: ", result
            best_p5 = result[1] if result[1] > best_p5 else best_p5

        print "best P@5: ", best_p5


if __name__ == '__main__':
    main()
