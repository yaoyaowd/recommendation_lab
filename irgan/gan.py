# https://github.com/geek-ai/irgan
# https://zhuanlan.zhihu.com/p/29860542

import linecache
import tensorflow as tf
from model import *
from util import *
import cPickle
import numpy as np


# Get batch data from training set
def get_batch_data(file, index, size):
    user = []
    item = []
    label = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip().split()
        user.append(int(line[0]))
        user.append(int(line[0]))
        item.append(int(line[1]))
        item.append(int(line[2]))
        label.append(1.)
        label.append(0.)
    return user, item, label


def generate_for_d(sess, model, filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]

        rating = sess.run(model.all_rating, {model.u: [u]})
        rating = np.array(rating[0]) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob)
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))
    return len(data)


def main():
    param = cPickle.load(open(WORK_DIR + "model_dns_ori.pkl"))
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM,
                        lamda=0.1 / BATCH_SIZE, param=None, initdelta=0.05, learning_rate=0.001)
    generator = IRGAN_GEN(USER_NUM, ITEM_NUM, EMB_DIM, param=param, learning_rate=0.001)
    #discriminator = IRGAN_DIS(USER_NUM, ITEM_NUM, EMB_DIM, learning_rate=0.001)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "gen ", simple_test(sess, generator)
        print "dis ", simple_test(sess, discriminator)

        best = 0.
        for epoch in range(15):
            if epoch >= 0:
                for d_epoch in range(50):
                    if d_epoch % 5 == 0:
                        train_size = generate_for_d(sess, generator, DIS_TRAIN_FILE)
                    index = 1
                    sum_loss = 0
                    while True:
                        if index > train_size:
                            break
                        if index + BATCH_SIZE <= train_size + 1:
                            input_user, input_item, input_label = get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                        else:
                            input_user, input_item, input_label = get_batch_data(DIS_TRAIN_FILE, index, train_size - index + 1)
                        index += BATCH_SIZE

                        loss, _ = sess.run([discriminator.pre_loss, discriminator.d_updates],
                                           feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                                      discriminator.label: input_label})
                        sum_loss += np.sum(loss)
                    sum_loss /= train_size

                    if d_epoch % 10 == 9:
                        result = simple_test(sess, discriminator)
                        print "epoch ", epoch, "sum loss: ", sum_loss, "iter: ", d_epoch, "dis: ", result

                # Train G
                for g_epoch in range(50):  # 50
                    sum_loss = 0
                    total_train = 0
                    for u in user_pos_train:
                        sample_lambda = 0.2
                        pos = user_pos_train[u]

                        rating = sess.run(generator.all_logits, {generator.u: u})
                        exp_rating = np.exp(rating)
                        prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                        pn = (1 - sample_lambda) * prob
                        pn[pos] += sample_lambda * 1.0 / len(pos)
                        # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

                        sample = sorted(np.random.choice(np.arange(ITEM_NUM), 2 * len(pos), p=pn))
                        ###########################################################################
                        # Get reward and adapt it with importance sampling
                        ###########################################################################
                        reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample})
                        if u == 0:
                            print 'reward', reward
                            print 'weight', prob[sample] / pn[sample]
                        reward = reward * prob[sample] / pn[sample]
                        ###########################################################################
                        # Update G
                        ###########################################################################
                        u_e, i_e, i_prob, gan_loss, _ = sess.run(
                            [generator.u_embedding, generator.i_embedding,
                             generator.i_prob, generator.gan_loss,
                             generator.gan_updates],
                            {generator.u: u, generator.i: sample, generator.reward: reward})
                        sum_loss += gan_loss
                        total_train += len(sample)
                    sum_loss /= total_train

                    result = simple_test_train(sess, generator)
                    print "train:", g_epoch, "loss:", sum_loss, "gen:", result
                    result = simple_test(sess, generator)
                    print "epoch ", epoch, "iter: ", g_epoch, \
                        "loss: ", sum_loss, "total: ", total_train, \
                        "gen: ", result


if __name__ == '__main__':
    main()
