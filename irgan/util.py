import numpy as np
import multiprocessing

WORK_DIR = '/Users/dwang/irgan/item_recommendation/ml-100k/'


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size+2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0
    return dcg_at_k(r, k) / dcg_max


def load_data(filename):
    ret = {}
    with open(WORK_DIR + filename) as input:
        for line in input:
            line = line.split()
            uid, iid, r = int(line[0]), int(line[1]), float(line[2])
            if r > 3.99:
                if uid not in ret:
                    ret[uid] = []
                ret[uid].append(iid)
    return ret


USER_NUM = 943
ITEM_NUM = 1683
EMB_DIM = 5
BATCH_SIZE = 16
NUM_CORES = multiprocessing.cpu_count()
DIS_TRAIN_FILE = WORK_DIR + "dis-train.txt"

user_pos_train = load_data('movielens-100k-train.txt')
user_pos_test = load_data('movielens-100k-test.txt')
all_users = user_pos_train.keys()
all_users.sort()
ALL_ITEMS = set(range(ITEM_NUM))


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]
    test_items = list(ALL_ITEMS - set(user_pos_train[u]))
    item_score = [(i, rating[i]) for i in test_items]
    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [x[0] for x in item_score]
    r = [1 if i in user_pos_test[u] else 0 for i in item_sort]
    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


def simple_test(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(processes=4)
    batch_size = 128
    test_users = user_pos_test.keys()
    test_user_num = len(test_users)
    for index in range(0, test_user_num, batch_size):
        user_batch = test_users[index:index + batch_size]
        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re
    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def simple_test_train_one_user(x):
    rating = x[0]
    u = x[1]
    test_items = list(ALL_ITEMS)
    item_score = [(i, rating[i]) for i in test_items]
    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [x[0] for x in item_score]
    r = [1 if i in user_pos_train[u] else 0 for i in item_sort]
    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


def simple_test_train(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(processes=4)
    batch_size = 128
    test_users = user_pos_train.keys()
    test_user_num = len(test_users)
    for index in range(0, test_user_num, batch_size):
        user_batch = test_users[index:index + batch_size]
        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_train_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re
    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret