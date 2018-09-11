import tensorflow as tf

with tf.Session() as sess:
    # a = tf.constant([1, 0.5, 0.3, 0.3, 0.5, 1], dtype=tf.float32)
    # w = tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float32)
    # m = tf.constant([[1, 0.5, 0.3, 0.3, 0.5, 1],[1, 0.5, 0.3, 0.3, 0.5, 1]], dtype=tf.float32)
    # print 'tensor a:', sess.run(a)
    #
    # c = tf.reduce_sum(tf.multiply(a, w))
    # print 'a * w:', sess.run(c)
    #
    # m2 = tf.square(tf.reduce_sum(tf.multiply(m, a)))
    # print sess.run(m2)
    #
    # m3 = tf.reduce_sum(tf.square(tf.multiply(m, a)))
    # print sess.run(m3)
    #
    # aa = tf.stack([a for i in range(2)])
    # m4 = tf.concat([aa, m], 1)
    # print sess.run(tf.reduce_sum(m4, 1))

    # uuii = tf.constant([[1, 0.5, 0.3, 0.3, 0.5, 1], [1, 0.5, 0.3, 0.3, 0.5, 1]], dtype=tf.float32)
    # w = tf.constant([[1], [0.5], [0.3], [0.3], [0.5], [1]])
    # fm = tf.constant([[1, 0.5, 0.3, 0.3, 0.5, 1], [1, 0.5, 0.3, 0.3, 0.5, 1]], dtype=tf.float32)
    #
    # sw = tf.matmul(uuii, w)
    # print sess.run(sw)
    #
    # sq = tf.square(tf.reduce_sum(tf.matmul(uuii, fm, transpose_b=True), 1))
    # print sess.run(sw + tf.transpose(tf.expand_dims(sq, 0)))
    #
    # sqq = tf.reduce_sum(tf.square(tf.matmul(uuii, fm, transpose_b=True)),1)
    # print sess.run(sqq)
    # self.positive_fm = tf.reduce_sum(tf.multiply(self.x_positive, self.w), 1) \
    #                    + 0.5 * (tf.square(tf.reduce_sum(tf.matmul(self.x_positive, self.fm), 1))
    #                             - tf.reduce_sum(tf.square(tf.matmul(self.x_positive, self.fm)), 1))
    # print b.shape


    embed_w = tf.constant([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.9,0.1]], dtype=tf.float32)
    normal_embed_w = tf.nn.l2_normalize(embed_w, dim=1)

    print sess.run(normal_embed_w)

    sparse_ids = tf.SparseTensor(values=[0,1,2,3],
                                 indices=[[0,0],[0,1],[1,0],[1,1]],
                                 dense_shape=[2,4])
    lookup_embed = tf.nn.embedding_lookup_sparse(normal_embed_w, sparse_ids, None, combiner='mean')
    print sess.run(lookup_embed)
