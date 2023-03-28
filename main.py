from time import localtime, strftime, time

import faiss
import os
import random
import numpy as np
import _pickle as pkl

from params import args
from metrics import ComputeMetrics
from tf_ver_ctrl import tf, keras
from multiprocessing import Process, Queue

np.set_printoptions(threshold=np.inf)

EMB_DIM = 32
hit_and_ndcg = [10, 20, 50, 100, 200, 300]
MATCH_NUMS = max(hit_and_ndcg)
dataset = args.dataset
BATCH_SIZE = args.batch_size
neg_num = BATCH_SIZE
NEG_IID_LIST_LEN = 200

path = f"./datasets/{dataset}/"
train_data_path = path + f"{dataset}-no_id_ordered_by_count-train.tfrecords"
test_data_path = path + f"{dataset}-no_id_ordered_by_count-test.tfrecords"
feature_counts_path = path + "sparse_features_max_idx.pkl"
# popularity table 
pop_dict_path = path + "item_popularity_table.pkl"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
EPOCHS = args.epochs
model_type = 'dccl'
log_path = './logs'
print(log_path)
metrics_file = os.path.join(log_path, f"{dataset}/{model_type}_results.txt")

metrics_q = Queue()
metrics_computer = ComputeMetrics(metrics_q, hit_and_ndcg,  metrics_file)
metrics_computer_p = Process(target=metrics_computer.compute_metrics)
metrics_computer_p.daemon = True
metrics_computer_p.start()

def pop_func(pop_tensor, pop_coeff):
    pop_tensor = tf.multiply(pop_tensor, pop_coeff)
    pop_tensor = tf.where(pop_tensor >= 1.0, tf.ones_like(pop_tensor), pop_tensor)
    pop_curve = tf.exp(-pop_tensor)
    # mask = tf.where(pop_tensor > pop_threshold, tf.ones_like(pop_tensor), pop_curve)
    return pop_curve

def CSELoss(y_pred, label = None, mask = None, neg_num=128):
    if (label is None):
        n = tf.shape(y_pred)[0]
        y_true = tf.eye(n, dtype = tf.float32)
    else:
        y_true = tf.cast(label, dtype = tf.float32)

    N = tf.shape(y_pred)[0]
    y_pred = tf.math.exp(y_pred)
    ratio = 1.0 - (neg_num + 1)*1.0/BATCH_SIZE
    pos_pred = tf.multiply(y_pred, y_true)
    ner = tf.reduce_sum(pos_pred, axis = -1, keepdims = False) + 1e-1
    if (ratio > 1e-5):
        mask = tf.math.greater(tf.random.uniform(shape = [N, N], minval=0,
                                                 maxval = 1, dtype=tf.float32, seed = 102), ratio)
        mask = tf.math.logical_or(mask, label)
        mask = tf.cast(mask, dtype = tf.float32)
        der_pred = tf.multiply(y_pred, mask)
        der = tf.reduce_sum(der_pred, axis = -1, keepdims = False) + 1e-1
    else:
        der = tf.reduce_sum(y_pred, axis = -1, keepdims = False) + 1e-1
    loss = -tf.math.log(tf.math.divide_no_nan(ner, der))
    if mask is None:
        pass
    else:
        loss = loss * mask
    loss = tf.reduce_sum(loss)
    return loss

def eval_on_test(sess, index):
    index.reset()
    all_items = sess.run([all_item_emb])[0]
    index.add(all_items)
    while(True):
        try:
            test_user_query, tuids, target_items = sess.run([user_emb, test_uids, test_iids])
            S, I = index.search(test_user_query, MATCH_NUMS)
            metrics_q.put((I, target_items, False, False))
        except tf.errors.OutOfRangeError:
            sess.run(test_iterator.initializer)    
            metrics_q.put((None, None, True, False))
            break
    return

features = {
            'uid': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'iid': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
            'neg_iid_list': tf.io.FixedLenFeature(shape=(NEG_IID_LIST_LEN, ), dtype=tf.int64)
}


if __name__ == '__main__':
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    train_data = tf.data.TFRecordDataset(train_data_path)
    train_data = train_data.map(lambda x: tf.io.parse_single_example(x, features))

    train_data = train_data.shuffle(BATCH_SIZE * 100)
    train_data = train_data.repeat(1).batch(BATCH_SIZE).prefetch(10)

    test_data = tf.data.TFRecordDataset(test_data_path)
    test_data = test_data.map(lambda x: tf.io.parse_single_example(x, features))
    test_data = test_data.repeat(1).padded_batch(BATCH_SIZE).prefetch(10)

    train_iterator = train_data.make_initializable_iterator()
    test_iterator = test_data.make_initializable_iterator()
    train_batch = train_iterator.get_next()
    test_batch = test_iterator.get_next()

    feature_counts = pkl.load(open(feature_counts_path, "rb"))
    item_size = feature_counts["iid"]
    print(feature_counts)

    item_pop_dict = pkl.load(open(pop_dict_path, 'rb'))
    item_pop_dict[0] = 0.0
    item_pop_dict[1] = 0.0
    item_pop_size = len(item_pop_dict)
    item_pop_list = [item_pop_dict[p] for p in range(item_size)]
    item_pop_max = max(item_pop_list)
    item_pop_min = min(item_pop_list)
    item_pop_tensor = tf.constant(item_pop_list)
    item_pop_norm_tensor = tf.constant([e / item_pop_max for e in item_pop_list])
    # pop_threshold = np.percentile(item_pop_list, args.pop_percent)
    print("item pop size: ", item_pop_size, ", max pop: ", item_pop_max, ", min pop: ", item_pop_min)

    sorted_pop_dict = sorted(item_pop_dict.items(), key=lambda x:x[1])
    sorted_pop_list = [x[0] for x in sorted_pop_dict]

    embd_initializer = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)
    l2_alpha = 0.0001
    embd_regularizer = keras.regularizers.l2(l2_alpha)
    # user id
    uid_int_emb_layer = keras.layers.Embedding(
        feature_counts["uid"], EMB_DIM, embeddings_initializer = embd_initializer,
        embeddings_regularizer = embd_regularizer,
        mask_zero=True, name='uid_interest_emb_layer'
    )
    uid_conf_emb_layer = keras.layers.Embedding(
        feature_counts["uid"], EMB_DIM, embeddings_initializer = embd_initializer,
        embeddings_regularizer = embd_regularizer,
        mask_zero=True, name='uid_conformity_emb_layer'
    )
    # item id
    iid_cont_emb_layer = keras.layers.Embedding(
        item_size, EMB_DIM, embeddings_initializer = embd_initializer,
        embeddings_regularizer = embd_regularizer,
        mask_zero=True, name='iid_content_emb_layer'
    )
    iid_pop_emb_layer = keras.layers.Embedding(
        item_size, EMB_DIM, embeddings_initializer = embd_initializer,
        embeddings_regularizer = embd_regularizer,
        mask_zero=True, name='iid_popularity_emb_layer'
    )

    iid_index = tf.constant(np.arange(item_size), dtype = tf.int64)
    all_item_cont_emb = iid_cont_emb_layer(iid_index)
    all_item_pop_emb = iid_pop_emb_layer(iid_index)
    all_item_emb = tf.concat([all_item_cont_emb, all_item_pop_emb], axis=-1)
    # all_item_emb_norm = tf.math.l2_normalize(all_item_emb, axis=1)

    def contrastive_model(uids, iids, neg_iid_list, is_train):
        uid_int_emb = uid_int_emb_layer(uids)
        uid_conf_emb = uid_conf_emb_layer(uids)
        user_emb= tf.concat([uid_int_emb, uid_conf_emb], axis=-1)
        if (not is_train):
            return user_emb

        if (is_train):
            uids_1 = tf.reshape(uids,[-1, 1])
            uids_2 = tf.reshape(uids,[1, -1])
            y_true = tf.equal(uids_1, uids_2)
            # instance-level
            item_cont_emb = iid_cont_emb_layer(iids)
            item_pop_emb = iid_pop_emb_layer(iids)
            item_cont_emb_norm = tf.math.l2_normalize(item_cont_emb, axis = -1)
            item_pop_emb_norm = tf.math.l2_normalize(item_pop_emb, axis = -1)

            pos_item_pop = tf.gather(item_pop_norm_tensor, iids)
            mask_item_cont = pop_func(pos_item_pop, args.pop_coeff)
            mask_item_pop = tf.ones_like(mask_item_cont) - mask_item_cont

            user_int_emb_norm = tf.math.l2_normalize(uid_int_emb, axis = -1)
            user_conf_emb_norm = tf.math.l2_normalize(uid_conf_emb, axis = -1)

            ui_int_score = tf.matmul(user_int_emb_norm, item_cont_emb_norm, transpose_b = True)
            ui_int_score = ui_int_score * args.score_coeff

            item_pop_1 = tf.reshape(pos_item_pop, [-1, 1])
            item_pop_2 = tf.reshape(pos_item_pop, [1, -1])
            pop_select = tf.cast(tf.math.greater_equal(item_pop_1, item_pop_2), dtype=tf.float32)
            ui_conf_score = tf.matmul(user_conf_emb_norm, item_pop_emb_norm, transpose_b = True) * pop_select
            ui_conf_score = ui_conf_score * args.score_coeff

            iids_1 = tf.reshape(iids,[-1, 1])
            iids_2 = tf.reshape(iids,[1, -1])
            iid_eq = tf.cast(tf.equal(iids_1, iids_2), dtype = tf.float32)
            y_true = tf.greater(tf.matmul(tf.cast(y_true, dtype = tf.float32), iid_eq), 0.5)
            ui_int_loss = CSELoss(ui_int_score, y_true, mask_item_cont, neg_num = neg_num)
            ui_conf_loss = CSELoss(ui_conf_score, y_true, mask_item_pop, neg_num = neg_num)

            neg_iids_list = tf.reshape(neg_iid_list, [-1, NEG_IID_LIST_LEN])
            rown = tf.shape(user_emb)[0]
            r = tf.random_uniform(shape=[rown], minval=0, maxval=(NEG_IID_LIST_LEN-1), dtype=tf.int32)
            r = tf.reshape(r, [rown, 1])
            neg_iids = tf.batch_gather(neg_iid_list, r)
            neg_iids = tf.reshape(neg_iids, [rown])

            pos_iids_cont_emb = iid_cont_emb_layer(iids)
            pos_iids_pop_emb = iid_pop_emb_layer(iids)
            neg_iids_cont_emb = iid_cont_emb_layer(neg_iids)
            neg_iids_pop_emb = iid_pop_emb_layer(neg_iids)
            pos_iids_emb = tf.concat([pos_iids_cont_emb, pos_iids_pop_emb],axis=-1)
            neg_iids_emb = tf.concat([neg_iids_cont_emb, neg_iids_pop_emb],axis=-1)
            pos_score = tf.reduce_sum(tf.multiply(user_emb, pos_iids_emb), axis = -1, keepdims = True)
            neg_score = tf.reduce_sum(tf.multiply(user_emb, neg_iids_emb), axis = -1, keepdims = True)
            loss_total = tf.reduce_sum(-tf.log(tf.nn.sigmoid(pos_score  - neg_score)+1e-9))


            loss = args.dccl_int_weight * ui_int_loss + args.dccl_conf_weight * ui_conf_loss + \
                   loss_total
            optimizer = keras.optimizers.Adam(args.lr)
            model_vars = tf.trainable_variables()
            grads = tf.gradients(loss, model_vars)
            train_op = optimizer.apply_gradients(zip(grads, model_vars))
            return loss, train_op

    uids = train_batch['uid']
    iids = train_batch['iid']
    neg_iid_list = train_batch['neg_iid_list']
    loss, train_op = contrastive_model(uids, iids, neg_iid_list, True)

    test_uids  = test_batch["uid"]
    test_iids = test_batch["iid"]
    user_emb = contrastive_model(test_uids, test_iids, None, False)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, EMB_DIM * 2, flat_config)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    print('-' * 120)
    with tf.Session(config=config) as sess:
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        #print(train_handle, test_handle)
        sess.run(tf.initializers.global_variables())
        prev_time = time()
        epoch = 0
        cycle = 0
        sess.run(train_iterator.initializer)
        #sess.run(test_iterator.initializer)
        exit = False
        while True:
            cycle += 1
            iterator_num = 0
            total_loss = 0.0
            prev_time = time()
            while True:
                iterator_num += 1
                try:
                    batch_loss, _ = sess.run([loss, train_op])#, feed_dict={handle: train_handle})
                except tf.errors.OutOfRangeError:
                    epoch += 1
                    if (epoch >= EPOCHS):
                        exit = True
                    else:
                        sess.run(train_iterator.initializer)
                    break
                total_loss += batch_loss
                print('\r' + '-' * 32 + ' ' * 6 + f'batch_loss: {batch_loss:.8f}' + ' ' * 6  + '-' * 32, end='')
            curr_time = time()
            time_elapsed = curr_time - prev_time
            print(f'\ntrain_loss of iteration-{epoch}: {(total_loss / iterator_num):.8f}    ' +
                '-' * 36 + f'    time elapsed: {time_elapsed:.2f}s')
            sess.run(test_iterator.initializer)
            eval_on_test(sess, index)
            if (exit):
                break
        metrics_q.put((None, None, False, True))
        metrics_computer_p.join()
