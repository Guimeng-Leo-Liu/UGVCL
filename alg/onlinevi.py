import numpy as np
import pandas as pd
import tensorflow as tf
from helper_functions import *
import time
import Adam
import matplotlib.pyplot as plt
import math
from mlp import init_weights

dimZ = 50
dimH = 500
dimH1 = 500
dimH2 = 500
dimH3 = 500
dimX = 784
one = [np.ones((dimZ, dimH1)), np.ones((dimH1,)), np.ones((dimH1, dimH2)), np.ones((dimH2,)), np.ones((dimH2, dimH3)), np.ones((dimH3,)), np.ones((dimH3, dimX)), np.ones((dimX,))]


def plt_hist(values,name):

    plt.figure(figsize=(16, 8), dpi=100)

    if name == 'snr':
        bins = np.arange(math.floor(np.min(values)),math.ceil(np.max(values))+1,0.001)
    else:
        bins = np.arange(math.floor(np.min(values)), math.floor(np.max(values)) + 1, 0.01)

    # bins = np.arange(np.min(time), np.max(time), 0.1)
    print(np.max(values))
    print(np.min(values))
    # print(bins)
    plt.hist(values, bins)
    # plt.xticks(list(range(int(np.min(values)), int(np.max(values))))[::2])

    plt.ylim(0, 16)

    plt.xlabel('values')
    plt.ylabel('numbers')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(name, loc = 'right')
    plt.show()


def print_var(sess, name):
    # this is for check if the variable has been change
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    print(values[0])
    # plt_hist(values[0], name)



def get_q_theta_params():
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen_shared' in var.name]
    param_dict = {}
    for var in var_list:
        param_dict[var.name] = var	# make sure here is not a copy!
    return param_dict

def get_headnet_params(task):
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen_%d_head' % task in var.name]
    param_dict = {}
    for var in var_list:
        param_dict[var.name] = var	# make sure here is not a copy!
    return param_dict

def init_shared_prior():
    q_params = get_q_theta_params()
    prior_params = {}
    for name in q_params.keys():
        shape = q_params[name].get_shape().as_list()
        prior_params[name] = np.zeros(shape, dtype='f')

    return prior_params
    
def update_shared_prior(sess, prior_params):  # only the gen-shared part is updated
    q_params = get_q_theta_params()
    for name in prior_params.keys():
        prior_params[name] = sess.run(q_params[name])

    return prior_params
    
def update_q_sigma(sess):
    q_params = get_q_theta_params()
    for name in q_params.keys():
        if 'log_sig' in name:
            shape = q_params[name].get_shape().as_list()
            sess.run(tf.assign(q_params[name], np.ones(shape)* -6.0))
    print ('reset the log sigma of q to -5')


def re_initialize(sess, mask, one):
    q_params = get_q_theta_params()
    # for name in q_params.keys():
    for name, i in zip(q_params.keys(), range(len(q_params.keys()))):

        ind = int(i / 2)
        if 'mu_W' in name:
            inv_mask = np.array(one[ind], dtype=object) - np.array(mask[ind],dtype=object)
            sess.run(tf.assign(tf.trainable_variables()[i],
                               tf.trainable_variables()[i] * mask[ind] + inv_mask * init_weights(one[ind].shape[0], one[ind].shape[1])))

def get_last_n_top(sess, task, dynamic_prunerate, p1, p2): #####################

    shared_q_params = get_q_theta_params()
    N_layer = int(len(shared_q_params.keys()) / 4)
    top = []
    last = []
    snr = []
    for i in range(N_layer):    # one layer has four params
    # for i in range(N_layer-1):  # one layer has four params
        index = i*4
        snr1 = abs(tf.trainable_variables()[index]/ tf.trainable_variables()[index+2])
        snr2 = abs(tf.trainable_variables()[index+1] / tf.trainable_variables()[index+3])
        snr1 = sess.run(snr1)
        snr2 = sess.run(snr2)
        snr.append(snr1)
        snr.append(snr2.reshape((snr2.shape[0], 1)))

    print('==========================check weight mu and sigma==========================')
    print('==========================snr==========================')

    for i in range(N_layer*2):  # weight ans bias are combination of two params

        if dynamic_prunerate:
            #p2 = 1 / (11 - 2) # for inverse to task
            # l = math.floor(i/2) 
            if task == 1:
                sum = 0
                pruned = 1 - p1
            elif task == 2:
                sum = p1
                pruned = (1 - sum) * (1 - p2)
            else:
                sum = p1
                pruned = (1 - sum) * (1 - p2)
                for _ in range(task - 2):
                    sum += (1 - sum) * p2
                    #p2 = 1 / (8 - i) # 这里只有inverse to task时才用
                    pruned *= 1 - p2

            m = np.zeros(snr[i].shape)
            threshold = np.quantile(pd.DataFrame(snr[i]).fillna(0), sum + pruned)
            print(threshold)
            m[snr[i] > threshold] = 1
            print(pd.value_counts(m.flatten()))

        else:
            assert p2 < ((1 - p1) / 9)

            if task == 1:
                pruned = 1 - p1
            else:
                pruned = 1 - p2
            m = np.zeros(snr[i].shape)
            threshold = np.quantile(pd.DataFrame(snr[i]).fillna(0), pruned)
            print(threshold)
            m[snr[i] > threshold] = 1
            print(pd.value_counts(m.flatten()))

        if (i % 2) == 1:   # bias和weight的shape不一样，分开处理
            last.append(m.reshape((m.shape[0], ))) # important
            top.append(1 - m.reshape((m.shape[0], ))) # not important
        else:
            last.append(m)
            top.append(1 - m)

    return last, top


def get_lr_mask(sess, mask, task, one):
    temp_var = []

    print('=========================================get_lr_mask=========================================')
    print('==========================variables before mask==========================')

    # print_var(sess, 'weights before get_lr_mask')

    shared_q_params = get_q_theta_params()
    N_layer = int(len(shared_q_params.keys()) / 4)
    for i in range(N_layer):
    # for i in range(N_layer - 1):
        index = i * 4

        # store the top p for restore later, cuz top p should be fixed after retrain and can't change during training in following tasks
        temp_var.append(sess.run(tf.trainable_variables()[index]) * mask[task-1][int(index / 2)])
        temp_var.append(np.zeros(tf.trainable_variables()[index+1].shape))
        temp_var.append(sess.run(tf.trainable_variables()[index+2]) * mask[task-1][int(index / 2)])
        temp_var.append(np.zeros(tf.trainable_variables()[index+3].shape))

        # pick last (1-p)
        sess.run(tf.assign(tf.trainable_variables()[index], tf.trainable_variables()[index] * (np.array(one[int(index / 2)], dtype=object) - np.array(
                               mask[task - 1][int(index / 2)], dtype=object))))
        # sess.run(tf.assign(tf.trainable_variables()[index + 1], tf.trainable_variables()[index + 1] * (np.array(one[int(index / 2) + 1], dtype=object) - np.array(
        #                            mask[task - 2][int(index / 2) + 1], dtype=object))))
        sess.run(tf.assign(tf.trainable_variables()[index + 2], tf.trainable_variables()[index + 2] * (np.array(one[int(index / 2)], dtype=object) - np.array(
                               mask[task - 1][int(index / 2)], dtype=object))))
        # sess.run(tf.assign(tf.trainable_variables()[index + 3], tf.trainable_variables()[index + 3] * (np.array(one[int(index / 2) + 1], dtype=object) - np.array(
        #                            mask[task - 2][int(index / 2) + 1], dtype=object))))


    print('==========================variables after mask==========================')
    # print_var(sess, 'weights after get_lr_mask')
    print('==========================temp variables==========================')
    # print(temp_var[0])
    # print('temp_var', plt_hist(temp_var[0], 'temp_var get_lr_mask'))

    return temp_var

def get_weight_mask(sess, mask, task, one, printhist = False):
    temp_var = []

    if printhist:
        print('=========================================get_weight_mask=========================================')
        print('==========================variables before mask==========================')
        print_var(sess, 'weights before get_weight_mask')

    shared_q_params = get_q_theta_params()
    N_layer = int(len(shared_q_params.keys()) / 4)

    # print('==========================check temp var==========================')
    # var1 = sess.run(tf.trainable_variables()[0])
    # print(var1)
    # print((np.array(one[0], dtype=object) - np.array(mask[task - 1][0], dtype=object)))
    # temp_varivable1 = var1 * (np.array(one[0], dtype=object) - np.array(mask[task - 1][0], dtype=object))
    # print(temp_varivable1)


    for i in range(N_layer):
    # for i in range(N_layer - 1):
        index = i * 4 # 4 parameter for each layer

        # pick out last (1-p)% for restore later
        var1 = sess.run(tf.trainable_variables()[index])
        temp_varivable1 = var1* (np.array(one[int(index / 2)], dtype=object) - np.array(mask[task][int(index / 2)],
                                                                           dtype=object))
        # var2 = sess.run(tf.trainable_variables()[index + 1])
        # temp_varivable2 = var2 * (np.array(one[int(index / 2) + 1], dtype=object) - np.array(mask[task - 1][int(index / 2) + 1],
        #                                                                        dtype=object))
        var3 = sess.run(tf.trainable_variables()[index + 2])
        temp_varivable3 = var3 * (np.array(one[int(index / 2)], dtype=object) - np.array(mask[task][int(index / 2)],
                                                                                         dtype=object))
        # var4 = sess.run(tf.trainable_variables()[index + 3])
        # temp_varivable4 = var4 * (np.array(one[int(index / 2) + 1], dtype=object) - np.array(mask[task - 1][int(index / 2) + 1],
        #                                                                        dtype=object))
        temp_var.append(temp_varivable1)
        temp_var.append(np.zeros(tf.trainable_variables()[index+1].shape))
        temp_var.append(temp_varivable3)
        temp_var.append(np.zeros(tf.trainable_variables()[index+3].shape))

        # pick out top p% for retrain later
        sess.run(tf.assign(tf.trainable_variables()[index],
                           tf.trainable_variables()[index] * mask[task][int(index / 2)]))
        # sess.run(tf.assign(tf.trainable_variables()[index + 1],
        #                    tf.trainable_variables()[index + 1] * np.ones(tf.trainable_variables()[index+1].shape)))
        sess.run(tf.assign(tf.trainable_variables()[index + 2],
                           tf.trainable_variables()[index + 2] * mask[task][int(index / 2)]))
        # sess.run(tf.assign(tf.trainable_variables()[index + 3],
        #                    tf.trainable_variables()[index + 3] * np.ones(tf.trainable_variables()[index+3].shape)))

    if printhist:
        print('==========================variables after mask==========================')
        print_var(sess, 'weights after get_weight_mask')
        print('==========================temp variables==========================')
        print(temp_var[0])
        # print(plt_hist(temp_var[0], 'temp_var get_weight_mask'))

    return temp_var

def restore_mask(sess, temp_var, printhist = False):
    shared_q_params = get_q_theta_params()
    for i in range(int(len(shared_q_params.keys()))):
    # for i in range(int(len(shared_q_params.keys())) - 4):
        sess.run(tf.assign(tf.trainable_variables()[i],
                           tf.trainable_variables()[i] + temp_var[i]))

    if printhist:
        print('==================================================restore==================================================')
        # print_var(sess, 'restore')


def KL_param(shared_prior_params, task, regularise_headnet=False):
    # first get q params
    shared_q_params = get_q_theta_params()
    N_layer = int(len(shared_q_params.keys()) / 4)  # one layer has four params
    # then compute kl between prior and q
    kl_total = 0.0
    # for the shared network
    for l in range(N_layer):
        suffices = ['W', 'b']
        for suffix in suffices:
            mu_q = shared_q_params['gen_shared_l%d_mu_' % l + suffix + ':0']
            log_sig_q = shared_q_params['gen_shared_l%d_log_sig_' % l + suffix + ':0']
            mu_p = shared_prior_params['gen_shared_l%d_mu_' % l + suffix + ':0']
            log_sig_p = shared_prior_params['gen_shared_l%d_log_sig_' % l + suffix + ':0']
            kl_total += tf.reduce_sum(KL(mu_q, log_sig_q, mu_p, log_sig_p))

    # for the head network
    if regularise_headnet:
        head_q_params = get_headnet_params(task)
        N_layer = len(head_q_params.keys()) / 4  # one layer has for params
        for l in range(N_layer):
            for suffix in ['W', 'b']:
                mu_q = shared_q_params['gen_head%d_l%d_mu_' % (task, l) + suffix + ':0']
                log_sig_q = shared_q_params['gen_head%d_l%d_log_sig_' % (task, l) + suffix + ':0']
                kl_total += tf.reduce_sum(KL(mu_q, log_sig_q, 0.0, 0.0))

    return kl_total


def lowerbound(x, enc, dec, ll, mask, K = 1, mu_pz = 0.0, log_sig_pz = 0.0):
    mu_qz, log_sig_qz = enc(x) # 训练集喂进encoder
    #z = sample_gaussian(mu_qz, log_sig_qz)
    kl_z = KL(mu_qz, log_sig_qz, mu_pz, log_sig_pz)

    if K > 1:
        print ('using K=%d theta samples for onlinevi' % K)

    logp = 0.0
    for _ in range(K): # 这里就是为了采样不同的z
        # see bayesian_generator.py, tiling z does not work!
        z = sample_gaussian(mu_qz, log_sig_qz)	# sample different z
        mu_x = dec(z, mask)	# sample different theta
        if ll == 'bernoulli':
            logp += log_bernoulli_prob(x, mu_x) / K
        if ll == 'l2':
            logp += log_l2_prob(x, mu_x) / K
        if ll == 'l1':
            logp += log_l1_prob(x, mu_x) / K
    return logp - kl_z

def construct_optimizer(sess, X_ph, enc, dec, ll, N_data, batch_size_ph, shared_prior_params, task, one, K, mask = None):

    # loss function
    # print(mask[task-1])
    bound = tf.reduce_mean(lowerbound(X_ph, enc, dec, ll, mask[0], K))
    kl_theta = KL_param(shared_prior_params, task)
    loss_total = -bound + kl_theta / N_data
    batch_size = X_ph.get_shape().as_list()[0]

    # now construct optimizers
    lr_ph = tf.placeholder(tf.float32, shape=())
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen' in var.name]
    N_param = np.sum([np.prod(var.get_shape().as_list()) for var in var_list])

    print('==========================================================================================')
    if task == 1:
        opt = Adam.AdamOptimizer(learning_rate=lr_ph).minimize(loss_total)
        ops = [opt, bound, kl_theta]

    else:
        op = []

        # the lr_mask here is the last (1-p)% in previous task
        for i in range(int(16/4)):
            index = i*4
            dec_op1 = Adam.AdamOptimizer(learning_rate=lr_ph * (np.array(one[int(index/2)]) - np.array(mask[task-1][int(index/2)]))).minimize(loss_total, var_list=tf.trainable_variables()[index])
            dec_op2 = Adam.AdamOptimizer(learning_rate=lr_ph * (np.array(np.ones(tf.trainable_variables()[index+1].shape)))).minimize(loss_total, var_list=tf.trainable_variables()[index+1])
            dec_op3 = Adam.AdamOptimizer(learning_rate=lr_ph * (np.array(one[int(index/2)]) - np.array(mask[task-1][int(index/2)]))).minimize(loss_total, var_list=tf.trainable_variables()[index+2])
            dec_op4 = Adam.AdamOptimizer(learning_rate=lr_ph * (np.array(np.ones(tf.trainable_variables()[index+3].shape)))).minimize(loss_total, var_list=tf.trainable_variables()[index+3])
            op.append(dec_op1)
            op.append(dec_op2)
            op.append(dec_op3)
            op.append(dec_op4)
        enc_var = tf.trainable_variables()[16:]
        enc_op = Adam.AdamOptimizer(learning_rate=lr_ph).minimize(loss_total, var_list=enc_var)
        op.append(enc_op)
        train_op = tf.group(op[0], op[1], op[2], op[3], op[4], op[5], op[6], op[7], op[8], op[9], op[10], op[11], op[12], op[13], op[14], op[15], op[16])
        ops = [train_op, bound, kl_theta]




    def train(sess, X, lr):
        _, logp, kl = sess.run(ops, feed_dict={X_ph: X, lr_ph: lr,
                                    batch_size_ph: X.shape[0]})
        return logp, kl / N_param




    def fit(sess, X, n_iter, lr):
        N = X.shape[0]
        print ("training for %d epochs" % n_iter)
        begin = time.time()
        n_iter_vae = int(N / batch_size)
        # for iteration in range(1, 20):
        for iteration in range(1, n_iter + 1):
            ind_s = np.random.permutation(range(N))
            bound_total = 0.0
            kl_total = 0.0
            for j in range(0, n_iter_vae): # 每次循环用不同的打乱后的输入图像
                indl = j * batch_size
                indr = (j+1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr-N)]))
                logp, kl = train(sess, X[ind], lr)
                bound_total += logp / n_iter_vae
                kl_total += kl / n_iter_vae
            end = time.time()
            print ("Iter %d, bound=%.2f, kl=%.2f, time=%.2f" \
                  % (iteration, bound_total, kl_total, end - begin))
            begin = end

    return fit

def re_construct_optimizer(sess, X_ph, enc, dec, ll, N_data, batch_size_ph, shared_prior_params, task, K, mask, lr_mask):

    # loss function
    bound = tf.reduce_mean(lowerbound(X_ph, enc, dec, ll, mask[task], K))
    kl_theta = KL_param(shared_prior_params, task)
    loss_total = -bound + kl_theta / N_data
    batch_size = X_ph.get_shape().as_list()[0]

    # now construct optimizers
    lr_ph = tf.placeholder(tf.float32, shape=())
    t_vars = tf.trainable_variables()
    var_list = [var for var in t_vars if 'gen' in var.name]
    N_param = np.sum([np.prod(var.get_shape().as_list()) for var in var_list])

    retrain_op = []

    extra_lr = 1

    for i in range(int(16 / 4)):
        index = i * 4
        # 这里要考虑一下retrianing时需不需要把bias也全部训练一次
        dec_op1 = Adam.AdamOptimizer(learning_rate=lr_ph * extra_lr * lr_mask[int(index / 2)]).minimize(loss_total, var_list= tf.trainable_variables()[index])
        dec_op2 = Adam.AdamOptimizer(learning_rate=lr_ph * extra_lr * (np.array(np.ones(tf.trainable_variables()[index+1].shape)))).minimize(loss_total, var_list= tf.trainable_variables()[index + 1])
        dec_op3 = Adam.AdamOptimizer(learning_rate=lr_ph * extra_lr * lr_mask[int(index / 2)]).minimize(loss_total, var_list= tf.trainable_variables()[index + 2])
        dec_op4 = Adam.AdamOptimizer(learning_rate=lr_ph * extra_lr * (np.array(np.ones(tf.trainable_variables()[index+3].shape)))).minimize(loss_total, var_list= tf.trainable_variables()[index + 3])
        retrain_op.append(dec_op1)
        retrain_op.append(dec_op2)
        retrain_op.append(dec_op3)
        retrain_op.append(dec_op4)
    enc_var = tf.trainable_variables()[16:]
    enc_op = Adam.AdamOptimizer(learning_rate=lr_ph* extra_lr).minimize(loss_total, var_list=enc_var)
    retrain_op.append(enc_op)
    retrain_op = tf.group(retrain_op[0], retrain_op[1], retrain_op[2], retrain_op[3], retrain_op[4], retrain_op[5], retrain_op[6],\
                          retrain_op[7], retrain_op[8], retrain_op[9], retrain_op[10], retrain_op[11], retrain_op[12],\
                          retrain_op[13], retrain_op[14], retrain_op[15], retrain_op[16])
    retrain_ops = [retrain_op, bound, kl_theta]

    def retrain(sess, X, lr):
        _, logp, kl = sess.run(retrain_ops, feed_dict={X_ph: X, lr_ph: lr, batch_size_ph: X.shape[0]})

        return logp, kl / N_param

    def re_fit(sess, X, n_iter, lr):

        N = X.shape[0]
        n_iter_vae = int(N / batch_size)
        # retrain
        # print(mask)
        # temp_var = get_weight_mask(sess, mask, task, one, printhist = True)  # pruning weight

        print("retraining for %d epochs" % int(n_iter / 2))

        begin = time.time()
        for iteration in range(1, int((n_iter / 2) + 1)):
        # for iteration in range(1, 10):
            ind_s = np.random.permutation(range(N))
            bound_total = 0.0
            kl_total = 0.0
            for j in range(0, n_iter_vae):
                indl = j * batch_size
                indr = (j + 1) * batch_size
                ind = ind_s[indl:min(indr, N)]
                if indr > N:
                    ind = np.concatenate((ind, ind_s[:(indr - N)]))
                logp, kl = retrain(sess, X[ind], lr)     # retrain
                bound_total += logp / n_iter_vae
                kl_total += kl / n_iter_vae
            end = time.time()
            print("Iter %d, bound=%.2f, kl=%.2f, time=%.2f" \
                  % (iteration, bound_total, kl_total, end - begin))
            begin = end

        # restore_mask(sess, temp_var, printhist = True)   # let the pruned weights come back from zero

    return re_fit
