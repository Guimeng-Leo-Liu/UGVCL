import numpy as np
import tensorflow as tf
import sys, os
sys.path.extend(['alg/', 'models/'])
from visualisation import plot_images
from encoder_no_shared import encoder, recon
from utils import init_variables, save_params, load_params, load_data
from eval_test_ll import construct_eval_func
import pickle

dimZ = 50
n_channel = 128
batch_size = 50
lr = 1e-4
K_mc = 10
checkpoint = -1


data_path = ''# TODO

def main(data_name, method, percentage, dimZ, n_channel, batch_size, K_mc, checkpoint, lbd):
    # set up dataset specific stuff
    from config import config
    labels, n_iter, dimX, dimH, dimH1, dimH2, dimH3, shape_high, ll = config(data_name, n_channel)
    one = [np.ones((dimZ, dimH1)), np.ones((dimH1,)), np.ones((dimH1, dimH2)), np.ones((dimH2,)), np.ones((dimH2, dimH3)), np.ones((dimH3,)), np.ones((dimH3, dimX)), np.ones((dimX,))]

    if data_name == 'mnist':
        from mnist import load_mnist
    if data_name == 'notmnist':
        from notmnist import load_notmnist
    if data_name == 'cifar10':
        from cifar10 import load_cifar10

    # import functionalities
    if method == 'onlinevi':
        from bayesian_generator import generator, construct_gen
        from onlinevi import construct_optimizer, init_shared_prior, \
                     update_shared_prior, update_q_sigma, get_weight_mask, restore_mask, \
                    get_last_n_top, re_construct_optimizer, get_lr_mask, print_var, re_initialize
    if method in ['ewc', 'noreg', 'laplace', 'si']:
        from generator import generator_head, generator_shared, generator, construct_gen
        if method in ['ewc', 'noreg']:
            from vae_ewc import construct_optimizer, lowerbound
        if method == 'ewc': from vae_ewc import update_ewc_loss, compute_fisher
        if method == 'laplace':
            from vae_laplace import construct_optimizer, lowerbound
            from vae_laplace import update_laplace_loss, compute_fisher, init_fisher_accum
        if method == 'si':
            from vae_si import construct_optimizer, lowerbound, update_si_reg

    if 'D' in percentage:
        dynamic_prunerate = True
        p1, p2 = percentage.split("D", 2)
        p1 = int(p1) / 100; p2 = int(p2) / 100
    if 'C' in percentage:
        dynamic_prunerate = False
        p1, p2 = percentage.split("C", 2)
        p1 = int(p1) / 100; p2 = int(p2) / 100

    # then define model
    n_layers = 3
    batch_size_ph = tf.placeholder(tf.int32, shape=(), name='batch_size')
    # dec_shared = generator_shared(dimX, dimH, n_layers_shared, 'sigmoid', 'gen')
    # dec = generator(generator_head(dimZ, dimH, n_layers_head, 'gen'), dec_shared)
    dec = generator(dimZ, dimH1, dimH2, dimH3, dimX, n_layers, 'sigmoid', 'gen')


    # initialise sessions
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    sess = tf.Session()
    string = method
    if method in ['ewc', 'laplace', 'si']:
        string = string + '_lbd%.1f' % lbd
    if method == 'onlinevi' and K_mc > 1:
        string = string + '_K%d' % K_mc + '_%s' % percentage ###############################################
    path_name = data_name + '_%s/' % string
    print('path_name',path_name)
    if not os.path.isdir('save/'):
        os.mkdir('save/')
    if not os.path.isdir('save/'+path_name):
        os.mkdir('save/'+path_name)
        print ('create path save/' + path_name)
    filename = 'save/' + path_name + 'checkpoint'
    if checkpoint < 0:
        print ('training from scratch')
        old_var_list = init_variables(sess)
    else:
        load_params(sess, filename, checkpoint)
    checkpoint += 1

    # visualise the samples
    N_gen = 10**2
    path = 'figs/' + path_name
    if not os.path.isdir('figs/'):
        os.mkdir('figs/')
    if not os.path.isdir(path):
        os.mkdir(path)
        print ('create path ' + path)
    X_ph = tf.placeholder(tf.float32, shape=(batch_size, dimX), name = 'x_ph')

    # now start fitting
    N_task = len(labels)
    gen_ops = []
    X_valid_list = []
    X_test_list = []
    eval_func_list = []
    result_list = []
    mask = []
    mask.append(one)
    lr_mask = []
    if method == 'onlinevi':
        shared_prior_params = init_shared_prior() # 只初始化共享的参数
    if method in ['ewc', 'noreg']:
        ewc_loss = 0.0
    if method == 'laplace':
        F_accum = init_fisher_accum()
        laplace_loss = 0.0
    if method == 'si':
        old_params_shared = None
        si_reg = None
    # n_layers_head = 2
    # n_layers_enc = n_layers_shared + n_layers_head - 1
    for task in range(1, N_task+1):
        # first load data
        if data_name == 'mnist':
            X_train, X_test, _, _ = load_mnist(digits = labels[task-1], conv = False)
        if data_name == 'notmnist':
            X_train, X_test, _, _ = load_notmnist(data_path, digits = labels[task-1], conv = False)
        if data_name == 'cifar10':
            X_train, X_test, _, _ = load_cifar10(digits=labels[task - 1], conv=False)

        N_train = int(X_train.shape[0] * 0.9)
        X_valid_list.append(X_train[N_train:]) # 验证集0.1
        X_train = X_train[:N_train] # 训练集0.9
        X_test_list.append(X_test)

        # define the head net and the generator ops
        # dec = generator(generator_head(dimZ, dimH, n_layers_head, 'gen_%d' % task), dec_shared)
        enc = encoder(dimX, dimH, dimZ, n_layers, 'enc_%d' % task)
        # gen_ops.append(construct_gen(dec, dimZ, sampling=False)(N_gen))
        # print ('construct eval function...')
        # # evaluate test log-likelihood
        # eval_func_list.append(construct_eval_func(X_ph, enc, dec, ll, \
        #                                           batch_size_ph, K = 100, sample_W = False))

        # print(sess.run(tf.trainable_variables()[0])) #############################################################

        # then construct loss func and fit func
        print ('construct fit function...')
        if method == 'onlinevi':
            fit = construct_optimizer(sess, X_ph, enc, dec, ll, X_train.shape[0], batch_size_ph, \
                                                  shared_prior_params, task, one, K_mc, mask)

        if method in ['ewc', 'noreg']:
            bound = lowerbound(X_ph, enc, dec, ll)
            fit = construct_optimizer(X_ph, batch_size_ph, bound, X_train.shape[0], ewc_loss)
            if method == 'ewc':
                fisher, var_list = compute_fisher(X_ph, batch_size_ph, bound, X_train.shape[0])

        if method == 'laplace':
            bound = lowerbound(X_ph, enc, dec, ll)
            fit = construct_optimizer(X_ph, batch_size_ph, bound, X_train.shape[0], laplace_loss)
            fisher, var_list = compute_fisher(X_ph, batch_size_ph, bound, X_train.shape[0])

        if method == 'si':
            bound = lowerbound(X_ph, enc, dec, ll)
            fit, shared_var_list = construct_optimizer(X_ph, batch_size_ph, bound, X_train.shape[0],
                                                       si_reg, old_params_shared, lbd)
            if old_params_shared is None:
                old_params_shared = sess.run(shared_var_list)

        # initialise all the uninitialised stuff
        old_var_list = init_variables(sess, old_var_list)

        # start training for each task
        if method == 'si':
            new_params_shared, w_params_shared = fit(sess, X_train, n_iter, lr)
        else:
            fit(sess, X_train, n_iter, lr)

        # get mask and lr_mask
        mtrx = []

        # if task > 1:
        #     print('mask[0]', mask[task-2][1])

        # print(np.array(mask).shape)

        # for i in range(np.array(mask).shape[0]):
        #     for j in range(np.array(mask).shape[1]):
                # print(np.sum(mask[i][j]))


        if task == 1:
            print('==========================find top p% in each parameters==========================')
            lr_mask, _ = get_last_n_top(sess, task, dynamic_prunerate, p1, p2)
            mask.append(lr_mask)
        else:
            # find out the top p% (except those fixed position)

            temp_lr_var = get_lr_mask(sess, mask, task, one) # pick out those not fixed position
            lr_mask, _ = get_last_n_top(sess, task, dynamic_prunerate, p1, p2) # find top p%

            for i in range(np.array(lr_mask, dtype=object).shape[0]):
                mtrx.append(mask[task - 1][i] + lr_mask[i])
            mask.append(mtrx)
            restore_mask(sess, temp_lr_var)

        # print('===============lr_mask====================')
        # print(lr_mask[1])

# ###########################################################################
#         if method == 'onlinevi':
#             # update prior
#             print ('update prior...')
#             # temp_var = get_weight_mask(sess, mask, task)
#             shared_prior_params = update_shared_prior(sess, shared_prior_params)
#             # reset the variance of q
#             # restore_mask(sess, temp_var)
#             update_q_sigma(sess)
# #############################################################################
#         print(mask[task][0])
#         print(sess.run(tf.trainable_variables()[0]))  #############################################################

        # retrain for each task
        if method == 'onlinevi':
            re_fit = re_construct_optimizer(sess, X_ph, enc, dec, ll, X_train.shape[0], batch_size_ph, \
                                                  shared_prior_params, task, K_mc, mask, lr_mask)
            old_var_list = init_variables(sess, old_var_list)
            re_fit(sess, X_train, n_iter, lr)

        # print(sess.run(tf.trainable_variables()[0]))  #############################################################

        # plot samples
        # temp_var = get_weight_mask(sess, mask, task)


        gen_ops.append(construct_gen(dec, dimZ, mask[task], sampling=False)(N_gen)) ####################################
        print('construct eval function...')
        # evaluate test log-likelihood
        eval_func_list.append(construct_eval_func(X_ph, enc, dec, ll, \
                                                  batch_size_ph, mask[task], K=100, sample_W=False)) ####################################

        x_gen_list = sess.run(gen_ops, feed_dict={batch_size_ph: N_gen})
        for i in range(len(x_gen_list)):
            plot_images(x_gen_list[i], shape_high, path, \
                        data_name+'_gen_task%d_%d' % (task, i+1))

        x_list = [x_gen_list[i][:1] for i in range(len(x_gen_list))]
        x_list = np.concatenate(x_list, 0)

        tmp = np.zeros([10, dimX])
        tmp[:task] = x_list
        if task == 1:
            x_gen_all = tmp
        else:
            x_gen_all = np.concatenate([x_gen_all, tmp], 0)

        # print test-ll on all tasks
        tmp_list = []
        for i in range(len(eval_func_list)):
            print ('task %d' % (i+1),)
            test_ll = eval_func_list[i](sess, X_valid_list[i])
            tmp_list.append(test_ll)
        result_list.append(tmp_list)

        # save param values
        save_params(sess, filename, checkpoint)
        checkpoint += 1

        # restore_mask(sess, temp_var)


        # update regularisers/priors
        if method == 'ewc':
            # update EWC loss
            print ('update ewc loss...')
            X_batch = X_train[np.random.permutation(range(X_train.shape[0]))[:batch_size]]
            ewc_loss = update_ewc_loss(sess, ewc_loss, var_list, fisher, lbd, X_batch)
        if method == 'laplace':
            # update EWC loss
            print ('update laplace loss...')
            X_batch = X_train[np.random.permutation(range(X_train.shape[0]))[:batch_size]]
            laplace_loss, F_accum = update_laplace_loss(sess, F_accum, var_list, fisher, lbd, X_batch)

        ###########################################################################
        if method == 'onlinevi':
            # update prior
            print('update prior...')
            # temp_var = get_weight_mask(sess, mask, task)
            # shared_prior_params = update_shared_prior(sess, shared_prior_params)
            # reset the variance of q
            # restore_mask(sess, temp_var)
            update_q_sigma(sess)

            # re-initialize
            re_initialize(sess, mask[task], one)



        #############################################################################
        # if method == 'onlinevi':
        #     # update prior
        #     print ('update prior...')
        #     shared_prior_params = update_shared_prior(sess, shared_prior_params)
        #     # reset the variance of q
        #     update_q_sigma(sess)

        if method == 'si':
            # update regularisers/priors
            print ('update SI big omega matrices...')
            si_reg, _ = update_si_reg(sess, si_reg, new_params_shared, \
                                      old_params_shared, w_params_shared)
            old_params_shared = new_params_shared

    plot_images(x_gen_all, shape_high, path, data_name+'_gen_all')

    for i in range(len(result_list)):
        print (result_list[i])

    # save binary mask
    fname = 'save/' + data_name + '_%s_mask.pkl' % string
    f = open(fname, 'wb')
    pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('mask saved at ' + fname)
    f.close()

    # save results
    fname = 'results/' + data_name + '_%s.pkl' % string
    if not os.path.isdir('results/'):
        os.mkdir('results/')
    pickle.dump(result_list, open(fname, 'wb'))
    print ('test-ll results saved in', fname)

if __name__ == '__main__':
    data_name = str(sys.argv[1])
    method = str(sys.argv[2])
    assert method in ['noreg', 'laplace', 'ewc', 'si', 'onlinevi']
    if method == 'onlinevi':
        lbd = 1.0	# some placeholder, doesn't matter
        percentage = str(sys.argv[3])
    else:
        lbd = float(sys.argv[3])
    main(data_name, method, percentage, dimZ, n_channel, batch_size, K_mc, checkpoint, lbd)

