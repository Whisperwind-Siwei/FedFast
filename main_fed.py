#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    high_ratio = args.high_ratio
    split_ratio = 0.7
    strategy = args.strategy
    prune_ratio = args.prune_ratio
    low_step = args.low_step

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.load_dict:
            with open("./save/dict_users_mnist_iid{}_{}.pkl".format(args.iid, args.num_users), "rb") as file:
                dict_users = pickle.load(file)
        else:
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
            with open("./save/dict_users_mnist_iid{}_{}.pkl".format(args.iid, args.num_users), "wb") as file:
                pickle.dump(dict_users, file)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.load_dict:
            with open("./save/dict_users_cifar_iid{}_{}.pkl".format(args.iid, args.num_users), "rb") as file:
                dict_users = pickle.load(file)
        else:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            else:
                dict_users = cifar_noniid(dataset_train, args.num_users, 1, 0.5)
            with open("./save/dict_users_cifar_iid{}_{}.pkl".format(args.iid, args.num_users), "wb") as file:
                pickle.dump(dict_users, file)
    else:
        exit('Error: unrecognized dataset')

    high_idxs = []
    low_idxs = []
    for idx in range(args.num_users):
        if idx < split_ratio * args.num_users:
            high_idxs += list(dict_users[idx])
        else:
            low_idxs += list(dict_users[idx])
    dataset_high = DatasetSplit(dataset_train, high_idxs)
    dataset_low = DatasetSplit(dataset_train, low_idxs)

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    high_accuracy, low_accuracy = [], []

    for iter in range(args.epochs):
        net_glob.train()
        loss_locals = []
        w_locals = []
        quantity_high = []
        m = max(int(args.frac * args.num_users * high_ratio), 1)
        idxs_users_high = np.random.choice(range(int(args.num_users * high_ratio)), m, replace=False)
        for idx in idxs_users_high:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_quantity = len(dict_users[idx])
            for k in w.keys():
                w[k] *= local_quantity
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            quantity_high.append(local_quantity)
        if strategy and iter != 0:
            n = int(args.frac * args.num_users * (1 - high_ratio))
            idxs_users_low = np.random.choice(range(int(args.num_users * high_ratio), args.num_users), n, replace=False)
            w_low = []
            loss_low = []
            for idx in idxs_users_low:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                loss = local.forward(net=copy.deepcopy(net_glob).to(args.device))
                temp_net = copy.deepcopy(net_glob)
                temp_w = temp_net.state_dict()
                temp_w_delta = copy.deepcopy(w_delta)
                for k in temp_w_delta.keys():
                    if 'weight' in k:
                        mask = (torch.rand(temp_w_delta[k].size()) > prune_ratio).to(args.device)
                        temp_w_delta[k] = (temp_w_delta[k] * mask) * low_step
                for k in temp_w.keys():
                    temp_w[k] += temp_w_delta[k]
                temp_net.load_state_dict(temp_w)
                temp_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                temp_loss = temp_local.forward(net=copy.deepcopy(temp_net).to(args.device))
                loss_diff = loss - temp_loss
                for k in temp_w_delta.keys():
                    temp_w_delta[k] = temp_w_delta[k] * loss_diff * len(dict_users[idx])
                loss_low.append(abs(loss_diff) * len(dict_users[idx]))
                w_low.append(copy.deepcopy(temp_w_delta))

        # update global weights
        w_aggre = FedAvg(w_locals, quantity_high)
        w_delta = copy.deepcopy(w_aggre)
        for k in w_delta.keys():
            w_delta[k] = torch.sub(w_aggre[k], w_glob[k])
        w_final = copy.deepcopy(w_aggre)
        if strategy and iter != 0:
            w_low_agg = copy.deepcopy(w_low[0])
            sum_loss_low = sum(loss_low)
            for k in w_low_agg.keys():
                for i in range(1, len(w_low)):
                    w_low_agg[k] += w_low[i][k]
                w_low_agg[k] = torch.div(w_low_agg[k], sum_loss_low)
            for k in w_final.keys():
                w_final[k] += w_low_agg[k]
        w_glob = copy.deepcopy(w_final)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        val_acc_list.append(acc_test)
        acc_high, loss_high = test_img(net_glob, dataset_high, args)
        acc_low, loss_low = test_img(net_glob, dataset_low, args)
        high_accuracy.append(acc_high)
        low_accuracy.append(acc_low)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(val_acc_list)), val_acc_list)
    plt.ylabel('Test Accuracy')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                       args.iid, args.high_ratio, args.strategy,
                                                                       args.prune_ratio, args.low_step))
    np.save('./save/fed_{}_{}_{}_C{}_iid{}_{}_{}_{}_{}_test.npy'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                   args.iid, args.high_ratio, args.strategy,
                                                                   args.prune_ratio, args.low_step), val_acc_list)
    np.save('./save/fed_{}_{}_{}_C{}_iid{}_{}_{}_{}_{}_high.npy'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                   args.iid, args.high_ratio, args.strategy,
                                                                   args.prune_ratio, args.low_step), high_accuracy)
    np.save('./save/fed_{}_{}_{}_C{}_iid{}_{}_{}_{}_{}_low.npy'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                   args.iid, args.high_ratio, args.strategy,
                                                                   args.prune_ratio, args.low_step), low_accuracy)

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

