#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
File Name     : mnist_mlp_cpu_backup.py
Author        : 王峤宇
Created Date  : 3/16/26
Version       : 1.0
Description   : 
------------------------------------------------------------
Usage Example:
    python your_script_name.py [options]
Notes:
    - 请在此处补充代码使用说明、输入输出格式等
------------------------------------------------------------
"""

import argparse

import struct
import os
import sys
import time
import numpy as np
from .layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Dataset(object):
    def __init__(self, minst_dir, data_file, label_file):
        self._minst_dir = minst_dir
        self._data_file = data_file
        self._label_file = label_file

        _data = self.load_mnist(os.path.join(minst_dir, data_file), is_images=True)
        _label = self.load_mnist(os.path.join(minst_dir, label_file), is_images=False)

        self.data_with_label = np.append(_data, _label, axis=1)

    @staticmethod
    def load_data(minst_dir, data_file, label_file):
        Dataset( minst_dir, data_file, label_file)

        _data = Dataset.load_mnist(os.path.join(minst_dir, data_file), is_images=True)
        _label = Dataset.load_mnist(os.path.join(minst_dir, label_file), is_images=False)

        return np.append(_data, _label, axis=1)

    @staticmethod
    def load_mnist(file_dir, is_images=True):
        # 加载MNIST数据集的代码
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()

        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1

        data_size = num_images * num_rows * num_cols

        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])

        return mat_data


class MNIST_MLP(object):
    def __init__(self, batch_size=100000,
                 input_size=784, hidden1=32, hidden2=16, out_classes=10,
                 lr=0.01, max_epoch=2, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

        # data
        self.train_data = None
        self.test_data = None

        # build_model
        self.layers = []
        self.update_layers = []


    def load_train_data(self, minst_dir, data_file, label_file):
        self.train_data = Dataset.load_data(minst_dir, data_file, label_file)

    def load_test_data(self, minst_dir, data_file, label_file):
        self.test_data = Dataset.load_data(minst_dir, data_file, label_file)

    def build_model(self):
        # layers
        self.layers.append(fc1 := FullyConnectedLayer(self.input_size, self.hidden1))
        self.layers.append(relu1 := ReLULayer())
        self.layers.append(fc2 := FullyConnectedLayer(self.hidden1, self.hidden2))
        self.layers.append(relu2 := ReLULayer())
        self.layers.append(fc3 := FullyConnectedLayer(self.hidden2, self.out_classes))
        self.layers.append(softmax_loss := SoftmaxLossLayer())

        # update layers
        self.update_layers.append(fc1)
        self.update_layers.append(fc2)
        self.update_layers.append(fc3)

    def init_model(self):
        for layer in self.update_layers:
            layer.init_param()

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self):
        top_diff = None

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                top_diff = layer.backward()
            else:
                top_diff = layer.backward(top_diff)

    def update_param(self):
        for layer in self.update_layers:
            layer.update_param(self.lr)

    def _shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

    def save_model(self, param_dir):
        params = {}
        params['fc1_weight'], params['fc1_bias'] = self.layers[0].save_param()
        params['fc2_weight'], params['fc2_bias'] = self.layers[2].save_param()
        params['fc3_weight'], params['fc3_bias'] = self.layers[4].save_param()
        np.save(param_dir, params)

    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        self.layers[0].load_param(params['fc1_weight'], params['fc1_bias'])
        self.layers[2].load_param(params['fc2_weight'], params['fc2_bias'])
        self.layers[4].load_param(params['fc3_weight'], params['fc3_bias'])

    def get_loss(self, batch_labels):
        return self.layers[-1].get_loss(batch_labels)

    def train(self):
        assert self.train_data is not None, "Please load training data before training."

        max_batch = self.train_data.shape[0] // self.batch_size

        for idx_epoch in range(self.max_epoch):
            self._shuffle_data()

            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch * self.batch_size: (idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size: (idx_batch + 1) * self.batch_size, -1]

                prob = self.forward(batch_images)
                loss = self.get_loss(batch_labels)

                self.backward()
                self.update_param()

                if (idx_batch + 1) % self.print_iter == 0:
                    print(f"Epoch [{idx_epoch + 1}/{self.max_epoch}], Batch [{idx_batch + 1}/{max_batch}]")

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images=self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels=np.argmax(prob,axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size]=pred_labels

        accuracy = np.mean(pred_results == self.test_data[:,-1])
        print('Accuracy in test set:%f' % accuracy)

def build_mnist_mlp(param_dir='weight.npy', train=False):
    h1,h2,e=256, 512, 16
    mlp=MNIST_MLP(batch_size=10000, hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_train_data('../mnist_data', 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    mlp.load_test_data('../mnist_data', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    mlp.build_model()
    mlp.init_model()

    if (train):
        mlp.train()
        mlp.save_model('mlp-%d-%d-%depoch.npy'%(h1,h2,e))
    else:
        mlp.load_model(param_dir)

    return mlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/evaluate MNIST MLP')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden1', type=int, default=32)
    parser.add_argument('--hidden2', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--param', type=str, default='weight.npy')
    parser.add_argument('--data-dir', type=str, default='../dataset/minist')
    parser.add_argument('--train', action='store_true', help='Train model if set, otherwise load and evaluate', default=True)
    args = parser.parse_args()

    mlp = MNIST_MLP(batch_size=args.batch_size, hidden1=args.hidden1, hidden2=args.hidden2, out_classes=10,
                    lr=args.lr, max_epoch=args.epochs)
    mlp.load_train_data('../dataset/minist', 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    mlp.load_test_data('../dataset/minist', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    mlp.build_model()
    mlp.init_model()

    if args.train:
        mlp.train()
        mlp.save_model('mlp-%d-%d-%depoch.npy'%(args.hidden1,args.hidden2,args.epochs))
    else:
        mlp.load_model(args.param)

    mlp.evaluate()
