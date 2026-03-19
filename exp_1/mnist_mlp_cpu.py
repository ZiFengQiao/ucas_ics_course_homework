#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
File Name     : mnist_mlp_cpu.py  
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

import struct
import os
import numpy as np
from layers_1 import *


class Dataset(object):
    def __init__(self, minst_dir, data_file, label_file):
        self._minst_dir = minst_dir
        self._data_file = data_file
        self._label_file = label_file

        _data = self.load_mnist(os.path.join(minst_dir, data_file), is_images=True)
        _label = self.load_mnist(os.path.join(minst_dir, label_file), is_images=False)

        self.data_with_label = np.append(_data, _label, axis=1)


    def load_mnist(self, file_dir, is_images=True):
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
    def __init__(self, batch_size=64,
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
        self._train_data = None
        self._test_data = None

        # build_model
        self.layers = []
        self.update_layers = []

        self._build_model()

    def load_train_data(self, minst_dir, data_file, label_file):
        self._train_data = Dataset(minst_dir, data_file, label_file)

    def load_test_data(self, minst_dir, data_file, label_file):
        self._test_data = Dataset(minst_dir, data_file, label_file)

    def _build_model(self):
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

    def backward(self, label):
        top_diff = label
        for layer in reversed(self.layers):
            top_diff = layer.backward(top_diff)

    def update_param(self):
        for layer in self.update_layers:
            layer.update_param(self.lr)

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

    def train(self):
        assert self._train_data is not None, "Please load training data before training."

        max_batch = self._train_data.shape[0] // self.batch_size

        for idx_epoch in range(self.max_epoch):
            self._train_data.shuffle_data()

            for idx_batch in range(max_batch):
                batch_images = self._train_data.data_with_label[idx_batch * self.batch_size : (idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self._train_data.data_with_label[idx_batch * self.batch_size : (idx_batch + 1) * self.batch_size, -1]

                prob = self.forward(batch_images)
                loss = self.layers[-1].get_loss(batch_labels)

                self.backward()
                self.update_param()
                self.backward(label)
                self.update_param()

                if (idx_batch + 1) % self.print_iter == 0:
                    print(f"Epoch [{idx_epoch + 1}/{self.max_epoch}], Batch [{idx_batch + 1}/{max_batch}]")


if __name__ == '__main__':

    train_data = Dataset('./minist', 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    test_data = Dataset('./minist', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')



    print(len(train_data.data_with_label))