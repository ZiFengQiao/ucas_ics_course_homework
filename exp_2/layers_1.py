import sys
import numpy as np
import struct
import os
import time


def show_matrix(mat, name):
    # print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass


def show_time(time, name):
    # print(name + str(time))
    pass


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')

    def forward(self, input):
        start_time = time.time()
        self.input = input
        self.output = np.dot(input, self.weight) + self.bias
        show_time(time.time() - start_time, 'fc forward time: ')
        return self.output

    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)    # 权值梯度
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)     # 偏置梯度
        bottom_diff = np.dot(top_diff, self.weight.T)          # 向前传递的梯度

        return bottom_diff

    def get_gradient(self):
        return self.d_weight, self.d_bias

    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')

    def save_param(self):
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
        return self.weight, self.bias


class ReLULayer(object):
    def __init__(self):
        print('\t Relu layer')

    def forward(self, input):
        start_time = time.time()
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff.copy()
        bottom_diff[self.input < 0] = 0
        return bottom_diff


class SoftmaxLossLayer(object):
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        exp_sum = np.sum(input_exp, axis=1, keepdims=True)
        self.prob = input_exp / exp_sum
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff





