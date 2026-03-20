#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------
File Name     : layers_1.py  
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
import numpy as np

class Layer(object):
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, top_diff):
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    def __init__(self, num_input, num_output):
        self._num_input = num_input
        self._num_output = num_output

        # param
        self._weight = None
        self._bias = None

        # forward
        self.current_input = None
        self.current_output = None

        # backward
        self._current_weight_diff = None
        self._current_bias_diff = None
        self._current_param_need_update = False

    def init_param(self, std=0.01):
        # 初始化权重和偏置，权重使用正态分布随机初始化，偏置初始化为零

        self._weight = np.random.normal(loc=0.0, scale=std, size=(self._num_input, self._num_output))
        self._bias = np.zeros([1, self._num_output])

    def forward(self, input_data):
        # 点积运算，输入数据与权重矩阵相乘，并加上偏置项

        output = np.dot(input_data, self._weight) + self._bias

        self.current_input = input_data
        self.current_output = output
        return output

    def backward(self, top_diff):
        # 反向传播，计算权重和偏置的梯度
        # 向量转置后进行点积运算，计算权重的梯度；对偏置的梯度进行求和
        weight_diff = np.dot(self.current_input.T, top_diff)    # 权值梯度
        bias_diff = np.sum(top_diff, axis=0, keepdims=True)     # 偏置梯度
        bottom_diff = np.dot(top_diff, self._weight.T)          # 向前传递的梯度

        self._current_weight_diff = weight_diff
        self._current_bias_diff = bias_diff
        self._current_param_need_update = True

        return bottom_diff

    def update_param(self, learning_rate):
        # 更新权重和偏置，使用梯度下降法更新参数
        assert self._current_param_need_update, "Please call backward() before update_param()."

        self._weight -= learning_rate * self._current_weight_diff
        self._bias -= learning_rate * self._current_bias_diff

    def load_param(self, weight, bias):
        self._weight = weight
        self._bias = bias

    def save_param(self):
        return self._weight, self._bias

class ReLULayer(Layer):
    def __init__(self):
        # forward
        self._current_input = None
        self._current_output = None

        # backward
        self._current_top_diff = None

    def forward(self, input_data):
        output = np.maximum(0, input_data)

        self._current_input = input_data
        self._current_output = output
        return output

    def backward(self, top_diff):
        # bottom_diff = top_diff * 1 (x > 0)
        bottom_diff = np.where(self._current_input > 0, top_diff, 0)
        return bottom_diff

class SoftmaxLossLayer(Layer):
    def __init__(self):
        # forward
        self._current_input = None
        self._current_output = None

    def forward(self, input_data):
        input_exp = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)

        self._current_input = input_data
        self._current_output = prob
        return prob

    def backward(self):
        # 计算softmax层的梯度，使用交叉熵损失函数的梯度公式
        return self._current_top_diff

    def get_loss(self, label):
        batch_size = self._current_output.shape[0]

        # 计算交叉熵损失，使用softmax输出和标签之间的交叉熵损失公式
        label_one_hot = np.zeros_like(self._current_output)
        label_one_hot[np.arange(label.shape[0]), label] = 1.0

        self._current_top_diff = (self._current_output - label_one_hot) / batch_size
        loss = -np.sum(label_one_hot * np.log(self._current_output)) / batch_size
        return loss
