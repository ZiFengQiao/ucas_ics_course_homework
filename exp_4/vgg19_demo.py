# -*- coding: UTF-8 -*- 
import pycnnl
import time
import numpy as np
import os
import scipy.io
from PIL import Image
import imageio.v2 as imageio

class VGG19(object):
    def __init__(self):
        # set up net
        
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []
        self.filter_quant_params = []


        self.param_layer_name = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
        )

    def build_model(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path

       
        # TODO: 使用net的createXXXLayer接口搭建VGG19网络
        # creating layers
        self.net.setInputShape(1, 3, 224, 224)
        # conv1_1

        ch_in = 3
        ch_out = 64
        output_size = 224
        for layer_name in self.param_layer_name:
            if 'relu' in layer_name:
                self.net.createReLuLayer(layer_name)
            elif 'pool' in layer_name:
                # pool layer
                input_shape = pycnnl.IntVector(4)
                input_shape[0] = 1              # batch size
                input_shape[1] = ch_in          # channel in
                input_shape[2] = output_size    # input height
                input_shape[3] = output_size    # input width

                kernel_size = 2
                stride = 2
                self.net.createPoolingLayer(layer_name, input_shape, kernel_size, stride)
                output_size = output_size // 2

            elif 'conv' in layer_name:
                # conv layer
                input_shape = pycnnl.IntVector(4)
                input_shape[0] = 1              # batch size
                input_shape[1] = ch_in          # channel in
                input_shape[2] = output_size    # input height
                input_shape[3] = output_size    # input width

                padding = 1
                kernel_size = 3
                stride = 1
                dilation = 1
                self.net.createConvLayer(layer_name, input_shape, ch_out, kernel_size, stride, dilation, padding)
            elif 'flatten' in layer_name:
                pass
            elif 'fc' in layer_name:
                if layer_name == 'fc8':
                    ch_out = 1000
                else:
                    ch_out = 4096

                # flatten 操作
                ch_in = output_size * output_size * ch_in
                output_size = 1

                # fc8
                input_shapem3 = pycnnl.IntVector(4)
                input_shapem3[0] = 1
                input_shapem3[1] = 1
                input_shapem3[2] = 1
                input_shapem3[3] = ch_in
                weight_shapem3 = pycnnl.IntVector(4)
                weight_shapem3[0] = 1
                weight_shapem3[1] = 1
                weight_shapem3[2] = ch_in
                weight_shapem3[3] = ch_out
                output_shapem3 = pycnnl.IntVector(4)
                output_shapem3[0] = 1
                output_shapem3[1] = 1
                output_shapem3[2] = 1
                output_shapem3[3] = ch_out

                self.net.createMlpLayer(layer_name, input_shapem3, weight_shapem3, output_shapem3)

            elif 'softmax' in layer_name:
                # softmax
                input_shapes = pycnnl.IntVector(3)
                input_shapes[0] = 1
                input_shapes[1] = 1
                input_shapes[2] = 1000
                self.net.createSoftmaxLayer('softmax', input_shapes, 1)
                print("create softmax layer")

            ch_in = ch_out
            if 'pool' in layer_name:
                ch_out = min(ch_out*2, 512)

    def load_model(self):
        # loading params ... 
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        
        count = 0
        for idx in range(self.net.size()):
            if 'conv' in self.net.getLayerName(idx):
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # TODO：调整权重形状
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [out_channel, height, width, in_channel]
                weight = np.transpose(weight, [3, 0, 1, 2]).flatten().astype(np.float32).tolist()
                bias = bias.flatten().astype(np.float32).tolist()
                self.net.loadParams(idx, weight, bias)
                count += 1
            if 'fc' in self.net.getLayerName(idx):
                # Loading params may take quite a while. Please be patient.
                weight, bias = params['layers'][0][idx][0][0][0][0]
                weight = weight.reshape([weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                weight = weight.flatten().astype(np.float64)
                bias = bias.reshape(-1).astype(np.float64)

                self.net.loadParams(idx, weight, bias)
                count += 1

    def load_image(self, image_dir):
        # loading image
        self.image = image_dir
        image_mean = np.array([123.68, 116.779, 103.939])
        print('Loading and preprocessing image from ' + image_dir)
        input_image = imageio.imread(image_dir)
        pil_img = Image.fromarray(input_image)
        pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        input_image = np.array(pil_img, dtype=np.float32)
        input_image -= image_mean
        input_image = np.reshape(input_image, [1]+list(input_image.shape))
        # TODO：调整输入数据
        input_data = input_image.flatten().astype(np.float64)

        self.net.setInputData(input_data)

    def forward(self):
        return self.net.forward()
    
    def get_top5(self, label):
        start = time.time()
        self.forward()
        end = time.time()

        result = self.net.getOutputData()

        # loading labels
        labels = []
        with open('../synset_words.txt', 'r') as f:
            labels = f.readlines()

        # print results
        top1 = False
        top5 = False
        print('------ Top 5 of ' + self.image + ' ------')
        prob = sorted(list(result), reverse=True)[:6]
        if result.index(prob[0]) == label:
            top1 = True
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if idx == label:
                top5 = True
            print('%f - '%top + labels[idx].strip())

        print('inference time: %f'%(end - start))
        return top1,top5
    
    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        start = time.time()
        with open(file_list, 'r') as f:
            file_list = f.readlines()
            total_num = len(file_list)
            for line in file_list:
                image = line.split()[0].strip()
                label = int(line.split()[1].strip())
                vgg.load_image(image)
                top1,top5 = vgg.get_top5(label)
                if top1 :
                    top1_num += 1
                if top5 :
                    top5_num += 1
        end = time.time()

        print('Global accuracy : ')
        print('accuracy1: %f (%d/%d) '%(float(top1_num)/float(total_num), top1_num, total_num))
        print('accuracy5: %f (%d/%d) '%(float(top5_num)/float(total_num), top5_num, total_num))
        print('Total execution time: %f'%(end - start))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    vgg.evaluate('../file_list')
