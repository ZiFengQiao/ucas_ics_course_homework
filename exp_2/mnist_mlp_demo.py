import pycnnl
import time
import numpy as np
import struct
import os

class MNIST_MLP(object):
    def __init__(self):
        # set up net
        
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []  
        self.filter_quant_params = []
    
    def build_model(self, batch_size=10000, input_size=784, 
                    hidden_list=None, out_classes=10):

        if hidden_list == None:
            hidden_list = [256, 128]

        self.batch_size = batch_size
        self.out_classes = out_classes
        

        # creating layers
        self.net.setInputShape(batch_size, input_size, 1, 1) #设置输入参数

        # fc1
        last_hidden = input_size

        for idx, hidden in enumerate(hidden_list):
            input_shapem     = pycnnl.IntVector(4)
            input_shapem[0]  = batch_size
            input_shapem[1]  = 1
            input_shapem[2]  = 1
            input_shapem[3]  = last_hidden

            weight_shapem    = pycnnl.IntVector(4)
            weight_shapem[0] = batch_size
            weight_shapem[1] = 1
            weight_shapem[2] = last_hidden
            weight_shapem[3] = hidden

            output_shapem    = pycnnl.IntVector(4)
            output_shapem[0] = batch_size
            output_shapem[1] = 1
            output_shapem[2] = 1
            output_shapem[3] = hidden
    
            self.net.createMlpLayer(f'fc{idx}', input_shapem, weight_shapem, output_shapem)

            self.net.createReLuLayer(f'relu{idx}')



        # last fc layer
        input_shapem = pycnnl.IntVector(4)
        input_shapem[0] = batch_size
        input_shapem[1] = 1
        input_shapem[2] = 1
        input_shapem[3] = last_hidden

        weight_shapem = pycnnl.IntVector(4)
        weight_shapem[0] = batch_size
        weight_shapem[1] = 1
        weight_shapem[2] = last_hidden
        weight_shapem[3] = out_classes

        output_shapem = pycnnl.IntVector(4)
        output_shapem[0] = batch_size
        output_shapem[1] = 1
        output_shapem[2] = 1
        output_shapem[3] = out_classes
        self.net.createMlpLayer(f'fc{len(hidden_list)}', input_shapem, weight_shapem, output_shapem)

        # softmax
        input_shape_softmax=pycnnl.IntVector(3)
        input_shape_softmax[0]=batch_size
        input_shape_softmax[1]=1
        input_shape_softmax[2]=out_classes
        self.net.createSoftmaxLayer('softmax', input_shape_softmax, 1)

    def load_mnist(self, file_dir, is_images = True):
        # Read binary data
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        # Analysis file header
        if is_images:
            # Read images
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            # Read labels
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        try:
            mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        except:
            print('Error in unpacking data from file %s'%file_dir)
            raise
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
        return mat_data
    
    def load_data(self, data_path, label_path):
        print('Loading MNIST data from files...')
        test_images = self.load_mnist(data_path, is_images=True)
        test_labels = self.load_mnist(label_path, is_images=False)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        
        params = np.load(param_dir,allow_pickle=True,encoding="latin1").item()
        weigh1 = params['fc1_weight'].flatten().astype(np.float64)
        bias1 = params['fc1_bias'].flatten().astype(np.float64)
        self.net.loadParams(0, weigh1, bias1)

        weigh2 = params['fc2_weight'].flatten().astype(np.float64)
        bias2 = params['fc2_bias'].flatten().astype(np.float64)
        self.net.loadParams(2, weigh2, bias2)

        weigh3 = params['fc3_weight'].flatten().astype(np.float64)
        bias3 = params['fc3_bias'].flatten().astype(np.float64)
        self.net.loadParams(4, weigh3, bias3)
    
    def forward(self):
        return self.net.forward()

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(self.test_data.shape[0]//self.batch_size):
            # print("batch %d"%idx)
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            data = batch_images.flatten().tolist()
            self.net.setInputData(data)
            start = time.time()
            self.forward()
            end = time.time()
            print('inferencing time: %f'%(end - start))
            prob = self.net.getOutputData()
            prob = np.array(prob).reshape((self.batch_size, self.out_classes))
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        if self.test_data.shape[0] % self.batch_size >0: 
            last_batch = self.test_data.shape[0]//self.batch_size*self.batch_size
            batch_images = self.test_data[-last_batch:, :-1]
            data = batch_images.flatten().tolist()
            self.net.setInputData(data)
            self.forward()
            prob = self.net.getOutputData()
            pred_labels = np.argmax(prob, axis=1)
            pred_results[-last_batch:] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:,-1])
        print('Accuracy in test set: %f' % accuracy)

OUT = 10
def run_mnist():
    batch_size = 10000
    hidden_list = [256, 256, 256, 64, 16]

    c = OUT
    mlp = MNIST_MLP()
    mlp.build_model(batch_size=batch_size, hidden_list=hidden_list, out_classes=c)
    model_path = 'weight.npy'
    test_data = '../../mnist_data/t10k-images-idx3-ubyte'
    test_label = '../../mnist_data/t10k-labels-idx1-ubyte'

    mlp.load_data(test_data, test_label)

    mlp.load_model(model_path)

    for i in range(10):
        mlp.evaluate()

if __name__ == '__main__':
    run_mnist()
