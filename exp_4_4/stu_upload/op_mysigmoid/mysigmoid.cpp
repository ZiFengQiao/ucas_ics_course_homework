//Pytorch扩展头文件的引用
#include <torch/extension.h>
using namespace std;

//mysigmoid_cpu函数的具体实现
torch::Tensor mysigmoid(const torch::Tensor & dets) {
    //TODO: 将输入的tensor转化为浮点类型的vector
    auto input_data = dets.data_ptr<float>();
    int input_size = dets.numel();

    //TODO: 创建一个浮点类型的output_data，output_data为大小与输入相同的vector
    vector <float> output_data(input_size);

    //TODO: 对于输入向量的每个元素计算mysigmoid
    for (int i = 0; i < input_size; i++) {
        output_data[i] = 1 / (1 + exp(-input_data[i]));
    }

    //TODO: Create tensor options with dtype float32
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);

    //TODO: Create a tensor from the output vector
    auto foo= torch::from_blob(output_data.data(), {int64_t(output_data.size())}, opts).clone();

    //TODO: 将得到的tensor转换成所需的大小
    auto output = foo.view_as(dets);
    return output;
}

//TODO: 算子绑定为Pytorch的模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	// 绑定部分
    m.def("mysigmoid", &mysigmoid, "mysigmoid");
}       
