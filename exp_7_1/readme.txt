本实验为智能编程语言算子开发与集成实验，需要进行如下操作：

#1、补全mlu_custom_ext/mlu/src/bang_sigmoid_sample.mlu、mlu_custom_ext/mlu/src/bang_sigmoid.cpp/bang_sigmoid.cpp、/mlu_custom_ext/mlu_functions/mlu_functions.py、tests/test_sigmoid.py、tests/test_sigmoid_benchmark.py一系列文件

vim mlu_custom_ext/mlu/src/bang_sigmoid_sample.mlu
vim mlu_custom_ext/mlu/src/bang_sigmoid.cpp
vim /mlu_custom_ext/mlu_functions/mlu_functions.py
vim tests/test_sigmoid.py
vim tests/test_sigmoid_benchmark.py


#2、在根目录下完成自定义算子的编译

python setup.py install


#3、进入tests的目录进行精度和性能测试

cd tests
python test_sigmoid.py
python test_sigmoid_benchmark.py


