from setuptools import setup
from torch.utils import cpp_extension

setup(
    #TODO: 给出编译后的链接库名称
    name = "mysigmoid",
    ext_modules=[
        cpp_extension.CppExtension(
            name = "mysigmoid",
            sources = ["mysigmoid.cpp"]
        )
    ],
    # 执行编译命令设置
    cmdclass={						       
        'build_ext': cpp_extension.BuildExtension.with_options()
    }
)

print("generate .so PASS!\n")
