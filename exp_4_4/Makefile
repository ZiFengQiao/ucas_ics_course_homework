TARGETS = stu_upload/evaluate_cpu.py \
	stu_upload/test_mysigmoid.py \
	stu_upload/op_mysigmoid/mysigmoid.cpp \
	stu_upload/op_mysigmoid/setup.py

all: compress

# -j 选项表示在压缩文件中不包含目录结构，直接将文件放在压缩包的根目录下。
compress:
	zip -j upload.zip $(TARGETS)


clean:
	rm -f upload.zip