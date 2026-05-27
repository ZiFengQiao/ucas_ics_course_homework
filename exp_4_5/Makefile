TARGETS = AttModel.py \
		  eval.py \
		  modules.py \

all: compress

# -j 选项表示在压缩文件中不包含目录结构，直接将文件放在压缩包的根目录下。
compress:
	zip -j upload.zip $(TARGETS)


clean:
	rm -f upload.zip