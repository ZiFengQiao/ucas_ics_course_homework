# ====================================================================
# File Name     : Makefile
# Author        : PLMaple
# Created Date  : 2026-05-26
# Description   : 自动化提取指定文件并压缩为不含目录结构的纯 ZIP 压缩包
# ====================================================================

# ----- 可配置参数 -----
# 目标压缩包名称
ZIP_NAME := archive.zip

# 需要打包的指定文件列表（支持跨目录、支持通配符）
# 即使这些文件位于子目录中，压缩后也会被平铺到 ZIP 的根目录下
TARGET_FILES := \
    ./mlu_custom_ext/mlu/src/bang_sigmoid.cpp \
    ./mlu_custom_ext/mlu/src/bang_sigmoid_sample.mlu \
    ./mlu_custom_ext/mlu_functions/mlu_functions.py \
    ./tests/test_sigmoid.py \
    ./tests/test_sigmoid_benchmark.py \

# ----- 核心编译伪目标 -----
.PHONY: all zip clean info

# 默认目标
all: zip

# 压缩目标
zip:
	@echo "=================================================="
	@echo "开始打包文件 (不保留目录结构)..."
	@echo "=================================================="
	@# -q: 静默模式
	@# -j: 丢弃路径信息 (Junk paths)，只保留文件本身
	@# 如果文件不存在，zip 命令会报错退出
	zip -qj $(ZIP_NAME) $(TARGET_FILES)
	@echo "打包完成 -> $(ZIP_NAME)"
	@echo "压缩包内文件列表:"
	@unzip -l $(ZIP_NAME) | sed '1,3d; /^-/d; /files$$/d'

# 查看当前配置的文件列表
info:
	@echo "当前指定打包的文件列表:"
	@for file in $(TARGET_FILES); do \
		if [ -f $$file ]; then \
			echo "  [存在] $$file"; \
		else \
			echo "  [缺失] $$file" ; \
		fi; \
	done

# 清理生成的压缩包
clean:
	@if [ -f $(ZIP_NAME) ]; then \
		rm -f $(ZIP_NAME); \
		echo "已清理 $(ZIP_NAME)"; \
	else \
		echo "没有发现可清理的压缩文件"; \
	fi