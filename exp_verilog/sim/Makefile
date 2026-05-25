# 定义模块名称和对应的测试文件
MODULES = inner_product_4x16 matrix_vector_mult_4x4x16 matrix_mult_4x4x4x16
TEST_FILES = tb_inner_product.cpp tb_matrix_vector.cpp tb_matrix_mult.cpp

# 源文件路径
SRC_DIR = ../src
OBJ_DIR = obj_dir

# Verilator 配置
VERILATOR = verilator
VERILATOR_FLAGS = --cc --exe --build -Wall --trace -I$(SRC_DIR)
CXXFLAGS = -std=c++14

# 默认目标：编译所有模块
all: $(MODULES)

# 内积运算器
inner_product_4x16:
	@echo "=== 创建编译目录: $(OBJ_DIR)/$@ ==="
	mkdir -p $(OBJ_DIR)/$@
	@echo "=== 编译内积运算器 ==="
	$(VERILATOR) $(VERILATOR_FLAGS) --Mdir $(OBJ_DIR)/$@ \
		$(SRC_DIR)/$@.v \
		tb_inner_product.cpp \
		--top-module $@ \
		-CFLAGS "$(CXXFLAGS)"

# 矩阵乘向量
matrix_vector_mult_4x4x16:
	@echo "=== 创建编译目录: $(OBJ_DIR)/$@ ==="
	mkdir -p $(OBJ_DIR)/$@
	@echo "=== 编译矩阵乘向量 ==="
	$(VERILATOR) $(VERILATOR_FLAGS) --Mdir $(OBJ_DIR)/$@ \
		$(SRC_DIR)/$@.v \
		$(SRC_DIR)/inner_product_4x16.v \
		tb_matrix_vector.cpp \
		--top-module $@ \
		-CFLAGS "$(CXXFLAGS)"

# 矩阵乘法
matrix_mult_4x4x4x16:
	@echo "=== 创建编译目录: $(OBJ_DIR)/$@ ==="
	mkdir -p $(OBJ_DIR)/$@
	@echo "=== 编译矩阵乘法 ==="
	$(VERILATOR) $(VERILATOR_FLAGS) --Mdir $(OBJ_DIR)/$@ \
		$(SRC_DIR)/$@.v \
		$(SRC_DIR)/matrix_vector_mult_4x4x16.v \
		$(SRC_DIR)/inner_product_4x16.v \
		tb_matrix_mult.cpp \
		--top-module $@ \
		-CFLAGS "$(CXXFLAGS)"

# 清理编译产物
clean:
	@echo "=== 清理编译产物 ==="
	rm -rf $(OBJ_DIR)
	rm -f *.vcd

# 伪目标
.PHONY: all clean $(MODULES)
