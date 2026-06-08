"""
Fused Attention
===============
 
This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team
 
Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
 
"""

import torch
import torch_mlu

import triton
import triton.language as tl
import time
import numpy as np




@triton.jit
def _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        start_m,
        qk_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,
        N_CTX: tl.constexpr,
        IS_DIVISIBLE: tl.constexpr):
    """

    :param acc:             传递的累加结果
    :param l_i:             传递的累积和
    :param m_i:             传递的最大值
    :param q:               Q矩阵块
    :param K_block_ptr:     K矩阵块指针
    :param V_block_ptr:     V矩阵块指针
    :param start_m:         当前处理的Q块的起始位置，用于计算偏移和掩码
    :param qk_scale:        缩放因子，通常是1/sqrt(d_model)，用于调整注意力分数的范围
    :param BLOCK_M:         每次处理多少行Q
    :param BLOCK_DMODEL:    处理块的特征维度大小
    :param BLOCK_N:         每次处理多少列K/V
    :param STAGE:           当前处理阶段，1表示处理off-band，2表示处理on-band，3表示同时处理两者
    :param offs_m:          m偏移
    :param offs_n:          n偏移
    :param N_CTX:           QKV特征维度
    :param IS_DIVISIBLE:
    :return:
    """


    # 1. 根据STAGE确定处理范围，这是Flash Attention的核心分块策略
    if STAGE == 1:
        # stage1 处理KV-cache的off-band部分
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # stage2 处理当前块的数据，即on-band部分，使用 tl.multiple_of 确保起始位置是 BLOCK_M 的倍数，这是 Triton 内存对齐的关键
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # stage3 处理所有数据，适用于非因果注意力，直接设置范围为整个序列长度
        lo, hi = 0, N_CTX

    # 2. 使用tl.advance移动块指针，这是Triton内存访问的关键
    K_block_ptr = tl.advance(K_block_ptr, offs_m)  # tl.advance 用于移动块指针，第一个参数是行偏移，第二个参数是列偏移
    V_block_ptr = tl.advance(V_block_ptr, offs_n)  # tl.advance 用于移动块指针，第一个参数是行偏移，第二个参数是列偏移

    # 3. 主循环：按BLOCK_N大小处理K和V
    for start_n in range(lo, hi, BLOCK_N):  # 按 BLOCK_N 大小遍历 K 和 V 矩阵，这是分块处理的核心循环
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # 4. 加载K矩阵，使用条件边界检查
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")   # 当数据不能整除时，使用边界检查和零填充来处理边界情况

        # 5. 计算QK注意力分数，使用tl.dot进行矩阵乘法
        qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk += tl.dot(q, k)  # 使用 tl.dot 进行矩阵乘法，计算 QK 注意力分数

        # 6. 根据STAGE应用不同的注意力掩码策略
        if STAGE == 2:
            mask = offs_m[None, :] >= (start_n + offs_n[:, None])
            qk = qk * qk_scale + tl.where(mask, 0, float("-inf"))  # 创建因果掩码，使用 tl.where 条件选择，将未来位置设为负无穷
            # tl.where 返回由0和-inf组成的矩阵
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            qk -= m_ij[None, :]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 0) * qk_scale)
            qk = qk * qk_scale - m_ij[None, :]

        # 7. 使用tl.math.exp2计算指数，避免数值溢出
        p = tl.math.exp2(qk)  # 使用 tl.math.exp2 而不是 tl.exp，避免数值溢出，提高数值稳定性
        l_ij = tl.sum(p, 0)

        # 8. 更新累积器，使用数值稳定的更新策略
        alpha = tl.exp(m_i - m_ij)  # 使用数值稳定的更新策略，通过指数变换避免大数相减导致的精度损失
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # 9. 转换为float16以节省内存带宽
        qk_wram = qk.to(tl.float16)  # 转换为 float16 以节省内存带宽，这是 Triton 性能优化的常用技巧
        qkv = tl.dot(v, qk_wram)

        # 10. 更新输出累积器和统计信息
        acc = acc * alpha[None, :]
        acc += qkv
        m_i = m_ij
        l_i = l_i * alpha + l_ij  # 更新累积和，用于后续的 softmax 归一化

        # 11. 移动指针到下一个块
        V_block_ptr = tl.advance(V_block_ptr, offs_m)  # 移动到下一个数据块，V 指针在列方向移动，K 指针在行方向移动
        K_block_ptr = tl.advance(K_block_ptr, offs_n)  # 移动到下一个数据块，V 指针在列方向移动，K 指针在行方向移动

    return acc, l_i, m_i


@triton.jit
def _attn_eff_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        start_m,
        qk_scale,
        Mask_block_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,
        N_CTX: tl.constexpr,
        IS_DIVISIBLE: tl.constexpr):

    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, lo))
    Mask_block_ptr = tl.advance(Mask_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):         # 处理 kv上的 (N_CTX/BLOCK_N) 数据
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
            mask = tl.load(Mask_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            mask = tl.load(Mask_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk += tl.dot(k, q)

        if STAGE == 2:
            qk = qk * qk_scale + mask*1.44269504
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            qk -= m_ij[None, :]
        else:
            qk = qk * qk_scale + mask*1.44269504
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            qk -= m_ij[None, :]

        p = tl.math.exp2(qk)  # 求指数
        l_ij = tl.sum(p, 0)   # 求和
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk_wram = p.to(tl.float16)
        qkv = tl.dot(v, qk_wram)
        # -- update output accumulator --
        acc = acc * alpha[None, :]
        acc += qkv
        m_i = m_ij
        l_i = l_i * alpha + l_ij  # 更正后的上一个+本次
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i

@triton.jit
def _attn_eff_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        stride_mm,
        stride_mn,
        Z,
        H,
        causal_mask,
        N_CTX,
        Q_N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        STAGE: tl.constexpr,
        IS_DIVISIBLE: tl.constexpr):
    core_id = tl.program_id(0)
    core_dim = tl.num_programs(axis=0)
    cluster_id = tl.program_id(1)
    cluster_dim = tl.num_programs(axis=1)

    context_num = tl.cdiv(Q_N_CTX, BLOCK_M)  # 向上取整
    total_heads = Z * H * context_num
    task_heads = total_heads // cluster_dim
    task_remain_heads = total_heads - task_heads * cluster_dim
    task_heads += 1
    task_head_begin = cluster_id * task_heads
    if cluster_id >= task_remain_heads:
        task_heads -= 1
        task_head_begin = cluster_id * task_heads + task_remain_heads
    else:
        pass
    if task_heads <= 0:
        return

    core_heads = task_heads // core_dim
    core_remain_heads = task_heads - core_heads * core_dim
    core_heads += 1
    core_head_begin = core_id * core_heads
    if core_id >= core_remain_heads:
        core_heads -= 1
        core_head_begin = core_id * core_heads + core_remain_heads
    if core_heads <= 0:
        return
    head_begin = task_head_begin + core_head_begin
    head_end = head_begin + core_heads

    for head_idx in range(head_begin, head_end):  # 一个core处理 q上的 (Q_N_CTX/BLOCK_M) 数据
        start_m = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh  # add by zhangkun 20240627
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, Q_N_CTX),
            strides=(stride_qk, stride_qm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        Mask_block_ptr = tl.make_block_ptr(
            base=causal_mask,
            shape=(N_CTX, Q_N_CTX),
            strides=(stride_mn, stride_mm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(Q_N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/ln(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            acc, l_i, m_i = _attn_eff_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                Mask_block_ptr,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
                IS_DIVISIBLE)
        # stage 2: on-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 2 as its STAGE
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            tl.debug_barrier()
            acc, l_i, m_i = _attn_eff_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                Mask_block_ptr,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                2,
                offs_m,
                offs_n,
                N_CTX,
                IS_DIVISIBLE)
        # epilogue
        m_i += tl.math.log2(l_i)
        l_i_recip = 1 / l_i
        acc = acc * l_i_recip[None, :]
        acc = tl.trans(acc)
        m_ptrs = M + off_hz * Q_N_CTX + offs_m
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))

@triton.jit
def _attn_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        Z,
        H,
        N_CTX,
        Q_N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        STAGE: tl.constexpr,
        IS_DIVISIBLE: tl.constexpr):

    # 12. 获取程序ID，用于任务分配
    core_id = tl.program_id(0)          # 获取当前程序在网格中的位置和总程序数，用于任务分配
    core_dim = tl.num_programs(axis=0)  # 获取当前程序在网格中的位置和总程序数，用于任务分配
    cluster_id = tl.program_id(1)
    cluster_dim = tl.num_programs(axis=1)

    # 13. 计算上下文数量，使用tl.cdiv进行向上取整
    context_num = tl.cdiv(N_CTX, BLOCK_M)             # 使用 tl.cdiv 进行向上取整，计算需要处理的上下文块数量
    total_heads = Z * H * context_num

    # 14. 任务分配：将总任务分配给不同的cluster和core
    task_heads = total_heads // cluster_dim         # 将总任务平均分配给不同的 cluster，处理余数分配
    task_remain_heads = total_heads % cluster_dim   # 将总任务平均分配给不同的 cluster，处理余数分配
    task_heads += 1
    task_head_begin = cluster_id * task_heads
    if cluster_id >= task_remain_heads:
        task_heads -= 1
        task_head_begin = cluster_id * task_heads + task_remain_heads
    else:
        pass
    if task_heads <= 0:
        return

    core_heads = task_heads // core_dim
    core_remain_heads = task_heads - core_heads * core_dim
    core_heads += 1
    core_head_begin = core_id * core_heads
    if core_id >= core_remain_heads:
        core_heads -= 1
        core_head_begin = core_id * core_heads + core_remain_heads
    if core_heads <= 0:
        return
    head_begin = task_head_begin + core_head_begin
    head_end = head_begin + core_heads

    # Z是batchsize
    # H是head数
    # Q_N_CTX是query的序列长度，N_CTX是key和value的序列长度
    # 所以，这里是分头处理Q
    for head_idx in range(head_begin, head_end):  # 一个core处理 q上的 (Q_N_CTX/BLOCK_M) 数据
        start_m = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H

        # 15. 计算偏移量，使用to(tl.int64)进行类型转换
        q_offset = off_z.to(tl.int64)  # 使用 to(tl.int64) 进行类型转换，确保偏移量计算不会溢出
        kv_offset = off_h.to(tl.int64)  # 使用 to(tl.int64) 进行类型转换，确保偏移量计算不会溢出

        # 16. 创建块指针，这是Triton高效内存访问的核心
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, Q_N_CTX),
            strides=(stride_qk, stride_qm),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )  # 创建块指针，指定基地址、形状、步长、偏移、块形状和访问顺序
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(Q_N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )

        # 17. 初始化偏移量数组，使用tl.arange生成序列
        offs_m = tl.arange(BLOCK_M)  # 使用 tl.arange 生成连续的索引序列，用于后续的掩码计算
        offs_n = tl.arange(BLOCK_N)  # 使用 tl.arange 生成连续的索引序列，用于后续的掩码计算

        # 18. 初始化累积器和统计变量
        m_i = float("-inf")  # 初始化最大值和累积和，使用负无穷和1.0作为初始值
        l_i = 1.0            # 初始化最大值和累积和，使用负无穷和1.0作为初始值
        acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)

        # 19. 加载缩放因子，使用数值优化
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)

        # 加载Q矩阵
        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # 20. 分阶段处理：STAGE 1处理off-band，STAGE 2处理on-band
        if STAGE & 1:  # 使用位运算 STAGE & 1 判断是否需要执行第一阶段，这是 Flash Attention 的核心分阶段策略
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
                IS_DIVISIBLE)
        if STAGE & 2:
            tl.debug_barrier()  # 同步点，便于编译器优化
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                start_m,
                qk_scale,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                2,
                offs_m,
                offs_n,
                N_CTX,
                IS_DIVISIBLE)
        # 后处理：归一化输出
        m_i += tl.math.log2(l_i)
        l_i_recip = 1 / l_i
        acc = acc * l_i_recip[None, :]
        acc = tl.trans(acc)

        # 存储结果
        m_ptrs = M + off_hz * Q_N_CTX + offs_m
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))



empty = torch.empty(128, device="mlu")

FLASH_ATT_MODEL_PASSED = False
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, causal_mask):
        # 形状约束检查
        global FLASH_ATT_MODEL_PASSED
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv

        o = torch.empty_like(q)
        q_ctx_len = q.shape[-2]
        kv_ctx_len = k.shape[-2]

        # 动态调整块大小
        if q_ctx_len <= 128:
            if kv_ctx_len % 128 == 0:
                BLOCK_M = q_ctx_len
                BLOCK_N = 128
            else:
                BLOCK_M = q_ctx_len
                BLOCK_N = kv_ctx_len
        elif q_ctx_len < 256:
            if kv_ctx_len % 64 == 0:
                BLOCK_M = q_ctx_len
                BLOCK_N = 64
            else:
                BLOCK_M = q_ctx_len
                BLOCK_N = kv_ctx_len
        elif q_ctx_len >= 256:
            if kv_ctx_len % 64 == 0:
                BLOCK_M = 64
                BLOCK_N = 64
            elif kv_ctx_len % 32 == 0:
                BLOCK_M = 32
                BLOCK_N = 32
            else:
                BLOCK_M = 64
                BLOCK_N = kv_ctx_len

        # 对齐优化
        align_num = 64
        while BLOCK_M >= 512:
            BLOCK_M = (BLOCK_M // 2 + align_num - 1) // align_num * align_num
        while BLOCK_N >= 512:
            BLOCK_N = (BLOCK_N // 2 + align_num - 1) // align_num * align_num

        # 配置参数
        num_stages = 4 if Lk <= 64 else 3
        num_warps = 1
        stage = 3 if causal else 1
        if torch.mlu.get_device_capability()[0] == 9:
            num_warps = 8
            num_stages = 0
        num_stages = 0

        # 网格配置
        grid = (4, 8, 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]),
                        device=q.device, dtype=torch.float32)

        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0

        N_CTX = q.shape[2]
        IS_DIVISIBLE = False
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True

        # 调用内核
        if causal_mask is not None:
            _attn_eff_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                causal_mask.stride(2),
                causal_mask.stride(3),
                q.shape[0],
                q.shape[1],
                causal_mask,
                N_CTX=k.shape[2],
                Q_N_CTX=q.shape[2],
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=Lk,
                STAGE=stage,
                IS_DIVISIBLE=IS_DIVISIBLE,
                num_warps=num_warps,
                num_stages=num_stages
            )
        else:
            _attn_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                M,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                q.shape[0],
                q.shape[1],
                N_CTX=k.shape[2],
                Q_N_CTX=q.shape[2],
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=Lk,
                STAGE=stage,
                IS_DIVISIBLE=IS_DIVISIBLE,
                num_warps=num_warps,
                num_stages=num_stages
            )
        if not FLASH_ATT_MODEL_PASSED:
            print("FlashAtt Module PASS!")
            FLASH_ATT_MODEL_PASSED = True
        return o
attention = _attention.apply


# 测试函数
def test_op():
    torch.manual_seed(20)
    Z, H, N_CTX, D_HEAD = 1, 128, 257, 16
    causal = False
    causal_mask = []
    dtype = torch.float16
    use_data_from_file = False
    Q_N_CTX = N_CTX

    for n in range(N_CTX, N_CTX + 1):
        if not use_data_from_file:
            q = (torch.empty((Z, H, Q_N_CTX, D_HEAD), dtype=dtype,
                         device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
            k = (torch.empty((Z, H, n, D_HEAD), dtype=dtype,
                         device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
            v = (torch.empty((Z, H, n, D_HEAD), dtype=dtype,
                         device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
            if causal_mask is not None:
                causal_mask = (torch.empty((Z, 1, Q_N_CTX, n), dtype=dtype,
                             device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
        else:
            q_np = np.fromfile("query_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
            k_np = np.fromfile("key_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
            v_np = np.fromfile("value_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)

            q = torch.from_numpy(q_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_(True)
            k = torch.from_numpy(k_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_(True)
            v = torch.from_numpy(v_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_(True)

        sm_scale = 0.5
        dout = torch.randn_like(q)

        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)
        print("causal:", causal)

        # Triton实现
        tri_times = []
        for i in range(100):
            st = time.time()
            tri_out = attention(q, k, v, causal, sm_scale, causal_mask).half()
            ed = time.time()
            tri_times.append(ed - st)
            #print(f"triton run {i+1} cost: {ed - st:.6f}s")
        tri_avg_time = sum(tri_times[2:]) / (len(tri_times)-2)
        tri_out=tri_out.flatten()
        nan_mask = torch.isnan(tri_out)
        has_nan = torch.any(nan_mask)
        print("triton attention avg cost:", tri_avg_time)

        # SDPA实现
        sdpa_times = []
        for i in range(100):
            st = time.time()
            if causal_mask is not None:
                causal = False
            sdpa_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, dropout_p=0.0, is_causal=causal, scale=sm_scale)
            ed = time.time()
            sdpa_times.append(ed - st)
            #print(f"sdpa run {i+1} cost: {ed - st:.6f}s")
        sdpa_avg_time = sum(sdpa_times[2:]) / (len(sdpa_times)-2)
        sdpa_output=sdpa_output.flatten()
        print("scaled_dot_product_attention avg cost:", sdpa_avg_time)



        pytorch_valible=True
        if(pytorch_valible==True):
            # pytorch的实现
            pyt_times = []
            for i in range(100):
                st=time.time()
                M = torch.tril(torch.ones((Q_N_CTX, N_CTX), device="mlu"))
                qk = torch.matmul(q, k.transpose(2, 3)) 
                p=qk*sm_scale
                if(causal_mask is not None):
                    p=p+causal_mask
                elif causal:
                    p[:, :, M == 0] = float("-inf")
                
                if(1):
                    p = torch.softmax(p.float(), dim=-1).half()
                    pyt_out = torch.matmul(p, v)
                    
                else:
                    qkv=torch.matmul(p, v)
                    print("qkv:",qkv)
                    exp_p=torch.exp(p)
                    print("exp_p:",exp_p)
                    sum_exp_p = torch.sum(exp_p, dim=-1, keepdim=True)
                    print("sum_exp_p:",sum_exp_p)
                    softmax=exp_p/sum_exp_p
                    print("softmax:",softmax)
                    pyt_out = torch.matmul(softmax, v)
                #print("pyt_out:",pyt_out)
                pyt_out=pyt_out.flatten()
                ed=time.time()
                pyt_times.append(ed - st)
                #print(f"pytorch run {i+1} cost: {ed - st:.6f}s")
            pyt_avg_time = sum(pyt_times[2:]) / (len(pyt_times)-2)
            print("pytorch attention avg cost:",pyt_avg_time)
            
            # compare
            abs_tp_error = torch.sum(torch.abs(pyt_out - tri_out))
            rel_tp_error = torch.sum(torch.abs(pyt_out - tri_out)) / torch.sum(torch.clamp(torch.abs(pyt_out) + torch.abs(tri_out), min=0.00000001))
            print(f"abs_tp_error: {abs_tp_error.item():.4f}")
            print(f"rel_tp_error: {rel_tp_error.item():.6f}")
            abs_sp_error = torch.sum(torch.abs(pyt_out - sdpa_output))
            rel_sp_error = torch.sum(torch.abs(pyt_out - sdpa_output)) / torch.sum(torch.clamp(torch.abs(pyt_out) + torch.abs(sdpa_output), min=0.00000001))
            print(f"abs_sp_error: {abs_sp_error.item():.4f}")
            print(f"rel_sp_error: {rel_sp_error.item():.6f}")
        
        abs_ts_error = torch.sum(torch.abs(sdpa_output - tri_out))
        rel_ts_error = torch.sum(torch.abs(sdpa_output - tri_out)) / torch.sum(torch.clamp(torch.abs(sdpa_output) + torch.abs(tri_out), min=0.00000001))
        print(f"abs_ts_error: {abs_ts_error.item():.4f}")
        print(f"rel_ts_error: {rel_ts_error.item():.6e}")

        try:
            if (tri_avg_time < sdpa_avg_time) and (tri_avg_time < pyt_avg_time):
                print("time PASS!")   
            else:
                print("time FAIL!")
        except Exception as e:
            print("time check error:", e)

        try:
            rel_tp_ok = (rel_tp_error is not None) and (float(rel_tp_error.item()) < 1e-3)
            rel_ts_ok = (rel_ts_error is not None) and (float(rel_ts_error.item()) < 1e-3)
            if rel_tp_ok and rel_ts_ok:
                print("err PASS!")    
            else:
                print("err FAIL!")
        except Exception as e:
            print("error check error:", e)

        print("testop PASS!")



TORCH_HAS_FP8=False
BATCH, N_HEADS, KV_N_CTX, D_HEAD = 1, 2, 1024, 64
Q_N_CTX=1
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["triton"] + ["flash"],
                line_names=["Triton"] + ["Flash"],
                styles=[("red", "-"), ("blue", "-")],
                ylabel="ms",
                plot_name=f"fused-attention_opt-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal_{causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "D_HEAD": D_HEAD,
                    "dtype": torch.float16,
                    "mode": mode,
                    "causal": causal,
                },
            ))
 
 
if __name__ == '__main__':
    print("====================== 填空题测试 =======================")
    test_op() 
