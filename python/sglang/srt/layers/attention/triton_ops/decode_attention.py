# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging

import triton
import triton.language as tl

from sglang.srt.utils import is_hip

is_hip_ = is_hip()

# logger = logging.getLogger(__name__)

# # TODO: Remove this when triton>=3.2.0. This issue will not affect performance and accuracy.
# logger.warning(
#     "The following error message 'operation scheduled before its operands' can be ignored."
# )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)
            # acc

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if is_hip_:
        BLOCK = 8
    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if is_hip_:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )

@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr, # tensor([ 0,  5, 10], device='xpu:0', dtype=torch.int32)
    kv_indices, # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device='xpu:0')
    Att_Out,
    stride_qbs, # 1024
    stride_qh, # 64
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    tmp_p,
    stride_x,
    stride_y,
    tmp_v,
    v_stride_x,
    v_stride_y,
    tmp_q,
    q_stride_x,
    q_stride_y,
    tmp_k,
    k_stride_x,
    k_stride_y,
    tmp_qk,
    qk_stride_x,
    qk_stride_y,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    BX : tl.constexpr,
    BY : tl.constexpr,
    BZ : tl.constexpr,
    P_BLCK_NROWS : tl.constexpr,
    P_BLCK_NCOLS : tl.constexpr,
    V_BLCK_NROWS : tl.constexpr,
    V_BLCK_NCOLS : tl.constexpr,
    Q_BLCK_NROWS : tl.constexpr,
    Q_BLCK_NCOLS : tl.constexpr,
    K_BLCK_NROWS : tl.constexpr,
    K_BLCK_NCOLS : tl.constexpr,
    QK_BLCK_NROWS : tl.constexpr,
    QK_BLCK_NCOLS : tl.constexpr,
):
    cur_batch = tl.program_id(0) # [0, 1]
    cur_head_id = tl.program_id(1) # [0]
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H) # [0] always
    split_kv_id = tl.program_id(2) # [0 - 7]

    if BLOCK_H < kv_group_num: # both are 16
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num # 16 always
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H) # [0..15]
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H # [full True]
    mask_h = mask_h & (cur_head < q_head_num) # [full True]

    offs_d = tl.arange(0, BLOCK_DMODEL) # [0..63]
    offs_dv = tl.arange(0, BLOCK_DV) # [0..63]
    mask_d = offs_d < Lk # [True] x64
    mask_dv = offs_dv < Lv # [True] x64

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch) # [0, 5, (10 won't be loaded)]
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx # 5 always

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS) # 1
    split_kv_start = kv_len_per_split * split_kv_id # [0 - 7]
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len) # min([1 - 8], 5) | [1, 2, 3, 4, 5, 5, 5, 5]

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    # tl.device_print("acc tensor aaa", acc)
    bx = tl.program_id(0)  # X grid index
    by = tl.program_id(1)  # Y grid index
    bz = tl.program_id(2)  # Z grid index

    grid_idx = bx * (BY) * (BZ) + by * (BZ) + bz
    if split_kv_end > split_kv_start:
        # only one iter always
        # start_n: [0 - 4]
        # split_kv_end: [1, 2, 3, 4, 5]
        
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N) # [0 - 4] + 32
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            ) # [0, 5] ???
            # tl.device_print("kv kv_loc and end", kv_loc)
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            # tl.device_print("row_idx", k)
            # # ======== STORE TMP_Q =========
            # q_blck_row_idx = grid_idx * Q_BLCK_NROWS * q_stride_x + tl.arange(0, Q_BLCK_NROWS)[:, None] * q_stride_x
            # q_blck_col_idx = tl.arange(0, Q_BLCK_NCOLS)[None, :]

            # tl.store(tmp_q + q_blck_row_idx + q_blck_col_idx, q.to(tl.bfloat16))

            # ======== STORE TMP_K =========
            # k_blck_row_idx = grid_idx * K_BLCK_NROWS * k_stride_x + tl.arange(0, K_BLCK_NROWS)[:, None] * k_stride_x
            # k_blck_col_idx = tl.arange(0, K_BLCK_NCOLS)[None, :]

            # tl.store(tmp_k + k_blck_row_idx + k_blck_col_idx, k.to(tl.bfloat16))

            qk = tl.dot(q, k.to(q.dtype))

            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )
            # ======== STORE TMP_QK =========
            qk_blck_row_idx = grid_idx * QK_BLCK_NROWS * qk_stride_x + tl.arange(0, QK_BLCK_NROWS)[:, None] * qk_stride_x
            qk_blck_col_idx = tl.arange(0, QK_BLCK_NCOLS)[None, :]

            tl.store(tmp_qk + qk_blck_row_idx + qk_blck_col_idx, qk.to(tl.bfloat16))
        #     # tl.device_print("qk val", qk)

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            # tl.device_print("idx", offs_buf_v)
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )
            
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            # tl.device_print("qk val", v)
            
            # sample_p = tl.full(shape=(P_BLCK_NROWS, P_BLCK_NCOLS), value=grid_idx, dtype=tl.bfloat16)

            # ======== STORE TMP_P =========
            # p_blck_row_idx = grid_idx * P_BLCK_NROWS * stride_x + tl.arange(0, P_BLCK_NROWS)[:, None] * stride_x
            # p_blck_col_idx = tl.arange(0, P_BLCK_NCOLS)[None, :]

            # tl.store(tmp_p + p_blck_row_idx + p_blck_col_idx, p.to(tl.bfloat16))

            # ======== STORE TMP_V =========
            # v_blck_row_idx = grid_idx * V_BLCK_NROWS * v_stride_x + tl.arange(0, V_BLCK_NROWS)[:, None] * v_stride_x
            # v_blck_col_idx = tl.arange(0, V_BLCK_NCOLS)[None, :]

            # tl.store(tmp_v + v_blck_row_idx + v_blck_col_idx, v.to(tl.bfloat16))

            acc *= re_scale[:, None]
            te = tl.dot(p.to(v.dtype), v)
            acc += te

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )

import torch
import pickle

def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    # breakpoint()

    # [TODO] work around shmem limit on MI3xx
    if is_hip_ and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    # (2, 1, 8)
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), # 16 / 16
        NUM_KV_SPLITS,
    )
    # breakpoint()

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    import os
    DEVICE = os.environ.get("SRT_DEVICE", "cuda")

    # ===== DEFINE TMP_P =========
    P_BLCK_NROWS = 16
    P_BLCK_NCOLS = 32
    # torch.full((size,), -1, dtype=torch.float32, device='cuda')
    tmp_p = torch.full((grid[0] * grid[1] * grid[2] * P_BLCK_NROWS, P_BLCK_NCOLS), -1, dtype=torch.bfloat16, device=DEVICE)

    # ===== DEFINE TMP_V =========
    V_BLCK_NROWS = 32
    V_BLCK_NCOLS = 64
    tmp_v = torch.full((grid[0] * grid[1] * grid[2] * P_BLCK_NROWS, P_BLCK_NCOLS), -1, dtype=torch.bfloat16, device=DEVICE)

    # ===== DEFINE TMP_Q =========
    Q_BLCK_NROWS = 16
    Q_BLCK_NCOLS = 64
    tmp_q = torch.full((grid[0] * grid[1] * grid[2] * Q_BLCK_NROWS, Q_BLCK_NCOLS), -1, dtype=torch.bfloat16, device=DEVICE)

    # ===== DEFINE TMP_K =========
    K_BLCK_NROWS = 64
    K_BLCK_NCOLS = 32
    tmp_k = torch.full((grid[0] * grid[1] * grid[2] * K_BLCK_NROWS, K_BLCK_NCOLS), -1, dtype=torch.bfloat16, device=DEVICE)

    # ===== DEFINE TMP_QK =========
    QK_BLCK_NROWS = 16
    QK_BLCK_NCOLS = 32
    tmp_qk = torch.full((grid[0] * grid[1] * grid[2] * QK_BLCK_NROWS, QK_BLCK_NCOLS), -1, dtype=torch.bfloat16, device=DEVICE)

    # breakpoint()
    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        tmp_p,
        tmp_p.stride(0),
        tmp_p.stride(1),
        tmp_v,
        tmp_v.stride(0),
        tmp_v.stride(1),
        tmp_q,
        tmp_q.stride(0),
        tmp_q.stride(1),
        tmp_k,
        tmp_k.stride(0),
        tmp_k.stride(1),
        tmp_qk,
        tmp_qk.stride(0),
        tmp_qk.stride(1),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        BX=grid[0],
        BY=grid[1],
        BZ=grid[2],
        P_BLCK_NROWS=P_BLCK_NROWS,
        P_BLCK_NCOLS=P_BLCK_NCOLS,
        V_BLCK_NROWS=V_BLCK_NROWS,
        V_BLCK_NCOLS=V_BLCK_NCOLS,
        Q_BLCK_NROWS=Q_BLCK_NROWS,
        Q_BLCK_NCOLS=Q_BLCK_NCOLS,
        K_BLCK_NROWS=K_BLCK_NROWS,
        K_BLCK_NCOLS=K_BLCK_NCOLS,
        QK_BLCK_NROWS=QK_BLCK_NROWS,
        QK_BLCK_NCOLS=QK_BLCK_NCOLS,
        **extra_kargs,
    )
    print(tmp_p)
    # breakpoint()
    IDX = 1 if DEVICE == "cuda" else 2
    # with open(f"../../dump{IDX}_p.pkl", "wb") as f:
    #     pickle.dump(tmp_p.cpu(), f)
    # with open(f"../../dump{IDX}_v.pkl", "wb") as f:
    #     pickle.dump(tmp_v.cpu(), f)
    # with open(f"../../dump{IDX}_q_dbg.pkl", "wb") as f:
    #     pickle.dump(tmp_q.cpu(), f)
    # with open(f"../../dump{IDX}_k.pkl", "wb") as f:
    #     pickle.dump(tmp_k.cpu(), f)
    with open(f"../../dump{IDX}_qk.pkl", "wb") as f:
        pickle.dump(tmp_qk.cpu(), f)
    print("hey")


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    O,
    kv_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    v_buffer,
    kv_indptr,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        kv_indptr,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )
    # print("NORMAL:")
    # print(attn_logits)
    # print(q)
    # print(o)
    # print(v_buffer)
    # print(kv_indptr)
    # print(num_kv_splits)
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, kv_indptr, num_kv_splits)


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )
    # print("GROUPED:")
    # print(attn_logits)
    # print(q)
    # print(o)
    # print(v_buffer)
    # print(kv_indptr)
    # print(num_kv_splits)
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, kv_indptr, num_kv_splits)


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )
