import torch
import triton
import triton.language as tl

@triton.jit
def matmul_bf16_int4_kernel(
    x_ptr, 
    w_packed_ptr, 
    w_scale_ptr, 
    w_zero_ptr, 
    output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    scales = tl.load(w_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    zeros = tl.load(w_zero_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K):
        x_vals = tl.load(x_ptr + offs_m * K + k, mask=mask_m, other=0.0).to(tl.float32)

        pack_idx = k // 8
        nibble_idx = k % 8

        packed = tl.load(w_packed_ptr + offs_n * (K // 8) + pack_idx, mask=mask_n, other=0)

        vals_int = (packed >> (4 * nibble_idx)) & 0xF
        vals = (vals_int.to(tl.float32) - zeros) * scales

        acc += x_vals[:, None] * vals[None, :]

    outputs_vals = acc.to(tl.bfloat16)
    mask_outputs_vals = mask_m[:, None] & mask_n[None, :]
    tl.store(output_ptr + offs_m[:, None] * N + offs_n[None, :], outputs_vals, mask=mask_outputs_vals)