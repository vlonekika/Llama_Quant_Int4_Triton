import torch
import triton
import triton.language as tl

@triton.jit
def quantize_rowwise_int4_kernel(
    w_ptr, 
    packed_ptr, 
    scale_ptr, 
    zero_ptr,
    n_rows, 
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    row_w_offset = pid * n_cols
    row_k_offset = pid * (n_cols // 8)

    min_val = 1e6
    max_val = -1e6

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(w_ptr + row_w_offset + cols, mask=mask, other=0.0).to(tl.float32)

        x_min = tl.min(tl.where(mask, x, 1e6), axis=0)
        x_max = tl.max(tl.where(mask, x, -1e6), axis=0)
        min_val = tl.minimum(min_val, x_min)
        max_val = tl.maximum(max_val, x_max)

    data_range = tl.maximum(max_val - min_val, 1e-6)
    scale = data_range / 15.0
    zero_point = tl.extra.cuda.libdevice.rint(-min_val / scale)
    zero_point = tl.minimum(tl.maximum(zero_point, 0.0), 15.0)

    tl.store(scale_ptr + pid, scale.to(tl.float16))
    tl.store(zero_ptr + pid, zero_point.to(tl.float16))

    n_packs = n_cols // 8
    for pack_idx in range(n_packs):
        base_col = pack_idx * 8
        cols = base_col + tl.arange(0, 8)

        x = tl.load(w_ptr + row_w_offset + cols).to(tl.float32)

        q = tl.extra.cuda.libdevice.rint(x / scale + zero_point)
        q = tl.minimum(tl.maximum(q, 0.0), 15.0)
        q_int = q.to(tl.int32)

        shifts = tl.arange(0, 8).to(tl.int32) * 4
        packed_vec = q_int << shifts
        packed = tl.sum(packed_vec, axis=0).to(tl.int32)

        tl.store(packed_ptr + row_k_offset + pack_idx, packed)


def quantize_int4(w, BLOCK_SIZE=64):
    w = w.contiguous()
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    n_rows, n_cols = w.shape


    packed = torch.empty((n_rows, n_cols // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty(n_rows, dtype=torch.float16, device=w.device)
    zeros = torch.empty(n_rows, dtype=torch.float16, device=w.device)

    grid = lambda meta: (n_rows,)

    quantize_rowwise_int4_kernel[grid](
        w, 
        packed, 
        scales, 
        zeros,
        n_rows, 
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return packed, scales, zeros