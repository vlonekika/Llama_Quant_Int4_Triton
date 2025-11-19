# Llama_Quant_Int4_Triton
A collection of high-performance Triton (CUDA) kernels for 4-bit weight quantization and fast inference of low-bit LLMs

---

## Performance Overview

The tables below summarize the timings for **weight quantization** and **matrix multiplication** using Triton kernels compared to standard PyTorch operations. The best performing configurations are highlighted.

### **Quantization Timings**

| Method | Block Size | Time (ms) |
|--------|-----------|-----------|
| triton | 32        | 3.18      |
| **triton** | **64**        | **0.68**  |
| triton | 128       | 2.69      |
| triton | 256       | 2.50      |
| triton | 512       | 2.45      |
| torch  | -         | 177.74    |

> **Note:** Quantization was compared with PyTorch quantization in the X16@W4^T setup.

**Best quantization configuration:**  
- **Method:** triton  
- **Block Size:** 64  
- **Time:** 0.68 ms

---

### **Matrix Multiplication (Matmul) Timings**

> **Benchmark settings:** `hidden_size = 1024`, `batch_size = 512`

| Method | BLOCK_M | BLOCK_N | BLOCK_K | Time (ms) |
|--------|---------|---------|---------|-----------|
| triton | 32      | 32      | 16      | 20.14     |
| triton | 32      | 64      | 16      | 7.69      |
| triton | 32      | 64      | 32      | 7.47      |
| triton | 64      | 64      | 32      | 6.08      |
| triton | 64      | 128     | 32      | 6.24      |
| triton | 64      | 128     | 64      | 5.47      |
| triton | 128     | 64      | 64      | 5.10      |
| triton | **128** | **128** | **64**  | **4.70**  |
| triton | 128     | 128     | 128     | 4.98      |
| triton | 256     | 128     | 128     | 17.19     |
| torch  | -       | -       | -       | 66.64     |

**Best matmul configuration:**  
- **Method:** triton  
- **BLOCK_M / BLOCK_N / BLOCK_K:** 128 / 128 / 64  
- **Time:** 4.70 ms

---

## Triton vs PyTorch Inference Timings

> **Note:** Comparison between **X16@W4^T** (Triton) and **X16@W16^T** (PyTorch).  
> **Benchmark settings:** `hidden_size = 2048`, `token_cnt = [128, 512, 2048]`.

| Batch | Method | BLOCK_M | BLOCK_N | BLOCK_K | Time (ms) |
|-------|--------|---------|---------|---------|-----------|
| 128   | Triton X@W4^T | 32  | 32  | 16  | 7.61    |
| 128   | Triton X@W4^T | 32  | 64  | 16  | 5.15    |
| 128   | Triton X@W4^T | 32  | 64  | 32  | 4.77    |
| 128   | Triton X@W4^T | 64  | 64  | 32  | 3.37    |
| 128   | Triton X@W4^T | 64  | 128 | 32  | 3.16    |
| 128   | Triton X@W4^T | 64  | 128 | 64  | 3.16    |
| 128   | Triton X@W4^T | 128 | 64  | 64  | 2.46    |
| 128   | Triton X@W4^T | 128 | 128 | 64  | 4.67    |
| 128   | Triton X@W4^T | 128 | 128 | 128 | 4.67    |
| 128   | Triton X@W4^T | 256 | 128 | 128 | 12.51   |
| 128   | PyTorch X16@W16^T | - | - | - | 0.30  |
| 512   | Triton X@W4^T | 32  | 32  | 16  | 12.40   |
| 512   | Triton X@W4^T | 32  | 64  | 16  | 7.60    |
| 512   | Triton X@W4^T | 32  | 64  | 32  | 7.35    |
| 512   | Triton X@W4^T | 64  | 64  | 32  | 5.43    |
| 512   | Triton X@W4^T | 64  | 128 | 32  | 4.85    |
| 512   | Triton X@W4^T | 64  | 128 | 64  | 4.73    |
| 512   | Triton X@W4^T | 128 | 64  | 64  | 3.69    |
| 512   | Triton X@W4^T | 128 | 128 | 64  | 4.81    |
| 512   | Triton X@W4^T | 128 | 128 | 128 | 4.57    |
| 512   | Triton X@W4^T | 256 | 128 | 128 | 40.56   |
| 512   | PyTorch X16@W16^T | - | - | - | 0.80  |
| 2048  | Triton X@W4^T | 32  | 32  | 16  | 47.18   |
| 2048  | Triton X@W4^T | 32  | 64  | 16  | 32.55   |
| 2048  | Triton X@W4^T | 32  | 64  | 32  | 31.72   |
| 2048  | Triton X@W4^T | 64  | 64  | 32  | 22.81   |
| 2048  | Triton X@W4^T | 64  | 128 | 32  | 16.99   |
| 2048  | Triton X@W4^T | 64  | 128 | 64  | 17.04   |
| 2048  | Triton X@W4^T | 128 | 64  | 64  | 16.38   |
| 2048  | Triton X@W4^T | 128 | 128 | 64  | 17.13   |
| 2048  | Triton X@W4^T | 128 | 128 | 128 | 15.79   |
| 2048  | Triton X@W4^T | 256 | 128 | 128 | 265.04  |
| 2048  | PyTorch X16@W16^T | - | - | - | 3.30  |

**Best Triton configuration:**  
- **Batch:** 128  
- **BLOCK_M / BLOCK_N / BLOCK_K:** 128 / 64 / 64  
- **Time:** 2.46 ms

---

## FP16 vs INT4 Model Comparison

### 1️⃣ Memory Usage & Perplexity

| Model             | Perplexity | Memory Size (MB) |
|------------------|------------|----------------|
| FP16 (full precision)  | 28.03      | 4714.26        |
| INT4 Quantized    | 45.23      | 1002.26        |

> **Note:** INT4 quantization significantly reduces memory consumption at the cost of higher perplexity.

---

### 2️⃣ Average Time per Batch (Batch Size = 16)

| Sequence Length | FP16 Model (ms) | INT4 Quantized Model (ms) |
|-----------------|----------------|---------------------------|
| 128             | 155.45         | 3172.17                  |
| 512             | 355.38         | 7442.40                  |
| 2048            | 382.21         | 7446.27                  |

> **Observation:** FP16 model is much faster for small batches, while INT4 model greatly reduces memory but increases computation time per batch.
