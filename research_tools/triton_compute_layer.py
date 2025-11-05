import torch
import torch.nn as nn

from quant import quantize_int4
from matmul import matmul_int4


class QuantLinearINT4(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weights",
            torch.zeros((out_features, in_features // 8), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, dtype=torch.float16),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(out_features, dtype=torch.float16),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def transform_layer(cls, module):
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        linear = cls(in_features, out_features, bias)

        weight = module.weight.detach().to(torch.float16).contiguous().cuda()
        weights, scales, zeros = quantize_int4(weight)

        linear.weights.copy_(weights)
        linear.scales.copy_(scales)
        linear.zeros.copy_(zeros)

        if module.bias is not None:
            bias.data.copy_(bias.detach().to(torch.float16).cuda())

        return linear.cuda()

    def forward(self, x):

        x_dtype = x.dtype
        x_shape = x.shape[:-1]

        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        out = matmul_int4(
            x,
            self.weights,
            self.scales,
            self.zeros,
        )

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        if len(x_shape) > 1:
            out = out.reshape(*x_shape, -1)

        return out.to(x_dtype)