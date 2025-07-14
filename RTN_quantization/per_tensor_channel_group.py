import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_group_absmax_input_features(w, group_size, n_bits=8):
    """
    Quantize weights in groups along the input features dimension.
    
    Args:
        w: Weight tensor of shape (out_features, in_features)
        group_size: Number of input features per group
        n_bits: Number of bits for quantization
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    assert w.dim() == 2
    assert type(group_size) == int, "group_size must be an integer"
    # Reshape to group along input features
    # (out_features, in_features) -> (out_features, num_groups, group_size)
    w_grouped = w.view(out_features, -1, group_size)
    
    # Reshape to treat each group as a separate channel
    # (out_features, num_groups, group_size) -> (out_features * num_groups, group_size)
    w_reshaped = w_grouped.view(-1, group_size)
    
    # Apply per-channel quantization to each group
    quantized_w = quantize_weight_per_channel_absmax(w_reshaped, n_bits=n_bits)
    # Reshape back to original dimensions
    quantized_w = quantized_w.view(out_features, in_features)
    
    return quantized_w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_activation_per_group_absmax_token_dim(t, group_size, n_bits=8):
    """
    Per-group activation quantization that works with same input shapes as per_token.
    Groups features in the last dimension.
    """
    t_shape = t.shape
    last_dim = t_shape[-1]
    assert type(group_size) == int, "group_size must be an integer"
    assert last_dim % group_size == 0, f"Last dimension ({last_dim}) must be divisible by group_size ({group_size})"
    
    # Reshape to group the last dimension: (..., features) -> (..., num_groups, group_size)
    new_shape = t_shape[:-1] + (last_dim // group_size, group_size)
    t_grouped = t.view(new_shape)
    
    # Reshape to (...*num_groups, group_size) so each group becomes a "token"
    t_reshaped = t_grouped.view(-1, group_size)
    
    # Apply per-token quantization (each group gets its own scale)
    t_quantized = quantize_activation_per_token_absmax(t_reshaped, n_bits=n_bits)
    
    # Reshape back to original shape
    t_quantized = t_quantized.view(t_shape)
    
    return t_quantized

class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        group_size= None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        elif act_quant == "per_group_token":
            self.act_quant_name = "per_group_token"
            self.act_quant = partial(quantize_activation_per_group_absmax_token_dim, group_size=self.group_size, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False  , group_size=None
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            group_size=group_size
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        elif weight_quant == "per_group":
            new_module.weight = quantize_weight_per_group_absmax_input_features(
                module.weight, group_size,n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"
