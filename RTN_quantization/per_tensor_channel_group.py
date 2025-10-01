import torch
from torch import nn
from functools import partial
import torch.nn.functional as F 

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
def quantize_weight_per_channel_random_round_up_down_absmax(w, n_bits=8, state="up", percent=0.5):

    original_dtype = w.dtype
    out_features, in_features = w.shape

    num_channels_to_quantize = int(out_features * percent)

    random_indices = torch.randperm(out_features)[:num_channels_to_quantize]

    w_output = w.clone()

    # Process only the selected channels
    for idx in random_indices:
        # Get the channel
        channel = w[idx]

        # Calculate scale for this channel
        scale = channel.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scale = scale.clamp(min=1e-5) / q_max

        # Normalize the channel
        channel_normalized = channel / scale

        # Apply rounding based on state
        if state == "up":
            channel_quantized = channel_normalized.ceil()
        elif state == "down":
            channel_quantized = channel_normalized.floor()
        elif state == "RTN":
            channel_quantized = channel_normalized.round()
        elif state == "random":
            # Generate random mask for up/down rounding
            random_mask = torch.rand_like(channel_normalized) > 0.5
            channel_quantized = torch.where(
                random_mask,
                channel_normalized.ceil(),  # Round up
                channel_normalized.floor()   # Round down
            )
            
        else:
            raise ValueError(f"Invalid state: {state}. Must be 'up', 'down', or 'RTN'")

        # Scale back and update the output
        w_output[idx] = channel_quantized * scale

    return w_output.to(original_dtype)

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
    t.contiguous().view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.contiguous().view(-1, t_shape[-1])
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

    t_reshaped = t_grouped.view(-1, group_size)
    t_quantized = quantize_activation_per_token_absmax(t_reshaped, n_bits=n_bits)
    t_quantized = t_quantized.view(t_shape)
    
    return t_quantized


def cal_density(X:torch.Tensor, margin:float=0.9):
   
    B,H,W,C = X.shape
    X = X.view(B,1,H*W,C)
    # X = X.permute(0, 2, 1, 3)
    X = F.normalize(X, p=2, dim=-1)

    score_map =F.elu(X @ X.transpose(-1, -2)-margin,alpha=0 )
    # score_map = X @ X.transpose(-1, -2)
    scores = score_map.mean(-1)
    return scores 
@torch.no_grad()
def quantize_activation_low_high_density_activation(t, n_bits=8, quantizehigh=True):
    original_shape = t.shape
    original_dtype = t.dtype

    B, H, W, C = t.shape
    scores = cal_density(t) 
    scores = scores.squeeze(1).reshape(-1)  
    t_2d = t.view(B * H * W, C)

    # # threshold = scores.mean()
    # threshold = torch.median(scores)
    
    # # Create mask for tokens to quantize
    # if quantizehigh:
    #     token_mask = scores > threshold
    # else:
    #     token_mask = scores <= threshold
    
    # Sort scores to find percentile threshold
    percent =60
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    # Calculate the number of tokens to quantize based on percentage
    num_tokens = scores.numel()
    num_to_quantize = int(num_tokens * (percent / 100.0))
    # Create mask for tokens to quantize
    token_mask = torch.zeros_like(scores, dtype=torch.bool)
    if quantizehigh:
        # Quantize the top 'percent'% highest density tokens
    
        top_indices = sorted_indices[:num_to_quantize]
        token_mask[top_indices] = True
    else:
        # Quantize the bottom 'percent'% lowest density tokens
        bottom_indices = sorted_indices[-num_to_quantize:]
        token_mask[bottom_indices] = True
        
    output = t_2d.clone()
    tokens_to_quantize = t_2d[token_mask]
    # print("nu quantized/ total :", tokens_to_quantize.shape[0], t_2d.shape[0])
    if tokens_to_quantize.numel() > 0:
        scales = tokens_to_quantize.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)

        quantized_tokens = (tokens_to_quantize / scales).round() * scales

        output[token_mask] = quantized_tokens
    
    output = output.view(original_shape)
    output = output.to(original_dtype)
    return output

class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        group_size= None,
        n_bit = 8,
        quantizehigh= True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_bits = n_bit
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
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.n_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.n_bits)
        elif act_quant == "per_group_token":
            self.act_quant_name = "per_group_token"
            self.act_quant = partial(quantize_activation_per_group_absmax_token_dim, group_size=self.group_size, n_bits=self.n_bits)
        elif act_quant == "low_high_density_activation":
            self.act_quant_name = "low_high_density_activation"
            self.act_quant = partial(quantize_activation_low_high_density_activation, n_bits=self.n_bits,quantizehigh=quantizehigh)
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
        # print name of module
        
        q_x = self.act_quant(x)
        # import ipdb; ipdb.set_trace()
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)

        return q_y

    @staticmethod
    def from_float(
        module, n_bits_w,n_bits_ac,weight_quant="per_channel", act_quant="per_token", quantize_output=False  , group_size=None, quantize_weight = True,quantizehigh=True,up_down_RTN ="up"
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            group_size=group_size,
            n_bit=n_bits_ac ,
            quantizehigh= quantizehigh
        )
        if quantize_weight:
            if weight_quant == "per_channel":
                # new_module.weight = quantize_weight_per_channel_absmax(
                #     module.weight, n_bits=n_bits_w
                # )  # use 8-bit integer for weight
                if up_down_RTN == "up":
                    print("up")
                    new_module.weight =quantize_weight_per_channel_random_round_up_down_absmax(module.weight, n_bits=n_bits_w, state="up", percent=0.75) # up, down, RTN
                elif up_down_RTN =="down":
                    print("down")
                    new_module.weight =quantize_weight_per_channel_random_round_up_down_absmax(module.weight, n_bits=n_bits_w, state="down", percent=0.75) # up, down, RTN
                elif up_down_RTN =="RTN":
                    print("RTN")
                    new_module.weight =quantize_weight_per_channel_random_round_up_down_absmax(module.weight, n_bits=n_bits_w, state="RTN", percent=0.75) # up, down, RTN
                elif up_down_RTN =="random":
                    print("random")
                    new_module.weight =quantize_weight_per_channel_random_round_up_down_absmax(module.weight, n_bits=n_bits_w, state="random", percent=0.75) # up, down, RTN
                else:
                    new_module.weight = quantize_weight_per_channel_absmax(
                        module.weight, n_bits=n_bits_w
                    )  # use 8-bit integer for weight
                    # raise ValueError(f"Invalid up_down_RTN: {up_down_RTN}")
            elif weight_quant == "per_tensor":
                new_module.weight = quantize_weight_per_tensor_absmax(
                    module.weight, n_bits=n_bits_w
                )
            elif weight_quant == "per_group":
                new_module.weight = quantize_weight_per_group_absmax_input_features(
                    module.weight, group_size,n_bits=n_bits_w
                )
            else:
                raise ValueError(f"Invalid weight_quant: {weight_quant}")
            new_module.weight_quant_name = weight_quant
        else:
            new_module.weight = module.weight
            new_module.weight_quant_name = "None"
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"
