import torch
import torch.nn as nn


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5): 
    """_summary_
    Smooth LayerNorm and Linear layers by scaling their weights and biases based on activation scales.
    Original: Y = Linear(LayerNorm(X))
    Smoothed: Y = Linear(s⁻¹ * LayerNorm(X) * s) where s = scaling factor
    Args:
        ln (nn.LayerNorm): the LayerNorm layer to smooth
        fcs (list): the list of Linear layers to smooth
        act_scales (torch.Tensor) : the activation scales to use for smoothing
        alpha (float, optional): the smoothing factor. Defaults to 0.5.
    alpha should be bigger if activation outliers are more significant
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))