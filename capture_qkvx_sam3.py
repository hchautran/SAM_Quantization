#!/usr/bin/env python3
"""
Capture q, k, v, and x (input hidden states) from every Sam3ViTLayer
during a single image inference pass.

Captured tensors (per layer i):
  features[i]["x"]  — input to Sam3ViTLayer before layer_norm1
                       shape: [B, H, W, C]
  features[i]["q"]  — query after q_proj + RoPE
                       shape: [B, num_heads, H*W, head_dim]
  features[i]["k"]  — key after k_proj + RoPE
                       shape: [B, num_heads, H*W, head_dim]
  features[i]["v"]  — value after v_proj
                       shape: [B, num_heads, H*W, head_dim]

Usage:
    python capture_qkvx_sam3.py --image input_imgs/example.jpg --save features.pt

    # or import:
    from capture_qkvx_sam3 import capture_features
    features = capture_features(model, processor, image)
"""

import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Feature capture logic
# ─────────────────────────────────────────────────────────────────────────────

def _patch_attention(attn_module, layer_idx: int, store: Dict):
    """Replace Sam3ViTRoPEAttention.forward to intercept q, k, v post-RoPE."""
    from transformers.models.sam3.modeling_sam3 import apply_rotary_pos_emb_2d
    orig_forward = attn_module.forward

    def patched_forward(hidden_states, position_embeddings, **kwargs):
        batch_size, height, width, _ = hidden_states.shape
        seq_len = height * width
        new_shape = (batch_size, seq_len, attn_module.num_attention_heads, attn_module.head_dim)

        q = attn_module.q_proj(hidden_states).view(*new_shape).transpose(1, 2)
        k = attn_module.k_proj(hidden_states).view(*new_shape).transpose(1, 2)
        v = attn_module.v_proj(hidden_states).view(*new_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb_2d(q, k, cos=cos, sin=sin)

        store[layer_idx]["q"] = q_rope.detach().cpu()
        store[layer_idx]["k"] = k_rope.detach().cpu()
        store[layer_idx]["v"] = v.detach().cpu()

        # run original (which re-computes q/k/v internally — unavoidable)
        return orig_forward(hidden_states, position_embeddings, **kwargs)

    attn_module.forward = patched_forward
    return orig_forward


def attach_hooks(vit_model) -> Dict[int, Dict]:
    """
    Attach pre-hooks on every Sam3ViTLayer to capture x,
    and patch every Sam3ViTRoPEAttention to capture q/k/v.

    Returns the feature store and cleanup lists.
    """
    layers = vit_model.layers
    n = len(layers)
    store: Dict[int, Dict] = {i: {} for i in range(n)}
    hook_handles = []
    orig_forwards = []

    for i, layer in enumerate(layers):
        # capture x = input hidden_states to the layer (before layer_norm1)
        def make_pre_hook(idx):
            def pre_hook(_module, args):
                x = args[0] if isinstance(args, tuple) else args
                store[idx]["x"] = x.detach().cpu()
            return pre_hook

        h = layer.register_forward_pre_hook(make_pre_hook(i))
        hook_handles.append(h)

        # patch attention to capture q, k, v
        orig = _patch_attention(layer.attention, i, store)
        orig_forwards.append((layer.attention, orig))

    return store, hook_handles, orig_forwards


def detach_hooks(hook_handles, orig_forwards):
    for h in hook_handles:
        h.remove()
    for attn_mod, orig in orig_forwards:
        attn_mod.forward = orig


# ─────────────────────────────────────────────────────────────────────────────
# Sobel filter utilities
# ─────────────────────────────────────────────────────────────────────────────

# Sobel kernels: [1, 1, 3, 3]
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Sobel filter to a spatial feature map and return gradient magnitude.

    Parameters
    ----------
    x : Tensor [B, H, W, C]  (float32 on cpu)

    Returns
    -------
    mag : Tensor [B, H, W, C]  — per-channel gradient magnitude
    """
    B, H, W, C = x.shape
    # [B*C, 1, H, W]
    x_flat = x.permute(0, 3, 1, 2).reshape(B * C, 1, H, W).float()

    kx = _SOBEL_X
    ky = _SOBEL_Y

    gx = F.conv2d(x_flat, kx, padding=1)   # [B*C, 1, H, W]
    gy = F.conv2d(x_flat, ky, padding=1)

    mag = (gx ** 2 + gy ** 2).sqrt()       # [B*C, 1, H, W]
    return mag.view(B, C, H, W).permute(0, 2, 3, 1)  # [B, H, W, C]


def scan_global_layers(
    features: Dict[int, Dict],
    global_attn_indexes: List[int] = (7, 15, 23, 31),
) -> Dict[int, torch.Tensor]:
    """
    Run Sobel on the x feature of every global attention layer.

    Parameters
    ----------
    features           : output of capture_features()
    global_attn_indexes: layer indices with global (non-windowed) attention

    Returns
    -------
    dict  layer_idx → sobel magnitude Tensor [B, H, W, C]
    """
    results = {}
    for idx in global_attn_indexes:
        if idx not in features or "x" not in features[idx]:
            continue
        results[idx] = sobel_magnitude(features[idx]["x"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_sobel_heatmaps(
    sobel_maps: Dict[int, torch.Tensor],
    image: Image.Image = None,
    cmap: str = "hot",
    alpha: float = 0.6,
    save_path: str = None,
):
    """
    Plot Sobel gradient magnitude heatmaps for each global attention layer.

    Parameters
    ----------
    sobel_maps : output of scan_global_layers()  {layer_idx → [B, H, W, C]}
    image      : optional PIL Image to show alongside each heatmap
    cmap       : matplotlib colormap for the heatmap
    alpha      : opacity when overlaying heatmap on the original image
    save_path  : if given, save the figure to this path instead of showing

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    layer_idxs = sorted(sobel_maps.keys())
    n_layers = len(layer_idxs)
    has_image = image is not None
    n_cols = 3 if has_image else 2   # original | heatmap | overlay  (or  heatmap | overlay)
    if not has_image:
        n_cols = 1

    fig, axes = plt.subplots(n_layers, n_cols + (1 if has_image else 0),
                             figsize=(4 * (n_cols + (1 if has_image else 0)), 3.5 * n_layers),
                             squeeze=False)

    for row, idx in enumerate(layer_idxs):
        mag = sobel_maps[idx][0]              # [H, W, C]
        mag_mean = mag.mean(-1).numpy()       # [H, W]  — average over channels
        mag_norm = (mag_mean - mag_mean.min()) / (mag_mean.max() - mag_mean.min() + 1e-8)

        col = 0

        if has_image:
            img_resized = image.resize((mag_mean.shape[1], mag_mean.shape[0]))
            axes[row, col].imshow(img_resized)
            axes[row, col].set_title(f"Layer {idx} — input image", fontsize=9)
            axes[row, col].axis("off")
            col += 1

        # heatmap
        im = axes[row, col].imshow(mag_norm, cmap=cmap, vmin=0, vmax=1)
        axes[row, col].set_title(f"Layer {idx} — Sobel magnitude", fontsize=9)
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        col += 1

        # overlay on original image
        if has_image:
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            axes[row, col].imshow(img_np)
            axes[row, col].imshow(mag_norm, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
            axes[row, col].set_title(f"Layer {idx} — overlay (α={alpha})", fontsize=9)
            axes[row, col].axis("off")

    fig.suptitle("SAM3 ViT — Sobel gradient magnitude (global attention layers)", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────────────────────────

def capture_features(model, processor, image: Image.Image, device="cuda") -> Dict[int, Dict]:
    """
    Run one forward pass and return per-layer {x, q, k, v} tensors.

    Parameters
    ----------
    model       : Sam3Model (already on device)
    processor   : Sam3Processor
    image       : PIL Image
    device      : str

    Returns
    -------
    dict  layer_idx → {"x": Tensor, "q": Tensor, "k": Tensor, "v": Tensor}
    """
    vit = model.vision_encoder.backbone  # Sam3ViTModel
    store, hook_handles, orig_forwards = attach_hooks(vit)

    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(dtype=next(model.parameters()).dtype)

    with torch.no_grad():
        model.vision_encoder(pixel_values)

    detach_hooks(hook_handles, orig_forwards)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Capture q/k/v/x from SAM3 ViT layers")
    p.add_argument("--model-id", default="facebook/sam3")
    p.add_argument("--image", default=None, help="Path to input image (uses random tensor if omitted)")
    p.add_argument("--save", default="sam3_features.pt", help="Output .pt file")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    from transformers import Sam3Model, Sam3Processor

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"Loading {args.model_id} ...")
    model = Sam3Model.from_pretrained(args.model_id, torch_dtype=dtype).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.model_id)

    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"Image: {args.image}  size={image.size}")
    else:
        import numpy as np
        image = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
        print("Using random 1024×1024 image")

    print("Running inference and capturing features ...")
    features = capture_features(model, processor, image, device=args.device)

    # print summary
    n_layers = len(features)
    print(f"\nCaptured {n_layers} layers:")
    print(f"{'Layer':<8} {'x shape':<28} {'q shape':<32} {'k shape':<32} {'v shape'}")
    print("-" * 120)
    for i in range(n_layers):
        d = features[i]
        x_s = str(tuple(d["x"].shape)) if "x" in d else "missing"
        q_s = str(tuple(d["q"].shape)) if "q" in d else "missing"
        k_s = str(tuple(d["k"].shape)) if "k" in d else "missing"
        v_s = str(tuple(d["v"].shape)) if "v" in d else "missing"
        print(f"{i:<8} {x_s:<28} {q_s:<32} {k_s:<32} {v_s}")

    torch.save(features, args.save)
    print(f"\nSaved → {args.save}")


if __name__ == "__main__":
    main()
