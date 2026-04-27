from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from segment_anything.modeling.image_encoder import (
    Block as EncoderBlock,
    Attention,
    ImageEncoderViT,
    window_partition,
    window_unpartition,
    add_decomposed_rel_pos,
)


UNMERGED_TOKEN = 0
SRC_TOKEN = 1
DST_TOKEN = 2


def get_central_difference_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude over a `(B, H, W, C)` token grid."""
    padded = F.pad(x, (0, 0, 1, 1, 1, 1), mode="replicate")
    diff_x = (padded[:, 1:-1, 2:, :] - padded[:, 1:-1, :-2, :]) / 2.0
    diff_y = (padded[:, 2:, 1:-1, :] - padded[:, :-2, 1:-1, :]) / 2.0
    return torch.sqrt((diff_x.square() + diff_y.square()).sum(dim=-1))



def get_sobel_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude over a `(B, H, W, C)` token grid."""
    x = x.permute(0, 3, 1, 2)
    _, channels, _, _ = x.shape

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    )
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    )

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
    return torch.sqrt((grad_x.square() + grad_y.square()).mean(dim=1))



def generate_src_and_dst_idx(
    grad: torch.Tensor,
    sx: int = 2,
    sy: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate the src/dst partitioning logic used in `merge.py`."""
    batch, height, width = grad.shape
    if height % sy != 0 or width % sx != 0:
        raise ValueError(
            f"Expected height and width to be divisible by sy={sy} and sx={sx}, "
            f"but got {(height, width)}"
        )

    cell_rows = height // sy
    cell_cols = width // sx
    num_dst = cell_rows * cell_cols

    grad_cells = grad.view(batch, cell_rows, sy, cell_cols, sx)
    grad_cells = grad_cells.permute(0, 1, 3, 2, 4).contiguous().view(batch, num_dst, sy * sx)
    _, min_idx = grad_cells.min(dim=-1)

    row_offsets = torch.arange(cell_rows, device=grad.device).repeat_interleave(cell_cols)
    col_offsets = torch.arange(cell_cols, device=grad.device).repeat(cell_rows)

    local_row = min_idx // sx
    local_col = min_idx % sx
    global_row = row_offsets.unsqueeze(0) * sy + local_row
    global_col = col_offsets.unsqueeze(0) * sx + local_col
    dst_idx = global_row * width + global_col

    idx_buffer = torch.zeros(batch, height * width, device=grad.device, dtype=torch.long)
    idx_buffer.scatter_(dim=-1, index=dst_idx, src=-torch.ones_like(dst_idx))
    src_idx = idx_buffer.argsort(dim=-1)[:, num_dst:]
    return src_idx, dst_idx



def compute_merge_token_groups(
    input_attn: torch.Tensor,
    post_shortcut_x: torch.Tensor,
    sx: int = 2,
    sy: int = 2,
    grad_method: str = "sobel",
    r: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the `grad_bipartite_soft_matching` partitioning logic independently to
    `input_attn` and `post_shortcut_x` so their edge maps and merge maps can be
    compared directly.

    Args:
        input_attn: Tensor of shape `(B, H, W, C)`.
        post_shortcut_x: Tensor of shape `(B, H, W, C)`.
        sx: Horizontal cell size for dst token selection.
        sy: Vertical cell size for dst token selection.
        grad_method: `sobel` or `central_difference`.
        r: Number of src tokens to merge into dst tokens.

    Returns:
        edge_map_attn: `(B, H, W)` edge map computed from `input_attn`.
        edge_map_mlp: `(B, H, W)` edge map computed from `post_shortcut_x`.
        merge_map_attn: `(B, H, W)` map with labels `0=unmerged`, `1=src`, `2=dst`
            computed from `input_attn`.
        merge_map_mlp: `(B, H, W)` map with labels `0=unmerged`, `1=src`, `2=dst`
            computed from `post_shortcut_x`.
    """
    if input_attn.ndim != 4 or post_shortcut_x.ndim != 4:
        raise ValueError(
            "Expected `input_attn` and `post_shortcut_x` to have shape `(B, H, W, C)`"
        )
    if input_attn.shape != post_shortcut_x.shape:
        raise ValueError(
            f"Expected matching shapes, got {tuple(input_attn.shape)} and {tuple(post_shortcut_x.shape)}"
        )
    if sx <= 0 or sy <= 0:
        raise ValueError(f"Expected positive sx/sy, got sx={sx}, sy={sy}")
    if r < 0:
        raise ValueError(f"Expected non-negative r, got r={r}")
    if grad_method not in {"sobel", "central_difference"}:
        raise ValueError(
            "grad_method must be either 'sobel' or 'central_difference'"
        )

    def _single_tensor_merge_map(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, height, width, channels = x.shape
        num_tokens = height * width
        gather = torch.gather
        x = x.float()

        if grad_method == "sobel":
            edge_map = get_sobel_gradient(x)
        else:
            edge_map = get_central_difference_gradient(x)

        metric = F.normalize(x.reshape(batch, num_tokens, channels), p=2, dim=-1)

        if sx == width and sy == height:
            a_idx = torch.arange(1, num_tokens, device=x.device).view(1, -1).expand(batch, -1)
            b_idx = torch.zeros(batch, 1, device=x.device, dtype=torch.long)
            max_src = a_idx.shape[1]
            local_r = min(r, max_src)
            src_relative_idx = torch.arange(local_r, device=x.device).view(1, -1).expand(batch, -1)
            unm_relative_idx = torch.arange(local_r, max_src, device=x.device).view(1, -1).expand(batch, -1)
        else:
            a_idx, b_idx = generate_src_and_dst_idx(edge_map, sx=sx, sy=sy)
            src_metric = gather(
                metric,
                dim=1,
                index=a_idx.unsqueeze(-1).expand(batch, a_idx.shape[1], channels),
            )
            dst_metric = gather(
                metric,
                dim=1,
                index=b_idx.unsqueeze(-1).expand(batch, b_idx.shape[1], channels),
            )
            similarity = src_metric @ dst_metric.transpose(-1, -2)

            max_src = src_metric.shape[1]
            local_r = min(r, max_src)
            node_max, _ = similarity.max(dim=-1)
            edge_rank = node_max.argsort(dim=-1, descending=True)
            src_relative_idx = edge_rank[:, :local_r]
            unm_relative_idx = edge_rank[:, local_r:]

        src_absolute_idx = (
            gather(a_idx, dim=1, index=src_relative_idx) if local_r > 0 else a_idx[:, :0]
        )
        unm_absolute_idx = gather(a_idx, dim=1, index=unm_relative_idx)
        dst_absolute_idx = b_idx

        merge_map = torch.full(
            (batch, num_tokens),
            fill_value=-1,
            device=x.device,
            dtype=torch.long,
        )
        if unm_absolute_idx.numel() > 0:
            merge_map.scatter_(1, unm_absolute_idx, UNMERGED_TOKEN)
        if src_absolute_idx.numel() > 0:
            merge_map.scatter_(1, src_absolute_idx, SRC_TOKEN)
        if dst_absolute_idx.numel() > 0:
            merge_map.scatter_(1, dst_absolute_idx, DST_TOKEN)

        if (merge_map < 0).any():
            raise RuntimeError("Token grouping failed: some spatial locations were not assigned")

        return edge_map, merge_map.view(batch, height, width)

    edge_map_attn, merge_map_attn = _single_tensor_merge_map(input_attn)
    edge_map_mlp, merge_map_mlp = _single_tensor_merge_map(post_shortcut_x)
    return edge_map_attn, edge_map_mlp, merge_map_attn, merge_map_mlp

class MergeObserver:
    """Stores per-block features and derived edge/similarity statistics."""

    def __init__(
        self,
        gradient_method: str = "sobel",
        store_on_cpu: bool = True,
        detach_tensors: bool = True,
    ) -> None:
        if gradient_method not in {"sobel", "central_difference"}:
            raise ValueError(
                "gradient_method must be either 'sobel' or 'central_difference'"
            )

        self.gradient_method = gradient_method
        self.store_on_cpu = store_on_cpu
        self.detach_tensors = detach_tensors
        self.records: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    def clear(self) -> None:
        self.records.clear()

    reset = clear

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.detach_tensors:
            tensor = tensor.detach()
        if self.store_on_cpu:
            tensor = tensor.cpu()
        return tensor

    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_method == "sobel":
            return get_sobel_gradient(x)
        return get_central_difference_gradient(x)

    def _compute_similarity(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        tokens = x.reshape(batch, height * width, channels)
        tokens = F.normalize(tokens.float(), p=2, dim=-1)
        return torch.matmul(tokens, tokens.transpose(-1, -2))

    def capture(
        self,
        block_name: str,
        input_attn: torch.Tensor,
        input_attn_2: torch.Tensor,
        output_attn: torch.Tensor,
        post_shortcut_x: torch.Tensor,
        window_size: int,
        x_attn: torch.Tensor,
    ) -> Dict[str, Any]:
        edge_intensity = self._compute_gradient(input_attn.float())
        cosine_similarity = self._compute_similarity(post_shortcut_x)
        edge_flat = edge_intensity.reshape(edge_intensity.shape[0], -1)
        sort_idx = edge_flat.argsort(dim=-1, descending=True)

        record = {
            "block_name": block_name,
            "window_size": int(window_size),
            "num_windows": int(input_attn.shape[0]),
            "spatial_shape": tuple(int(v) for v in input_attn.shape[1:3]),
            "input_attn": self._prepare_tensor(input_attn),
            "input_attn_2": self._prepare_tensor(input_attn_2),
            "output_attn": self._prepare_tensor(output_attn),
            "x_attn": self._prepare_tensor(x_attn),
            "post_shortcut_x": self._prepare_tensor(post_shortcut_x),
            "edge_intensity": self._prepare_tensor(edge_intensity),
            "edge_intensity_flat": self._prepare_tensor(edge_flat),
            "cosine_similarity": self._prepare_tensor(cosine_similarity),
            "edge_sort_idx": self._prepare_tensor(sort_idx),
        }
        self.records[block_name].append(record)
        return record

    def get(self, block_name: str) -> List[Dict[str, Any]]:
        return self.records.get(block_name, [])

    def latest(self, block_name: str) -> Optional[Dict[str, Any]]:
        entries = self.get(block_name)
        return entries[-1] if entries else None

    def latest_by_index(self, block_idx: int) -> Optional[Dict[str, Any]]:
        return self.latest(f"image_encoder.blocks.{block_idx}")


class MergeBlockObserver(EncoderBlock):
    """Block wrapper that captures tensors around attention and the first residual."""

    def set_merge_observer(self, observer: MergeObserver, name: str) -> None:
        self._merge_observer = observer
        self._merge_block_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        input_attn = x

        if self.window_size > 0:
            height, width = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        input_attn_2 = x
        x ,x_attn = self.attn(x)
        output_attn = x
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (height, width))

        x = shortcut + x
        post_shortcut_x = x

        observer = getattr(self, "_merge_observer", None)
        if observer is not None:
            observer.capture(
                block_name=getattr(self, "_merge_block_name", self.__class__.__name__),
                input_attn=input_attn,
                input_attn_2=input_attn_2,
                output_attn=output_attn,
                post_shortcut_x=post_shortcut_x,
                window_size=self.window_size,
                x_attn=x_attn,
            )

        x = x + self.mlp(self.norm2(x))
        return x
class MergeAttentionObserver(Attention):
    """Attention wrapper that captures tensors for merge analysis."""


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x , attn
class ImageEncoderViTMergeObserver(ImageEncoderViT):
    """Image encoder wrapper that exposes the shared merge observer."""

    def set_merge_observer(
        self, observer: MergeObserver, name: str = "image_encoder"
    ) -> None:
        self._merge_observer = observer
        self._merge_encoder_name = name

    def clear_merge_observer(self) -> None:
        observer = getattr(self, "_merge_observer", None)
        if observer is not None:
            observer.clear()



def patch_sam_for_merging(
    model: torch.nn.Module,
    observer: Optional[MergeObserver] = None,
    gradient_method: str = "sobel",
    store_on_cpu: bool = True,
    detach_tensors: bool = True,
) -> MergeObserver:
    """Monkey-patch SAM encoder blocks to capture merge analysis tensors."""

    observer = observer or MergeObserver(
        gradient_method=gradient_method,
        store_on_cpu=store_on_cpu,
        detach_tensors=detach_tensors,
    )

    for name, module in model.named_modules():
        if isinstance(module, ImageEncoderViT):
            module.__class__ = ImageEncoderViTMergeObserver
            module.set_merge_observer(observer, name=name)
        if isinstance(module, EncoderBlock):
            module.__class__ = MergeBlockObserver
            module.set_merge_observer(observer, name=name)
        if isinstance(module, Attention):
            module.__class__ = MergeAttentionObserver

    return observer
