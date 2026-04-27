"""Basic encoder attention processors for SAM quantization."""

import os
import time
from typing import Callable, Dict, Optional

import torch
from segment_anything.modeling.image_encoder import (
    add_decomposed_rel_pos as _sam_add_decomposed_rel_pos,
    get_rel_pos,
)
from ..base import AttentionProcessor


def _get_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _format_ms(latency_ms: float) -> str:
    return f"{latency_ms:.3f}"


def _zero_relpos_timings() -> Dict[str, float]:
    return {
        "relpos_total": 0.0,
        "relpos_einsum_h": 0.0,
        "relpos_einsum_w": 0.0,
        "relpos_get_h": 0.0,
        "relpos_get_w": 0.0,
        "pos": 0.0,
        "attn_combine": 0.0,
    }


class _OnlineAttentionProfiler:
    """CUDA-aware online profiler for per-layer attention timing."""

    def __init__(self) -> None:
        self.enabled = _get_env_flag("SAM_PROFILE_ENCODER_ATTN", default=False)
        self.warmup = max(0, _get_env_int("SAM_PROFILE_ENCODER_ATTN_WARMUP", 3))
        self.print_every = max(1, _get_env_int("SAM_PROFILE_ENCODER_ATTN_PRINT_EVERY", 1))
        self.stats: Dict[str, Dict[str, object]] = {}

    def _get_entry(self, module_name: Optional[str]) -> Dict[str, object]:
        key = module_name or "<unnamed>"
        if key not in self.stats:
            self.stats[key] = {
                "calls": 0,
                "timed_calls": 0,
                "totals": {},
            }
        return self.stats[key]

    def begin_call(self, module_name: Optional[str]) -> bool:
        if not self.enabled:
            return False

        entry = self._get_entry(module_name)
        entry["calls"] = int(entry["calls"]) + 1
        return int(entry["calls"]) > self.warmup

    def measure(self, reference_tensor: torch.Tensor, fn: Callable[[], torch.Tensor]):
        if reference_tensor.is_cuda:
            stream = torch.cuda.current_stream(reference_tensor.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            result = fn()
            end_event.record(stream)
            end_event.synchronize()
            return result, start_event.elapsed_time(end_event)

        start_time = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start_time) * 1000.0

    def record(self, module_name: Optional[str], shapes: Dict[str, int], timings_ms: Dict[str, float]) -> None:
        if not self.enabled:
            return

        key = module_name or "<unnamed>"
        entry = self._get_entry(key)
        entry["timed_calls"] = int(entry["timed_calls"]) + 1
        timed_calls = int(entry["timed_calls"])
        totals = entry["totals"]

        for name, value in timings_ms.items():
            totals[name] = float(totals.get(name, 0.0)) + float(value)

        if timed_calls % self.print_every != 0:
            return

        avg_total = float(totals["attention_total"]) / timed_calls
        print(
            ""
            f"layer={key} "
            f"call={int(entry['calls'])} "
            f"profiled={timed_calls} "
            f"B={shapes['batch']} H={shapes['height']} W={shapes['width']} heads={shapes['heads']} "
            f"qk={_format_ms(timings_ms['qk'])}ms "
            f"relpos={_format_ms(timings_ms['relpos_total'])}ms "
            f"(get_h={_format_ms(timings_ms['relpos_get_h'])}ms "
            f"get_w={_format_ms(timings_ms['relpos_get_w'])}ms "
            f"rel_h={_format_ms(timings_ms['relpos_einsum_h'])}ms "
            f"rel_w={_format_ms(timings_ms['relpos_einsum_w'])}ms "
            f"pos={_format_ms(timings_ms['pos'])}ms "
            f"attn_combine={_format_ms(timings_ms['attn_combine'])}ms) "
            f"softmax={_format_ms(timings_ms['softmax'])}ms "
            f"attn_v={_format_ms(timings_ms['attn_v'])}ms "
            f"total={_format_ms(timings_ms['attention_total'])}ms "
            f"avg_total={_format_ms(avg_total)}ms",
            flush=True,
        )


def _profiled_add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size,
    k_size,
    profiler: _OnlineAttentionProfiler,
):
    """
    Preserve the SAM implementation while exposing timings for the two einsums
    and the full relative-position block.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh, relpos_get_h_ms = profiler.measure(rel_pos_h, lambda: get_rel_pos(q_h, k_h, rel_pos_h))
    Rw, relpos_get_w_ms = profiler.measure(rel_pos_w, lambda: get_rel_pos(q_w, k_w, rel_pos_w))

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    rel_h, relpos_einsum_h_ms = profiler.measure(q, lambda: torch.einsum("bhwc,hkc->bhwk", r_q, Rh))
    rel_w, relpos_einsum_w_ms = profiler.measure(q, lambda: torch.einsum("bhwc,wkc->bhwk", r_q, Rw))

    # attn, relpos_combine_ms = profiler.measure(
    #     attn,
    #     lambda: (
    #         attn.view(B, q_h, q_w, k_h, k_w)
    #         + rel_h[:, :, :, :, None]
    #         + rel_w[:, :, :, None, :]
    #     ).view(B, q_h * q_w, k_h * k_w),
    # )
    pos, pos_ms = profiler.measure(
        attn,
        lambda: (
            rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        ).reshape(B, q_h * q_w, k_h * k_w),
    )
    attn, attn_combine_ms = profiler.measure(attn, lambda: attn + pos)
    timings_ms = {
        "relpos_einsum_h": relpos_einsum_h_ms,
        "relpos_einsum_w": relpos_einsum_w_ms,
        "relpos_get_h": relpos_get_h_ms,
        "relpos_get_w": relpos_get_w_ms,
        "pos": pos_ms,
        "attn_combine": attn_combine_ms,
    }
    timings_ms["relpos_total"] = (
        timings_ms["relpos_get_h"]
        + timings_ms["relpos_get_w"]
        + timings_ms["relpos_einsum_h"]
        + timings_ms["relpos_einsum_w"]
        + timings_ms["pos"]
        + timings_ms["attn_combine"]
    )

    return attn, timings_ms


class EncoderAttentionProcessor(AttentionProcessor):
    """
    Processor for calibrating and processing image encoder attention layers.

    This processor is designed specifically for the ViT-based image encoder
    in SAM, which has a different architecture than the mask decoder.
    """

    def __init__(self, strategy_name: str = 'base'):
        super().__init__(strategy_name)
        self.stat = {}
        self.attention_profiler = _OnlineAttentionProfiler()

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        """
        Collect statistics for linear layers (QKV projections).

        Args:
            X: Input tensor
            Y: Output tensor from linear layer
            name: Module name
            linear_name: Linear layer name (e.g., 'qkv', 'proj')
        """
        pass

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass

    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """
        Separate QKV tensor into heads for encoder attention.

        Args:
            qkv: Combined QKV tensor of shape (B, H*W, 3, num_heads, C_per_head)
            num_heads: Number of attention heads

        Returns:
            q, k, v tensors each of shape (B, num_heads, H*W, C_per_head)
        """
        # qkv shape: (B, H*W, 3, num_heads, C_per_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, C_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            x: Input tensor
            module: Attention module
            module_name: Module name

        Returns:
            Processed output tensor
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        # should_profile = self.attention_profiler.begin_call(module_name)
        should_profile = False
        relpos_timings = _zero_relpos_timings()

        if should_profile:
            attn, qk_ms = self.attention_profiler.measure(
                q,
                lambda: (q * module.scale) @ k.transpose(-2, -1),
            )
        else:
            attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            if should_profile:
                attn, relpos_timings = _profiled_add_decomposed_rel_pos(
                    attn,
                    q,
                    module.rel_pos_h,
                    module.rel_pos_w,
                    (H, W),
                    (H, W),
                    self.attention_profiler,
                )
            else:
                attn = _sam_add_decomposed_rel_pos(
                    attn,
                    q,
                    module.rel_pos_h,
                    module.rel_pos_w,
                    (H, W),
                    (H, W),
                )

        if should_profile:
            attn, softmax_ms = self.attention_profiler.measure(attn, lambda: attn.softmax(dim=-1))
            x, attn_v_ms = self.attention_profiler.measure(
                attn,
                lambda: (attn @ v)
                .view(B, module.num_heads, H, W, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1),
            )
            timings_ms = {
                "qk": qk_ms,
                **relpos_timings,
                "softmax": softmax_ms,
                "attn_v": attn_v_ms,
            }
            timings_ms["attention_total"] = (
                timings_ms["qk"]
                + timings_ms["relpos_total"]
                + timings_ms["softmax"]
                + timings_ms["attn_v"]
            )
            self.attention_profiler.record(
                module_name,
                {
                    "batch": B,
                    "height": H,
                    "width": W,
                    "heads": module.num_heads,
                },
                timings_ms,
            )
        else:
            attn = attn.softmax(dim=-1)
            x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        x = module.proj(x)
        return x


class EncoderRecenterAttentionProcessor(AttentionProcessor):
    """
    Encoder processor that recenters QKV projections by subtracting spatial means.

    Processing Strategy:
    -------------------
    1. Computes spatial mean: x_mean = x.mean(H).mean(W)
    2. Subtracts mean: x_hat = x - x_mean
    3. Projects: qkv_hat = QKV(x_hat), qkv_mean = QKV(x_mean)
    4. Computes attention with recentered keys
    5. Applies attention to recentered values

    This helps with quantization by centering activations around zero.
    """

    def __init__(self, strategy_name: str = 'recentered'):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        pass

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass

    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """Separate QKV tensor into heads for encoder attention."""
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process with recentering for better quantization."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        x_mean = x.mean(1, keepdim=True).mean(2, keepdim=True)
        x_hat = x - x_mean
        qkv_hat = module.qkv(x_hat).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_mean = module.qkv(x_mean).reshape(B, 1, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q_hat, k_hat, v_hat = qkv_hat.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        q_mean, k_mean, v_mean = qkv_mean.reshape(3, B * module.num_heads, 1, -1).unbind(0)

        attn = (q_hat * module.scale) @ k_hat.transpose(-2, -1)

        if module.use_rel_pos:
            attn = _sam_add_decomposed_rel_pos(attn, q_hat + q_mean, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = ((attn @ v_hat) + v_mean).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x
