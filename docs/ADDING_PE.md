# Adding a new PE algorithm

This walks through adding a new token-compression or attention-replacement
patch to the **Perception Encoder (PE)** evaluation pipeline. Once
registered, the algorithm appears automatically in
[tasks/pe_imagenet/eval_pe_clip.py](../tasks/pe_imagenet/eval_pe_clip.py)
(zero-shot CLIP eval) and
[tasks/pe_imagenet/profile_pe.py](../tasks/pe_imagenet/profile_pe.py)
(per-block latency profiler).

PE patches use the **same shape as SAM patches**: subclass the encoder's
`Block` / `Attention` classes (here, `ResidualAttentionBlock` /
`SelfAttention`) and reassign `module.__class__` at apply time.
[ADDING_SAM.md](ADDING_SAM.md) explains the pattern in detail; this doc
covers PE-specific knobs.

---

## 1. Where the registry lives

[`PiToMe/algo/registry.py`](../PiToMe/algo/registry.py). Each PE algorithm
is one `PEAlgoSpec` entry:

```python
@dataclass
class PEAlgoSpec:
    name: str                          # CLI string (e.g. "tome_partial")
    apply: Callable                    # (model, **kwargs) -> Any
    remove: Optional[Callable] = None  # only needed for back-compat
    kwargs_from_args: Optional[Callable] = None
    accepts_ratio: bool = True
    description: str = ""
    category: str = "compress"         # "compress" | "partial" | "attention"
```

Eval and profile scripts only ever call two functions from this module:

  * `apply_pe(model, name, args, ratio)` — looks up the spec, builds
    kwargs from `kwargs_from_args(args, ratio)`, calls `spec.apply(...)`.
  * `remove_all_pe(model)` — walks the model and reverts **any subclass**
    of `ResidualAttentionBlock` / `SelfAttention` back to its stock class,
    then removes registered hooks and clears `model._tome_info`. You don't
    need to register your subclasses anywhere — `remove_all_pe` detects them
    structurally.

The reason for the `category` and `accepts_ratio` fields is sweep control:
`"attention"` patches like `flash_rope` swap the attention kernel without
changing token counts, so it makes no sense to sweep them across multiple
ratios — `accepts_ratio=False` makes the eval loop run them once per
sweep instead of once per `--ratio` value.

## 2. How PE patches work

Same idea as SAM:

  1. Subclass `SelfAttention` (and usually `ResidualAttentionBlock`) and
     override `forward()` with your reduced-work implementation.
  2. Walk `transformer.resblocks`, and for each existing
     `SelfAttention` / `ResidualAttentionBlock` instance, **reassign
     `module.__class__`** to your subclass. Weights stay in place.
  3. Stash patch state on `model._tome_info`. Every patched module reads
     from `self._tome_info` (the apply function points it at the same dict).

Two structural patch flavors live alongside each other in the codebase:

### Stage-compression — token count drops at boundaries

PE blocks are split into N stages; between stages, a designated
"compress" block runs the original block forward and then drops/merges
tokens. Built on the base classes in
[`_pe_stage.py`](../PiToMe/algo/_pe_stage.py):

  * `FlashRopePEAttention(SelfAttention)` — slices `rope.freq` to the
    surviving tokens (`info["active_idx"]`) and optionally routes through
    the fused FA2+RoPE cute kernel. Used unchanged by every stage-compress
    algo — you don't subclass it.
  * `StageCompressPEBlock(ResidualAttentionBlock)` — base for stage-end
    blocks. Subclasses override `compress(x, active_idx, info)
    -> (x_reduced, new_active_idx)` to define the merge rule.

So writing a new stage-compress algo is mostly: subclass
`StageCompressPEBlock`, implement `compress()`, then call
`apply_stage_compress(...)` to install it.

### Partial — full token count, K/V or MLP work reduction

Token sequence stays at S throughout; reduce work via merged K/V before
attention and/or merge → MLP → unmerge. No shared base class — each algo
defines its own `<Algo>PEPartialAttention(SelfAttention)` and
`<Algo>PEPartialBlock(ResidualAttentionBlock)` outright. The `apply`
function walks `transformer.resblocks` and reassigns `__class__`, mirroring
SAM almost exactly. See [tome/pe_partial.py](../PiToMe/algo/tome/pe_partial.py)
for the canonical example.

---

## 3. Walkthrough A — adding a stage-compress `mytome`

```python
# PiToMe/algo/mytome/pe_compress.py
from .._pe_stage import (
    apply_stage_compress, FlashRopePEAttention, StageCompressPEBlock,
)


class MyTomePECompressBlock(StageCompressPEBlock):
    def compress(self, x, active_idx, info):
        ratio: float = info["ratio"]
        has_cls: bool = info.get("use_cls_token", False)

        # Your matcher here. Must return (x_reduced, new_active_idx).
        # `active_idx` (S,) is the absolute position of each surviving
        # token in the original sequence — needed so RoPE picks the
        # right cos/sin. Return the new (smaller) active_idx matching
        # x_reduced.
        ...
        return x_reduced, new_active_idx


def apply_pe_mytome_patch(model, ratio=0.7, num_stages=4,
                          use_flash_rope=False, group_size=4,
                          compress_at_blocks=None, verbose=True):
    info = {"algo": "mytome", "ratio": ratio, "num_stages": num_stages,
            "group_size": group_size,
            "use_flash_rope": bool(use_flash_rope)}
    return apply_stage_compress(
        model,
        compress_block_class=MyTomePECompressBlock,
        attn_class=FlashRopePEAttention,
        info=info,
        num_stages=num_stages,
        use_flash_rope=use_flash_rope,
        compress_at_blocks=compress_at_blocks,
        verbose_tag="pe-mytome-stage" if verbose else "",
    )
```

Imports stay at module top — same shape as the SAM patches.
`_pe_stage.py` adds `perception_models/` to `sys.path` once at import
time (mirroring what each SAM patch does for `sam-hq/`), so
`core.vision_encoder.pe` resolves whether PE is pip-installed or only
present as a submodule.

Then register in [`registry.py`](../PiToMe/algo/registry.py)'s
`_register_builtins()`:

```python
from .mytome.pe_compress import apply_pe_mytome_patch
register_pe(PEAlgoSpec(
    name="mytome",
    apply=apply_pe_mytome_patch,
    kwargs_from_args=_kw_compress,
    category="compress",
    description="...",
))
```

`remove` is optional — `remove_all_pe` walks the model and reverts any
subclass of `ResidualAttentionBlock` / `SelfAttention` automatically.

---

## 4. Walkthrough B — adding a partial `mytome_partial`

A partial patch defines its own attention + block subclasses (not built on
`StageCompressPEBlock`):

```python
# PiToMe/algo/mytome/pe_partial.py
from .._pe_stage import (
    SelfAttention, ResidualAttentionBlock,
    _find_vision_transformer, _vit_uses_cls_token,
)


class MyTomePEPartialAttention(SelfAttention):
    def forward(self, x, attn_mask=None):
        info = self._tome_info
        ratio = info.get("ratio", 1.0)
        if ratio >= 1.0:
            return super().forward(x, attn_mask=attn_mask)
        # Custom attention with merged K/V here.
        # Stash any merge/unmerge closures on `info` for the block to reuse.
        ...


class MyTomePEPartialBlock(ResidualAttentionBlock):
    def forward(self, x, attn_mask=None):
        info = self._tome_info
        # Standard PE residual structure with optional merge/MLP/unmerge.
        x = x + self.drop_path1(self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask)))
        ...
        return x


def apply_pe_mytome_partial_patch(model, ratio=0.7, start_block=0,
                                  mlp_merge=True, verbose=True):
    transformer = _find_vision_transformer(model)
    n_blocks = len(transformer.resblocks)
    sb = max(0, min(int(start_block), n_blocks))

    info = {"algo": "mytome_partial", "ratio": float(ratio),
            "use_cls_token": _vit_uses_cls_token(model),
            "start_block": sb, "mlp_merge": bool(mlp_merge)}
    model._tome_info = info

    for idx in range(sb, n_blocks):
        blk = transformer.resblocks[idx]
        attn = blk.attn
        if isinstance(attn, SelfAttention) and attn.rope is not None:
            if not isinstance(attn, MyTomePEPartialAttention):
                attn.__class__ = MyTomePEPartialAttention
            attn._tome_info = info
        if not isinstance(blk, MyTomePEPartialBlock):
            blk.__class__ = MyTomePEPartialBlock
        blk._tome_info = info

    return n_blocks - sb
```

Re-export `SelfAttention` and `ResidualAttentionBlock` from `_pe_stage`
rather than importing directly from `core.vision_encoder.pe` — that keeps
the path-mutation in one place.

Compare this to SAM's
[tome/sam.py](../PiToMe/algo/tome/sam.py) — the shape is identical
(define subclasses, walk modules, reassign `__class__`, stash state).

Register:

```python
from .mytome.pe_partial import apply_pe_mytome_partial_patch
register_pe(PEAlgoSpec(
    name="mytome_partial",
    apply=apply_pe_mytome_partial_patch,
    kwargs_from_args=_kw_partial_basic,
    category="partial",
    description="...",
))
```

---

## 5. Run

Both scripts pick the new name up automatically:

```bash
python tasks/pe_imagenet/eval_pe_clip.py --algorithm mytome_partial \
    --ratio 0.7 0.5 --partial-start-block 5 --no-amp \
    --dataset imagenet1k --dataset-root ./data/imagenet

python tasks/pe_imagenet/profile_pe.py --tome-algo mytome_partial --tome-ratio 0.5
```

---

## 6. Picking `kwargs_from_args`

Maps the parsed argparse Namespace + the per-config ratio to the kwargs
your `apply` function expects. Pre-built builders for the common cases:

| Builder | Patch shape | CLI flags it forwards |
|---|---|---|
| `_kw_compress` | stage compression | `--num-stages`, `--group-size`, `--use-flash-rope`, `--compress-at-blocks` |
| `_kw_partial_basic` | partial w/o sparse-attn | `--partial-start-block`, `--mlp-merge` |
| `_kw_partial_sparsesam` | partial + sparsesam mask | above + `--group-size`, `--sparse-ratio` |

If your algorithm needs a CLI option that doesn't yet exist, add it in
**both** [eval_pe_clip.py](../tasks/pe_imagenet/eval_pe_clip.py) and
[profile_pe.py](../tasks/pe_imagenet/profile_pe.py), then write a custom
`kwargs_from_args` that reads it.

---

## 7. Categories and gating

  * **`"compress"`** — token count drops at boundaries. One run per ratio.
  * **`"partial"`** — full token count preserved. Eligible for
    `--sweep-mlp-merge` (which doubles the run grid by toggling `mlp_merge`).
  * **`"attention"`** — pure attention-kernel swap, no token math. Usually
    `accepts_ratio=False` so it runs once per sweep (e.g. `flash_rope`).

---

## 8. Smoke test before running a real eval

A 30-second sanity check catches the common failure modes (shape
mismatches, dtype confusion, broken `remove`) before you spend 20 minutes
on an ImageNet sweep:

```python
import torch, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'perception_models')
from core.vision_encoder import pe
from PiToMe.algo.registry import apply_pe, remove_all_pe

model = pe.CLIP.from_config("PE-Core-S16-384", pretrained=False).eval()

class A: ...
args = A(); args.ratio = [0.5]; args.partial_start_block = 0; args.mlp_merge = True

apply_pe(model, "mytome_partial", args=args, ratio=0.5)
x = torch.randn(2, 3, 384, 384)
out = model.encode_image(x); print(out.shape)         # (2, output_dim)

remove_all_pe(model)
out2 = model.encode_image(x); print(torch.allclose(out, out2))   # remove worked
```

The `remove_all_pe` step should restore bit-identical numerics — if it
doesn't, your `apply` is leaking state somewhere outside `_tome_info` /
class assignment.

---

## 9. Sweep + plot trade-off curves

```bash
ALGOS_SWEEP="tome_partial mytome_partial gradtome_partial" \
RATIOS_SWEEP="0.9 0.7 0.5 0.4 0.3 0.25" \
sh tasks/pe_imagenet/eval_pe.sh

python tasks/pe_imagenet/plot_pe_partial.py ./benchmark_results/pe_clip_*.csv \
    --metric acc1 --x ratio --out ./benchmark_results/pe_partial_tradeoff.png
```

One curve per `(algo, mlp_merge)`. Solid lines are `mlp_merge=True`,
dashed are `False`, and the baseline ("none") is drawn as a horizontal
reference.

---

## 10. Notes 

  * **Autocast + LayerNorm** — PyTorch's autocast policy upcasts LN
    outputs to fp32 mid-stack. If your patch dispatches based on
    `x.dtype`, you'll see fp32 instead of the model's fp16/bf16 and your
    cute-kernel cache key will miss. Key on the *weight* dtype instead —
    see `_kernel_dtype` in
    [sparsesam/pe_partial.py](../PiToMe/algo/sparsesam/pe_partial.py). Or
    run with `--no-amp` to keep the whole forward in `--dtype`.
  * **`bipartite_soft_matching` halves at most per call** (`r ≤ T/2`).
    For `ratio < 0.5`, use `_chained_bipartite_match` from
    [tome/pe_partial.py](../PiToMe/algo/tome/pe_partial.py) /
    [gradtome/pe_partial.py](../PiToMe/algo/gradtome/pe_partial.py),
    which composes multiple passes so the cumulative product equals the
    requested ratio.
  * **CLS handling** — pass `class_token=info["use_cls_token"]` to your
    matcher so the CLS token isn't dropped or merged. The flag is set by
    `apply_stage_compress` / `apply_pe_*_patch` based on whether the
    underlying `VisionTransformer` uses CLS.
  * **RoPE under compression** — when tokens get dropped, the surviving
    tokens' positions shift. `FlashRopePEAttention` re-indexes
    `rope.freq` from the `active_idx` you return. Stage-compress patches
    just need to return the new `active_idx`; the plumbing does the rest.
  * **Cute kernel availability** — `_get_kernel(dtype, head_dim)` returns
    `(None, None, None)` when no `_BLOCK_CANDIDATES` tile satisfies
    `can_implement` on the target GPU (typically due to SMEM or thread
    constraints). The diagnostic prints which candidates failed and why —
    read it before tweaking tile sizes.
