# Adding a new SAM algorithm

This walks through adding a new token-compression patch to the **SAM-HQ**
evaluation pipeline. Once registered, the algorithm appears automatically
in
[tasks/sam_hq44k/eval_hq44k.py](../tasks/sam_hq44k/eval_hq44k.py) (HQ44K
mIoU/throughput sweep) and
[tasks/sam_coco/eval_coco.py](../tasks/sam_coco/eval_coco.py) (COCO
val2017 with GT-box prompts).

If you haven't read the overview yet, start with
[ADDING_ALGORITHMS.md](ADDING_ALGORITHMS.md).

---

## 1. Where the registry lives

[`PiToMe/algo/registry.py`](../PiToMe/algo/registry.py). Each SAM
algorithm is one `SAMAlgoSpec` entry:

```python
@dataclass
class SAMAlgoSpec:
    name: str
    apply: Callable                            # (encoder, **kwargs) -> Any
    block_class: Optional[type] = None         # subclass of Block
    attn_class:  Optional[type] = None         # subclass of Attention
    kwargs_from_args: Optional[Callable] = None
    accepts_ratio: bool = True
    description: str = ""
```

Eval scripts only call two functions from this module:

  * `apply_sam(encoder, name, args, ratio, **extra)` — looks up the spec,
    builds kwargs from `kwargs_from_args(args, ratio)`, calls
    `spec.apply(encoder, **kwargs)`. Returns early if `name == "none"` or
    `ratio >= 1.0` (so a sweep can include the baseline trivially).
  * `remove_all_sam(encoder, mask_decoder=None)` — walks every module on
    the encoder and reverts any subclassed `Block`/`Attention` listed in
    a registered spec's `block_class`/`attn_class` fields back to the
    stock `Block`/`Attention` class. Also clears `encoder.tome_info` and
    any installed instance-level `forward` overrides.

That last point is the reason `block_class` and `attn_class` exist on the
spec: the registry needs to know which subclasses to look for so it can
revert them between sweep configs without you having to track which patch
is currently active.

---

## 2. How SAM patches work

The SAM-HQ image encoder is an MAE-style ViT. Each
`segment_anything.modeling.image_encoder.Block` contains an `Attention`
module and an MLP. To install a patch you:

  1. Subclass `Block` (and usually `Attention`) and override `forward()`
     with your reduced-work implementation.
  2. Walk `encoder.modules()`, and for each existing `Block`/`Attention`
     instance, **reassign `module.__class__`** to your subclass. This
     keeps the original weights and `nn.Module` registration intact —
     you're only changing which `forward()` runs.
  3. Stash any per-image patch state (ratio, caches) on
     `encoder.tome_info` so `remove_all_sam` can clean it up.

Removing the patch is just the reverse `__class__` reassignment, plus
deleting `tome_info`. `remove_all_sam` does both for you.

There's an extra wrinkle vs. PE: SAM-HQ has both **windowed local
blocks** (14×14 attention) and **global attention blocks**. Your block
subclass must handle `self.window_size > 0` (windowed via
`window_partition` / `window_unpartition`) and `self.window_size == 0`
(plain dense attention). The sparsesam patch is a good reference for the
two-branch shape — see
[sparsesam/sam.py](../PiToMe/algo/sparsesam/sam.py)'s `ToMeSAMBlock.forward`.

---

## 3. Three steps

### Step 1 — Write the patch

`PiToMe/algo/<myalgo>/sam.py`:

```python
from segment_anything.modeling.image_encoder import (
    ImageEncoderViT, Block, Attention,
)

class MyAlgoSAMBlock(Block):
    def forward(self, x):
        # x: (B, H, W, C). Use self._tome_info / self.attn / self.mlp etc.
        # Branch on self.window_size > 0 vs == 0.
        ...

class MyAlgoSAMAttention(Attention):
    def forward(self, x):
        # x: (B, H, W, C). Implement your reduced attention here.
        ...

def apply_patch(encoder: ImageEncoderViT,
                algo: str = "tome",
                ratio: float = 0.9,
                margin: float = 0.5,
                mlp_merge: bool = True) -> ImageEncoderViT:
    """Install the patch in-place. Return the same encoder."""
    encoder.tome_info = {
        "algo":      algo,
        "ratio":     ratio,
        "margin":    margin,
        "mlp_merge": mlp_merge,
        # plus any per-image caches your forwards need
    }

    for module in encoder.modules():
        if isinstance(module, Block) and not isinstance(module, MyAlgoSAMBlock):
            module.__class__ = MyAlgoSAMBlock
            module._tome_info = encoder.tome_info
        elif isinstance(module, Attention) and not isinstance(module, MyAlgoSAMAttention):
            module.__class__ = MyAlgoSAMAttention
            module._tome_info = encoder.tome_info

    return encoder
```

The full canonical skeleton (with windowed/global branches, ratio scheduling,
and per-image cache reset) is in
[tome/sam.py](../PiToMe/algo/tome/sam.py).

### Step 2 — Register

In [`registry.py`](../PiToMe/algo/registry.py)'s
`_register_sam_builtins()`:

```python
from .myalgo.sam import (
    apply_patch as _patch_my,
    MyAlgoSAMBlock as _B, MyAlgoSAMAttention as _A,
)
register_sam(SAMAlgoSpec(
    name="myalgo",
    apply=_wrap_apply_with_internal_algo(_patch_my, "tome"),
    block_class=_B, attn_class=_A,
    kwargs_from_args=_sam_kw_basic,
    description="What this patch does, one sentence.",
))
```

What each piece does:

  * **`_wrap_apply_with_internal_algo(_patch_my, "tome")`** — adapter that
    calls `_patch_my(encoder, algo="tome", ratio=…, margin=…, **extra)`.
    The `"tome"` baked-in lets you reuse the same patch module for both
    `"tome"` and `"pitome"` flavors by registering twice with different
    internal algo strings (this is how
    `tome` / `pitome` and `sparsesam` / `sparsesam_pitome` are wired). The
    wrapper inspects your `apply_patch` signature and only forwards
    optional kwargs (e.g. `mlp_merge`) to patches that actually accept
    them — old patches without the option keep working.
  * **`block_class=_B, attn_class=_A`** — what `remove_all_sam` looks for
    when reverting the subclassing.
  * **`kwargs_from_args=_sam_kw_basic`** — the standard adapter forwards
    `--ratio`, `--margin`, `--mlp-merge`. If your patch needs more
    knobs, add them to both eval scripts' argparse and write a custom
    builder.

### Step 3 — Run

Both eval scripts pick the new name up automatically:

```bash
python tasks/sam_hq44k/eval_hq44k.py --algos myalgo --ratios 0.9 0.7 \
    --batch-sizes 1 2 4 --num-samples 100 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l

python tasks/sam_coco/eval_coco.py --algos myalgo --ratios 0.9 0.7 \
    --coco-root ./data/coco --num-images 200 --ap \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l
```

`eval_hq44k.py` reports throughput, per-image encoder latency, peak
memory, mIoU, and Boundary IoU. `eval_coco.py` reports per-instance
mIoU/B-IoU on GT-box prompts and (with `--ap`) COCO segm AP.

---

## 4. Smoke test before running a real eval

This catches shape mismatches, broken `remove`, and dtype issues in
~10 seconds:

```python
import torch, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'sam-hq')
from segment_anything import sam_model_registry
from PiToMe.algo.registry import apply_sam, remove_all_sam

sam = sam_model_registry["vit_l"](checkpoint="./ckts/sam_hq_vit_l.pth").cuda().half()
encoder = sam.image_encoder

# Fake the args namespace your kwargs_from_args reads.
class A: ...
args = A(); args.ratios = [0.7]; args.margin = 0.5; args.mlp_merge = True

apply_sam(encoder, "myalgo", args=args, ratio=0.7)
x = torch.randn(1, 3, 1024, 1024, device="cuda", dtype=torch.float16)
out, _ = encoder(x); print(out.shape)              # expect (1, 256, 64, 64)

remove_all_sam(encoder)
out2, _ = encoder(x); print(torch.allclose(out, out2))   # remove worked
```

---

## 5. Notes 

  * **`apply_patch` signature must accept `algo`, `ratio`, `margin`** —
    the registry wrapper passes these positionally-by-name. Extra kwargs
    (`mlp_merge`, `group_size`, …) are fine to add — the wrapper
    introspects your signature and only forwards the ones you actually
    declare, so older patches without the option keep working.
  * **Subclass via `__class__` reassignment, not `Module.__init__`** —
    re-instantiating a `Block` would lose the trained weights. The
    `__class__` swap keeps every parameter and buffer in place; you're
    only changing which `forward()` runs.
  * **Stash state on `encoder.tome_info`**, not as instance attrs on each
    block, so `remove_all_sam` can clean it up with one `del`. If you
    need per-block state too, set `module._tome_info = encoder.tome_info`
    so they share the same dict.
  * **Local vs global blocks** — SAM-HQ `vit_l` has windowed (14×14) local
    blocks plus a few global attention blocks. Your block subclass needs
    both branches. The local branch typically does
    `window_partition` → patched attention → `window_unpartition`; the
    global branch runs patched attention on the full sequence. See
    [sparsesam/sam.py](../PiToMe/algo/sparsesam/sam.py)'s
    `ToMeSAMBlock.forward` for the canonical two-branch shape.
  * **Per-image cache reset** — if your patch caches anything per image
    (token permutations, masks), reset it inside a wrapped
    `encoder.forward` so caches don't leak across images during a sweep.
    See `_patched_forward` in
    [tome/sam.py](../PiToMe/algo/tome/sam.py) for the pattern.
  * **Profiler bypasses the registry** —
    [tasks/sam_profile/profile_encoder.py](../tasks/sam_profile/profile_encoder.py)
    imports `apply_patch` directly from `algo.sparsesam.sam`. To profile
    a new algorithm there, edit the script's `apply_tome_patch` /
    `remove_tome_patch` to point at your patch module.
