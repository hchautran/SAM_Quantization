import argparse
import time
import torch
import qgemm  # type: ignore
 


@torch.inference_mode()
def sym_quant_rowwise(x: torch.Tensor):
    assert x.dim() == 2 and x.dtype == torch.float16 and x.is_cuda
    scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 7.0
    q = qgemm.sym_quant(x, scale)
    return q, scale


def run_once(M: int, N: int, K: int, warmup: int, iters: int, dtype: str, check_error: bool = False):
    device = torch.device("cuda")
    torch.manual_seed(0)

    xa = torch.randn((M, K), device=device, dtype=torch.float16)
    xb = torch.randn((N, K), device=device, dtype=torch.float16)

    qa, sa = sym_quant_rowwise(xa)
    qb, sb = sym_quant_rowwise(xb)

    for _ in range(warmup):
        c = qgemm.matmul(qa, qb)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c = qgemm.matmul(qa, qb)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / iters
    macs = M * N * K
    tflops = (macs / 1e12) / (avg_ms / 1000.0)
    print(f"INT4 matmul:    M={M} N={N} K={K} | latency={avg_ms:.3f} ms | Throughput={tflops:.2f} TFLOPS (int4 muls)")

    for _ in range(warmup):
        c_fp = xa @ xb.t()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        c_fp = xa @ xb.t()
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms_fp = (end - start) * 1000.0 / iters
    tflops_fp = (macs / 1e12) / (avg_ms_fp / 1000.0)
    print(f"FP16 matmul:    M={M} N={N} K={K} | latency={avg_ms_fp:.3f} ms | Throughput={tflops_fp:.2f} TFLOPS")

    if check_error:
        s = sa @ sb.t() 
        c_deq = c.to(torch.float16) * s
        diff = (c_deq - c_fp).float()
        mae = diff.abs().mean().item()
        rmse = torch.sqrt((diff * diff).mean()).item()
        denom = c_fp.abs().mean().clamp(min=1e-6)
        rel_mae = (diff.abs().mean() / denom).item()
        print(f"Dequant error:  MAE={mae:.4e} | RMSE={rmse:.4e} | RelMAE={rel_mae:.4e}")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark qgemm int4 matmul kernel")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--check-error", action="store_true")
    # Multiple sizes support
    p.add_argument(
        "--sizes",
        type=str,
        nargs="*",
        help="List of sizes like 1024x1024x4096 (MxNxK). If provided, overrides --M/--N/--K.",
    )
    p.add_argument(
        "--preset",
        type=str,
        choices=["square", "llm"],
        help="Preset list of sizes: square (512/1024/2048), llm (token-projection-like).",
    )
    return p.parse_args()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required"
    args = parse_args()
    def parse_size_token(tok: str):
        tok = tok.lower().replace(",", "x").replace("*", "x")
        parts = [t for t in tok.split("x") if t]
        assert len(parts) == 3, f"Invalid size token: {tok}"
        return tuple(int(t) for t in parts)

    sizes = []
    if args.preset:
        if args.preset == "square":
            sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
        elif args.preset == "llm":
            # Typical projection-like sizes (batch flattened): moderate to avoid OOM
            sizes = [
                (1024, 1024, 4096),
                (2048, 2048, 4096),
                (1024, 1024, 8192),
            ]
    if args.sizes:
        sizes.extend([parse_size_token(s) for s in args.sizes])

    if sizes:
        for (M, N, K) in sizes:
            run_once(M, N, K, args.warmup, args.iters, dtype="i4", check_error=args.check_error)
    else:
        run_once(args.M, args.N, args.K, args.warmup, args.iters, dtype="i4", check_error=args.check_error)


