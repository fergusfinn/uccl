"""Standalone ragged-dispatch hammer for the conc-8192 wedge.

Reproduces the serving failure population in the bare UCCL eager path:
per-rank token counts wildly skewed (including the exact observed wedge
shape and zero-token ranks), eager host-synced dispatch+combine, looped.
A hang here reproduces the serving wedge in minutes without vLLM.

Run: 2 nodes x 4 GPUs via run-multinode.sh (same as test_ht_cudagraph).
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from utils import init_dist_under_torchrun
from buffer import Buffer

try:
    from uccl.ep import Config
except ImportError:
    sys.stderr.write("Failed to import uccl.ep\n")
    raise


def compute_buffer_sizes(num_sms: int, hidden: int, num_ranks: int):
    hidden_bytes = hidden * 2
    config = Config(num_sms, 8, 512, 16, 512)

    def align(size, margin=1.2, alignment=128):
        return ((int(size * margin) + alignment - 1) // alignment) * alignment

    return (
        align(config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)),
        align(config.get_rdma_buffer_size_hint(hidden_bytes, num_ranks)),
    )


# The exact per-rank shape of the instrumented serving wedge (2026-06-10,
# arm A: node0=[2057,64,64,64], node1=[1028,1028,64,64]), plus adversarial
# variants: zero-token ranks, single-talker, all-tiny, chunk-boundary
# remainders. Trial t uses SHAPES[t % len(SHAPES)].
SHAPES = [
    [2057, 64, 64, 64, 1028, 1028, 64, 64],   # observed wedge
    [8192, 64, 64, 64, 64, 64, 64, 64],        # max skew
    [0, 64, 0, 2057, 0, 1028, 0, 64],          # zero-token ranks
    [1, 1, 1, 1, 1, 1, 1, 8192],               # tiny + one giant
    [64, 64, 64, 64, 64, 64, 64, 64],          # uniform control
    [2048, 2049, 1, 0, 257, 255, 129, 127],    # chunk-boundary edges
]


def run_eager_step(buffer, rank, num_tokens, hidden, num_experts, num_topk,
                   config, combine_config, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed * 8192 + rank)
    n = max(num_tokens, 0)
    x = torch.randn((n, hidden), dtype=torch.bfloat16, generator=g) * 0.1
    if n > 0:
        x[:, 0] = rank
    scores = torch.randn((n, num_experts), dtype=torch.float32, generator=g)
    if n > 0:
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    else:
        topk_idx = torch.empty((0, num_topk), dtype=torch.int64, device="cuda")
    topk_weights = torch.ones((n, num_topk), dtype=torch.float32)

    (ntpr, ntprr, ntpe, is_in_rank, _) = buffer.get_dispatch_layout(
        topk_idx, num_experts
    )
    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x,
        num_tokens_per_rank=ntpr,
        num_tokens_per_rdma_rank=ntprr,
        is_token_in_rank=is_in_rank,
        num_tokens_per_expert=ntpe,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_worst_tokens=0,  # eager host-synced path, as in the wedge
        config=config,
    )
    recv_x = recv_x * 2.0
    combined_x, _, _ = buffer.combine(recv_x, handle, config=combine_config)
    if n > 0:
        owner = is_in_rank.sum(dim=1).to(torch.bfloat16).unsqueeze(1)
        want = 2.0 * owner * x
        if not torch.allclose(combined_x.float(), want.float(), rtol=0.02, atol=1e-3):
            err = (combined_x.float() - want.float()).abs().max().item()
            raise AssertionError(f"[rank {rank}] seed={seed} value mismatch {err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--trials", type=int, default=120)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    assert num_ranks == 8, "shapes table assumes 8 ranks"

    num_nvl_bytes, num_rdma_bytes = compute_buffer_sizes(
        args.num_sms, args.hidden, num_ranks
    )
    buffer = Buffer(
        group, num_nvl_bytes, num_rdma_bytes,
        low_latency_mode=False, num_qps_per_rank=args.num_sms,
        explicitly_destroy=True,
    )
    config = buffer.get_dispatch_config(num_ranks)
    combine_config = buffer.get_combine_config(num_ranks)

    # Phase R: no-sync replay -> eager alternation. Reproduces the
    # host-counter clobber: a replayed worst-mode graph's notify kernels
    # write counter sums with no host poll pacing them; the next eager
    # dispatch's host-side -1 reset lands while the graph is in flight and
    # gets clobbered, deadlocking the eager kernel's wait-for--1. No
    # torch.cuda.synchronize between iterations, by design.
    gn = 256
    xg = torch.randn((gn, args.hidden), dtype=torch.bfloat16) * 0.1
    sg = torch.randn((gn, args.num_experts), dtype=torch.float32)
    tig = torch.topk(sg, args.num_topk, dim=-1, largest=True, sorted=False)[1]
    twg = torch.ones((gn, args.num_topk), dtype=torch.float32)

    def graph_step():
        (a_, b_, c_, d_, _) = buffer.get_dispatch_layout(tig, args.num_experts)
        rx, _, _, _, h, _ = buffer.dispatch(
            xg, num_tokens_per_rank=a_, num_tokens_per_rdma_rank=b_,
            is_token_in_rank=d_, num_tokens_per_expert=c_,
            topk_idx=tig, topk_weights=twg,
            num_worst_tokens=gn * num_ranks, config=config,
        )
        rx2 = rx * 2.0
        cx, _, _ = buffer.combine(rx2, h, config=combine_config)
        return cx

    graph_step()  # eager warmup of the worst path
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_step()
    torch.cuda.synchronize()
    dist.barrier(group)
    if rank == 0:
        print("[ragged] phase R: no-sync replay/eager alternation", flush=True)
    for it in range(200):
        g.replay()
        # deliberately NO synchronize: the eager call below runs its
        # host-side counter resets while the graph may still be in flight
        shape = SHAPES[it % len(SHAPES)]
        run_eager_step(
            buffer, rank, shape[rank], args.hidden, args.num_experts,
            args.num_topk, config, combine_config, seed=5000 + it,
        )
        if it % 50 == 49:
            torch.cuda.synchronize()
            dist.barrier(group)
            if rank == 0:
                print(f"[ragged] phase R: {it + 1}/200", flush=True)
    torch.cuda.synchronize()
    dist.barrier(group)
    if rank == 0:
        print("[ragged] phase R OK", flush=True)

    for trial in range(args.trials):
        shape = SHAPES[trial % len(SHAPES)]
        if rank == 0 and trial % len(SHAPES) == 0:
            print(f"[ragged] trial {trial}/{args.trials}", flush=True)
        run_eager_step(
            buffer, rank, shape[rank], args.hidden, args.num_experts,
            args.num_topk, config, combine_config, seed=1000 + trial,
        )
        torch.cuda.synchronize()
    dist.barrier(group)
    if rank == 0:
        print(f"RAGGED PASS: {args.trials} eager trials over "
              f"{len(SHAPES)} skew shapes, no hang, values correct", flush=True)
    buffer.destroy()


if __name__ == "__main__":
    main()
