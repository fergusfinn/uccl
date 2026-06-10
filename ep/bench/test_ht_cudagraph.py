"""MWE: capture UCCL DeepEP-HT internode dispatch+combine in a CUDA graph.

Question under test: with num_worst_tokens set (no host count sync), can the
high-throughput internode path be stream-captured and replayed correctly with
fresh data AND fresh routing each replay — i.e. the sglang-style "whole step
in one graph" shape?

Run (2 nodes x 4 GPUs, from ep/bench so `utils`/`buffer` import):
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$N \
    --master_addr=$ADDR --master_port=12355 test_ht_cudagraph.py

Phases:
  0. eager warmup + analytic-oracle check (combine(dispatch(x)) == x * owner_rank_count)
  1. capture layout+dispatch+fake-expert+combine into one torch.cuda.CUDAGraph
  2. N replays, fresh x/routing per replay, validate oracle + cross-check vs eager
  3. mixed eager/replay interleaving (vLLM serving would mix prefill-eager with
     decode-replay against the same Buffer)
  4. timing: eager loop vs replay loop
Exit code 0 = all checks pass on all ranks.
"""

import argparse
import os
import sys
import time

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


def make_inputs(seed, rank, num_tokens, hidden, num_experts, num_topk):
    """Deterministic per-(seed, rank) inputs. x encodes (rank, token) identity
    so misrouted/corrupted tokens are detectable, not just statistically wrong."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed * 4096 + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, generator=g) * 0.1
    # Stamp identity into the first two hidden dims (bf16-exact small ints).
    x[:, 0] = rank
    x[:, 1] = torch.arange(num_tokens, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, generator=g)
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32)
    return x, topk_idx, topk_weights


def run_step(
    buffer, x, topk_idx, topk_weights, num_experts, num_worst_tokens, config,
    combine_config,
):
    """One full HT step: layout -> dispatch -> fake expert (x2) -> combine.
    Everything device-side; with num_worst_tokens>0 there is no host sync."""
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(topk_idx, num_experts)
    recv_x, _, _, expert_counts, handle, _ = buffer.dispatch(
        x,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_worst_tokens=num_worst_tokens,
        config=config,
    )
    # Stand-in for expert compute: in-place so the combined values prove the
    # "expert" output (not the dispatch input) made the round trip.
    recv_x *= 2.0
    combined_x, _, _ = buffer.combine(recv_x, handle, config=combine_config)
    return combined_x, is_token_in_rank, expert_counts


def expected_combined(x, is_token_in_rank):
    # dispatch sends one copy per owner rank; fake expert doubles it; combine
    # sums copies. => combined == 2 * owner_count * x
    owner_count = is_token_in_rank.sum(dim=1).to(torch.bfloat16).unsqueeze(1)
    return 2.0 * owner_count * x


def check(name, got, want, rank, rtol=0.02, atol=1e-3):
    # bf16 tolerance: combine reduces hierarchically (NVL then RDMA) with
    # bf16 intermediates, so outputs differ from any reference by a couple
    # of ULP (rel ~2^-8). Routing errors are O(1) relative and still caught.
    if not torch.allclose(got.float(), want.float(), atol=atol, rtol=rtol):
        bad = (got.float() - want.float()).abs().max().item()
        nbad = (~torch.isclose(got.float(), want.float(), atol=atol, rtol=rtol)).sum().item()
        raise AssertionError(
            f"[rank {rank}] {name}: mismatch max_abs_err={bad} n_bad={nbad} "
            f"shape={tuple(got.shape)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--timing-iters", type=int, default=30)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
    rank, num_ranks, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    assert num_ranks > num_local_ranks, "need >=2 nodes for the internode path"

    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_worst_tokens = num_tokens * num_ranks

    num_nvl_bytes, num_rdma_bytes = compute_buffer_sizes(args.num_sms, hidden, num_ranks)
    buffer = Buffer(
        group,
        num_nvl_bytes,
        num_rdma_bytes,
        low_latency_mode=False,
        num_qps_per_rank=args.num_sms,
        explicitly_destroy=True,
    )
    config = buffer.get_dispatch_config(num_ranks)
    # 8-rank combine config from Buffer.get_combine_config is broken for
    # 2 RDMA ranks (rdma_chunked_send_tokens=6 < num_warps_per_forwarder=8);
    # use the fixed values (uccl commit c09c8a7a) without touching the repo.
    if num_ranks == 8:
        combine_config = Config(Buffer.num_sms, 4, 256, 8, 128)
    else:
        combine_config = buffer.get_combine_config(num_ranks)

    def log(msg):
        if rank == 0:
            print(f"[mwe] {msg}", flush=True)

    log(
        f"world={num_ranks} tokens/rank={num_tokens} hidden={hidden} "
        f"experts={num_experts} topk={num_topk} worst_tokens={num_worst_tokens} "
        f"nvl={num_nvl_bytes/1e9:.2f}GB rdma={num_rdma_bytes/1e9:.2f}GB"
    )

    # ---- Phase 0: eager isolation: normal (host-sync) vs worst-tokens ------
    # Runs both modes on identical inputs and diffs each stage, so a failure
    # localizes to dispatch-layout vs combine-reduction.
    for seed in range(3):
        x, topk_idx, topk_weights = make_inputs(
            seed, rank, num_tokens, hidden, num_experts, num_topk
        )
        layout = buffer.get_dispatch_layout(topk_idx, num_experts)
        (ntpr, ntprr, ntpe, is_in_rank, _) = layout

        def do_dispatch(worst):
            return buffer.dispatch(
                x,
                num_tokens_per_rank=ntpr,
                num_tokens_per_rdma_rank=ntprr,
                is_token_in_rank=is_in_rank,
                num_tokens_per_expert=ntpe,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_worst_tokens=worst,
                config=config,
            )

        recv_n, _, _, nlist_n, h_n, _ = do_dispatch(0)  # normal: host-synced
        recv_w, _, _, counts_w, h_w, _ = do_dispatch(num_worst_tokens)
        torch.cuda.synchronize()
        check(
            f"seed={seed} device expert counts vs host list",
            counts_w.cpu().float(),
            torch.tensor(nlist_n, dtype=torch.float32, device="cpu"),
            rank, rtol=0.0, atol=0.0,
        )
        real = recv_n.size(0)
        check(
            f"seed={seed} dispatch worst-vs-normal recv rows (real={real})",
            recv_w[:real], recv_n, rank,
        )
        c_n, _, _ = buffer.combine(recv_n * 2.0, h_n, config=combine_config)
        c_w, _, _ = buffer.combine(recv_w * 2.0, h_w, config=combine_config)
        torch.cuda.synchronize()
        check(
            f"seed={seed} normal-path oracle",
            c_n, expected_combined(x, is_in_rank), rank,
        )
        check(f"seed={seed} combine worst-vs-normal", c_w, c_n, rank)
    torch.cuda.synchronize()
    dist.barrier(group)
    log("phase 0 OK: eager worst-tokens dispatch+combine matches the normal path")

    # ---- Phase 1: capture ----------------------------------------------------
    # Static input buffers; everything downstream lives inside the graph.
    x_st, topk_idx_st, topk_weights_st = make_inputs(
        100, rank, num_tokens, hidden, num_experts, num_topk
    )

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            combined_st, owner_mask_st, counts_st = run_step(
                buffer,
                x_st,
                topk_idx_st,
                topk_weights_st,
                num_experts,
                num_worst_tokens,
                config,
                combine_config,
            )
    except Exception as exc:
        print(f"[rank {rank}] CAPTURE FAILED: {exc!r}", flush=True)
        raise
    torch.cuda.synchronize()
    dist.barrier(group)
    log("phase 1 OK: dispatch+combine captured into one CUDA graph")

    # ---- Phase 2: replays with fresh data and fresh routing -------------------
    for trial in range(args.replays):
        seed = 200 + trial
        x, topk_idx, topk_weights = make_inputs(
            seed, rank, num_tokens, hidden, num_experts, num_topk
        )
        x_st.copy_(x)
        topk_idx_st.copy_(topk_idx)
        topk_weights_st.copy_(topk_weights)
        graph.replay()
        torch.cuda.synchronize()
        check(
            f"replay trial={trial}",
            combined_st,
            expected_combined(x_st, owner_mask_st),
            rank,
        )
        # Cross-check one trial in depth against a fresh eager run.
        if trial in (0, args.replays // 2):
            eager_combined, _, eager_counts = run_step(
                buffer, x, topk_idx, topk_weights, num_experts, num_worst_tokens, config, combine_config
            )
            torch.cuda.synchronize()
            check(f"replay-vs-eager trial={trial}", combined_st, eager_combined, rank)
            check(
                f"replay-vs-eager expert counts trial={trial}",
                counts_st.float(), eager_counts.float(), rank, rtol=0.0, atol=0.0,
            )
    dist.barrier(group)
    log(f"phase 2 OK: {args.replays} replays, fresh routing each, all correct")

    # ---- Phase 3: interleave eager calls and replays --------------------------
    for trial in range(10):
        seed = 300 + trial
        x, topk_idx, topk_weights = make_inputs(
            seed, rank, num_tokens, hidden, num_experts, num_topk
        )
        if trial % 2 == 0:
            x_st.copy_(x)
            topk_idx_st.copy_(topk_idx)
            topk_weights_st.copy_(topk_weights)
            graph.replay()
            torch.cuda.synchronize()
            check(
                f"mixed-replay trial={trial}",
                combined_st,
                expected_combined(x_st, owner_mask_st),
                rank,
            )
        else:
            combined, owner_mask, _ = run_step(
                buffer, x, topk_idx, topk_weights, num_experts, num_worst_tokens, config, combine_config
            )
            torch.cuda.synchronize()
            check(
                f"mixed-eager trial={trial}",
                combined,
                expected_combined(x, owner_mask),
                rank,
            )
    dist.barrier(group)
    log("phase 3 OK: eager and replayed steps interleave safely on one Buffer")

    # ---- Phase 3b: serving-shaped interleave --------------------------------
    # Multiple graph shapes captured on ONE Buffer, replays interleaved
    # across shapes and with host-synced eager dispatches (worst=0), the
    # way vLLM mixes prefill-eager and decode-replay across many capture
    # sizes. Each shape gets its own static buffers and worst sizing.
    shapes = [s for s in (32, 128, num_tokens) if s <= num_tokens]
    multi = {}
    for s in shapes:
        xs, tis, tws = make_inputs(400 + s, rank, s, hidden, num_experts, num_topk)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cs, oms, _ = run_step(
                buffer, xs, tis, tws, num_experts, s * num_ranks, config,
                combine_config,
            )
        torch.cuda.synchronize()
        multi[s] = (g, xs, tis, tws, cs, oms)
    dist.barrier(group)
    log(f"phase 3b: captured shapes {shapes} on one Buffer")
    for trial in range(30):
        s = shapes[(trial * 7) % len(shapes)]
        g, xs, tis, tws, cs, oms = multi[s]
        seed = 500 + trial
        x, ti, tw = make_inputs(seed, rank, s, hidden, num_experts, num_topk)
        xs.copy_(x); tis.copy_(ti); tws.copy_(tw)
        g.replay()
        torch.cuda.synchronize()
        check(f"multi-shape replay trial={trial} shape={s}", cs,
              expected_combined(xs, oms), rank)
        if trial % 3 == 2:
            # host-synced eager step (worst=0), like a serving prefill
            xe, tie, twe = make_inputs(seed + 1000, rank, num_tokens, hidden,
                                       num_experts, num_topk)
            ce, ome, _ = run_step(
                buffer, xe, tie, twe, num_experts, 0, config, combine_config,
            )
            torch.cuda.synchronize()
            check(f"multi-shape eager trial={trial}", ce,
                  expected_combined(xe, ome), rank)
    dist.barrier(group)
    log("phase 3b OK: cross-shape replay + host-synced eager interleave correct")

    # ---- Phase 4: timing -------------------------------------------------------
    def timed(fn, iters):
        torch.cuda.synchronize()
        dist.barrier(group)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    eager_ms = timed(
        lambda: run_step(
            buffer, x_st, topk_idx_st, topk_weights_st, num_experts,
            num_worst_tokens, config, combine_config,
        ),
        args.timing_iters,
    )
    replay_ms = timed(graph.replay, args.timing_iters)
    log(
        f"phase 4: eager {eager_ms:.3f} ms/step vs replay {replay_ms:.3f} ms/step "
        f"({eager_ms / replay_ms:.2f}x)"
    )

    dist.barrier(group)
    buffer.destroy()
    if rank == 0:
        print("MWE PASS: HT internode dispatch+combine is CUDA-graph capturable "
              "and replay-correct under changing data and routing", flush=True)


if __name__ == "__main__":
    main()
