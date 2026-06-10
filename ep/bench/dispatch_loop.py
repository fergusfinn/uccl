"""Sustained dispatch loop for wire-utilization measurement.

Runs buffer.dispatch() at a fixed config in a tight loop for LOOP_DURATION_S
seconds (torchrun-style, one process per GPU). Bracket externally with CXI
telemetry snapshots to measure true NIC bytes/sec. LOOP_CACHED=1 reuses the
dispatch handle (skips the notify/count exchange per iteration).
"""
import gc, os, time
import torch
import torch.distributed as dist
from buffer import Buffer
from utils import init_dist_under_torchrun
from test_internode import compute_buffer_sizes
from uccl.ep import Config


def measure(buffer, group, rank, local_world):
    duration = float(os.environ.get("LOOP_DURATION_S", "60"))
    num_tokens = int(os.environ.get("LOOP_TOKENS", "4096"))
    hidden = int(os.environ.get("LOOP_HIDDEN", "7168"))
    num_topk = int(os.environ.get("LOOP_TOPK", "8"))
    num_experts = int(os.environ.get("LOOP_EXPERTS", "288"))
    nvl_chunk = int(os.environ.get("LOOP_NVL_CHUNK", "32"))
    nvl_buf = int(os.environ.get("LOOP_NVL_BUF", "256"))
    rdma_chunk = int(os.environ.get("LOOP_RDMA_CHUNK", "64"))
    rdma_buf = int(os.environ.get("LOOP_RDMA_BUF", "128"))
    cached = os.environ.get("LOOP_CACHED", "0") == "1"

    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32,
                         device="cuda").abs() + 1.0
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32,
                              device="cuda")
    (num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
     is_token_in_rank, _) = buffer.get_dispatch_layout(topk_idx, num_experts)
    config = Config(24, nvl_chunk, nvl_buf, rdma_chunk, rdma_buf)
    args = dict(x=x, num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                topk_idx=topk_idx, topk_weights=topk_weights,
                num_tokens_per_expert=num_tokens_per_expert, config=config)
    if cached:
        recv = buffer.dispatch(**args)
        args = dict(x=x, handle=recv[4], config=config)
    for _ in range(5):
        buffer.dispatch(**args)
    torch.cuda.synchronize()
    dist.barrier(group)
    if rank == 0:
        print(f"[loop] start duration={duration}s tokens={num_tokens} "
              f"cached={cached}", flush=True)
    t0 = time.time()
    iters = 0
    while time.time() - t0 < duration:
        buffer.dispatch(**args)
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    rdma_tokens = int(num_tokens_per_rdma_rank.sum().item()) - int(
        num_tokens_per_rdma_rank[rank // local_world].item())
    bytes_per_iter = rdma_tokens * hidden * 2
    print(f"[loop] rank={rank} iters={iters} elapsed={elapsed:.2f}s "
          f"rdma_tokens={rdma_tokens} bytes/iter={bytes_per_iter} "
          f"offered_GBps={iters * bytes_per_iter / elapsed / 1e9:.2f}",
          flush=True)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "4"))
    rank, world, group = init_dist_under_torchrun(local_rank, local_world)
    hidden = int(os.environ.get("LOOP_HIDDEN", "7168"))
    nvl_b, rdma_b = compute_buffer_sizes(24, hidden, world)
    buffer = Buffer(group, nvl_b, rdma_b, low_latency_mode=False,
                    num_qps_per_rank=24, explicitly_destroy=True)
    measure(buffer, group, rank, local_world)
    # All measurement tensors died with measure()s frame; flush deferred
    # frees before the buffer tears down the CUDA context.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier(group)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
