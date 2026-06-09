"""
Regression smoke for the zero-token internode layout path.

Run with the normal multi-node launcher, for example:

  JOBID=<2-node-job> NODES=2 NTASKS_PER_NODE=4 WORK=<uccl-checkout> \
    <run-multinode.sh> env PYTHONPATH="$WORK/ep/bench:$WORK:$WORK/ep/deep_ep_wrapper" \
    LOCAL_WORLD_SIZE=4 python3 ep/bench/test_zero_layout_guard.py

The test passes a guarded RDMA-rank count buffer to the raw runtime
get_dispatch_layout(num_tokens=0) entry point. The guard catches regressions
where the zero-token path clears num_ranks ints instead of num_rdma_ranks ints.
"""

import os

import torch
import torch.distributed as dist

from buffer import Buffer
from test_internode import compute_buffer_sizes
from utils import init_dist_under_torchrun


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "4"))
    buffer = None

    try:
        rank, world_size, group = init_dist_under_torchrun(
            local_rank, local_world_size
        )
        num_nodes = world_size // local_world_size

        hidden = 1024
        num_experts = 64
        num_topk = 4
        num_sms = 24
        num_nvlink_bytes, num_rdma_bytes = compute_buffer_sizes(
            num_sms, hidden, world_size
        )

        buffer = Buffer(
            group,
            num_nvlink_bytes,
            num_rdma_bytes,
            low_latency_mode=False,
            explicitly_destroy=True,
        )

        guard_value = 1234567
        topk_idx = torch.empty((0, num_topk), dtype=torch.int64, device="cuda")
        num_tokens_per_rank = torch.full(
            (world_size,), guard_value, dtype=torch.int32, device="cuda"
        )
        rdma_with_guard = torch.full(
            (num_nodes + world_size + 8,),
            guard_value,
            dtype=torch.int32,
            device="cuda",
        )
        num_tokens_per_rdma_rank = rdma_with_guard[:num_nodes]
        rdma_guard = rdma_with_guard[num_nodes:]
        num_tokens_per_expert = torch.full(
            (num_experts,), guard_value, dtype=torch.int32, device="cuda"
        )
        is_token_in_rank = torch.empty(
            (0, world_size), dtype=torch.bool, device="cuda"
        )

        buffer.runtime.get_dispatch_layout(
            topk_idx.data_ptr(),
            0,
            num_topk,
            num_experts,
            num_tokens_per_rank.data_ptr(),
            num_tokens_per_rdma_rank.data_ptr(),
            num_tokens_per_expert.data_ptr(),
            is_token_in_rank.data_ptr(),
            None,
            False,
            False,
            buffer._ll_compute_stream_ptr(torch.device("cuda", local_rank)),
        )
        torch.cuda.synchronize()

        assert torch.equal(num_tokens_per_rank, torch.zeros_like(num_tokens_per_rank))
        assert torch.equal(
            num_tokens_per_rdma_rank, torch.zeros_like(num_tokens_per_rdma_rank)
        )
        assert torch.equal(
            num_tokens_per_expert, torch.zeros_like(num_tokens_per_expert)
        )
        if not torch.equal(rdma_guard, torch.full_like(rdma_guard, guard_value)):
            raise AssertionError(
                f"rank {rank}: rdma guard was clobbered: {rdma_guard.cpu().tolist()}"
            )

        dist.barrier(group)
        if rank == 0:
            print(
                f"[zero-layout-guard] pass world={world_size} num_nodes={num_nodes}",
                flush=True,
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
