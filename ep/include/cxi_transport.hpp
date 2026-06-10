#pragma once

#ifdef USE_CXI

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

namespace uccl::cxi {

struct EndpointInfo {
  uint64_t mr_key = 0;
  uint64_t size = 0;
  uint64_t host_mr_key = 0;
  uint64_t host_size = 0;
  std::vector<uint8_t> ep_name;
};

// fi_context2 must stay the first member: the CQ returns &ctx and we map it
// back to the owning WriteContext by address.
struct WriteContext {
  fi_context2 ctx{};
  std::vector<uint64_t> wr_ids;  // ring command ids retired on completion
  bool in_use = false;
};

class Transport {
 public:
  Transport() = default;
  Transport(Transport const&) = delete;
  Transport& operator=(Transport const&) = delete;
  ~Transport();

  void init(int device_index = -1);
  void register_cuda_buffer(void* ptr, size_t size);
  void register_host_buffer(void* ptr, size_t size);
  EndpointInfo local_info() const;
  fi_addr_t insert_peer(EndpointInfo const& peer);
  void write(fi_addr_t peer, void* local, size_t bytes, uint64_t remote_offset,
             uint64_t remote_key, WriteContext* ctx);
  // Non-throwing on queue-full: returns 0 on success, -FI_EAGAIN when the TX
  // queue is exhausted (caller should poll() and retry). Throws on hard
  // errors. Increments outstanding() on success.
  int try_write(fi_addr_t peer, void* local, size_t bytes,
                uint64_t remote_offset, uint64_t remote_key,
                WriteContext* ctx);
  void inject_atomic_add64(fi_addr_t peer, int64_t value,
                           uint64_t remote_offset, uint64_t remote_key);
  // Returns 0 on success, -FI_EAGAIN when the TX queue is exhausted.
  int try_inject_atomic_add64(fi_addr_t peer, int64_t value,
                              uint64_t remote_offset, uint64_t remote_key);
  void wait(WriteContext* ctx);
  bool wait_all(std::vector<WriteContext*> const& ctxs,
                std::atomic<bool> const* progress_run = nullptr);
  // Non-blocking CQ poll. Fills `done` with up to `max` completed contexts
  // and returns the count. Decrements outstanding(). Throws on CQ error.
  size_t poll(WriteContext** done, size_t max);
  // Block until all outstanding writes complete (or progress_run goes
  // false). Completed contexts are marked !in_use but their wr_ids are NOT
  // delivered to the caller — only use when retirement bookkeeping has been
  // handled elsewhere or does not matter (teardown).
  bool drain(std::atomic<bool> const* progress_run = nullptr);
  size_t outstanding() const { return outstanding_; }

  // Pooled contexts for the async path. Contexts handed out by
  // acquire_context() must be returned via release_context() after poll()
  // reports them complete. Never mix pooled and caller-owned (stack)
  // contexts on the same transport.
  WriteContext* acquire_context();
  void release_context(WriteContext* ctx);

 private:
  std::vector<std::unique_ptr<WriteContext>> ctx_pool_;
  std::vector<WriteContext*> free_ctxs_;
  fi_info* info_ = nullptr;
  fid_fabric* fabric_ = nullptr;
  fid_domain* domain_ = nullptr;
  fid_ep* ep_ = nullptr;
  fid_cq* cq_ = nullptr;
  fid_av* av_ = nullptr;
  fid_mr* mr_ = nullptr;
  fid_mr* host_mr_ = nullptr;
  void* cuda_ptr_ = nullptr;
  size_t cuda_size_ = 0;
  void* host_ptr_ = nullptr;
  size_t host_size_ = 0;
  size_t outstanding_ = 0;
};

}  // namespace uccl::cxi

#endif  // USE_CXI
