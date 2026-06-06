#pragma once

#ifdef USE_CXI

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <vector>

#include <cuda_runtime_api.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

namespace uccl::cxi {

struct EndpointInfo {
  uint64_t mr_key = 0;
  uint64_t size = 0;
  uint64_t host_mr_key = 0;
  uint64_t host_size = 0;
  std::vector<uint8_t> ep_name;
};

struct WriteContext {
  fi_context2 ctx{};
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
  void inject_atomic_add64(fi_addr_t peer, int64_t value,
                           uint64_t remote_offset, uint64_t remote_key);
  void wait(WriteContext* ctx);
  bool wait_all(std::vector<WriteContext*> const& ctxs,
                std::atomic<bool> const* progress_run = nullptr);

 private:
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
};

}  // namespace uccl::cxi

#endif  // USE_CXI
