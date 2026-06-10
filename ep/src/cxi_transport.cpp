#ifdef USE_CXI

#include "cxi_transport.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <sched.h>
#include <unistd.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>

namespace uccl::cxi {
namespace {

void check_fi(char const* what, int ret) {
  if (ret != 0) {
    throw std::runtime_error(std::string(what) + ": " + fi_strerror(-ret));
  }
}

void check_cuda(char const* what, cudaError_t ret) {
  if (ret != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(ret));
  }
}

fi_threading threading_hint() {
  char const* value = std::getenv("UCCL_CXI_THREADING");
  if (!value || std::strcmp(value, "endpoint") == 0) return FI_THREAD_ENDPOINT;
  if (std::strcmp(value, "completion") == 0) return FI_THREAD_COMPLETION;
  if (std::strcmp(value, "domain") == 0) return FI_THREAD_DOMAIN;
  if (std::strcmp(value, "fid") == 0) return FI_THREAD_FID;
  if (std::strcmp(value, "safe") == 0) return FI_THREAD_SAFE;
  if (std::strcmp(value, "unspec") == 0) return FI_THREAD_UNSPEC;
  throw std::runtime_error(std::string("Invalid UCCL_CXI_THREADING=") + value);
}

}  // namespace

Transport::~Transport() {
  if (host_mr_) fi_close(&host_mr_->fid);
  if (mr_) fi_close(&mr_->fid);
  if (av_) fi_close(&av_->fid);
  if (cq_) fi_close(&cq_->fid);
  if (ep_) fi_close(&ep_->fid);
  if (domain_) fi_close(&domain_->fid);
  if (fabric_) fi_close(&fabric_->fid);
  if (info_) fi_freeinfo(info_);
}

void Transport::init(int device_index) {
  fi_info* hints = fi_allocinfo();
  if (!hints) throw std::runtime_error("fi_allocinfo failed");

  hints->fabric_attr->prov_name = strdup("cxi");
  if (device_index >= 0) {
    std::ostringstream domain_name;
    domain_name << "cxi" << device_index;
    hints->domain_attr->name = strdup(domain_name.str().c_str());
  }
  hints->ep_attr->type = FI_EP_RDM;
  hints->caps = FI_TAGGED | FI_MSG | FI_HMEM | FI_RMA | FI_READ | FI_WRITE |
                FI_ATOMIC | FI_REMOTE_WRITE | FI_DIRECTED_RECV |
                FI_LOCAL_COMM | FI_REMOTE_COMM;
  hints->mode = FI_CONTEXT | FI_CONTEXT2;
    hints->domain_attr->threading = threading_hint();
  hints->domain_attr->control_progress = FI_PROGRESS_UNSPEC;
  hints->domain_attr->data_progress = FI_PROGRESS_UNSPEC;
  hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_ENDPOINT |
                                FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                                FI_MR_PROV_KEY;
  hints->domain_attr->mr_key_size = 2;
  hints->tx_attr->msg_order = FI_ORDER_SAS;
  hints->rx_attr->msg_order = FI_ORDER_SAS;

  try {
    check_fi("fi_getinfo(cxi)", fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr,
                                           0, hints, &info_));
    {
      char const* dc = std::getenv("UCCL_CXI_DELIVERY_COMPLETE");
      if (dc && dc[0] == (char)0x31) {
        info_->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
        fprintf(stderr, "[CXI] FI_DELIVERY_COMPLETE enabled on TX\n");
      }
    }
    check_fi("fi_fabric", fi_fabric(info_->fabric_attr, &fabric_, nullptr));
    check_fi("fi_domain", fi_domain(fabric_, info_, &domain_, nullptr));
    check_fi("fi_endpoint", fi_endpoint(domain_, info_, &ep_, nullptr));

    fi_cq_attr cq_attr{};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.size = 4096;
    check_fi("fi_cq_open", fi_cq_open(domain_, &cq_attr, &cq_, nullptr));
    check_fi("fi_ep_bind(cq)",
             fi_ep_bind(ep_, &cq_->fid, FI_TRANSMIT | FI_RECV));

    fi_av_attr av_attr{};
    av_attr.type = FI_AV_TABLE;
    check_fi("fi_av_open", fi_av_open(domain_, &av_attr, &av_, nullptr));
    check_fi("fi_ep_bind(av)", fi_ep_bind(ep_, &av_->fid, 0));

#ifdef FI_OPT_CUDA_API_PERMITTED
    bool cuda_api_permitted = false;
    check_fi("fi_setopt(FI_OPT_CUDA_API_PERMITTED)",
             fi_setopt(&ep_->fid, FI_OPT_ENDPOINT,
                       FI_OPT_CUDA_API_PERMITTED, &cuda_api_permitted,
                       sizeof(cuda_api_permitted)));
#endif

    check_fi("fi_enable", fi_enable(ep_));
  } catch (...) {
    fi_freeinfo(hints);
    throw;
  }
  fi_freeinfo(hints);
}

void Transport::register_cuda_buffer(void* ptr, size_t size) {
  if (!domain_ || !ep_) throw std::runtime_error("CXI transport not initialized");
  cuda_ptr_ = ptr;
  cuda_size_ = size;

  cudaPointerAttributes attrs{};
  check_cuda("cudaPointerGetAttributes", cudaPointerGetAttributes(&attrs, ptr));

  iovec iov{};
  iov.iov_base = ptr;
  iov.iov_len = size;

  fi_mr_attr mr_attr{};
  mr_attr.mr_iov = &iov;
  mr_attr.iov_count = 1;
  mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                   FI_REMOTE_WRITE | FI_REMOTE_READ;
  mr_attr.iface = FI_HMEM_CUDA;
  mr_attr.device.cuda = attrs.device;

  check_fi("fi_mr_regattr(cuda)", fi_mr_regattr(domain_, &mr_attr, 0, &mr_));
  if (info_->domain_attr->mr_mode & FI_MR_ENDPOINT) {
    check_fi("fi_mr_bind(ep)", fi_mr_bind(mr_, &ep_->fid, 0));
    check_fi("fi_mr_enable", fi_mr_enable(mr_));
  }
}

void Transport::register_host_buffer(void* ptr, size_t size) {
  if (!domain_ || !ep_) throw std::runtime_error("CXI transport not initialized");
  host_ptr_ = ptr;
  host_size_ = size;

  cudaPointerAttributes attrs{};
  auto const cuda_status = cudaPointerGetAttributes(&attrs, ptr);
  cudaGetLastError();
  if (cuda_status == cudaSuccess &&
      (attrs.type == cudaMemoryTypeDevice ||
       attrs.type == cudaMemoryTypeManaged)) {
    iovec iov{};
    iov.iov_base = ptr;
    iov.iov_len = size;

    fi_mr_attr mr_attr{};
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                     FI_REMOTE_WRITE | FI_REMOTE_READ;
    mr_attr.iface = FI_HMEM_CUDA;
    mr_attr.device.cuda = attrs.device;

    check_fi("fi_mr_regattr(control cuda)",
             fi_mr_regattr(domain_, &mr_attr, 0, &host_mr_));
  } else {
    check_fi("fi_mr_reg(host)",
             fi_mr_reg(domain_, ptr, size,
                       FI_SEND | FI_RECV | FI_READ | FI_WRITE |
                           FI_REMOTE_WRITE | FI_REMOTE_READ,
                       0, 0, 0, &host_mr_, nullptr));
  }
  if (info_->domain_attr->mr_mode & FI_MR_ENDPOINT) {
    check_fi("fi_mr_bind(host ep)", fi_mr_bind(host_mr_, &ep_->fid, 0));
    check_fi("fi_mr_enable(host)", fi_mr_enable(host_mr_));
  }
}

EndpointInfo Transport::local_info() const {
  if (!ep_ || !mr_) throw std::runtime_error("CXI endpoint or MR missing");
  EndpointInfo out;
  out.mr_key = fi_mr_key(mr_);
  out.size = cuda_size_;
  if (host_mr_) {
    out.host_mr_key = fi_mr_key(host_mr_);
    out.host_size = host_size_;
  }
  out.ep_name.resize(512);
  size_t len = out.ep_name.size();
  check_fi("fi_getname", fi_getname(&ep_->fid, out.ep_name.data(), &len));
  out.ep_name.resize(len);
  return out;
}

fi_addr_t Transport::insert_peer(EndpointInfo const& peer) {
  if (!av_) throw std::runtime_error("CXI AV missing");
  fi_addr_t addr = FI_ADDR_UNSPEC;
  int ret = fi_av_insert(av_, peer.ep_name.data(), 1, &addr, 0, nullptr);
  if (ret != 1) check_fi("fi_av_insert", ret < 0 ? ret : -FI_EINVAL);
  return addr;
}

namespace {
[[noreturn]] void throw_cq_error(fid_cq* cq) {
  fi_cq_err_entry err{};
  fi_cq_readerr(cq, &err, 0);
  char buf[256];
  char const* msg =
      fi_cq_strerror(cq, err.prov_errno, err.err_data, buf, sizeof(buf));
  throw std::runtime_error(std::string("CXI CQ error: ") +
                           (msg ? msg : fi_strerror(err.err)));
}
}  // namespace

// Consume up to `max` completions, marking contexts done. Returns number
// consumed; fills `done` (may be null when the caller doesn't need them).
size_t Transport::poll(WriteContext** done, size_t max) {
  if (!cq_) throw std::runtime_error("CXI CQ missing");
  if (outstanding_ == 0 || max == 0) return 0;
  size_t n = 0;
  while (n < max) {
    fi_cq_entry entries[16]{};
    size_t const want = std::min<size_t>(16, max - n);
    ssize_t rc = fi_cq_read(cq_, entries, want);
    if (rc > 0) {
      for (ssize_t i = 0; i < rc; ++i) {
        auto* c = reinterpret_cast<WriteContext*>(entries[i].op_context);
        c->in_use = false;
        if (outstanding_ > 0) --outstanding_;
        if (done) done[n] = c;
        ++n;
      }
      continue;
    }
    if (rc == -FI_EAGAIN) break;
    if (rc == -FI_EAVAIL) throw_cq_error(cq_);
    check_fi("fi_cq_read", static_cast<int>(rc));
  }
  return n;
}

int Transport::try_write(fi_addr_t peer, void* local, size_t bytes,
                         uint64_t remote_offset, uint64_t remote_key,
                         WriteContext* ctx) {
  if (!ep_ || !mr_) throw std::runtime_error("CXI endpoint or MR missing");
  if (!ctx) throw std::runtime_error("CXI write context is null");
  uintptr_t const local_addr = reinterpret_cast<uintptr_t>(local);
  uintptr_t const cuda_base = reinterpret_cast<uintptr_t>(cuda_ptr_);
  uintptr_t const host_base = reinterpret_cast<uintptr_t>(host_ptr_);
  fid_mr* local_mr = nullptr;
  if (local_addr >= cuda_base && local_addr + bytes <= cuda_base + cuda_size_) {
    local_mr = mr_;
  } else if (host_mr_ && local_addr >= host_base &&
             local_addr + bytes <= host_base + host_size_) {
    local_mr = host_mr_;
  } else {
    throw std::runtime_error("CXI write local range is outside registered MRs");
  }

  // CXI advertises FI_MR_VIRT_ADDR, but the validated CUDA write path on
  // Isambard requires offset-zero RMA addressing with provider keys.
  ssize_t rc = fi_write(ep_, local, bytes, fi_mr_desc(local_mr), peer,
                        remote_offset, remote_key, &ctx->ctx);
  if (rc == -FI_EAGAIN) return -FI_EAGAIN;
  check_fi("fi_write(cuda)", static_cast<int>(rc));
  ctx->in_use = true;
  ++outstanding_;
  return 0;
}

void Transport::write(fi_addr_t peer, void* local, size_t bytes,
                      uint64_t remote_offset, uint64_t remote_key,
                      WriteContext* ctx) {
  uint32_t spins = 0;
  for (;;) {
    int rc = try_write(peer, local, bytes, remote_offset, remote_key, ctx);
    if (rc == 0) return;
    // TX queue full: make progress by consuming completions, then retry.
    // NOTE: completions consumed here are not reported to any caller, so
    // this blocking variant is only safe on transports whose retirement
    // bookkeeping happens via wait_all() (the sync path).
    poll(nullptr, 16);
    if ((++spins & 0x3ff) == 0) sched_yield();
  }
}

int Transport::try_inject_atomic_add64(fi_addr_t peer, int64_t value,
                                       uint64_t remote_offset,
                                       uint64_t remote_key) {
  if (!ep_) throw std::runtime_error("CXI endpoint missing");
  ssize_t rc = fi_inject_atomic(ep_, &value, 1, peer, remote_offset,
                                remote_key, FI_INT64, FI_SUM);
  if (rc == -FI_EAGAIN) return -FI_EAGAIN;
  check_fi("fi_inject_atomic(add64)", static_cast<int>(rc));
  return 0;
}

void Transport::inject_atomic_add64(fi_addr_t peer, int64_t value,
                                    uint64_t remote_offset,
                                    uint64_t remote_key) {
  uint32_t spins = 0;
  while (try_inject_atomic_add64(peer, value, remote_offset, remote_key) ==
         -FI_EAGAIN) {
    poll(nullptr, 16);
    if ((++spins & 0x3ff) == 0) sched_yield();
  }
}

void Transport::wait(WriteContext* ctx) {
  if (!cq_ || !ctx) throw std::runtime_error("CXI CQ or context missing");
  uint32_t spins = 0;
  while (ctx->in_use) {
    if (poll(nullptr, 16) == 0) {
      if ((++spins & 0x3ff) == 0) sched_yield();
    }
  }
}

bool Transport::wait_all(std::vector<WriteContext*> const& ctxs,
                         std::atomic<bool> const* progress_run) {
  if (!cq_) throw std::runtime_error("CXI CQ missing");
  if (ctxs.empty()) return true;

  auto all_done = [&]() {
    for (WriteContext* ctx : ctxs) {
      if (!ctx) throw std::runtime_error("CXI write context is null");
      if (ctx->in_use) return false;
    }
    return true;
  };

  uint32_t spins = 0;
  while (!all_done()) {
    if (poll(nullptr, 16) == 0) {
      if (progress_run && !progress_run->load(std::memory_order_acquire)) {
        return false;
      }
      if ((++spins & 0x3ff) == 0) sched_yield();
    }
  }
  return true;
}

WriteContext* Transport::acquire_context() {
  if (!free_ctxs_.empty()) {
    WriteContext* c = free_ctxs_.back();
    free_ctxs_.pop_back();
    c->wr_ids.clear();
    return c;
  }
  ctx_pool_.push_back(std::make_unique<WriteContext>());
  return ctx_pool_.back().get();
}

void Transport::release_context(WriteContext* ctx) {
  if (!ctx) return;
  ctx->wr_ids.clear();
  ctx->in_use = false;
  free_ctxs_.push_back(ctx);
}

bool Transport::drain(std::atomic<bool> const* progress_run) {
  uint32_t spins = 0;
  while (outstanding_ > 0) {
    if (poll(nullptr, 16) == 0) {
      if (progress_run && !progress_run->load(std::memory_order_acquire)) {
        return false;
      }
      if ((++spins & 0x3ff) == 0) sched_yield();
    }
  }
  return true;
}

}  // namespace uccl::cxi

#endif  // USE_CXI
