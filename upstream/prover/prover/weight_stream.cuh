// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Weight streaming, as in DeepSpeed, "ZeRO-Inference", and Sheng et al., "FlexGen"
// (arXiv:2303.06865): WeightStream keeps the weights of every layer group (a fixed number of
// consecutive layers) in pinned host memory and copies them to the GPU group by group, into a few
// buffers, so that the GPU need not hold all the weights at once. The caller sets its options,
// passes it to BlockWeights::fill(), calls start(), and then brackets each use of a group by
// ready() and done(), group after group. Where it detects a caller's error (a call out of order,
// weights not loaded), it prints "internal error: weight stream: .." and calls exit(4); the exit
// code of the prover program `cp_prove` is then set by its handler end_on_exit() (ends.h).
//
// Terms. In backquotes: the word that identifiers and some messages use; a buffer is their `room`.
//   experts     the routed experts of a mixture-of-experts layer: three weight tensors per layer
//   unit        what is loaded into a buffer at one time: one part (`Part`) of a group's weights;
//               with experts_by_layer, one layer's experts (with experts_by_tensor, one of their
//               tensors), numbered e; with ends_by_turns, also the embedding or the output head
//   pass        (`walk`) a pass over the weights: one visit of all groups in order, from 0 to the
//               last, driven by ready() and done(); the function pass() makes a separate one
//   batch pass  (`wave`) the requests that are proved together during one pass
//   turn        one request's work on one group, in two phases (`stretch`): forward computation,
//               then the row passes (Weights::row_pass(), weights.h), which read the weights again
#pragma once
#include "prover/ends.h"
#include "prover/fork_memory.h"
#include "weights_blocks/block_weights.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <cuda.h>
#include <map>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <vector>

// A GPU buffer that can grow. It reserves a fixed range of GPU addresses once and maps memory into
// it as needed, so pointers into it stay valid as it grows.
// Public source: mapping memory into reserved addresses: CUDA C++ Programming Guide, "Virtual
// Memory Management".
// See also: NVIDIA, CUDA Programming Guide, "Virtual Memory Management",
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
struct GrowingRoom {
  char *base = nullptr;
  size_t mapped = 0;
  static constexpr size_t addresses = (size_t)64 << 30, piece = (size_t)256 << 20;
  // Memory is mapped in steps of this length. A shorter step leaves less of it unused. Set it
  // before the first need().
  size_t step = piece;
  // The largest size, in bytes, of the buffer. The first need() reserves a range of addresses for
  // it, unless lay_at() has given it a part of a longer range.
  size_t most = addresses;
  void lay_at(char *first, size_t bytes) { base = first, most = bytes; }
  static char *reserve(size_t bytes) {
    CUdeviceptr b = 0;
    const CUresult r = cuMemAddressReserve(&b, bytes, 0, 0, 0);
    if (r != CUDA_SUCCESS) {
      const char *m = nullptr;
      cuGetErrorString(r, &m);
      printf("weight stream: could not reserve GPU addresses: cuMemAddressReserve failed: %s\n", m ? m : "unknown error");
      name_the_end("cuda-call");
      exit(1);
    }
    return (char *)b;
  }
  // Ensures that at least `bytes` bytes are mapped at `base`; exits if it cannot.
  void need(size_t bytes) {
    auto check = [](CUresult r, const char *what) {
      if (r != CUDA_SUCCESS) {
        const char *m = nullptr;
        cuGetErrorString(r, &m);
        printf("weight stream: could not reserve or map GPU memory: %s failed: %s\n", what, m ? m : "unknown error");
        name_the_end("cuda-call");
        exit(1);
      }
    };
    if (!base) base = reserve(addresses);
    if ((bytes + step - 1) / step * step > most) { // memory is mapped in whole steps
      printf("weight stream: the weights of one GPU buffer need %zu bytes of GPU addresses, mapped in steps of %zu, "
             "but the buffer has %zu\n", bytes, step,
             most);
      exit(2);
    }
    while (mapped < bytes) {
      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      int device = 0;
      CK(cudaGetDevice(&device));
      prop.location.id = device;
      CUmemGenericAllocationHandle handle;
      check(cuMemCreate(&handle, step, &prop, 0), "cuMemCreate");
      check(cuMemMap((CUdeviceptr)base + mapped, step, 0, handle, 0), "cuMemMap");
      CUmemAccessDesc access{};
      access.location = prop.location;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      check(cuMemSetAccess((CUdeviceptr)base + mapped, step, &access, 1), "cuMemSetAccess");
      mapped += step;
    }
    fork_memory.device_buffers[base] = mapped; // so that wipe() zeroes it
    fork_memory.purpose_of[base] = "rooms for streamed weights";
  }
};

// Loads a buffer by a kernel: GPU threads read the pinned host memory at `from` themselves and
// write it to `to`. n: the number of 16-byte pieces. In the main loop each thread issues four reads
// before its first write, so that up to blocks x threads x 64 bytes are in flight. The reads use
// __ldcs(), the load with the streaming cache hint, because nothing is read twice.
// Public source: kernels that read registered host memory: NVIDIA, CUDA Runtime API, "Memory
// Management", cudaHostRegister() (where unified virtual addressing is in use, registering with
// cudaHostRegisterDefault also gives kernels access to the memory) and cudaHostGetDevicePointer().
static __global__ void k_load_a_room(uint4 *to, const uint4 *from, size_t n) {
  const size_t step = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i + 3 * step < n; i += 4 * step) {
    const uint4 a = __ldcs(from + i), b = __ldcs(from + i + step), c = __ldcs(from + i + 2 * step), d = __ldcs(from + i + 3 * step);
    to[i] = a, to[i + step] = b, to[i + 2 * step] = c, to[i + 3 * step] = d;
  }
  for (; i < n; i += step) to[i] = __ldcs(from + i);
}

// Implements wb::Placement (prover/weights_blocks/block_weights.h): BlockWeights::fill() asks it
// where each tensor goes and tells it when a layer's experts or a group are filled.
// Each load puts a group at the addresses that place() returned for it when the group was filled.
// What belongs to no group stays on the GPU unless ends_by_turns is set. With two buffers, one is
// loaded while kernels read the other. Unless no_walk_follows is set, passes follow one another
// without a further call (see done()).
struct WeightStream : wb::Placement {
  // The options from here to ends_by_turns must be set before the first group is filled. Which
  // options go together ("Only with ..") is the caller's duty: start() checks it only for
  // experts_by_tensor.
  //
  // If true, the routed experts' tensors (wb::Form::RoutedBlocks), which are most of a group's
  // bytes, have one separate buffer, and experts_done(g) starts the load of the next group's
  // experts into it.
  bool one_room_for_experts = false;
  // Only with one_room_for_experts. If true, the routed experts are loaded separately from
  // their group. Each layer's experts have a number e = 0, 1, .., in the order of the layers,
  // and are loaded into buffer e % experts_rooms (with experts_by_tensor: room_of_tensor(e)).
  // experts_ready(e) and experts_used(e) bracket each use; start() sets the order of the uses.
  bool experts_by_layer = false;
  // Only with experts_by_layer: the number of the experts' buffers, from 1 to Part::MOST_ROOMS;
  // otherwise place_routed() exits.
  int experts_rooms = 1;
  // Only with experts_by_layer and without ends_by_turns. If true, a number e denotes one routed
  // tensor, not the three of a layer together, and a buffer holds one tensor. A layer's tensors are
  // numbered in the order in which they are placed: gate, up, down. experts_rooms must then be at
  // least 3 * routed_layers_a_group, so that each routed tensor of a group has a buffer to itself.
  // Buffers beyond that number are used for the first tensors of the next group.
  bool experts_by_tensor = false;
  int routed_layers_a_group = 1; // set with experts_by_tensor
  // Only with experts_by_layer. If true, everything except the routed experts has one buffer,
  // not two.
  bool one_room_for_the_rest = false;
  // Only with experts_by_layer. If true, the embedding and the head are streamed too, through the
  // experts' buffers.
  bool ends_by_turns = false;
  // Null, or a pointer to a flag; set before the first pass. It is for experts_by_layer, when the
  // requests of a batch pass have their turns in a group one after another, all with the same uses.
  // The caller must keep the flag false during the turn of every request except the batch pass's
  // last, and true otherwise. Null counts as true.
  const bool *last_turn = nullptr;
  // Set before a pass. If true, the caller promises to call restart() or pass() before the next
  // pass; both clear this flag and rooms_will_be_lent. A buffer that done() or experts_done()
  // releases is then not loaded with a group of the next pass, and counts as empty: a ready() that
  // needs it exits. With experts_by_layer this holds only for the buffers of the weights other than the experts.
  // Public source: software pipelining of prefetches, as S. P. VanderWiel and D. J. Lilja describe it for cache
  // prefetch instructions: "When Caches Aren't Enough: Data Prefetching Techniques", IEEE Computer, July 1997. Here the
  // items are weight groups copied into GPU buffers, and the loads left out are the next pass's.
  bool no_walk_follows = false;
  // Set before a pass, together with no_walk_follows and only with experts_by_tensor, if the caller
  // will borrow the buffers through room() when the last group of that pass is done. An experts'
  // buffer that is released for the last time in that pass is then not loaded with a tensor of the
  // next pass either, and counts as empty.
  bool rooms_will_be_lent = false;
  // Set to false while no request's turn runs concurrently with the loads, as in the calls of
  // pass() for a batch's opening (the test on the weight table): load() then uses the copy engine,
  // not the loading kernel.
  bool turns_run = true;
  // For measurements; set before the first pass. While it and turns_run are true, each wait for a
  // buffer is timed by two events in the calling thread's stream. waited() returns, for the waits
  // since its last call, their number (`asks`), their total time, and how many took more than a
  // millisecond (`long_ones`). It blocks until the last of them is over.
  bool count_waits = false;
  struct Waited {
    double seconds = 0;
    size_t asks = 0, long_ones = 0;
  };
  Waited waited() {
    Waited w;
    w.asks = waits_used;
    for (size_t i = 0; i < waits_used; i++) {
      float ms = 0;
      CK(cudaEventSynchronize(waits[i].second));
      CK(cudaEventElapsedTime(&ms, waits[i].first, waits[i].second));
      w.seconds += ms / 1e3, w.long_ones += ms > 1;
    }
    waits_used = 0;
    return w;
  }

  void *place(size_t bytes, int group, wb::Form form) override {
    if (group < 0) return wb::Placement::place(bytes, group, form);
    if (one_room_for_the_rest) rest.n_rooms = 1;
    Part &p = one_room_for_experts && form == wb::Form::RoutedBlocks ? experts : rest;
    GrowingRoom &room = p.room_of(group);
    const size_t at = p.used_now;
    p.used_now += (bytes + 255) & ~(size_t)255;
    room.need(p.used_now + slack);
    return room.base + at;
  }
  void *place_routed(size_t bytes, int group, int layer) override {
    if (!experts_by_layer) return place(bytes, group, wb::Form::RoutedBlocks);
    if (experts_by_tensor || layers_with_experts.empty() || layers_with_experts.back() != layer) {
      layers_with_experts.push_back(layer);
      routed_bytes.push_back(0);
    }
    if (experts_rooms < 1 || experts_rooms > Part::MOST_ROOMS) {
      printf("internal error: weight stream: %d GPU buffers for the experts; 1 to %d are supported\n", experts_rooms,
             Part::MOST_ROOMS);
      exit(4);
    }
    experts.n_rooms = experts_rooms;
    if (experts_by_tensor) {
      const int e = (int)layers_with_experts.size() - 1;
      if (!e) lay_the_rooms_back_to_back();
      experts.room_by_number.push_back(room_of_tensor(e));
      experts.room_of(e).step = TENSOR_ROOMS_STEP;
    }
    GrowingRoom &room = experts.room_of((int)layers_with_experts.size() - 1);
    const size_t at = routed_bytes.back();
    routed_bytes.back() += (bytes + 255) & ~(size_t)255;
    room.need(routed_bytes.back() + slack);
    return room.base + at;
  }
  // With experts_by_tensor: the buffer of tensor number e. A down tensor may be longer than a
  // layer's other two, so the down tensors have separate buffers, 0 .. experts_rooms / 3 - 1. The
  // other tensors have the rest. Each of the two sets of buffers is used cyclically.
  // experts_rooms must be at least 3, or a down tensor's number divides by zero here. WeightStream
  // does not check that in time: start() compares experts_rooms with 3 * routed_layers_a_group, but
  // runs after this function's first calls. The prover program `cp_prove` sets experts_by_tensor
  // only for the `-memory` keyword `tensor-rooms=N` with N from 3 to 16 (arguments.h); N becomes
  // experts_rooms.
  // Public source: a pool of fixed-size buffers in size groups is described, for buffering an
  // application's file I/O, in the documentation of the Hermes I/O buffering system, page
  // "Buffer Pool", https://grc.iit.edu/docs/hermes/components/buffer-pool. Here each group is
  // used in rotation instead of through a free list.
  int room_of_tensor(int e) const {
    const int for_down = experts_rooms / 3;
    return e % 3 == 2 ? e / 3 % for_down : for_down + (e / 3 * 2 + e % 3) % (experts_rooms - for_down);
  }
  // With experts_by_tensor: the experts' buffers are mapped in steps of this length
  // (GrowingRoom::step), and their lengths are multiples of it.
  static constexpr size_t TENSOR_ROOMS_STEP = (size_t)32 << 20;
  // With experts_by_tensor: puts the experts' buffers one after another, in the order of their
  // numbers, in a single range of addresses, each long enough for the longest tensor that
  // will_place() announced for it. A caller that borrows all of them through room() then has one
  // range of memory. A buffer that no tensor uses gets no addresses. will_place() must have
  // announced every tensor or none; if none, each buffer reserves a separate range.
  // Public source: NVIDIA's TensorRT documentation ("How TensorRT Works", Memory,
  // https://docs.nvidia.com/deeplearning/tensorrt/11.0.0/architecture/how-trt-works.html) describes
  // lending idle working memory to other uses. Here the memory lent is not working memory: it is the weight stream's
  // GPU buffers, which hold weights and are loaded again afterwards (see room()).
  // Public sources for the groups of buffers: see room_of_tensor(); for the announced sizes: see will_place().
  void lay_the_rooms_back_to_back() {
    std::vector<size_t> longest(experts_rooms, 0);
    size_t all = 0;
    for (const ToldUnit &unit : told)
      if (unit.part == &experts) {
        size_t &room = longest[room_of_tensor(unit.number)];
        room = std::max(room, (unit.bytes + slack + TENSOR_ROOMS_STEP - 1) & ~(TENSOR_ROOMS_STEP - 1));
      }
    for (size_t bytes : longest) all += bytes;
    if (!all) return;
    char *at = GrowingRoom::reserve(all);
    for (int r = 0; r < experts_rooms; at += longest[r], r++)
      if (longest[r]) experts.rooms[r].lay_at(at, longest[r]), experts.rooms[r].step = TENSOR_ROOMS_STEP;
  }
  // Records in `told` how many bytes each unit will have, rounded as in place(), place_routed() and
  // place_end(). HostMemoryMaker allocates the units' host memory from this list, in its order, and
  // lay_the_rooms_back_to_back() sizes the experts' buffers from it. If a unit's later size differs
  // from the announced one, host_memory_of() exits.
  // Public source: planning memory from sizes known from shapes and types alone, before any weight is loaded: Hugging
  // Face Accelerate does this with a model on PyTorch's meta device to compute its device map (Accelerate
  // documentation, "Handling big models"); here BlockWeights::fill() announces the sizes in a separate loop.
  // Public source: asking for a size in a first call: CUB's device-wide calls (NVIDIA CCCL
  // documentation, CUB, "Device-Wide Primitives", "Determining Temporary Storage Requirements",
  // https://nvidia.github.io/cccl/cub/api_docs/device_wide.html).
  void will_place(size_t bytes, int group, wb::Form form, int layer) override {
    const size_t n = (bytes + 255) & ~(size_t)255;
    const bool routed = form == wb::Form::RoutedBlocks;
    if (group < 0) {
      if (ends_by_turns) told.push_back({&experts, told_experts++, n});
      return;
    }
    if (routed && experts_by_layer) { // a layer's three routed tensors are announced one after the other
      if (experts_by_tensor || told_layer != layer) told.push_back({&experts, told_experts++, 0}), told_layer = layer;
      told.back().bytes += n;
      return;
    }
    const Part *p = routed && one_room_for_experts ? &experts : &rest;
    for (auto unit = told.rbegin(); unit != told.rend(); unit++)
      if (unit->part == p && unit->number == group) {
        unit->bytes += n;
        return;
      }
    told.push_back({p, group, n});
  }
  // Returns the address of the host copy of a tensor that is stored as blocks. The memory is pinned
  // before it is returned, so that no weight in it can be swapped out. Weights elsewhere can be
  // swapped out: the C library's file buffer reads on past a model file's header, into the weights.
  uint8_t *host_twin(const void *placed, size_t bytes) override {
    const char *at = (const char *)placed;
    Part *p = &rest;
    int number = (int)rest.image.size();
    size_t used = rest.used_now;
    // Looks for `placed` in the experts' units that are being filled: one, or with
    // experts_by_tensor a layer's three, which are in three buffers.
    for (int e = experts_by_layer ? (int)experts.image.size() : number, i = 0;
         one_room_for_experts && i < (experts_by_tensor ? 3 : 1) && (!experts_by_layer || e < (int)routed_bytes.size());
         e++, i++) {
      const GrowingRoom &room = experts.room_of(e);
      if (at >= room.base && at < room.base + room.mapped)
        p = &experts, number = e, used = experts_by_layer ? routed_bytes.at(e) : experts.used_now;
    }
    const size_t first = (size_t)(at - p->room_of(number).base);
    if (at < p->room_of(number).base || first + bytes > used) {
      printf(
          "internal error: weight stream: a host copy was requested of bytes outside the unit that is being filled\n");
      exit(4);
    }
    Part::Filling &filling = p->filling[number];
    if (!filling.host) filling.bytes = used + slack, filling.host = host_memory_of(*p, number, filling.bytes);
    filling.from_the_file.push_back({first, bytes});
    return (uint8_t *)filling.host + first;
  }
  // With ends_by_turns: the numbers e of the embedding's cells, end_unit[0], and of the head,
  // end_unit[1]; they follow the last layer's number. A pass needs the embedding before group 0's
  // forward computation and again before its row passes, and the head after the last group.
  int end_unit[2] = {-1, -1};
  void *place_end(size_t bytes, int which) override {
    if (!ends_by_turns) return wb::Placement::place_end(bytes, which);
    layers_with_experts.push_back(-1 - which);
    routed_bytes.push_back((bytes + 255) & ~(size_t)255);
    GrowingRoom &room = experts.room_of((int)layers_with_experts.size() - 1);
    room.need(routed_bytes.back() + slack);
    return room.base;
  }
  void end_filled(int which, cudaStream_t stream) override {
    if (!ends_by_turns) return;
    end_unit[which] = (int)experts.image.size();
    experts.used_now = routed_bytes[end_unit[which]];
    carry_to_the_host(experts, end_unit[which], stream);
  }
  // With experts_by_layer: completes the host copy of one layer's routed experts. The layers must
  // come in the order in which place_routed() placed them; otherwise it exits.
  void routed_filled(int layer, cudaStream_t stream) override {
    if (!experts_by_layer) return;
    for (int tensor = 0; tensor < (experts_by_tensor ? 3 : 1); tensor++) {
      const int e = (int)experts.image.size();
      if (e >= (int)layers_with_experts.size() || layers_with_experts[e] != layer) {
        printf("internal error: weight stream: the routed experts of layer %d were filled out of order\n", layer);
        exit(4);
      }
      experts.used_now = routed_bytes[e];
      carry_to_the_host(experts, e, stream);
    }
  }
  // The result of pinned_host_memory(): the memory, or why there is none. That function neither
  // prints nor exits, because it also runs on HostMemoryMaker's thread, which must not end the
  // program; held() does both.
  struct HostMemory {
    char *p = nullptr;
    size_t mapped = 0;                    // the requested bytes, rounded up to huge pages
    bool no_memory = false;               // the operating system gave none
    cudaError_t registered = cudaSuccess; // the status of cudaHostRegister()
  };
  // Allocates pinned host memory, zeroed; asks for huge pages only if huge_pages is set. The memory
  // is not in fork_memory.host_buffers, so ForkMemory::wipe() does not zero it
  // (listed in prover/docs/ends-in-short.md, "In memory and on disk").
  // The results of the madvise() calls are not checked.
  // Memory from the operating system is touched by several threads and then registered, to shorten
  // cudaHostRegister(). The idea is public: PyTorch's notes, "CUDA semantics", "Optimizing memory
  // usage with PYTORCH_CUDA_ALLOC_CONF", pinned_use_cuda_host_register and
  // pinned_num_register_threads (https://docs.pytorch.org/docs/2.9/notes/cuda.html). Here mmap()
  // gives the memory, and 16 threads zero it.
  // Public source: cudaHostRegister(): NVIDIA, CUDA Runtime API, "Memory Management"
  // (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).
  // Public source: the hint for huge pages: madvise(2), MADV_HUGEPAGE: "The kernel will also
  // allocate huge pages directly when the region is naturally aligned to the huge page size"
  // (https://man7.org/linux/man-pages/man2/madvise.2.html). Hence the address is a multiple of
  // 2 MiB, the huge page's size on x86-64.
  static HostMemory pinned_host_memory(size_t bytes) {
    const size_t huge = (size_t)2 << 20, n = (bytes + huge - 1) & ~(huge - 1);
    HostMemory made;
    made.mapped = n;
    void *raw = mmap(nullptr, n + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
      made.no_memory = true;
      return made;
    }
    char *p = (char *)(((uintptr_t)raw + huge - 1) & ~(uintptr_t)(huge - 1));
    if (huge_pages) madvise(p, n, MADV_HUGEPAGE); // a hint only: the memory also works without huge pages
    madvise(p, n, MADV_DONTDUMP); // keeps the weights out of core files
    madvise(p, n, MADV_DONTFORK); // and out of child processes
    const size_t piece = (size_t)64 << 20;
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    for (int w = 0; w < 16; w++)
      workers.emplace_back([&] {
        for (size_t o; (o = next.fetch_add(piece)) < n;) memset(p + o, 0, std::min(piece, n - o));
      });
    for (auto &w : workers) w.join();
    made.registered = cudaHostRegister(p, n, cudaHostRegisterDefault);
    made.p = p;
    return made;
  }
  // Returns the memory; exits if the allocation or its registration failed.
  static char *held(const HostMemory &made) {
    if (made.no_memory) {
      printf("weight stream: could not get %.1f GB of host memory for a layer group's weights. All the weights are "
             "kept in host memory: free host memory, or use a machine that has more\n",
             made.mapped / 1e9);
      stop(End::HostMemory);
    }
    CK(made.registered);
    return made.p;
  }
  // Completes the host copy of a group's weights; with experts_by_layer, of those other than the
  // routed experts (see routed_filled()). The groups must come in order, from 0; otherwise it
  // exits.
  void group_filled(int group, cudaStream_t stream) override {
    if (group < 0) return;
    if (group != (int)rest.image.size()) {
      printf("internal error: weight stream: group %d was filled out of order\n", group);
      exit(4);
    }
    for (Part *p : parts())
      if (p == &rest || !experts_by_layer) carry_to_the_host(*p, group, stream);
    experts_before.push_back((int)experts.image.size());
  }
  // With experts_by_layer: the numbers e of group g's experts, as the first number and one past the
  // last.
  std::pair<int, int> experts_of(int g) const { return {experts_before[g], experts_before[g + 1]}; }
  int layer_of_experts(int e) const { return layers_with_experts[e]; }
  int experts_of_layer(int layer) const { // with experts_by_tensor: the first of its three numbers
    return (int)(std::find(layers_with_experts.begin(), layers_with_experts.end(), layer) -
                 layers_with_experts.begin());
  }
  // Whether the experts numbered e hold the routed tensor that a row pass of 'kind' reads: 7 gate, 8 up, 10 down.
  bool holds_kind(int e, int kind) const {
    return !experts_by_tensor || e - experts_of_layer(layers_with_experts[e]) == (kind == 7 ? 0 : kind == 8 ? 1 : 2);
  }
  // layer_ready() and layer_used() are experts_ready() and experts_used() for all of a layer's
  // routed experts. With experts_by_tensor the forward computation reads the down tensor last, so a
  // caller may set this flag (prover/prover/layer_setup.cuh does): layer_ready() then waits for the
  // other two tensors only, and the caller must call layer_down_ready() before the first kernel
  // that reads down. Without experts_by_tensor the flag has no effect.
  // Public source: see give_back() (a consumer waits for each stage of a pipeline only when it
  // needs that stage).
  bool down_is_asked_for_apart = false;
  void layer_ready(int layer) {
    const int tensors = !experts_by_tensor ? 1 : down_is_asked_for_apart ? 2 : 3;
    for (int e = experts_of_layer(layer), i = 0; i < tensors; i++) experts_ready(e + i);
  }
  void layer_down_ready(int layer) {
    if (experts_by_tensor && down_is_asked_for_apart) experts_ready(experts_of_layer(layer) + 2);
  }
  void layer_used(int layer) {
    for (int e = experts_of_layer(layer), i = 0; i < (experts_by_tensor ? 3 : 1); i++) experts_used(e + i);
  }

  // Call once, after the last group is filled. It sets the order of the experts' uses and queues
  // the first loads. A production build does not read the environment variable WEIGHT_FEED
  // (TESTENV(), prover/common/setting.h).
  void start() {
    for (cudaEvent_t down : brought_down) { // the loads below read the units' host memory
      CK(cudaEventSynchronize(down));
      CK(cudaEventDestroy(down));
    }
    brought_down.clear();
    {
      int least = 0, greatest = 0;
      CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
      CK(cudaStreamCreateWithPriority(&feed, cudaStreamNonBlocking, least)); // lowest priority: the loading kernel is meant to give way to other kernels
    }
    if (const char *word = TESTENV("WEIGHT_FEED")) { // for tests and tuning: "engine", or the loading kernel's blocks and threads
      if (!strcmp(word, "engine"))
        feed_blocks = 0;
      else if (sscanf(word, "%d,%d", &feed_blocks, &feed_threads) != 2 || feed_blocks < 1 || feed_threads < 32 || feed_threads > 1024) {
        printf("weight stream: WEIGHT_FEED wants the word engine, or blocks,threads\n");
        exit(2);
      }
    }
    for (Part *p : parts())
      for (int r = 0; r < p->n_rooms; r++) {
        CK(cudaEventCreateWithFlags(&p->loaded[r], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&p->released[r], cudaEventDisableTiming));
      }
    if (experts_by_tensor &&
        (!experts_by_layer || ends_by_turns || experts.n_rooms < 3 * routed_layers_a_group)) {
      printf("internal error: weight stream: experts_by_tensor is set without experts_by_layer, with ends_by_turns, or "
             "with fewer rooms than a layer group has routed tensors\n");
      exit(4);
    }
    // experts_order is the order in which one request uses the experts during a pass: for each
    // group g, its experts' numbers e, stretches_of(g) times, each time in the order of use_of()
    // (in group 0 with ends_by_turns, each time after the embedding).
    // span_begins divides it into spans: one for each group and a last one for a pass's end (the
    // head, with ends_by_turns). step() defines how a batch pass goes through the spans.
    for (int g = 0; experts_by_layer && g < (int)rest.image.size(); g++) {
      span_begins.push_back(experts_order.size());
      for (int stretch = 0; stretch < stretches_of(g); stretch++) {
        if (g == 0 && ends_by_turns) experts_order.push_back(end_unit[0]);
        for (int i = 0; i < experts_of(g).second - experts_of(g).first; i++)
          experts_order.push_back(use_of(g, i));
      }
    }
    span_begins.push_back(experts_order.size());
    if (ends_by_turns) experts_order.push_back(end_unit[1]);
    span_begins.push_back(experts_order.size());
    if (rest.n_rooms != (one_room_for_the_rest ? 1 : 2)) {
      printf("internal error: weight stream: %d GPU buffers for the weights other than the experts; %d expected\n",
             rest.n_rooms, one_room_for_the_rest ? 1 : 2);
      exit(4);
    }
    restart();
  }
  // How many times a turn in group g goes through the group's numbers: once in each phase. But a
  // group with only one number has a single use of it, which spans both phases, except group 0 with
  // ends_by_turns, whose phases each begin with the embedding.
  int stretches_of(int g) const {
    return experts_of(g).second - experts_of(g).first > 1 || (g == 0 && ends_by_turns) ? 2 : 1;
  }
  // The number that group g uses at step i of a phase. The caller must use the experts in this
  // order; otherwise experts_used() exits.
  // Public source: doing all work on one block of data, block after block: the data shackle of Kodukula, Ahmed and
  // Pingali, "Data-centric Multi-level Blocking", PLDI 1997, section 4.1, stated for cache blocking by a compiler.
  // Public source: reading chunks in the order of their delivery, stated for table scans from disk: the CScan operator
  // of Cooperative Scans, Switakowski, Boncz, Zukowski, "From Cooperative Scans to Predictive Buffer Management",
  // Proceedings of the VLDB Endowment, Vol. 5, No. 12, 2012 (PVLDB 5(12)), section 2, https://arxiv.org/pdf/1208.4170.
  // Here the blocks are the stream's units, and start() fixes the order in advance for loader and reader alike.
  int use_of(int g, int i) const { return experts_of(g).first + i; }
  // Waits until the GPU is idle, overwrites all buffers and queues the first loads, so that the
  // next pass finds nothing on the GPU that an earlier pass loaded.
  void restart() {
    CK(cudaDeviceSynchronize());
    loads_queued = 0;
    for (Part *p : parts()) {
      for (auto &room : p->rooms)
        if (room.base) CK(cudaMemset(room.base, 0xEE, room.mapped));
      std::fill(p->in_room, p->in_room + Part::MOST_ROOMS, -1);
    }
    CK(cudaDeviceSynchronize());
    first_loads();
  }
  // Whether group g is in its buffers or its load is queued. With experts_by_layer the experts are
  // not looked at.
  bool holds(int g) const {
    if (g < 0 || g >= (int)rest.image.size() || rest.in_room[rest.room_number(g)] != g) return false;
    return !one_room_for_experts || experts_by_layer || experts.in_room[experts.room_number(g)] == g;
  }
  // Makes the calling thread's CUDA stream wait until group g is loaded into its buffers (with
  // experts_by_layer: its weights other than the experts). Exits unless the load is done or queued.
  void ready(int g) {
    for (Part *p : parts())
      if (p == &rest || !experts_by_layer) ready(*p, g);
  }
  // With experts_by_layer: makes the calling thread's CUDA stream wait until the experts numbered e
  // are loaded. Exits like ready().
  void experts_ready(int e) { ready(experts, e); }
  // Ends a use: call this after queuing, on the calling thread's stream, the use's last read of the
  // experts numbered e. It releases their buffer to the experts that use it next in the batch
  // pass's order (step()). If those are the same experts, they stay loaded.
  //
  // All loads run on one stream (`feed`), in the order in which they are queued. So the call that
  // releases a buffer does not always queue its load: if a buffer that this turn has yet to release
  // will take experts that are used earlier, that load goes first. The waiting load is queued in
  // the turn's last call at the latest. Until then its buffer counts as empty, and the link to the
  // GPU may be idle for as long as the turn takes from this release to the later one.
  //
  // The order of the loads depends on nothing in a request: only on the experts' numbers and
  // buffers, rooms_will_be_lent, *last_turn and, in the last span (a pass's end, after all groups),
  // the number of requests in the batch pass.
  // Public source: serving first the waiting job that is closest to its deadline is
  // earliest-deadline-first scheduling (Wikipedia, "Earliest deadline first scheduling",
  // https://en.wikipedia.org/wiki/Earliest_deadline_first_scheduling); the page says that without
  // preemption it is optimal only among the algorithms "that do not allow inserted idle time". Here
  // the jobs are copies, which cannot be interrupted; a copy's deadline is its experts' next use;
  // and idle time is inserted.
  void experts_used(int e) {
    if (experts_order[experts_at] != e) {
      printf("internal error: weight stream: the experts numbered %d are released, but the order of the pass expects "
             "number %d\n", e,
             experts_order[experts_at]);
      exit(4);
    }
    const size_t at = experts_at;
    const int span = span_of(at);
    const bool again = walked_again(span);
    const int next = next_in_the_room(at, again);
    step();
    // Whether `tenant`, which follows `after` in a buffer, must be loaded. With rooms_will_be_lent
    // a lower number is not loaded: it belongs to the next pass, since with experts_by_tensor,
    // which that flag requires, a buffer's numbers never decrease within a pass.
    auto is_loaded = [this](int tenant, int after) { return tenant != after && !(rooms_will_be_lent && tenant < after); };
    if (next != e) {
      // The test mode `scribble` (ForkMemory::Test, fork_memory.h) overwrites the buffer, so that a
      // later read cannot find these experts by chance.
      if (fork_memory.test.scribble)
        CK(cudaMemsetAsync(experts.room_of(e).base, 0xEE, experts.room_of(e).mapped, cudaStreamPerThread));
      release(experts, e);
      if (is_loaded(next, e)) loads_held.push_back({next, uses_until(at, next, again)});
    }
    // The number of uses, from this one, until the first use of any experts that the rest of this
    // turn will still load.
    size_t soonest_to_come = SIZE_MAX;
    for (size_t later = at + 1; later < span_begins[span + 1]; later++)
      if (const int tenant = next_in_the_room(later, again); is_loaded(tenant, experts_order[later]))
        soonest_to_come = std::min(soonest_to_come, later - at + uses_until(later, tenant, again));
    std::sort(loads_held.begin(), loads_held.end(), // no two are equal: a use is of one number
              [](const HeldLoad &a, const HeldLoad &b) { return a.uses_to_go < b.uses_to_go; });
    size_t queued = 0;
    while (queued < loads_held.size() && loads_held[queued].uses_to_go < soonest_to_come)
      load_what_is_released(experts, loads_held[queued++].number);
    loads_held.erase(loads_held.begin(), loads_held.begin() + queued);
    for (HeldLoad &held : loads_held) held.uses_to_go--; // counted from the turn's next use
  }
  // How many released buffers wait for experts_used() to queue their loads; 0 between turns.
  size_t loads_held_back() const { return loads_held.size(); }
  // The experts that a pass uses first in buffer r; r if it uses none there.
  int first_in_room(int r) const {
    for (int e : experts_order)
      if (experts.room_number(e) == r) return e;
    return r;
  }
  // The smallest number above e and below `end` that shares e's buffer; 0 if there is none.
  int next_later_in_the_room(int e, int end) const {
    for (int next = e + 1; next < end; next++)
      if (experts.room_number(next) == experts.room_number(e)) return next;
    return 0;
  }
  // The experts that use the same buffer next after use number `at` of experts_order. If `again`,
  // another request goes through the span of `at` after the current turn, so the span's last use
  // is followed by the span's first use.
  int next_in_the_room(size_t at, bool again) const {
    const int span = span_of(at);
    size_t next = at;
    do next = again && next + 1 == span_begins[span + 1] ? span_begins[span] : (next + 1) % experts_order.size();
    while (experts.room_number(experts_order[next]) != experts.room_number(experts_order[at]));
    return experts_order[next];
  }
  // How many uses after use number `at` the experts numbered `number` are used next, in the order
  // that next_in_the_room() follows.
  size_t uses_until(size_t at, int number, bool again) const {
    const int span = span_of(at);
    size_t next = at, uses = 0;
    do {
      next = again && next + 1 == span_begins[span + 1] ? span_begins[span] : (next + 1) % experts_order.size();
      uses++;
    } while (experts_order[next] != number);
    return uses;
  }
  // Call after queuing the last use of group g on the calling thread's stream. Each of its buffers
  // is then loaded with the next group, cyclically, that shares it (but see no_walk_follows). With
  // experts_by_layer this concerns only the buffer of the weights other than the experts.
  void done(int g) {
    experts_done(g);
    if (fork_memory.test.scribble &&
        experts_by_layer) // a later read of this group, or a missing load, must not find it by chance
      CK(cudaMemsetAsync(rest.room_of(g).base, 0xEE, rest.room_of(g).mapped, cudaStreamPerThread));
    give_back(rest, g);
  }
  // With one_room_for_experts and without experts_by_layer: call after queuing the last use of
  // group g's routed experts on the calling thread's stream. The call may be omitted: done() makes
  // it. A second call for the same group does nothing, unless there is only one group, which is
  // then loaded once more (not with no_walk_follows).
  void experts_done(int g) {
    if (!one_room_for_experts || experts_by_layer || experts.in_room[0] != g) return;
    // Scribble mode: see experts_used().
    if (fork_memory.test.scribble)
      CK(cudaMemsetAsync(experts.rooms[0].base, 0xEE, experts.rooms[0].mapped, cudaStreamPerThread));
    give_back(experts, g);
  }
  // room(r) and room_bytes(r) take r from 0 to n_rooms() - 1; neither checks it.
  static constexpr int n_rooms() { return 2 + Part::MOST_ROOMS; }
  // The mapped bytes of buffer r, which room(r) would return; the buffer is not borrowed.
  size_t room_bytes(int r) const {
    return (r < 2 ? rest : experts).rooms[r < 2 ? r : r - 2].mapped;
  }
  // The length in bytes of the longest contiguous range of memory among the buffers that room()
  // returns. Adjacent buffers count as one range.
  size_t longest_stretch_of_rooms() const {
    size_t longest = 0, stretch = 0;
    const char *end = nullptr;
    for (int r = 0; r < n_rooms(); r++) {
      const size_t bytes = room_bytes(r);
      if (!bytes) continue;
      const char *first = (r < 2 ? rest : experts).rooms[r < 2 ? r : r - 2].base;
      stretch = first == end ? stretch + bytes : bytes;
      end = first + bytes;
      longest = std::max(longest, stretch);
    }
    return longest;
  }
  // Lets the caller borrow buffer r while no group is needed: returns its first address and mapped
  // bytes. It waits for nothing: the caller must order its use after the buffer's queued loads and
  // reads. restart() and later loads overwrite the buffer. A buffer that has bytes counts as empty
  // from this call on: outside pass(), ready() exits with an internal error until restart() has
  // loaded it again. Buffers 0 and 1 are for the weights other than the experts; buffer 1 has no
  // bytes with one_room_for_the_rest. Buffers 2 to n_rooms() - 1 are the experts' and have none
  // without one_room_for_experts.
  // Public source: giving GPU memory that held weights to other data, and fetching the weights
  // again from the host's copy, is described for the KV cache of an inference server by MIRAGE:
  // R. Li, S. Pal, V. N. Pullu, P. Sinha, J. Ryoo, L. K. John, N. J. Yadwadkar, "MIRAGE: KV Cache
  // Optimization through Parameter Remapping for Multi-tenant LLM Serving", arXiv 2507.11507,
  // version 1 (later versions bear the title "Oneiros: KV Cache Optimization .."), abstract and
  // section 1, https://arxiv.org/pdf/2507.11507v1.
  // Public source: the same between workloads is described by vLLM's documentation of its Sleep
  // Mode.
  // Public source: PatrickStar reuses parameter memory for gradients in training: J. Fang, Z. Zhu,
  // S. Li, H. Su, Y. Yu, J. Zhou, Y. You, "PatrickStar: Parallel Training of Pre-trained Models via
  // Chunk-based Dynamic Memory Management", arXiv 2108.05818, section 6.1 "Design on a Single GPU"
  // and Figure 5, https://arxiv.org/pdf/2108.05818.
  // See also lay_the_rooms_back_to_back().
  std::pair<char *, size_t> room(int r) {
    Part &p = r < 2 ? rest : experts;
    const int i = r < 2 ? r : r - 2;
    if (p.rooms[i].mapped) p.in_room[i] = -1;
    return {p.rooms[i].base, p.rooms[i].mapped};
  }
  // Visits all groups once, separately from the passes that ready() and done() drive. Work that
  // f(g) queues on the calling thread's stream runs after group g is loaded. The GPU must be idle
  // at the call (not checked). pass() loads the first groups itself, so the buffers may have been
  // borrowed through room() since the last done(). After pass() every buffer counts as empty: call
  // restart() before the next ready().
  template <class F> void pass(F f) {
    pass(f, [](int) {});
  }
  // With experts_by_layer, routed(layer) is called before f(g) for each of group g's layers with
  // experts; the work that it queues runs after those experts are loaded. No experts are loaded
  // twice. With ends_by_turns, after all groups, routed(-1) is called likewise for the embedding
  // and routed(-2) for the head. f(g) must not read the experts: when routed(layer) has returned,
  // their buffers may be released to later experts.
  template <class F, class R> void pass(F f, R routed) {
    const int groups = (int)rest.image.size(), all_experts = (int)experts.image.size();
    first_loads(true);
    for (int g = 0; g < groups; g++) {
      ready(g);
      // With experts_by_tensor, routed(layer) is called once, after the wait for the layer's third
      // tensor; none of the layer's buffers is given back before that.
      const int together = experts_by_tensor ? 3 : 1;
      for (int e = experts_by_layer ? experts_of(g).first : 0; experts_by_layer && e < experts_of(g).second; e++) {
        experts_ready(e);
        if ((e - experts_of(g).first) % together != together - 1) continue;
        routed(layers_with_experts[e]);
        for (int x = e - together + 1; x <= e; x++)
          if (const int next = next_later_in_the_room(x, all_experts)) give_back(experts, x, next);
      }
      f(g);
      // The experts' load is queued first: the next group needs it. The other weights' load is for
      // the group after the next.
      if (one_room_for_experts && !experts_by_layer && g + 1 < groups) give_back(experts, g);
      if (g + rest.n_rooms < groups) give_back(rest, g);
    }
    for (int e = ends_by_turns ? end_unit[0] : all_experts; e < all_experts; e++) {
      experts_ready(e);
      routed(layers_with_experts[e]);
      if (e + experts.n_rooms < all_experts) give_back(experts, e, e + experts.n_rooms);
    }
    for (Part *p : parts())
      for (int &in : p->in_room) in = -1;
  }
  // How many loads have been queued since the last restart().
  size_t loads() const { return loads_queued; }
  // The bytes of all units together, and of the largest group; each unit counts with its slack.
  double total_bytes() const {
    double n = 0;
    for (size_t used : rest.used) n += used;
    for (size_t used : experts.used) n += one_room_for_experts ? used : 0;
    return n;
  }
  size_t largest_group() const {
    size_t n = 0;
    for (size_t g = 0; g < rest.used.size(); g++) n = std::max(n, group_bytes(g));
    return n;
  }

private:
  // Kernels may read a little past a tensor's end. After a unit's last tensor
  // come this many zero bytes, which are copied and loaded with the unit.
  static constexpr size_t slack = 256;
  // A part of the weights, with separate buffers. Its units are numbered by group or, for `experts`
  // with experts_by_layer, by e. n_rooms: how many of `rooms` it may use. used_now: the bytes
  // placed so far in the unit that is being filled (for a number e: see routed_bytes). WeightStream
  // never frees its buffers or host copies.
  struct Part {
    static constexpr int MOST_ROOMS = 16;
    int n_rooms;
    explicit Part(int rooms_at_first) : n_rooms(rooms_at_first) { std::fill(in_room, in_room + MOST_ROOMS, -1); }
    GrowingRoom rooms[MOST_ROOMS];
    size_t used_now = 0;
    std::vector<char *> image; // each unit's pinned host copy
    // filling[g], while unit g is being filled: its host memory and length, once host_twin() has
    // requested the memory, and the ranges of it that host_twin() has returned (offset in the
    // buffer, length).
    struct Filling {
      char *host = nullptr;
      size_t bytes = 0;
      std::vector<std::pair<size_t, size_t>> from_the_file;
    };
    std::map<int, Filling> filling;
    // used[g]: unit g's bytes, with the slack. loaded[r], released[r]: events recorded after buffer
    // r's load and after the last read of it. in_room[r]: the unit loaded or queued; -1 if the
    // buffer counts as empty.
    std::vector<size_t> used;
    cudaEvent_t loaded[MOST_ROOMS], released[MOST_ROOMS];
    int in_room[MOST_ROOMS];
    // The buffer of unit g: g % n_rooms, unless room_by_number has an entry for g.
    std::vector<int> room_by_number;
    int room_number(int g) const { return g < (int)room_by_number.size() ? room_by_number[g] : g % n_rooms; }
    GrowingRoom &room_of(int g) { return rooms[room_number(g)]; }
  };
  Part rest{2}, experts{1};
  std::vector<Part *> parts() {
    return one_room_for_experts ? std::vector<Part *>{&rest, &experts} : std::vector<Part *>{&rest};
  }
  size_t group_bytes(size_t g) const {
    size_t n = rest.used[g];
    if (experts_by_layer)
      for (int e = experts_of((int)g).first; e < experts_of((int)g).second; e++) n += experts.used[e];
    else if (one_room_for_experts)
      n += experts.used[g];
    return n;
  }
  // With experts_by_layer. layers_with_experts[e], routed_bytes[e]: the layer (-1 embedding,
  // -2 head) and bytes of number e. experts_before[g]: how many numbers precede group g.
  // experts_order: the uses in one request's pass. experts_at: the next use.
  std::vector<int> layers_with_experts, experts_before{0}, experts_order;
  std::vector<size_t> routed_bytes;
  size_t experts_at = 0;
  // span_begins: where each span of experts_order begins, then the order's length. turns_walked:
  // how many requests of this batch pass have gone through the span of the next use.
  // requests_in_wave: set when a group's last turn ends, and read only in the last span; until
  // then it has the previous batch pass's number, or 1 after restart() or pass().
  std::vector<size_t> span_begins;
  int turns_walked = 0, requests_in_wave = 1;
  // A load whose buffer is released and which experts_used() has not queued yet: the experts'
  // number, and how many uses ahead their next use is.
  struct HeldLoad {
    int number;
    size_t uses_to_go;
  };
  std::vector<HeldLoad> loads_held;
  // The span that contains use number `at` of experts_order: a group's number, or the number of
  // groups for the last span. A group that uses no experts has an empty span.
  int span_of(size_t at) const {
    return (int)(std::upper_bound(span_begins.begin(), span_begins.end(), at) - span_begins.begin()) - 1;
  }
  // Whether another request goes through the span after the current turn. In a group's span: if
  // last_turn points to false. In the last span: until as many requests have gone through it as had
  // turns in the groups.
  bool walked_again(int span) const {
    return span + 2 < (int)span_begins.size() ? last_turn && !*last_turn : turns_walked + 1 < requests_in_wave;
  }
  // Moves on to the next use. After a span's last use comes that span's first use again, once for
  // each other request of the batch pass; then the next span's first use, and after the last span
  // the next pass's first use.
  void step() {
    const int span = span_of(experts_at);
    if (++experts_at < span_begins[span + 1]) return;
    const bool again = walked_again(span);
    turns_walked++;
    if (again) {
      experts_at = span_begins[span];
      return;
    }
    if (span + 2 < (int)span_begins.size()) requests_in_wave = turns_walked; // as many go through the last span
    turns_walked = 0;
    experts_at %= experts_order.size();
  }
  // What will_place() has recorded: a unit and its bytes without the slack, in the order of first
  // announcement.
  struct ToldUnit {
    const Part *part;
    int number;
    size_t bytes;
  };
  std::vector<ToldUnit> told;
  int told_experts = 0, told_layer = -1;
  // Allocates the units' host memory on a separate host thread, in the order of `asks`, while the
  // calling thread reads the model file. It stays at most AHEAD allocations ahead of those taken,
  // unless a call of take() waits for a later one. It makes no CUDA call after the last allocation.
  struct HostMemoryMaker {
    static constexpr size_t AHEAD = 3;
    std::vector<size_t> asks;      // each allocation's bytes; 0: skipped, counts as taken
    std::vector<HostMemory> made;  // made[i] is the allocation for asks[i]
    size_t taken = 0, awaited = 0; // allocations returned; `take` waits for all before this number
    bool ends = false;
    int device = 0;
    std::mutex lock;
    std::condition_variable moved;
    std::thread thread;
    void run() {
      const cudaError_t on_device = cudaSetDevice(device);
      std::unique_lock<std::mutex> held_lock(lock);
      while (made.size() < asks.size()) {
        moved.wait(held_lock, [&] { return ends || made.size() < taken + AHEAD || made.size() < awaited; });
        if (ends) return;
        const size_t bytes = asks[made.size()];
        held_lock.unlock();
        HostMemory piece;
        if (bytes) piece = pinned_host_memory(bytes);
        if (on_device != cudaSuccess) piece.registered = on_device;
        held_lock.lock();
        made.push_back(piece);
        taken += !bytes;
        moved.notify_all();
      }
    }
    HostMemory take(size_t i) {
      std::unique_lock<std::mutex> held_lock(lock);
      awaited = std::max(awaited, i + 1);
      moved.notify_all();
      moved.wait(held_lock, [&] { return made.size() > i; });
      taken++;
      moved.notify_all();
      return made[i];
    }
    ~HostMemoryMaker() {
      {
        std::lock_guard<std::mutex> held_lock(lock);
        ends = true;
      }
      moved.notify_all();
      if (thread.joinable()) thread.join();
    }
  };
  HostMemoryMaker maker;
  // A unit that will_place() announced gets the allocation made for it. Any other unit (nothing was
  // announced, or the part is empty in this group) gets memory that is allocated in this call.
  // The first call that finds `told` not empty starts `maker`'s thread on the list as it then is,
  // so every will_place() must come before it (not checked).
  // Public sources: see will_place().
  char *host_memory_of(const Part &p, int number, size_t bytes) {
    if (!told.empty() && !maker.thread.joinable()) {
      for (const ToldUnit &unit : told) maker.asks.push_back(unit.bytes + slack);
      CK(cudaGetDevice(&maker.device));
      maker.thread = std::thread([this] { maker.run(); });
    }
    for (size_t i = 0; i < told.size(); i++)
      if (told[i].part == &p && told[i].number == number) {
        if (maker.asks[i] != bytes) {
          printf(
              "internal error: weight stream: a unit needs %zu bytes of host memory, but %zu were announced "
              "beforehand\n",
              bytes, maker.asks[i]);
          exit(4);
        }
        return held(maker.take(i));
      }
    return held(pinned_host_memory(bytes));
  }
  // One event per unit, recorded after the unit's copies to the host; start() waits for them.
  std::vector<cudaEvent_t> brought_down;
  // Completes a unit's host copy, so that it has the buffer's bytes: copies from the GPU the ranges
  // that host_twin() has not returned (storage forms generated on the GPU, gaps, the slack). The
  // copies are only queued. So the buffer's next unit must be written on the same `stream`, after
  // them and after the kernels that check this unit.
  void carry_to_the_host(Part &p, int number, cudaStream_t stream) {
    p.room_of(number).need(p.used_now + slack); // a group without experts has an empty part
    CK(cudaMemsetAsync(p.room_of(number).base + p.used_now, 0, slack, stream));
    p.used_now += slack;
    Part::Filling filling = p.filling[number];
    p.filling.erase(number);
    if (filling.host && filling.bytes != p.used_now) {
      printf("internal error: weight stream: a unit grew after its host memory was requested\n");
      exit(4);
    }
    char *host = filling.host ? filling.host : host_memory_of(p, number, p.used_now);
    auto bring = [&](size_t first, size_t end) {
      if (end > first)
        CK(cudaMemcpyAsync(host + first, p.room_of(number).base + first, end - first, cudaMemcpyDeviceToHost, stream));
    };
    std::sort(filling.from_the_file.begin(), filling.from_the_file.end());
    size_t at = 0;
    for (const auto &given : filling.from_the_file) bring(at, given.first), at = given.first + given.second;
    bring(at, p.used_now);
    cudaEvent_t down;
    CK(cudaEventCreateWithFlags(&down, cudaEventDisableTiming));
    CK(cudaEventRecord(down, stream));
    brought_down.push_back(down);
    p.image.push_back(host);
    p.used.push_back(p.used_now);
    p.used_now = 0;
  }
  void ready(Part &p, int g) {
    check_holds(p, g, "asked for");
    if (!count_waits || !turns_run) { // no turns run: the loads limit an opening's passes anyway
      CK(cudaStreamWaitEvent(cudaStreamPerThread, p.loaded[p.room_number(g)], 0));
      return;
    }
    // The GPU waits, in the calling thread's stream; the host does not. The first event completes
    // when the stream's earlier work is done, the second when the load is done too: the time
    // between them is how long this stream was idle for weights.
    if (waits_used == waits.size()) {
      std::pair<cudaEvent_t, cudaEvent_t> two;
      CK(cudaEventCreate(&two.first));
      CK(cudaEventCreate(&two.second));
      waits.push_back(two);
    }
    CK(cudaEventRecord(waits[waits_used].first, cudaStreamPerThread));
    CK(cudaStreamWaitEvent(cudaStreamPerThread, p.loaded[p.room_number(g)], 0));
    CK(cudaEventRecord(waits[waits_used++].second, cudaStreamPerThread));
  }
  // waits: the event pairs of count_waits, reused; waits_used: how many are recorded since the
  // last waited(). feed: the stream of all loads, created by start().
  std::vector<std::pair<cudaEvent_t, cudaEvent_t>> waits;
  size_t waits_used = 0;
  cudaStream_t feed = nullptr;
  int feed_blocks = 4, feed_threads = 256; // `k_load_a_room`'s launch; 0 blocks: copy engine only
  size_t loads_queued = 0; // since the last restart()
  void check_holds(const Part &p, int g, const char *what) const {
    if (p.in_room[p.room_number(g)] == g) return;
    printf(
        "internal error: weight stream: group %d is %s, but its GPU buffer holds group %d (-1 means no group is "
        "loaded, as before start(), after pass(), after room(), or after a walk with no_walk_follows; restart() loads "
        "again; or, within a turn, the buffer is released and its next load is not queued yet)\n",
        g, what, p.in_room[p.room_number(g)]);
    exit(4);
  }
  // Queues each buffer's first load. If for_a_pass, the loads are for pass(), which uses the
  // experts in the order of their numbers.
  // Public source: see no_walk_follows.
  void first_loads(bool for_a_pass = false) {
    if (experts_by_tensor) {
      // Each buffer's first tensor, lowest number first: that is the order of first use, with or
      // without for_a_pass. The second group's other weights are queued after the first group's
      // tensors. A small model may have fewer tensors than buffers: a buffer that no tensor uses
      // has no memory and is never loaded.
      std::vector<int> first; // in ascending order
      std::vector<bool> has_a_tenant(experts.n_rooms, false);
      for (int e = 0; e < (int)experts.image.size(); e++)
        if (!has_a_tenant[experts.room_number(e)]) has_a_tenant[experts.room_number(e)] = true, first.push_back(e);
      // What the end of a pass has already loaded or queued for the next pass is not loaded again.
      // restart() and room() mark as empty what they overwrite or let a caller borrow.
      auto load_unless_there = [this](Part &p, int number) {
        if (p.in_room[p.room_number(number)] != number) load(p, number);
      };
      bool second_rest = rest.image.size() > 1 && rest.n_rooms > 1;
      load_unless_there(rest, 0);
      for (size_t i = 0; i < first.size(); i++) {
        if (second_rest && first[i] >= experts_of(0).second) load_unless_there(rest, 1), second_rest = false;
        load_unless_there(experts, first[i]);
      }
      if (second_rest) load_unless_there(rest, 1);
    }
    for (int g = 0; g < Part::MOST_ROOMS && !experts_by_tensor; g++)
      for (Part *p : parts())
        if (g < p->n_rooms && g < (int)p->image.size())
          load(*p, p == &experts && experts_by_layer && !for_a_pass ? first_in_room(g) : g);
    experts_at = 0, turns_walked = 0, requests_in_wave = 1;
    loads_held.clear();
    no_walk_follows = rooms_will_be_lent = false;
  }
  // Releases unit g's buffer and queues the load of the unit that uses it next: next_sharing() (but
  // see no_walk_follows); the form with three arguments loads `next`.
  void give_back(Part &p, int g) {
    const int next = next_sharing(p, g);
    if (no_walk_follows && next <= g) { // `next` belongs to the next pass: see no_walk_follows
      check_holds(p, g, "released");
      p.in_room[p.room_number(g)] = -1;
      return;
    }
    give_back(p, g, next);
  }
  // The next group after g, cyclically, that shares g's buffer; g itself if there is no other.
  static int next_sharing(const Part &p, int g) {
    int next = g;
    do next = (next + 1) % (int)p.image.size();
    while (p.room_number(next) != p.room_number(g));
    return next;
  }
  // Public source: releasing each buffer right after its last read is the pipeline of several buffers,
  // filled by one side and read by the other, of NVIDIA's CUDA Programming Guide (section 4.10, Pipelines,
  // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html), stated
  // there for stages in shared memory; here the stages are buffers in GPU memory that a copy stream
  // fills from host memory.
  void give_back(Part &p, int g, int next) {
    release(p, g);
    load_what_is_released(p, next);
  }
  // The two halves of give_back(). Between them the buffer counts as empty. Besides earlier loads,
  // the load waits only for work queued earlier on the stream of the thread that calls release().
  // Every read of the buffer, also in a batch pass's earlier turns, must be on that stream or
  // ordered before it.
  void release(Part &p, int g) {
    check_holds(p, g, "released");
    const int r = p.room_number(g);
    CK(cudaEventRecord(p.released[r], cudaStreamPerThread));
    p.in_room[r] = -1;
  }
  void load_what_is_released(Part &p, int next) {
    CK(cudaStreamWaitEvent(feed, p.released[p.room_number(next)], 0));
    load(p, next);
  }
  // Queues the load of unit g into its buffer on the `feed` stream.
  // While turns run (turns_run), a small kernel, k_load_a_room(), does the load and the copy engine
  // (cudaMemcpyAsync()) is left to the turns. In a trace on an H100, copies from host memory to the
  // GPU were served one after another in the order of issue, whichever streams they were on. A load
  // is hundreds of megabytes; a turn issues small copies (a matrix of random numbers, a round's few
  // words) and waits for each. A small copy issued after a queued load waits for all of that load,
  // so the loads' time adds to the turns' time instead of overlapping with it. Prover::fold_walks()
  // (fold_walk.cuh) clears turns_run for the passes of a batch's opening, in which nothing small is
  // copied and the host-to-GPU link limits the speed: then the copy engine loads, which uses the
  // link better than the kernel. No byte of a proof depends on the choice.
  // Observed at full size (Llama 4 Maverick's shape) on one H100 with `-in-step 6` (six requests in
  // a batch pass), a single run of each, not set up as a timing measurement: the batch passes took
  // 90.5 s with the copy engine and 73 to 80 s with the kernel in five launch shapes; the opening's
  // passes took 34.6 s with the copy engine and 38.2 s with the kernel.
  void load(Part &p, int g) {
    if (const size_t bytes = p.used[g]) {
      char *const to = p.room_of(g).base;
      void *from = nullptr;
      if (turns_run && feed_blocks) CK(cudaHostGetDevicePointer(&from, (void *)p.image[g], 0));
      const size_t pieces = from && (uintptr_t)from % 16 == 0 && (uintptr_t)to % 16 == 0 ? bytes / 16 : 0;
      if (pieces) {
        k_load_a_room<<<feed_blocks, feed_threads, 0, feed>>>((uint4 *)to, (const uint4 *)from, pieces);
        CK(cudaGetLastError());
      }
      if (bytes > 16 * pieces) // all of it, or a tail of under 16 bytes
        CK(cudaMemcpyAsync(to + 16 * pieces, (const char *)p.image[g] + 16 * pieces, bytes - 16 * pieces, cudaMemcpyHostToDevice, feed));
    }
    CK(cudaEventRecord(p.loaded[p.room_number(g)], feed));
    loads_queued++;
    p.in_room[p.room_number(g)] = g;
  }

public:
  // For a turn that runs a group's forward computation and no row passes: call it after that
  // computation. It steps through the rest of the turn's uses (experts_ready(), experts_used())
  // without work between them: the second phase's uses or, in a group whose only number has a
  // single use, the end of that use.
  void row_passes_passed_by(int g) {
    if (!experts_by_layer) return;
    auto pass_by = [this](int e) {
      experts_ready(e);
      experts_used(e);
    };
    if (g == 0 && ends_by_turns) pass_by(end_unit[0]); // the embedding comes first: see `start`
    for (int i = 0; i < experts_of(g).second - experts_of(g).first; i++) pass_by(use_of(g, i));
  }
  // For a reader of group g's experts that is not a request: an earlier batch's opening, which
  // reads the weights during a batch pass of one request (Prover::the_rider_folds(), service.cuh).
  // Call it when that reader's last read is queued on the calling thread's stream. It steps through
  // all uses of one turn in the group without work. If *last_turn is true, it counts as the batch
  // pass's last turn in the group: the buffers are released and loaded with the experts that use
  // them next (but see rooms_will_be_lent).
  void turn_passed_by(int g) {
    for (int stretch = 0; stretch < stretches_of(g); stretch++) row_passes_passed_by(g);
  }
  // The option -huge-pages (off by default: prover/prover/arguments.h says why). Set it before the
  // first group's filling begins. Static because pinned_host_memory() is: HostMemoryMaker, which
  // calls that function, has no pointer to the WeightStream.
  static inline bool huge_pages = false;
};
