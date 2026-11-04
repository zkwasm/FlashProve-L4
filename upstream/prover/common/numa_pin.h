// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// near_host_pin() pins the calling thread to the CPUs of a GPU's NUMA node, so that page-locked
// (pinned) host buffers that the thread allocates afterwards come from that node by default. Call
// it before the first cudaMallocHost, cudaHostAlloc or cudaHostRegister. Linux only.
// Threads that this thread creates later inherit its affinity mask. Once a thread is pinned or has
// inherited such a mask, a call for a GPU on another node changes nothing.
// Public sources. The idea is public: binding a thread to the CPUs near a GPU: NVIDIA, NVML API
// Reference, "CPU and Memory Affinity", nvmlDeviceSetCpuAffinity()
// (https://docs.nvidia.com/deploy/nvml-api/group__nvmlAffinity.html). This file reads the node and
// its CPUs from sysfs and does not use that library.
// Public source: local allocation, from the allocating CPU's node first: the Linux kernel's
// documentation, "NUMA Memory Policy", "System Default Policy"
// (https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <sched.h>
#include <cuda_runtime.h>
// Pins only to CPUs already in the thread's mask. Returns how many, or 0, without a message, if it
// did not pin; a failure of sched_getaffinity() is one such case, and so is a cpulist that cannot
// be read to its end (a letter in it, for example). The text of cpulist is not checked otherwise.
static inline int near_host_pin(int device = 0) {
  char id[64];
  if (cudaDeviceGetPCIBusId(id, 64, device) != cudaSuccess) return 0;
  for (char *c = id; *c; c++) *c = (char)tolower(*c);
  char path[256];
  snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/numa_node", id);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  int node = -1;
  if (fscanf(f, "%d", &node) != 1) node = -1;
  fclose(f);
  if (node < 0) return 0;
  snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
  f = fopen(path, "r");
  if (!f) return 0;
  char buf[4096];
  if (!fgets(buf, sizeof buf, f)) {
    fclose(f);
    return 0;
  }
  fclose(f);
  cpu_set_t allowed, s;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof allowed, &allowed)) return 0;
  CPU_ZERO(&s);
  int n = 0;
  char *p = buf;
  while (*p && *p != '\n') {
    char *was = p;
    int a = (int)strtol(p, &p, 10), b = a;
    if (*p == '-') {
      p++;
      b = (int)strtol(p, &p, 10);
    }
    for (int i = a; i <= b; i++)
      if (i < CPU_SETSIZE && CPU_ISSET(i, &allowed)) {
        CPU_SET(i, &s);
        n++;
      }
    if (*p == ',') p++;
    if (p == was) return 0;
  }
  if (!n || sched_setaffinity(0, sizeof s, &s)) return 0;
  printf("# GPU %s is on NUMA node %d; this thread is pinned to %d of its CPUs\n", id, node, n);
  return n;
}
