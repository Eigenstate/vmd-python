
// memory allocator for both normal and pinned (page-locked) memory allocations
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "dmabufalloc.h"

void * DMABufAlloc(size_t sz, int pagelocked) {
  if (!pagelocked) {
//printf("normal alloc: %d\n", sz);
    return malloc(sz);
  } else {
    void *mem = NULL;
    int rc;
//printf("pinned alloc: %d\n", sz);
    rc = cuMemAllocHost(&mem, sz);
    if (rc != CUDA_SUCCESS) {
//printf("pinned allocation failed!  rc=%d\n", rc);
      return NULL;
    }
    return mem;
  }
}

void DMABufFree(void *mem, int pagelocked) {
  if (!pagelocked) {
    free(mem);
  } else {
    cuMemFreeHost(mem);
  }
}
 
