/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAParPrefixOps.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   GPU-accelerated parallel prefix operations (sum, min, max etc)
 ***************************************************************************/

#include <stdlib.h>
#include "CUDAParPrefixOps.h"

#if defined(VMDUSECUB)
#include <cub/cub.cuh>
#else
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#endif


#if 0
#define CUERR { cudaError_t err; \
  cudaDeviceSynchronize(); \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  }}
#else
#define CUERR
#endif

#if defined(VMDUSECUB)

// VMDScanSum functor
struct VMDScanSum {
  template <typename T>
  //  CUB_RUNTIME_FUNCTION __forceinline__
  __device__ __forceinline__
  T operator()(const T &a, const T &b) const {
    return a + b;
  }
};

// 
// Exclusive scan
//
template <typename T>
long dev_excl_scan_sum_tmpsz(T *in_d, long nitems, T *out_d, T ival) {
  size_t tsz = 0;
  VMDScanSum sum_op;
  cub::DeviceScan::ExclusiveScan((T*) NULL, tsz, (T*) NULL, (T*) NULL, 
                                 sum_op, ival, nitems);
  return (long) tsz;
}

template <typename T>
void dev_excl_scan_sum(T *in_d, long nitems, T *out_d,
                       void *scanwork_d, long tlsz, T ival) {
  VMDScanSum sum_op;
  int autoallocate=0;
  size_t tsz = tlsz;

  if (scanwork_d == NULL) {
    autoallocate=1;
    cub::DeviceScan::ExclusiveScan((T*) NULL, tsz, (T*) NULL, (T*) NULL, 
                                   sum_op, ival, nitems);
//    printf("One-time alloc scan tmp size: %ld\n", tsz);
    cudaMalloc(&scanwork_d, tsz); 
  }

  cub::DeviceScan::ExclusiveScan(scanwork_d, tsz, in_d, out_d, 
                                 sum_op, ival, nitems);
  if (autoallocate)
    cudaFree(scanwork_d);
}


// 
// Inclusive scan
//
template <typename T>
long dev_incl_scan_sum_tmpsz(T *in_d, long nitems, T *out_d) {
  size_t tsz = 0;
  VMDScanSum sum_op;
  cub::DeviceScan::InclusiveScan((T*) NULL, tsz, (T*) NULL, (T*) NULL, 
                                 sum_op, nitems);
  return (long) tsz;
}


template <typename T>
void dev_incl_scan_sum(T *in_d, long nitems, T *out_d,
                       void *scanwork_d, long tlsz) {
  VMDScanSum sum_op;
  int autoallocate=0;
  size_t tsz = tlsz;

  if (scanwork_d == NULL) {
    autoallocate=1;
    cub::DeviceScan::InclusiveScan((T*) NULL, tsz, (T*) NULL, (T*) NULL, 
                                   sum_op, nitems);
//    printf("One-time alloc scan tmp size: %ld\n", tsz);
    cudaMalloc(&scanwork_d, tsz); 
  }

  cub::DeviceScan::InclusiveScan(scanwork_d, tsz, in_d, out_d, 
                                 sum_op, nitems);
  if (autoallocate)
    cudaFree(scanwork_d);
}

#else


// 
// Exclusive scan
//
template <typename T>
long dev_excl_scan_sum_tmpsz(T *in_d, long nGroups, T *out_d, T ival) {
  return 0;
}


template <typename T>
void dev_excl_scan_sum(T *in_d, long nGroups, T *out_d,
                       void *scanwork_d, long tsz, T ival) {
  // Prefix scan
  // XXX thrust is performing allocation/deallocations per-invocation,
  //     which shows up on the SP profile collections as taking a surprising
  //     amount of time.  We need to find a way to provide it with completely
  //     pre-allocated temp workspaces if at all possible.
  //     One Thrust cached allocation scheme (which works for GCC > 4.4) 
  //     is described in an example here:
  //       https://github.com/thrust/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu
  thrust::exclusive_scan(thrust::cuda::par, in_d, in_d + nGroups, out_d);
}


// 
// Inclusive scan
//
template <typename T>
long dev_incl_scan_sum_tmpsz(T *in_d, long nGroups, T *out_d) {
  return 0;
}

template <typename T>
void dev_incl_scan_sum(T *in_d, long nGroups, T *out_d,
                       void *scanwork_d, long tsz) {
  // Prefix scan
  // XXX thrust is performing allocation/deallocations per-invocation,
  //     which shows up on the SP profile collections as taking a surprising
  //     amount of time.  We need to find a way to provide it with completely
  //     pre-allocated temp workspaces if at all possible.
  //     One Thrust cached allocation scheme (which works for GCC > 4.4) 
  //     is described in an example here:
  //       https://github.com/thrust/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu
  thrust::inclusive_scan(thrust::cuda::par, in_d, in_d + nGroups, out_d);
}


#endif



//
// Force instantiation of required templates...
//
#define INST_DEV_EXCL_SCAN_SUM_TMPSZ(T) template long dev_excl_scan_sum_tmpsz<T>(T*, long, T*, T);
#define INST_DEV_EXCL_SCAN_SUM(T) template void dev_excl_scan_sum<T>(T*, long, T*, void*, long, T);

INST_DEV_EXCL_SCAN_SUM_TMPSZ(long)
INST_DEV_EXCL_SCAN_SUM_TMPSZ(int)
INST_DEV_EXCL_SCAN_SUM_TMPSZ(short)
INST_DEV_EXCL_SCAN_SUM_TMPSZ(unsigned long)
INST_DEV_EXCL_SCAN_SUM_TMPSZ(unsigned int)
INST_DEV_EXCL_SCAN_SUM_TMPSZ(unsigned short)

INST_DEV_EXCL_SCAN_SUM(long)
INST_DEV_EXCL_SCAN_SUM(int)
INST_DEV_EXCL_SCAN_SUM(short)
INST_DEV_EXCL_SCAN_SUM(unsigned long)
INST_DEV_EXCL_SCAN_SUM(unsigned int)
INST_DEV_EXCL_SCAN_SUM(unsigned short)


#if 0
inline __host__ __device__ uint2 operator+(uint2 a, uint2 b) {
  return make_uint2(a.x + b.x, a.y + b.y);
}

template long dev_excl_scan_sum_tmpsz<uint2>(uint2*, long, uint2*, uint2);
template void dev_excl_scan_sum<uint2>(uint2*, long, uint2*, void*, long, uint2);
#endif

template void dev_incl_scan_sum<float>(float*, long, float*, void*, long);


