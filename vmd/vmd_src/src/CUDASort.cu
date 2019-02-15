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
 *      $RCSfile: CUDASort.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $        $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   GPU-accelerated sorting operations 
 ***************************************************************************/

#include <stdlib.h>
#include "CUDASort.h"

#if defined(VMDUSECUB)
#include <cub/cub.cuh>
#else
#include <thrust/sort.h> // need thrust sorting primitives
#include <thrust/device_ptr.h> // need thrust sorting primitives
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

//
// Ascending key-value radix sort
//
template <typename KeyT, typename ValT>
long dev_radix_sort_by_key_tmpsz(KeyT *keys_d, ValT *vals_d, long nitems) {
  size_t tsz = 0;
  cub::DeviceRadixSort::SortPairs(NULL, tsz, 
                                  (const KeyT *) NULL, (KeyT *) NULL,
                                  (const ValT *) NULL, (ValT *) NULL,
                                  nitems, 0, sizeof(KeyT) * 8, 0, false);
  return (long) tsz; 
}


template <typename KeyT, typename ValT>
int dev_radix_sort_by_key(KeyT *keys_d, ValT *vals_d, long nitems,
                          KeyT *keyswork_d, ValT *valswork_d,
                          void *sortwork_d, long tlsz,
                          KeyT min_key, KeyT max_key) {
  int tmp_autoallocate=0;
  int keys_autoallocate=0;
  int vals_autoallocate=0;
  size_t tsz = tlsz;
  
  if (keyswork_d == NULL) {
    keys_autoallocate=1;
//    printf("One-time alloc keyswork size: %ld\n", nitems * sizeof(KeyT));
    cudaMalloc(&keyswork_d, nitems * sizeof(KeyT));
  }

  if (valswork_d == NULL) {
    vals_autoallocate=1;
//    printf("One-time alloc valswork size: %ld\n", nitems * sizeof(ValT));
    cudaMalloc(&valswork_d, nitems * sizeof(ValT));
  }

  if (sortwork_d == NULL) {
    tmp_autoallocate=1;
    cub::DeviceRadixSort::SortPairs(NULL, tsz, 
                                    (const KeyT *) NULL, (KeyT *) NULL,
                                    (const ValT *) NULL, (ValT *) NULL,
                                    nitems, 0, sizeof(KeyT) * 8, 0, false);
//    printf("One-time alloc sort tmp size: %ld\n", tsz);
    cudaMalloc(&sortwork_d, tsz);
  }


  //
  // One of the benefits of the bitwise nature of the stages in 
  // a radix sort is that we can trim the range of bits to only
  // those that are used within the input.  This reduces the number
  // of radix sort stages, thereby improving performance.  Since the
  // CUB implementation works on unsigned integral key types, we can 
  // determine the starting and ending bit positions to sort on if we
  // know the range of the input values.  If we don't know the precise
  // range of input values, we can still benefit from any upper or lower
  // bound value.  Worst case we use all of the bits in the key type, so
  // in that case it would be helpful to use a narrow key type.
  // 

  // default, worst-case scenario is that we must sort on all bit columns
  int begin_bit = 0;
  int end_bit = sizeof(KeyT) * 8;

  // if the caller provided a non-zero max key, we'll compute begin/end
  // bit columns from the LSB and MSB of the provided min/max keys
  if (max_key != ((KeyT) 0)) {
    // compute LSB from min key
    KeyT less = min_key - 1;
    KeyT lsb_val = (less | min_key) ^ less;
    int lsb = int(log2((double) lsb_val));

    int msb = int(log2((double) max_key)+0.5); // compute MSB from max key

    begin_bit = lsb;   // CUB starts sort on LSB bit column
    end_bit = msb + 1; // CUB sorts up to one bit beyond the MSB bit column
  }

  // XXX radix sort doesn't occur in-place, it needs to bounce back and 
  // forth between double-buffered work areas in multiple sweeps
  cub::DeviceRadixSort::SortPairs(NULL, tsz, 
                                  keys_d, keyswork_d, vals_d, valswork_d,
                                  nitems, begin_bit, end_bit, 0, false);

  // copy results from output work area 
  cudaMemcpyAsync(keys_d, keyswork_d, nitems * sizeof(KeyT), cudaMemcpyDeviceToDevice, 0);
  cudaMemcpyAsync(vals_d, valswork_d, nitems * sizeof(ValT), cudaMemcpyDeviceToDevice, 0);

  if (tmp_autoallocate) {
    cudaFree(sortwork_d);
  }

  cudaStreamSynchronize(0);

  if (keys_autoallocate) {
    cudaFree(keyswork_d);
  }

  if (vals_autoallocate) {
    cudaFree(valswork_d);
  }

  return 0;
}


#else

//
// Ascending key-value radix sort
//
template <typename KeyT, typename ValT>
long dev_radix_sort_by_key_tmpsz(KeyT *keys_d, ValT *vals_d, long nitems) {
  return 0;
}


template <typename KeyT, typename ValT>
int dev_radix_sort_by_key(KeyT *keys_d, ValT *vals_d, long nitems,
                          KeyT *keyswork_d, ValT *valswork_d,
                          void *sortwork_d, long tsz,
                          KeyT min_key, KeyT max_key) {
  // Thrust: device pointers are wrapped with vector iterators
  try {
    // It is common to encounter thrust memory allocation issues, so
    // we have to catch thrown exceptions here, otherwise we're guaranteed
    // to eventually have a crash.  If we get a failure, we have to bomb
    // out entirely and fall back to the CPU.

    // XXX thrust is performing allocation/deallocations per-invocation,
    //     which shows up on the SP profile collections as taking a surprising
    //     amount of time.  We need to find a way to provide it with completely
    //     pre-allocated temp workspaces if at all possible.
    //     One Thrust cached allocation scheme (which works for GCC > 4.4)
    //     is described in an example here:
    //       https://github.com/thrust/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu
    thrust::sort_by_key(thrust::device_ptr<unsigned int>(keys_d),
                        thrust::device_ptr<unsigned int>(keys_d + nitems),
                        thrust::device_ptr<unsigned int>(vals_d));
  }
  catch (std::bad_alloc) {
    printf("CUDA Thrust memory allocation failed: %s line %d\n", __FILE__, __LINE__);
    return -1;
  }
  catch (thrust::system::system_error) {
    printf("CUDA Thrust sort_by_key() failed: %s line %d\n", __FILE__, __LINE__);
    return -1;
  }

  return 0;
}

#endif



//
// Force instantiation of required templates...
//
#define INST_DEV_RADIX_SORT_BY_KEY_TMPSZ(KT, VT) template long dev_radix_sort_by_key_tmpsz<KT, VT>(KT*, VT*, long);
#define INST_DEV_RADIX_SORT_BY_KEY(KT, VT) template int dev_radix_sort_by_key<KT, VT>(KT*, VT*, long, KT*, VT*, void *, long, KT, KT);

INST_DEV_RADIX_SORT_BY_KEY_TMPSZ(unsigned int, unsigned int);
INST_DEV_RADIX_SORT_BY_KEY(unsigned int, unsigned int);


