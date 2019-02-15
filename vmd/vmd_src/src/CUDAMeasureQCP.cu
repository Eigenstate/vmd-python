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
 *      $RCSfile: CUDAMeasureQCP.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.7 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated QCP calculation
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "Inform.h"
#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 
#include "Measure.h"

typedef struct {
  int selected;
  int first;
  int last;
  int step;
  float *rmsdmat;
  int padcnt;
  int framecrdsz;
  float *crds;
} gpuqcprmsdthreadparms;

// #if __CUDA_ARCH__ < 300
// #error The CUDA QCP RMSD kernels only support Kepler and later GPUs
// #endif

//
// Device global variable for block count used in single-pass
// device-wide parallel reductions.
//
__device__ unsigned int glob_block_count = 0;


//
// Warp-wide sum reduction
//
template <typename T>
__inline__ __device__ T warp_sum_reduction(T v) {
  for (int offset = warpSize/2; offset > 0; offset >>=1 ) {
#if CUDART_VERSION >= 9000
    v += __shfl_down_sync(0xffffffff, v, offset);
#else
    v += __shfl_down(v, offset); 
#endif
  }
  return v;
}


//
// Block-wide sum reduction
// XXX would break with blockDim > 1024 (more than 32 warps) 
//
template <typename T>
__inline__ __device__ T block_sum_reduction(T v) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  v = warp_sum_reduction(v);

  __syncthreads(); // needed when called multiple times in a row

  if (lane == 0)
    shared[wid] = v;

  __syncthreads();

  if (threadIdx.x < blockDim.x / warpSize) {
    v = shared[lane];
  } else {
    v = 0;
  }

  if (wid ==0)
    v = warp_sum_reduction(v);

  return v;
}


//
// Device-wide QCP inner product kernel 
//   Notes:  This kernel is designed to compute a QCP inner product for
//   a single pairwise RMSD between two "large" structures, where "large"
//   means that we have sufficient parallelism (due to the total number of
//   atoms) to completely saturate GPU with work.  
//   Since the algorithm does linear work and only a relatively small number 
//   of FLOPS per atomic coordinate pair, this "large structure" case 
//   is inherently memory bandwidth bound, at least in isolation.  There may
//   an L2 cache performance benefit associated with scheduling back-to-back 
//   trajectory frame pair RMSD calculations where one of the frames is held
//   constant while looping over several others, but we could do much
//   better by writing a special-case kernel that would incorporate looping
//   internally within each thread block, thereby keeping one of the two 
//   coordinate sets entirely resident in machine registers, so long as the
//   problem was small enough to fit within the maximum CUDA 1-D grid size.
//
__global__ static void 
vmd_qcp_innerprod_soa_devicewide(double *pr,
                                 float *crdx1, float *crdy1, float *crdz1,
                                 float *crdx2, float *crdy2, float *crdz2,
                                 const int natoms, const float *weight) {
  unsigned int tid  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int tcnt = gridDim.x * blockDim.x;

  __shared__ int isLastBlock[1];
  if(threadIdx.x == 0) {
    isLastBlock[0] = 0;
  }
  __syncthreads();

  double G, a0, a1, a2, a3, a4, a5, a6, a7, a8;
  G=a0=a1=a2=a3=a4=a5=a6=a7=a8=0.0;

  int start = tid; 
  int stop  = natoms;
  if (weight != NULL) {
    for (int i=start; i<stop; i+=tcnt) {
      float fw = weight[i];
      float fx1 = crdx1[i];
      float fy1 = crdy1[i];
      float fz1 = crdz1[i];

      float fx2 = crdx2[i];
      float fy2 = crdy2[i];
      float fz2 = crdz2[i];

      G += fw * ((fx1*fx1 + fy1*fy1 + fz1*fz1) + (fx2*fx2 + fy2*fy2 + fz2*fz2));

      a0 += fx1 * fx2;
      a1 += fx1 * fy2;
      a2 += fx1 * fz2;

      a3 += fy1 * fx2;
      a4 += fy1 * fy2;
      a5 += fy1 * fz2;

      a6 += fz1 * fx2;
      a7 += fz1 * fy2;
      a8 += fz1 * fz2;
    }
  } else {
    for (int i=start; i<stop; i+=tcnt) {
      float fx1 = crdx1[i];
      float fy1 = crdy1[i];
      float fz1 = crdz1[i];

      float fx2 = crdx2[i];
      float fy2 = crdy2[i];
      float fz2 = crdz2[i];

      G += ((fx1*fx1 + fy1*fy1 + fz1*fz1) + (fx2*fx2 + fy2*fy2 + fz2*fz2));

      a0 += fx1 * fx2;
      a1 += fx1 * fy2;
      a2 += fx1 * fz2;

      a3 += fy1 * fx2;
      a4 += fy1 * fy2;
      a5 += fy1 * fz2;

      a6 += fz1 * fx2;
      a7 += fz1 * fy2;
      a8 += fz1 * fz2;
    }
  }

  __syncthreads();

  G *= 0.5; 

  // block-wide sum reduction of inner product
  double bG  = block_sum_reduction(G);
  double ba0 = block_sum_reduction(a0);
  double ba1 = block_sum_reduction(a1);
  double ba2 = block_sum_reduction(a2);
  double ba3 = block_sum_reduction(a3);
  double ba4 = block_sum_reduction(a4);
  double ba5 = block_sum_reduction(a5);
  double ba6 = block_sum_reduction(a6);
  double ba7 = block_sum_reduction(a7);
  double ba8 = block_sum_reduction(a8);

  __syncthreads();

  // thread 0 in each block writes out per-block partial sums
  if (threadIdx.x == 0) {
    pr[(0*gridDim.x)+blockIdx.x] = ba0;
    pr[(1*gridDim.x)+blockIdx.x] = ba1;
    pr[(2*gridDim.x)+blockIdx.x] = ba2;
    pr[(3*gridDim.x)+blockIdx.x] = ba3;
    pr[(4*gridDim.x)+blockIdx.x] = ba4;
    pr[(5*gridDim.x)+blockIdx.x] = ba5;
    pr[(6*gridDim.x)+blockIdx.x] = ba6;
    pr[(7*gridDim.x)+blockIdx.x] = ba7;
    pr[(8*gridDim.x)+blockIdx.x] = ba8;
    pr[(9*gridDim.x)+blockIdx.x] = bG;

    __threadfence(); // ensure all prior memory writes post before continuing

    // increment atomic counter of number of blocks that have finished
    // their work, so that the last block to finish knows to do the final
    // parallel reduction of all of the individual per-block partial sums
    unsigned int old_block_count = atomicInc(&glob_block_count, gridDim.x);
    isLastBlock[0] = ( old_block_count == (gridDim.x -1) );
  }

  __syncthreads();

  // the last block performs the final reduction of all of the individual
  // per-block partial sums
  if (isLastBlock[0]) {
    glob_block_count = 0;
    __threadfence(); // ensure block_count memory write posts before continuing

    // each thread loops over all 10 items doing the individual reductions
    for (int l=0; l < 10; ++l) {
      float sum = 0; 
      double *pr_a = pr+(l*gridDim.x);

      // each thread reduces all 10 items
      for (int j = threadIdx.x; j < gridDim.x; j += blockDim.x) {
        sum += pr_a[j];
      }

      sum = block_sum_reduction(sum);
      if (threadIdx.x==0)
        pr_a[0]=sum;
    }
  }
}


//
// Single-thread-block-per-structure-pair QCP inner product kernel 
//   Notes:  This kernel is designed to compute QCP inner products for
//   many pairwise RMSDs between "small" structures, where "small"
//   means that we have sufficient parallelism (due to the total number of
//   atoms) to supply a single CUDA thread block with sufficient work for
//   roughly 256 threads per thread block.
//   Since the algorithm does linear work and only a relatively small number 
//   of FLOPS per atomic coordinate pair, the "small structure" case 
//   is memory bandwidth bound in the worst case, but L1/L2 cache bandwidth
//   bound in the best cases.  
//   There is an opportunity to write a special case variant of this kernel
//   that stores the atomic coordinates for one of the structures in machine
//   registers such that they could be repeatedly reused, but this creates
//   some additional challenges in performing parallel sum reductions.
//
__global__ static void 
vmd_qcp_innerprod_soa_blockperpair(double *results,
                                   float *crdx1, float *crdy1, float *crdz1,
                                   float *crdx2, float *crdy2, float *crdz2,
                                   const int natoms, const float *weight,
                                   const int num_structs) {
  unsigned int tid  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int tcnt = gridDim.x * blockDim.x;

  double G, a0, a1, a2, a3, a4, a5, a6, a7, a8;

  // the grid of thread blocks loop over all of the structures
  for (int struct_id = blockIdx.x; struct_id < num_structs; struct_id+=gridDim.x) {
    G=a0=a1=a2=a3=a4=a5=a6=a7=a8=0.0;
    int struct_offset = struct_id * natoms * sizeof(float);
    int start = struct_offset + tid;
    int stop  = struct_offset + natoms;
    if (weight != NULL) {
      for (int i=start; i<stop; i+=tcnt) {
        float fw = weight[i];
        float fx1 = crdx1[i];
        float fy1 = crdy1[i];
        float fz1 = crdz1[i];

        float fx2 = crdx2[i];
        float fy2 = crdy2[i];
        float fz2 = crdz2[i];

        G += fw * ((fx1*fx1 + fy1*fy1 + fz1*fz1) + (fx2*fx2 + fy2*fy2 + fz2*fz2));

        a0 += fx1 * fx2;
        a1 += fx1 * fy2;
        a2 += fx1 * fz2;

        a3 += fy1 * fx2;
        a4 += fy1 * fy2;
        a5 += fy1 * fz2;

        a6 += fz1 * fx2;
        a7 += fz1 * fy2;
        a8 += fz1 * fz2;
      }
    } else {
      for (int i=start; i<stop; i+=tcnt) {
        float fx1 = crdx1[i];
        float fy1 = crdy1[i];
        float fz1 = crdz1[i];

        float fx2 = crdx2[i];
        float fy2 = crdy2[i];
        float fz2 = crdz2[i];

        G += ((fx1*fx1 + fy1*fy1 + fz1*fz1) + (fx2*fx2 + fy2*fy2 + fz2*fz2));

        a0 += fx1 * fx2;
        a1 += fx1 * fy2;
        a2 += fx1 * fz2;

        a3 += fy1 * fx2;
        a4 += fy1 * fy2;
        a5 += fy1 * fz2;

        a6 += fz1 * fx2;
        a7 += fz1 * fy2;
        a8 += fz1 * fz2;
      }
    }

    __syncthreads();

    G *= 0.5; 

    // block-wide sum reduction of inner product
    double bG  = block_sum_reduction(G);
    double ba0 = block_sum_reduction(a0);
    double ba1 = block_sum_reduction(a1);
    double ba2 = block_sum_reduction(a2);
    double ba3 = block_sum_reduction(a3);
    double ba4 = block_sum_reduction(a4);
    double ba5 = block_sum_reduction(a5);
    double ba6 = block_sum_reduction(a6);
    double ba7 = block_sum_reduction(a7);
    double ba8 = block_sum_reduction(a8);

    __syncthreads();

    // thread 0 in each block writes out per-block partial sums
    if (threadIdx.x == 0) {
      int addr = struct_id * 10;
      results[addr    ] = ba0;
      results[addr + 1] = ba1;
      results[addr + 2] = ba2;
      results[addr + 3] = ba3;
      results[addr + 4] = ba4;
      results[addr + 5] = ba5;
      results[addr + 6] = ba6;
      results[addr + 7] = ba7;
      results[addr + 8] = ba8;
      results[addr + 9] = bG;
    }
  }
}


// compute lower-triangular indices i,j from linear array index
static int idx2sub_tril(long N, long ind, long *J, long *I) {
  if (ind > (N*(N+1)/2)) {
    return -1; // out of bounds
  }

  long i, j;
  // j = ceil(0.5*(2*N+1 - sqrt(-8*ind + 1 + 4*N*(N+1))));
  // i = ind - (j-1)*N + j*(j-1)/2;

  // i = floor((2*N+1 - sqrt((2*N+1)*(2*N+1) - 8*ind)) / 2);
  // XXX deal with ambiguous types for sqrt() on Solaris
  double tmp2np1 = 2*N+1;
  i = floor((tmp2np1 - sqrt(tmp2np1*tmp2np1 - 8.0*ind)) / 2);
  j = ind - N*i + i*(i-1)/2 + i;

  *I = i;
  *J = j;

  return 0;
}


static void * measure_rmsdmat_qcp_thread(void *voidparms) {
  int threadid;
  gpuqcprmsdthreadparms *parms = NULL;
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);
  wkf_threadpool_worker_getid(voidparms, &threadid, NULL);

  //
  // copy in per-thread parameters
  //
  float *rmsdmat = parms->rmsdmat;

#if 0
  int framecrdsz = parms->framecrdsz;
  float *crds = parms->crds;
#endif
  int selected = parms->selected;
  int first  = parms->first;
  int last   = parms->last;
  int step   = parms->step;

#if 1
  printf("QCP GPU[%d] running...\n", threadid);
#endif

  int framecount = (last - first + 1) / step;

  wkf_tasktile_t tile;
  while (wkf_threadlaunch_next_tile(voidparms, 8, &tile) != WKF_SCHED_DONE) {
    long idx;

    for (idx=tile.start; idx<tile.end; idx++) {
      long i, j;

      // compute i,j from idx...
      // only compute off-diagonal elements, so we use (framecount-1)
      if (idx2sub_tril(framecount-1, idx, &i, &j)) {
        printf("qcpthread[%d]: work idx %ld out of triangle!\n", threadid, idx);
        break;
      }

      // calculate the (weighted) inner product of two structures
      double A[9];
      double E0 = 0;

#if 0
      float *xj = crds + (j * 3 * framecrdsz);
      float *yj = xj + framecrdsz;
      float *zj = xj + framecrdsz*2;

      float *xi = crds + (i * 3 * framecrdsz);
      float *yi = xi + framecrdsz;
      float *zi = xi + framecrdsz*2;

      E0 = InnerProductSOA(A, xj, yj, zj, xi, yi, zi,
                           selected, NULL /* weight */);
#endif

      // calculate the RMSD & rotational matrix
      FastCalcRMSDAndRotation(NULL, A, &rmsdmat[j*framecount + i],
                              E0, selected, -1);
    }
  }

  return NULL;
}


int qcp_soa_gpu(wkf_threadpool_t *devpool, // VMD GPU worker thread pool
                int selected, float *crds, int framecrdsz, int padcnt,
                int first, int last, int step, int framecount,
                float *rmsdmat) {
  //
  // copy in per-thread parameters
  //
  gpuqcprmsdthreadparms parms;
  memset(&parms, 0, sizeof(parms));

  parms.selected = selected;
  parms.rmsdmat = rmsdmat;
  parms.padcnt = padcnt;
  parms.framecrdsz = framecrdsz;
  parms.crds = crds;
  parms.first = first;
  parms.last = last;
  parms.step = step;

  // spawn child threads to do the work
  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=(framecount-1)*(framecount-1)/2; // only compute off-diag elements

#if 1
  wkf_threadpool_sched_dynamic(devpool, &tile);
  wkf_threadpool_launch(devpool, measure_rmsdmat_qcp_thread, &parms, 1);
#else
  int i, j;
  for (j=0; j<framecount; j++) {
    float *xj = crds + (j * 3 * framecrdsz);
    float *yj = xj + framecrdsz;
    float *zj = xj + framecrdsz*2;
    for (i=0; i<j; i++) {
      // calculate the (weighted) inner product of two structures
      double A[9];

      float *xi = crds + (i * 3 * framecrdsz);
      float *yi = xi + framecrdsz;
      float *zi = xi + framecrdsz*2;

      double E0 = InnerProductSOA(A, xj, yj, zj, xi, yi, zi,
                                  sel->selected, NULL /* weight */);

      // calculate the RMSD & rotational matrix
      FastCalcRMSDAndRotation(NULL, A, &rmsdmat[j*framecount + i],
                              E0, sel->selected, -1);
    }
  }
#endif

  return 0;
}  
