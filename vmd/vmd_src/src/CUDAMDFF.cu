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
 *      $RCSfile: CUDAMDFF.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.79 $      $Date: 2019/01/17 21:38:54 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated MDFF functions
 *
 * "GPU-Accelerated Analysis and Visualization of Large Structures 
 *  Solved by Molecular Dynamics Flexible Fitting"
 *  John E. Stone, Ryan McGreevy, Barry Isralewitz, and Klaus Schulten.
 *  Faraday Discussions, 169:265-283, 2014.
 *  Online full text available at http://dx.doi.org/10.1039/C4FD00005F
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#if CUDART_VERSION >= 9000
#include <cuda_fp16.h> // need to explicitly include for CUDA 9.0
#endif
#if CUDART_VERSION < 4000
#error The VMD MDFF feature requires CUDA 4.0 or later
#endif

#include <float.h> // FLT_MAX etc


#include "Inform.h"
#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 
#include "CUDASpatialSearch.h"

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "VolumetricData.h"
#include "VolMapCreate.h" // volmap_write_dx_file()

#include "CUDAMDFF.h"

#include <tcl.h>
#include "TclCommands.h"

#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#else
#define CUERR
#endif



static int check_gpu_compute20(cudaError_t &err) {
  cudaDeviceProp deviceProp;
  int dev;
  if (cudaGetDevice(&dev) != cudaSuccess)
    return -1;
  
  memset(&deviceProp, 0, sizeof(cudaDeviceProp));
  
  if (cudaGetDeviceProperties(&deviceProp, dev) != cudaSuccess) {
    err = cudaGetLastError(); // eat error so next CUDA op succeeds
    return -1;
  }

  // this code currently requires compute capability 2.0
  if (deviceProp.major < 2)
    return -1;

  return 0;
}



//
// Various math operators for vector types not already
// provided by the regular CUDA headers
//
// "+" operator
inline __host__ __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

// "+=" operator
inline __host__ __device__ void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __host__ __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// "+=" operator
inline __host__ __device__ void operator+=(float4 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}



//
// density format conversion routines
//

// no-op conversion for float to float
inline __device__ void convert_density(float & df, float df2) {
  df = df2;
}

// Convert float (32-bit) to half-precision (16-bit floating point) stored
// into an unsigned short (16-bit integer type).
inline __device__ void convert_density(unsigned short & dh, float df2) {
  dh = __float2half_rn(df2);
}


//
// Restrict macro to make it easy to do perf tuning tests
//
#if 0
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif


// 
// Parameters for linear-time range-limited gaussian density kernels
//
#define GGRIDSZ   8.0f
#define GBLOCKSZX 8
#define GBLOCKSZY 8

#define GBLOCKSZZ    2
#define GUNROLL      4

#define TOTALBLOCKSZ (GBLOCKSZX * GBLOCKSZY * GBLOCKSZZ)


//
// All of the current kernel versions require support for atomic ops,
// so they only work with compute capability 2.x or greater...
//

//
// global atomic counter
//
__device__ unsigned int tbcatomic[3] = {0, 0, 0};

__device__ void reset_atomic_counter(unsigned int *counter) {
  counter[0] = 0;
  __threadfence();
}


//
// completely inlined device function to calculate the min/max
// bounds of the acceleration grid region that must be traversed to
// satisfy the current potential grid point
//
inline __device__ void calc_ac_cell_ids(int3 &abmin, int3 &abmax,
                                        int3 acncells,
                                        float3 acoriginoffset,
                                        float gridspacing, 
                                        float acgridspacing,
                                        float invacgridspacing) {
  // compute ac grid index of lower corner minus gaussian radius
  abmin.x = (acoriginoffset.x + (blockIdx.x * blockDim.x) * gridspacing - acgridspacing) * invacgridspacing;
  abmin.y = (acoriginoffset.y + (blockIdx.y * blockDim.y) * gridspacing - acgridspacing) * invacgridspacing;
  abmin.z = (acoriginoffset.z + (blockIdx.z * blockDim.z * GUNROLL) * gridspacing - acgridspacing) * invacgridspacing;

  // compute ac grid index of upper corner plus gaussian radius
  abmax.x = (acoriginoffset.x + ((blockIdx.x+1) * blockDim.x) * gridspacing + acgridspacing) * invacgridspacing;
  abmax.y = (acoriginoffset.y + ((blockIdx.y+1) * blockDim.y) * gridspacing + acgridspacing) * invacgridspacing;
  abmax.z = (acoriginoffset.z + ((blockIdx.z+1) * blockDim.z * GUNROLL) * gridspacing + acgridspacing) * invacgridspacing;

  abmin.x = (abmin.x < 0) ? 0 : abmin.x;
  abmin.y = (abmin.y < 0) ? 0 : abmin.y;
  abmin.z = (abmin.z < 0) ? 0 : abmin.z;
  abmax.x = (abmax.x >= acncells.x-1) ? acncells.x-1 : abmax.x;
  abmax.y = (abmax.y >= acncells.y-1) ? acncells.y-1 : abmax.y;
  abmax.z = (abmax.z >= acncells.z-1) ? acncells.z-1 : abmax.z;
}



inline __device__ void calc_densities(int xindex, int yindex, int zindex,
                                      const float4 *sorted_xyzr, 
                                      int3 abmin,
                                      int3 abmax,
                                      int3 acncells,
                                      float3 acoriginoffset,
                                      const uint2 * cellStartEnd,
                                      float gridspacing, 
                                      float &densityval1
#if GUNROLL >= 2
                                      ,float &densityval2
#endif
#if GUNROLL >= 4
                                      ,float &densityval3
                                      ,float &densityval4
#endif
                                      ) {
  float coorx = acoriginoffset.x + gridspacing * xindex;
  float coory = acoriginoffset.y + gridspacing * yindex;
  float coorz = acoriginoffset.z + gridspacing * zindex;

  int acplanesz = acncells.x * acncells.y;
  int xab, yab, zab;
  for (zab=abmin.z; zab<=abmax.z; zab++) {
    for (yab=abmin.y; yab<=abmax.y; yab++) {
      for (xab=abmin.x; xab<=abmax.x; xab++) {
        int abcellidx = zab * acplanesz + yab * acncells.x + xab;
        uint2 atomstartend = cellStartEnd[abcellidx];
        if (atomstartend.x != GRID_CELL_EMPTY) {
          unsigned int atomid;
          for (atomid=atomstartend.x; atomid<atomstartend.y; atomid++) {
            float4 atom = sorted_xyzr[atomid];
            float dx = coorx - atom.x;
            float dy = coory - atom.y;
            float dxy2 = dx*dx + dy*dy;
  
            float dz = coorz - atom.z;
            float r21 = (dxy2 + dz*dz) * atom.w;
            densityval1 += __expf(r21);

#if GUNROLL >= 2
            float dz2 = dz + gridspacing;
            float r22 = (dxy2 + dz2*dz2) * atom.w;
            densityval2 += __expf(r22);
#endif
#if GUNROLL >= 4
            float dz3 = dz2 + gridspacing;
            float r23 = (dxy2 + dz3*dz3) * atom.w;
            densityval3 += __expf(r23);

            float dz4 = dz3 + gridspacing;
            float r24 = (dxy2 + dz4*dz4) * atom.w;
            densityval4 += __expf(r24);
#endif
          }
        }
      }
    }
  }
}



#if GUNROLL == 1
#define MAINDENSITYLOOP \
  float densityval1=0.0f; \
  calc_densities(xindex, yindex, zindex, sorted_xyzr, abmin, abmax, acncells, \
                 acoriginoffset, cellStartEnd, gridspacing, densityval1);
#elif GUNROLL == 2
#define MAINDENSITYLOOP \
  float densityval1=0.0f; \
  float densityval2=0.0f; \
  calc_densities(xindex, yindex, zindex, sorted_xyzr, abmin, abmax, acncells, \
                 acoriginoffset, cellStartEnd, gridspacing,                   \
                 densityval1, densityval2);
#elif GUNROLL == 4
#define MAINDENSITYLOOP \
  float densityval1=0.0f; \
  float densityval2=0.0f; \
  float densityval3=0.0f; \
  float densityval4=0.0f; \
  calc_densities(xindex, yindex, zindex, sorted_xyzr, abmin, abmax, acncells, \
                 acoriginoffset, cellStartEnd, gridspacing,                   \
                 densityval1, densityval2, densityval3, densityval4);
#endif


//
// compute density map
//
__global__ static void gaussdensity_fast(int natoms,
                                         const float4 *sorted_xyzr, 
                                         int3 volsz,
                                         int3 acncells,
                                         float3 acoriginoffset,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float gridspacing, unsigned int z, 
                                         float *densitygrid) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GUNROLL;
  unsigned int outaddr = zindex * volsz.x * volsz.y + 
                         yindex * volsz.x + 
                         xindex;
  zindex += z;

  // early exit if this thread is outside of the grid bounds
  if (xindex >= volsz.x || yindex >= volsz.y || zindex >= volsz.z)
    return;

  // compute ac grid index of lower corner minus gaussian radius
  int3 abmin, abmax;
  calc_ac_cell_ids(abmin, abmax, acncells, acoriginoffset,
                   gridspacing, acgridspacing, invacgridspacing);

  // density loop macro
  MAINDENSITYLOOP

  densitygrid[outaddr            ] = densityval1;
#if GUNROLL >= 2
  int planesz = volsz.x * volsz.y;
  if (++zindex < volsz.z) // map isn't always an even multiple of block size
    densitygrid[outaddr +   planesz] = densityval2;
#endif
#if GUNROLL >= 4
  if (++zindex < volsz.z) // map isn't always an even multiple of block size
    densitygrid[outaddr + 2*planesz] = densityval3;
  if (++zindex < volsz.z) // map isn't always an even multiple of block size
    densitygrid[outaddr + 3*planesz] = densityval4;
#endif
}



//
// compute density differences
//
__global__ static void gaussdensity_diff(int natoms,
                                         const float4 *sorted_xyzr, 
                                         int3 volsz,
                                         int3 acncells,
                                         float3 acoriginoffset,
                                         float acgridspacing,
                                         float invacgridspacing,
                                         const uint2 * cellStartEnd,
                                         float gridspacing, unsigned int z, 
                                         int3 refvolsz,
                                         int3 refoffset,
                                         const float *refmap,
                                         float *diffmap) {
  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GUNROLL;
  unsigned int refaddr = (zindex+refoffset.z) * refvolsz.x * refvolsz.y + 
                         (yindex+refoffset.y) * refvolsz.x + 
                         (xindex+refoffset.x);
  unsigned int outaddr = zindex * volsz.x * volsz.y + 
                         yindex * volsz.x + 
                         xindex;
  zindex += z;

  // early exit if this thread is outside of the grid bounds
  if (xindex >= volsz.x || yindex >= volsz.y || zindex >= volsz.z)
    return;

  // compute ac grid index of lower corner minus gaussian radius
  int3 abmin, abmax;
  calc_ac_cell_ids(abmin, abmax, acncells, acoriginoffset,
                   gridspacing, acgridspacing, invacgridspacing);

  // density loop macro
  MAINDENSITYLOOP

  diffmap[outaddr            ] = refmap[refaddr               ] - densityval1;
#if GUNROLL >= 2
  int planesz = volsz.x * volsz.y;
  int refplanesz = refvolsz.x * refvolsz.y;
  if (++zindex < volsz.z) // map isn't always an even multiple of block size
    diffmap[outaddr +   planesz] = refmap[refaddr +   refplanesz] - densityval2;
#endif
#if GUNROLL >= 4
  if (++zindex < volsz.z) // map isn't always an even multiple of block size
    diffmap[outaddr + 2*planesz] = refmap[refaddr + 2*refplanesz] - densityval3;
  if (++zindex < volsz.z) // map isn't always an even multiple of block size
    diffmap[outaddr + 3*planesz] = refmap[refaddr + 3*refplanesz] - densityval4;
#endif
}



//
// sum of absolute differences
//
__device__ float sumabsdiff_sumreduction(int tid, int totaltb, 
                                         float *sumabsdiffs_s,
                                         float *sumabsdiffs) {
  float sumabsdifftotal = 0.0f;

  // use precisely one warp to do the final reduction
  if (tid < warpSize) {
    for (int i=tid; i<totaltb; i+=warpSize) {
      sumabsdifftotal += sumabsdiffs[i];
    }

    // write to shared mem
    sumabsdiffs_s[tid] = sumabsdifftotal;
  }
  __syncthreads(); // all threads must hit syncthreads call...

  // perform intra-warp parallel reduction...
  // general loop version of parallel sum-reduction
  for (int s=warpSize>>1; s>0; s>>=1) {
    if (tid < s) {
      sumabsdiffs_s[tid] += sumabsdiffs_s[tid + s];
    }
    __syncthreads(); // all threads must hit syncthreads call...
  }

  return sumabsdiffs_s[0];
}


__global__ static void gaussdensity_sumabsdiff(int totaltb,
                                               int natoms,
                                               const float4 *sorted_xyzr, 
                                               int3 volsz,
                                               int3 acncells,
                                               float3 acoriginoffset,
                                               float acgridspacing,
                                               float invacgridspacing,
                                               const uint2 * cellStartEnd,
                                               float gridspacing, 
                                               unsigned int z, 
                                               int3 refvolsz,
                                               int3 refoffset,
                                               const float *refmap,
                                               float *sumabsdiff) {
  int tid = threadIdx.z*blockDim.x*blockDim.y + 
            threadIdx.y*blockDim.x + threadIdx.x;

#if __CUDA_ARCH__ >= 200
  // setup shared variable
  __shared__ bool isLastBlockDone;
  if (tid == 0) 
    isLastBlockDone = 0;
  __syncthreads();
#endif

  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GUNROLL;
  unsigned int refaddr = (zindex+refoffset.z) * refvolsz.x * refvolsz.y +
                         (yindex+refoffset.y) * refvolsz.x +
                         (xindex+refoffset.x);
  zindex += z;

  float thread_sumabsdiff = 0.0f;

  // can't early exit if this thread is outside of the grid bounds,
  // so we store a diff of 0.0 instead.
  if (xindex < volsz.x && yindex < volsz.y && zindex < volsz.z) {
    // compute ac grid index of lower corner minus gaussian radius
    int3 abmin, abmax;
    calc_ac_cell_ids(abmin, abmax, acncells, acoriginoffset,
                     gridspacing, acgridspacing, invacgridspacing);

    // density loop macro
    MAINDENSITYLOOP

    thread_sumabsdiff += fabsf(refmap[refaddr               ] - densityval1);
#if GUNROLL >= 2
    int refplanesz = refvolsz.x * refvolsz.y;
    if (++zindex < volsz.z) // map isn't always an even multiple of block size
      thread_sumabsdiff += fabsf(refmap[refaddr +   refplanesz] - densityval2);
#endif
#if GUNROLL >= 4
    if (++zindex < volsz.z) // map isn't always an even multiple of block size
      thread_sumabsdiff += fabsf(refmap[refaddr + 2*refplanesz] - densityval3);
    if (++zindex < volsz.z) // map isn't always an even multiple of block size
      thread_sumabsdiff += fabsf(refmap[refaddr + 3*refplanesz] - densityval4);
#endif
  }

  // all threads write their local sums to shared memory...
  __shared__ float sumabsdiff_s[TOTALBLOCKSZ];

  sumabsdiff_s[tid] = thread_sumabsdiff;
  __syncthreads(); // all threads must hit syncthreads call...

  // use precisely one warp to do the thread-block-wide reduction
  if (tid < warpSize) {
    float tmp = 0.0f;
    for (int i=tid; i<TOTALBLOCKSZ; i+=warpSize) {
      tmp += sumabsdiff_s[i];
    }

    // write to shared mem
    sumabsdiff_s[tid] = tmp;
  }
  __syncthreads(); // all threads must hit syncthreads call...

  // perform intra-warp parallel reduction...
  // general loop version of parallel sum-reduction
  for (int s=warpSize>>1; s>0; s>>=1) {
    if (tid < s) {
      sumabsdiff_s[tid] += sumabsdiff_s[tid + s];
    }
    __syncthreads(); // all threads must hit syncthreads call...
  }

  // check if we are the last thread block to finish and finalize results
  if (tid == 0) {   
    unsigned int bid = blockIdx.z * gridDim.x * gridDim.y +
                       blockIdx.y * gridDim.x + blockIdx.x;
    sumabsdiff[bid] = sumabsdiff_s[0];
#if __CUDA_ARCH__ >= 200
    __threadfence();

    unsigned int value = atomicInc(&tbcatomic[0], totaltb);
    isLastBlockDone = (value == (totaltb - 1));
  }
  __syncthreads();
  if (isLastBlockDone) {
    float totalsumabsdiff = sumabsdiff_sumreduction(tid, totaltb, sumabsdiff_s, sumabsdiff); 

    if (tid == 0)
      sumabsdiff[totaltb] = totalsumabsdiff;
 
    reset_atomic_counter(&tbcatomic[0]);
#else
    if (bid==0)
      sumabsdiff[totaltb] = 0.0f;
#endif
  }
}



//
// cross correlation
//
inline __device__ float calc_cc(float sums_ref, float sums_synth,
                                float squares_ref, float squares_synth,
                                int lsize, float lcc) {
  float cc = 0.0f;

  // detect and prevent cases that would cause division by zero
  // compute CC if at least one 1 pair of voxels was compared...
  if (lsize > 0) {
    float mean_ref     = sums_ref / lsize;
    float mean_synth   = sums_synth / lsize;
    float stddev_ref   = sqrtf(squares_ref / lsize - mean_ref*mean_ref);
    float stddev_synth = sqrtf(squares_synth / lsize - mean_synth*mean_synth);

    // To prevent corner cases that may contain only a single sample from
    // generating Inf or NaN correlation results, we test that the standard 
    // deviations are not only non-zero (thereby preventing Inf), but we go 
    // further, and test that the standard deviation values are 
    // greater than zero, as this type of comparison will return false 
    // in the presence of NaN, allowing us to handle that case better.
    // We can do the greater-than-zero comparison since it should not be
    // possible to get a negative result from the square root.
    if (stddev_ref > 0.0f && stddev_synth > 0.0f) {
      cc = (lcc - lsize*mean_ref*mean_synth) /
           (lsize * stddev_ref * stddev_synth);

      // report a CC of zero for blocks that have too few samples:
      // for now, we use a hard-coded threshold for testing purposes
      if (lsize < 32)
        cc = 0.0f; 

      // clamp out-of-range CC values, can be caused by too few samples...
      cc = (cc > -1.0f) ? cc : ((cc < 1.0f) ? cc : 1.0f);
    } else {
      cc = 0.0f;
    }
  }

  return cc;
}



// #define VMDUSESHUFFLE 1
#if defined(VMDUSESHUFFLE) && __CUDA_ARCH__ >= 300 && CUDART_VERSION >= 9000
// New warp shuffle-based CC sum reduction for Kepler and later GPUs.
inline __device__ void cc_sumreduction(int tid, int totaltb, 
                                float4 &total_cc_sums,
                                float &total_lcc,
                                int &total_lsize,
                                float4 *tb_cc_sums,
                                float *tb_lcc,
                                int *tb_lsize) {
  total_cc_sums = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  total_lcc = 0.0f;
  total_lsize = 0;

  // use precisely one warp to do the final reduction
  if (tid < warpSize) {
    for (int i=tid; i<totaltb; i+=warpSize) {
      total_cc_sums += tb_cc_sums[i];
      total_lcc     += tb_lcc[i];
      total_lsize   += tb_lsize[i];
    }

    // perform intra-warp parallel reduction...
    // general loop version of parallel sum-reduction
    for (int mask=warpSize/2; mask>0; mask>>=1) {
      total_cc_sums.x += __shfl_xor_sync(0xffffffff, total_cc_sums.x, mask);
      total_cc_sums.y += __shfl_xor_sync(0xffffffff, total_cc_sums.y, mask);
      total_cc_sums.z += __shfl_xor_sync(0xffffffff, total_cc_sums.z, mask);
      total_cc_sums.w += __shfl_xor_sync(0xffffffff, total_cc_sums.w, mask);
      total_lcc     += __shfl_xor_sync(0xffffffff, total_lcc, mask);
      total_lsize   += __shfl_xor_sync(0xffffffff, total_lsize, mask);
    }
  }
}
#else
// shared memory based CC sum reduction 
inline __device__ void cc_sumreduction(int tid, int totaltb, 
                                float4 &total_cc_sums,
                                float &total_lcc,
                                int &total_lsize,
                                float4 *tb_cc_sums,
                                float *tb_lcc,
                                int *tb_lsize) {
  total_cc_sums = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  total_lcc = 0.0f;
  total_lsize = 0;

  // use precisely one warp to do the final reduction
  if (tid < warpSize) {
    for (int i=tid; i<totaltb; i+=warpSize) {
      total_cc_sums += tb_cc_sums[i];
      total_lcc     += tb_lcc[i];
      total_lsize   += tb_lsize[i];
    }

    // write to shared mem
    tb_cc_sums[tid] = total_cc_sums;
    tb_lcc[tid]     = total_lcc;
    tb_lsize[tid]   = total_lsize;
  }
  __syncthreads(); // all threads must hit syncthreads call...

  // perform intra-warp parallel reduction...
  // general loop version of parallel sum-reduction
  for (int s=warpSize>>1; s>0; s>>=1) {
    if (tid < s) {
      tb_cc_sums[tid] += tb_cc_sums[tid + s];
      tb_lcc[tid]     += tb_lcc[tid + s];
      tb_lsize[tid]   += tb_lsize[tid + s];
    }
    __syncthreads(); // all threads must hit syncthreads call...
  }

  total_cc_sums = tb_cc_sums[0];
  total_lcc = tb_lcc[0];
  total_lsize = tb_lsize[0];
}
#endif


inline __device__ void thread_cc_sum(float ref, float density,
                                     float2 &thread_cc_means, 
                                     float2 &thread_cc_squares, 
                                     float &thread_lcc,
                                     int &thread_lsize) {
  if (!isnan(ref)) {
    thread_cc_means.x += ref;
    thread_cc_means.y += density;
    thread_cc_squares.x += ref*ref;
    thread_cc_squares.y += density*density;
    thread_lcc += ref * density;
    thread_lsize++;     
  }
}


//
// A simple tool to detect cases where we're getting significant
// floating point truncation when accumulating values...
//
#if 0
#define CKACCUM(a, b)                                                   \
  {                                                                     \
     float tmp = a;                                                     \
     a += b;                                                            \
     float trunc = a - tmp;                                             \
     if (b > 1e-5f && (((b - trunc) / b) > 0.01))                       \
       printf("truncation: sum: %f incr: %f trunc: %f trunc_rem: %f\n", \
               a, b, trunc, (b-trunc));                                 \
  }                                                                     
#else
#define CKACCUM(a, b) \
  a+=b;
#endif


__global__ static void gaussdensity_cc(int totaltb,
                                       int natoms,
                                       const float4 *sorted_xyzr, 
                                       int3 volsz,
                                       int3 acncells,
                                       float3 acoriginoffset,
                                       float acgridspacing,
                                       float invacgridspacing,
                                       const uint2 * cellStartEnd,
                                       float gridspacing, 
                                       unsigned int z, 
                                       float threshold,
                                       int3 refvolsz,
                                       int3 refoffset,
                                       const float *refmap,
                                       float4 *tb_cc_sums,
                                       float *tb_lcc,
                                       int *tb_lsize,
                                       float *tb_CC) {
  int tid = threadIdx.z*blockDim.x*blockDim.y + 
            threadIdx.y*blockDim.x + threadIdx.x;

#if __CUDA_ARCH__ >= 200
  // setup shared variable
  __shared__ bool isLastBlockDone;
  if (tid == 0) 
    isLastBlockDone = 0;
  __syncthreads();
#endif

  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GUNROLL;
  unsigned int refaddr = (zindex+refoffset.z) * refvolsz.x * refvolsz.y +
                         (yindex+refoffset.y) * refvolsz.x +
                         (xindex+refoffset.x);
  zindex += z;

  float2 thread_cc_means = make_float2(0.0f, 0.0f);
  float2 thread_cc_squares = make_float2(0.0f, 0.0f);
  float  thread_lcc = 0.0f;
  int    thread_lsize = 0;

  // can't early exit if this thread is outside of the grid bounds,
  // so we store a diff of 0.0 instead.
  if (xindex < volsz.x && yindex < volsz.y && zindex < volsz.z) {
    // compute ac grid index of lower corner minus gaussian radius
    int3 abmin, abmax;
    calc_ac_cell_ids(abmin, abmax, acncells, acoriginoffset,
                     gridspacing, acgridspacing, invacgridspacing);

    // density loop macro
    MAINDENSITYLOOP

    // for each density value, compare vs. threshold, check grid bounds
    // since these maps aren't necessarily a multiple of the thread block
    // size and we're unrolled in the Z-dimension, and finally, if 
    // if accepted, continue with summation of the CC contributions 
    float ref;
    if (densityval1 >= threshold) {
      ref = refmap[refaddr               ];
      thread_cc_sum(ref, densityval1, thread_cc_means, thread_cc_squares, thread_lcc, thread_lsize);
    }
#if GUNROLL >= 2
    int refplanesz = refvolsz.x * refvolsz.y;
    if ((densityval2 >= threshold) && (++zindex < volsz.z)) { 
      ref = refmap[refaddr +   refplanesz];
      thread_cc_sum(ref, densityval2, thread_cc_means, thread_cc_squares, thread_lcc, thread_lsize);
    }
#endif
#if GUNROLL >= 4
    if ((densityval3 >= threshold) && (++zindex < volsz.z)) { 
      ref = refmap[refaddr + 2*refplanesz];
      thread_cc_sum(ref, densityval3, thread_cc_means, thread_cc_squares, thread_lcc, thread_lsize);
    }
    if ((densityval4 >= threshold) && (++zindex < volsz.z)) { 
      ref = refmap[refaddr + 3*refplanesz];
      thread_cc_sum(ref, densityval4, thread_cc_means, thread_cc_squares, thread_lcc, thread_lsize);
    }
#endif
  }


#if defined(VMDUSESHUFFLE) && __CUDA_ARCH__ >= 300 && CUDART_VERSION >= 9000
  // all threads write their local sums to shared memory...
  __shared__ float2 tb_cc_means_s[TOTALBLOCKSZ];
  __shared__ float2 tb_cc_squares_s[TOTALBLOCKSZ];
  __shared__ float tb_lcc_s[TOTALBLOCKSZ];
  __shared__ int tb_lsize_s[TOTALBLOCKSZ];

  tb_cc_means_s[tid] = thread_cc_means;
  tb_cc_squares_s[tid] = thread_cc_squares;
  tb_lcc_s[tid] = thread_lcc;
  tb_lsize_s[tid] = thread_lsize;
  __syncthreads(); // all threads must hit syncthreads call...

  // use precisely one warp to do the thread-block-wide reduction
  if (tid < warpSize) {
    float2 tmp_cc_means = make_float2(0.0f, 0.0f);
    float2 tmp_cc_squares = make_float2(0.0f, 0.0f);
    float tmp_lcc = 0.0f;
    int tmp_lsize = 0;
    for (int i=tid; i<TOTALBLOCKSZ; i+=warpSize) {
      tmp_cc_means   += tb_cc_means_s[i];
      tmp_cc_squares += tb_cc_squares_s[i];
      tmp_lcc        += tb_lcc_s[i];
      tmp_lsize      += tb_lsize_s[i];
    }

    // perform intra-warp parallel reduction...
    // general loop version of parallel sum-reduction
    for (int mask=warpSize/2; mask>0; mask>>=1) {
      tmp_cc_means.x   += __shfl_xor_sync(0xffffffff, tmp_cc_means.x, mask);
      tmp_cc_means.y   += __shfl_xor_sync(0xffffffff, tmp_cc_means.y, mask);
      tmp_cc_squares.x += __shfl_xor_sync(0xffffffff, tmp_cc_squares.x, mask);
      tmp_cc_squares.y += __shfl_xor_sync(0xffffffff, tmp_cc_squares.y, mask);
      tmp_lcc          += __shfl_xor_sync(0xffffffff, tmp_lcc, mask);
      tmp_lsize        += __shfl_xor_sync(0xffffffff, tmp_lsize, mask);
    }

    // write per-thread-block partial sums to global memory,
    // if a per-thread-block CC output array is provided, write the 
    // local CC for this thread block out, and finally, check if we 
    // are the last thread block to finish, and finalize the overall
    // CC results for the entire grid of thread blocks.
    if (tid == 0) {   
      unsigned int bid = blockIdx.z * gridDim.x * gridDim.y +
                         blockIdx.y * gridDim.x + blockIdx.x;

      tb_cc_sums[bid] = make_float4(tmp_cc_means.x, tmp_cc_means.y,
                                    tmp_cc_squares.x, tmp_cc_squares.y);
      tb_lcc[bid]     = tmp_lcc;
      tb_lsize[bid]   = tmp_lsize;

      if (tb_CC != NULL) {
        float cc = calc_cc(tb_cc_means_s[0].x, tb_cc_means_s[0].y,
                           tb_cc_squares_s[0].x, tb_cc_squares_s[0].y,
                           tb_lsize_s[0], tb_lcc_s[0]);

        // write local per-thread-block CC to global memory
        tb_CC[bid]   = cc;
      }

      __threadfence();

      unsigned int value = atomicInc(&tbcatomic[0], totaltb);
      isLastBlockDone = (value == (totaltb - 1));
    }
  }
  __syncthreads();

  if (isLastBlockDone) {
    float4 total_cc_sums;
    float total_lcc;
    int total_lsize;
    cc_sumreduction(tid, totaltb, total_cc_sums, total_lcc, total_lsize,
                    tb_cc_sums, tb_lcc, tb_lsize); 

    if (tid == 0) {
      tb_cc_sums[totaltb] = total_cc_sums;
      tb_lcc[totaltb] = total_lcc;
      tb_lsize[totaltb] = total_lsize;
    }
 
    reset_atomic_counter(&tbcatomic[0]);
  }

#else

  // all threads write their local sums to shared memory...
  __shared__ float2 tb_cc_means_s[TOTALBLOCKSZ];
  __shared__ float2 tb_cc_squares_s[TOTALBLOCKSZ];
  __shared__ float tb_lcc_s[TOTALBLOCKSZ];
  __shared__ int tb_lsize_s[TOTALBLOCKSZ];

  tb_cc_means_s[tid] = thread_cc_means;
  tb_cc_squares_s[tid] = thread_cc_squares;
  tb_lcc_s[tid] = thread_lcc;
  tb_lsize_s[tid] = thread_lsize;
  __syncthreads(); // all threads must hit syncthreads call...

  // use precisely one warp to do the thread-block-wide reduction
  if (tid < warpSize) {
    float2 tmp_cc_means = make_float2(0.0f, 0.0f);
    float2 tmp_cc_squares = make_float2(0.0f, 0.0f);
    float tmp_lcc = 0.0f;
    int tmp_lsize = 0;
    for (int i=tid; i<TOTALBLOCKSZ; i+=warpSize) {
      tmp_cc_means   += tb_cc_means_s[i];
      tmp_cc_squares += tb_cc_squares_s[i];
      tmp_lcc        += tb_lcc_s[i];
      tmp_lsize      += tb_lsize_s[i];
    }

    // write to shared mem
    tb_cc_means_s[tid]   = tmp_cc_means;
    tb_cc_squares_s[tid] = tmp_cc_squares;
    tb_lcc_s[tid]        = tmp_lcc;
    tb_lsize_s[tid]      = tmp_lsize;
  }
  __syncthreads(); // all threads must hit syncthreads call...

  // perform intra-warp parallel reduction...
  // general loop version of parallel sum-reduction
  for (int s=warpSize>>1; s>0; s>>=1) {
    if (tid < s) {
      tb_cc_means_s[tid]   += tb_cc_means_s[tid + s];
      tb_cc_squares_s[tid] += tb_cc_squares_s[tid + s];
      tb_lcc_s[tid]        += tb_lcc_s[tid + s];
      tb_lsize_s[tid]      += tb_lsize_s[tid + s];
    }
    __syncthreads(); // all threads must hit syncthreads call...
  }
//#endif

  // write per-thread-block partial sums to global memory,
  // if a per-thread-block CC output array is provided, write the 
  // local CC for this thread block out, and finally, check if we 
  // are the last thread block to finish, and finalize the overall
  // CC results for the entire grid of thread blocks.
  if (tid == 0) {   
    unsigned int bid = blockIdx.z * gridDim.x * gridDim.y +
                       blockIdx.y * gridDim.x + blockIdx.x;
    tb_cc_sums[bid] = make_float4(tb_cc_means_s[0].x, tb_cc_means_s[0].y,
                                  tb_cc_squares_s[0].x, tb_cc_squares_s[0].y);
    tb_lcc[bid]     = tb_lcc_s[0];
    tb_lsize[bid]   = tb_lsize_s[0];

    if (tb_CC != NULL) {
      float cc = calc_cc(tb_cc_means_s[0].x, tb_cc_means_s[0].y,
                         tb_cc_squares_s[0].x, tb_cc_squares_s[0].y,
                         tb_lsize_s[0], tb_lcc_s[0]);

      // write local per-thread-block CC to global memory
      tb_CC[bid]   = cc;
    }

#if __CUDA_ARCH__ >= 200
    __threadfence();

    unsigned int value = atomicInc(&tbcatomic[0], totaltb);
    isLastBlockDone = (value == (totaltb - 1));
  }
  __syncthreads();
  if (isLastBlockDone) {
    float4 total_cc_sums;
    float total_lcc;
    int total_lsize;
    cc_sumreduction(tid, totaltb, total_cc_sums, total_lcc, total_lsize,
                    tb_cc_sums, tb_lcc, tb_lsize); 

    if (tid == 0) {
      tb_cc_sums[totaltb] = total_cc_sums;
      tb_lcc[totaltb] = total_lcc;
      tb_lsize[totaltb] = total_lsize;
    }
 
    reset_atomic_counter(&tbcatomic[0]);
#else
    // code path for older GPUs that lack atomic ops
    if (bid == 0) {
      tb_cc_sums[totaltb] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      tb_lcc[totaltb] = 0.0f;
      tb_lsize[totaltb] = 0;
    }
#endif
  }
#endif
}



//
// kernels for computing min/max/mean/stddev for a map
//
#if 0
inline __device__ void mmms_reduction(int tid, int totaltb,
                                      float4 &total_mmms,
                                      int &total_lsize,
                                      float4 *tb_mmms,
                                      int *tb_lsize) {
  total_mmms = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  total_lsize = 0;

  // use precisely one warp to do the final reduction
  if (tid < warpSize) {
    for (int i=tid; i<totaltb; i+=warpSize) {
      total_mmms += tb_mmms[i];
      total_lsize   += tb_lsize[i];
    }

    // write to shared mem
    tb_mmms[tid] = total_mmms;
    tb_lsize[tid]   = total_lsize;
  }
  __syncthreads(); // all threads must hit syncthreads call...

  // perform intra-warp parallel reduction...
  // general loop version of parallel sum-reduction
  for (int s=warpSize>>1; s>0; s>>=1) {
    if (tid < s) {
      tb_mmms[tid] += tb_mmms[tid + s];
      tb_lsize[tid]   += tb_lsize[tid + s];
    }
    __syncthreads(); // all threads must hit syncthreads call...
  }

  total_mmms = tb_mmms[0];
  total_lsize = tb_lsize[0];
}


inline __device__ void thread_mmms_reduce(float val,
                                          float2 &thread_minmax,
                                          float2 &thread_mean_square,
                                          int &thread_lsize) {
  if (!isnan(val)) {
    if (thread_lsize == 0) {
      thread_minmax.x = val;
      thread_minmax.y = val;
    } else {
      thread_minmax.x = fminf(thread_minmax.x, val);
      thread_minmax.y = fmaxf(thread_minmax.y, val);
    }

    thread_mean_square.x += val;
    thread_mean_square.y += val*val;
    thread_lsize++;    
  }
}


__global__ static void calc_minmax_mean_stddev(int totaltb,
                                               int3 volsz,
                                               float threshold,
                                               const float *map,
                                               float4 *tb_mmms,
                                               int *tb_lsize) {
  int tid = threadIdx.z*blockDim.x*blockDim.y + 
            threadIdx.y*blockDim.x + threadIdx.x;

#if __CUDA_ARCH__ >= 200
  // setup shared variable
  __shared__ bool isLastBlockDone;
  if (tid == 0) 
    isLastBlockDone = 0;
  __syncthreads();
#endif

  unsigned int xindex  = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
  unsigned int zindex  = ((blockIdx.z * blockDim.z) + threadIdx.z) * GUNROLL;
  unsigned int mapaddr = zindex * volsz.x * volsz.y +
                         yindex * volsz.x +
                         xindex;

  float2 thread_minmax = make_float2(0.0f, 0.0f);
  float2 thread_mean_square = make_float2(0.0f, 0.0f);
  int    thread_lsize = 0;

  // can't early exit if this thread is outside of the grid bounds
  if (xindex < volsz.x && yindex < volsz.y && zindex < volsz.z) {

    // for each value, check grid bounds since these maps aren't necessarily 
    // a multiple of the thread block size and we're unrolled in the 
    // Z-dimension, and finally, if accepted, continue with reduction
    float val;
    val = map[mapaddr            ];
    thread_mmms_reduce(val, thread_minmax, thread_mean_square, thread_lsize);
#if GUNROLL >= 2
    int planesz = volsz.x * volsz.y;
    if (++zindex < volsz.z) {
      val = map[mapaddr +   planesz];
      thread_mmms_reduce(val, thread_minmax, thread_mean_square, thread_lsize);
    }
#endif
#if GUNROLL >= 4
    if (++zindex < volsz.z) {
      val = map[mapaddr + 2*planesz];
      thread_mmms_reduce(val, thread_minmax, thread_mean_square, thread_lsize);
    }
    if (++zindex < volsz.z) {
      val = map[mapaddr + 3*planesz];
      thread_mmms_reduce(val, thread_minmax, thread_mean_square, thread_lsize);
    }
#endif
  }

  // all threads write their local sums to shared memory...
  __shared__ float2 tb_minmax_s[TOTALBLOCKSZ];
  __shared__ float2 tb_mean_square_s[TOTALBLOCKSZ];
  __shared__ int tb_lsize_s[TOTALBLOCKSZ];

  tb_minmax_s[tid]      = thread_minmax;
  tb_mean_square_s[tid] = thread_mean_square;
  tb_lsize_s[tid]       = thread_lsize;
  __syncthreads(); // all threads must hit syncthreads call...

  // use precisely one warp to do the thread-block-wide reduction
  if (tid < warpSize) {
    float2 tmp_minmax = thread_minmax;
    float2 tmp_mean_square = make_float2(0.0f, 0.0f);
    int tmp_lsize = 0;
    for (int i=tid; i<TOTALBLOCKSZ; i+=warpSize) {
      tmp_minmax.x = fminf(tb_minmax_s[i].x, tmp_minmax.x);
      tmp_minmax.x = fmaxf(tb_minmax_s[i].y, tmp_minmax.y);
      tmp_mean_square += tb_mean_square_s[i];
      tmp_lsize       += tb_lsize_s[i];
    }

    // write to shared mem
    tb_minmax_s[tid]   = tmp_minmax;
    tb_mean_square_s[tid] = tmp_mean_square;
    tb_lsize_s[tid]      = tmp_lsize;
  }
  __syncthreads(); // all threads must hit syncthreads call...

  // perform intra-warp parallel reduction...
  // general loop version of parallel sum-reduction
  for (int s=warpSize>>1; s>0; s>>=1) {
    if (tid < s) {
      tb_minmax_s[tid]      += tb_minmax_s[tid + s];
      tb_mean_square_s[tid] += tb_mean_square_s[tid + s];
      tb_lsize_s[tid]       += tb_lsize_s[tid + s];
    }
    __syncthreads(); // all threads must hit syncthreads call...
  }

  // write per-thread-block partial sums to global memory,
  // check if we are the last thread block to finish, and finalize 
  // the overall results for the entire grid of thread blocks.
  if (tid == 0) {
    unsigned int bid = blockIdx.z * gridDim.x * gridDim.y +
                       blockIdx.y * gridDim.x + blockIdx.x;
    tb_mmms[bid] = make_float4(tb_minmax_s[0].x, tb_minmax_s[0].y,
                               tb_mean_square_s[0].x, tb_mean_square_s[0].y);
    tb_lsize[bid]   = tb_lsize_s[0];
#if __CUDA_ARCH__ >= 200
    __threadfence();

    unsigned int value = atomicInc(&tbcatomic[0], totaltb);
    isLastBlockDone = (value == (totaltb - 1));
  }
  __syncthreads();
  if (isLastBlockDone) {
    float4 total_mmms;
    int total_lsize;
    mmms_reduction(tid, totaltb, total_mmms, total_lsize,
                   tb_mmms, tb_lsize);

    if (tid == 0) {
      tb_mmms[totaltb] = total_mmms;
      tb_lsize[totaltb] = total_lsize;
    }

    reset_atomic_counter(&tbcatomic[0]);
#else
    // code path for older GPUs that lack atomic ops
    if (bid == 0) {
      tb_mmms[totaltb] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      tb_lsize[totaltb] = 0;
    }
#endif
  }
}

#endif


static int vmd_cuda_build_accel(cudaError_t &err,
                                const long int natoms,
                                const int3 volsz,
                                const float gausslim,
                                const float radscale,
                                const float maxrad,
                                const float gridspacing,
                                const float *xyzr_f, 
                                float4 *& xyzr_d,
                                float4 *& sorted_xyzr_d,
                                uint2 *& cellStartEnd_d,
                                float &acgridspacing,
                                int3 &accelcells) {
  // compute grid spacing for the acceleration grid
  acgridspacing = gausslim * radscale * maxrad;

  // XXX allow override of value for testing
  if (getenv("VMDACGRIDSPACING") != NULL) {
    acgridspacing = atof(getenv("VMDACGRIDSPACING"));
  }

  accelcells.x = max(int((volsz.x*gridspacing) / acgridspacing), 1);
  accelcells.y = max(int((volsz.y*gridspacing) / acgridspacing), 1);
  accelcells.z = max(int((volsz.z*gridspacing) / acgridspacing), 1);
#if 0
  printf("  Accel grid(%d, %d, %d) spacing %f\n",
         accelcells.x, accelcells.y, accelcells.z, acgridspacing);
#endif

  // pack atom coordinates and radii into float4 types and copy to the device
  cudaMalloc((void**)&xyzr_d, natoms * sizeof(float4));

  // pre-process the atom coordinates and radii as needed
  // short-term fix until a new CUDA kernel takes care of this
  int i, i4;
  float4 *xyzr = (float4 *) malloc(natoms * sizeof(float4));
  float log2e = log2(2.718281828);
  for (i=0,i4=0; i<natoms; i++,i4+=4) {
    xyzr[i].x = xyzr_f[i4    ];
    xyzr[i].y = xyzr_f[i4 + 1];
    xyzr[i].z = xyzr_f[i4 + 2];

    float scaledrad = xyzr_f[i4 + 3] * radscale;
    float arinv = -1.0f * log2e / (2.0f*scaledrad*scaledrad);

    xyzr[i].w = arinv;
  }
  cudaMemcpy(xyzr_d, xyzr, natoms * sizeof(float4), cudaMemcpyHostToDevice);
  free(xyzr);

  // build uniform grid acceleration structure
  if (vmd_cuda_build_density_atom_grid(natoms, xyzr_d, sorted_xyzr_d, 
                                       cellStartEnd_d, 
                                       accelcells, 1.0f / acgridspacing) != 0) {
    cudaFree(xyzr_d);
    return -1;
  }

  return 0;
}



static int vmd_cuda_destroy_accel(float4 *& xyzr_d,
                                  float4 *& sorted_xyzr_d,
                                  uint2 *& cellStartEnd_d) {
  cudaFree(xyzr_d);
  xyzr_d = NULL;

  cudaFree(sorted_xyzr_d);
  sorted_xyzr_d = NULL;

  cudaFree(cellStartEnd_d);
  cellStartEnd_d = NULL;

  return 0;
}


//
// Simple helper to calculate the CUDA kernel launch parameters for
// the majority of the density kernel variants based on the dimensions
// of the computed density map
//
static void vmd_cuda_kernel_launch_dims(int3 volsz, dim3 &Bsz, dim3 &Gsz) {
  Bsz = dim3(GBLOCKSZX, GBLOCKSZY, GBLOCKSZZ);
  Gsz = dim3((volsz.x+Bsz.x-1) / Bsz.x, 
             (volsz.y+Bsz.y-1) / Bsz.y,
             (volsz.z+(Bsz.z*GUNROLL)-1) / (Bsz.z * GUNROLL));

#if 0
  printf("GPU kernel launch dimensions: Bsz: %dx%dx%d Gsz: %dx%dx%d\n",
         Bsz.x, Bsz.y, Bsz.z, Gsz.x, Gsz.y, Gsz.z);
#endif
}


static int vmd_cuda_gaussdensity_calc(long int natoms,
                                      float3 acoriginoffset,
                                      int3 acvolsz,
                                      const float *xyzr_f, 
                                      float *volmap, 
                                      int3 volsz, float maxrad,
                                      float radscale, float gridspacing,
                                      float gausslim,
                                      int3 refvolsz,
                                      int3 refoffset,
                                      const float *refmap,
                                      float *diffmap, int verbose) {
  float4 *xyzr_d = NULL;
  float4 *sorted_xyzr_d = NULL;
  uint2 *cellStartEnd_d = NULL;

  wkf_timerhandle globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

  cudaError_t err;
  if (check_gpu_compute20(err) != 0) 
    return -1;

  // Allocate output arrays for the gaussian density map and 3-D texture map
  // We test for errors carefully here since this is the most likely place
  // for a memory allocation failure due to the size of the grid.
  unsigned long volmemsz = volsz.x * volsz.y * volsz.z * sizeof(float);
  float *devdensity = NULL;
  if (cudaMalloc((void**)&devdensity, volmemsz) != cudaSuccess) {
    err = cudaGetLastError(); // eat error so next CUDA op succeeds
    return -1;
  }

#if 1
  float *devrefmap = NULL;
  float *devdiffmap = NULL;
  if (refmap && diffmap) {
    unsigned long refmemsz = refvolsz.x * refvolsz.y * refvolsz.z * sizeof(float);
    if (cudaMalloc((void**)&devrefmap, refmemsz) != cudaSuccess) {
      err = cudaGetLastError(); // eat error so next CUDA op succeeds
      return -1;
    }
    cudaMemcpy(devrefmap, refmap, refmemsz, cudaMemcpyHostToDevice);

    if (cudaMalloc((void**)&devdiffmap, volmemsz) != cudaSuccess) {
      err = cudaGetLastError(); // eat error so next CUDA op succeeds
      return -1;
    }
  }
#endif

  int3 accelcells = make_int3(1, 1, 1);
  float acgridspacing = 0.0f;
  if (vmd_cuda_build_accel(err, natoms, acvolsz, gausslim, radscale, maxrad,
                           gridspacing, xyzr_f, xyzr_d, sorted_xyzr_d,
                           cellStartEnd_d, acgridspacing, accelcells) != 0) {
    return -1;
  }

  double sorttime = wkf_timer_timenow(globaltimer);

  dim3 Bsz, Gsz;
  vmd_cuda_kernel_launch_dims(volsz, Bsz, Gsz);

  gaussdensity_fast<<<Gsz, Bsz, 0>>>(natoms, 
                                     sorted_xyzr_d,
                                     volsz,
                                     accelcells,
                                     acoriginoffset,
                                     acgridspacing,
                                     1.0f / acgridspacing,
                                     cellStartEnd_d,
                                     gridspacing, 0,
                                     devdensity);
  cudaDeviceSynchronize(); 
  double densitykerneltime = wkf_timer_timenow(globaltimer);

#if 1
  if (devrefmap && devdiffmap) {
#if 1
    if (verbose) {
      printf("CUDA: refvol(%d,%d,%d) refoffset(%d,%d,%d)\n",
             refvolsz.x, refvolsz.y, refvolsz.z,
             refoffset.x, refoffset.y, refoffset.z);
    }
#endif
    gaussdensity_diff<<<Gsz, Bsz, 0>>>(natoms, 
                                       sorted_xyzr_d,
                                       volsz,
                                       accelcells,
                                       acoriginoffset,
                                       acgridspacing,
                                       1.0f / acgridspacing,
                                       cellStartEnd_d,
                                       gridspacing, 0,
                                       refvolsz,
                                       refoffset,
                                       devrefmap,
                                       devdiffmap);
    cudaDeviceSynchronize(); 
  }
#endif
  double diffkerneltime = wkf_timer_timenow(globaltimer);


#if 1
  //
  // sum of absolute differences
  //
  if (devrefmap) {
    unsigned int blockcount = Gsz.x * Gsz.y * Gsz.z;
    float *devsumabsdiffs = NULL;
    if (cudaMalloc((void**)&devsumabsdiffs, (blockcount+1) * sizeof(float)) != cudaSuccess) {
      err = cudaGetLastError(); // eat error so next CUDA op succeeds
      return -1;
    }
    gaussdensity_sumabsdiff<<<Gsz, Bsz, 0>>>(blockcount,
                                             natoms, 
                                             sorted_xyzr_d,
                                             volsz,
                                             accelcells,
                                             acoriginoffset,
                                             acgridspacing,
                                             1.0f / acgridspacing,
                                             cellStartEnd_d,
                                             gridspacing, 0,
                                             refvolsz,
                                             refoffset,
                                             devrefmap,
                                             devsumabsdiffs);
    cudaDeviceSynchronize(); 

    float *sumabsdiffs = new float[blockcount+1];
    cudaMemcpy(sumabsdiffs, devsumabsdiffs, (blockcount+1)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devsumabsdiffs);

#if 0
    printf("Map of block-wise sumabsdiffs:\n");
    for (int i=0; i<blockcount; i++) {
      printf("  tb[%d] absdiff %f\n", i, sumabsdiffs[i]);
    }
#endif

#if 1
    float sumabsdifftotal = 0.0f;
    for (int j=0; j<blockcount; j++) {
      sumabsdifftotal += sumabsdiffs[j];
    }
    if (verbose) {
      printf("Sum of absolute differences: %f\n", sumabsdifftotal);
      printf("Kernel sum of absolute differences: %f\n", sumabsdiffs[blockcount]);
    }
#endif

    delete [] sumabsdiffs;
  }
#endif
  double sumabsdiffkerneltime = wkf_timer_timenow(globaltimer);


#if 1
  //
  // cross correlation
  //
  if (devrefmap) {
    unsigned int blockcount = Gsz.x * Gsz.y * Gsz.z;
    float4 *dev_cc_sums = NULL;
    float *dev_lcc = NULL;
    int *dev_lsize = NULL;
    float *dev_CC = NULL;
 
    if (cudaMalloc((void**)&dev_cc_sums, (blockcount+1) * sizeof(float4)) != cudaSuccess) {
      err = cudaGetLastError(); // eat error so next CUDA op succeeds
      return -1;
    }

    if (cudaMalloc((void**)&dev_lcc, (blockcount+1) * sizeof(float)) != cudaSuccess) {
      err = cudaGetLastError(); // eat error so next CUDA op succeeds
      return -1;
    }

    if (cudaMalloc((void**)&dev_lsize, (blockcount+1) * sizeof(int)) != cudaSuccess) {
      err = cudaGetLastError(); // eat error so next CUDA op succeeds
      return -1;
    }

    if (cudaMalloc((void**)&dev_CC, (blockcount+1) * sizeof(float)) != cudaSuccess) {
      err = cudaGetLastError(); // eat error so next CUDA op succeeds
      return -1;
    }


    gaussdensity_cc<<<Gsz, Bsz, 0>>>(blockcount,
                                     natoms, 
                                     sorted_xyzr_d,
                                     volsz,
                                     accelcells,
                                     acoriginoffset,
                                     acgridspacing,
                                     1.0f / acgridspacing,
                                     cellStartEnd_d,
                                     gridspacing, 0,
                                     -FLT_MAX, // accept all densities...
                                     refvolsz,
                                     refoffset,
                                     devrefmap,
                                     dev_cc_sums,
                                     dev_lcc,
                                     dev_lsize,
                                     dev_CC);
    cudaDeviceSynchronize(); 

    float4 *cc_sums = new float4[blockcount+1];
    cudaMemcpy(cc_sums, dev_cc_sums, (blockcount+1)*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaFree(dev_cc_sums);

    float *lcc = new float[blockcount+1];
    cudaMemcpy(lcc, dev_lcc, (blockcount+1)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_lcc);

    int *lsize = new int[blockcount+1];
    cudaMemcpy(lsize, dev_lsize, (blockcount+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_lsize);

    float *CC = new float[blockcount+1];
    cudaMemcpy(CC, dev_CC, (blockcount+1)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_CC);

#if 0
    float4 tmp_cc_sums = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float tmp_lcc = 0.0f;
    int tmp_lsize = 0;
    for (int j=0; j<blockcount; j++) {
      tmp_cc_sums.x += cc_sums[j].x;
      tmp_cc_sums.y += cc_sums[j].y;
      tmp_cc_sums.z += cc_sums[j].z;
      tmp_cc_sums.w += cc_sums[j].w;
      tmp_lcc += lcc[j]; 
      tmp_lsize += lsize[j]; 
    }

    if (verbose) {
      printf("CC sums:\n");
      printf("    mean_ref: %f\n", tmp_cc_sums.x);
      printf("    mean_syn: %f\n", tmp_cc_sums.y);
      printf("    stddev_ref: %f\n", tmp_cc_sums.z);
      printf("    stddev_syn: %f\n", tmp_cc_sums.w);
      printf("    lcc: %f\n", tmp_lcc);
      printf("    lsize: %d\n", tmp_lsize);

      printf("Kernel CC sums:\n");
      printf("    mean_ref: %f\n", cc_sums[blockcount].x);
      printf("    mean_syn: %f\n", cc_sums[blockcount].y);
      printf("    stddev_ref: %f\n", cc_sums[blockcount].z);
      printf("    stddev_syn: %f\n", cc_sums[blockcount].w);
      printf("    lcc: %f\n", lcc[blockcount]);
      printf("    lsize: %d\n", lsize[blockcount]);
    }
#endif

    float mean_ref     = cc_sums[blockcount].x / lsize[blockcount];
    float mean_synth   = cc_sums[blockcount].y / lsize[blockcount];
    float stddev_ref   = sqrtf(cc_sums[blockcount].z / lsize[blockcount] - mean_ref*mean_ref);
    float stddev_synth = sqrtf(cc_sums[blockcount].w / lsize[blockcount] - mean_synth*mean_synth);
    float cc = (lcc[blockcount] - lsize[blockcount]*mean_ref*mean_synth) / 
               (lsize[blockcount] * stddev_ref * stddev_synth);

#if 1
    if (verbose) {
      printf("Final MDFF Kernel CC results:\n");
      printf("  mean_ref: %f\n", mean_ref);
      printf("  mean_synth: %f\n", mean_synth);
      printf("  stdev_ref: %f\n", stddev_ref);
      printf("  stdev_synth: %f\n", stddev_synth);
      printf("  CC: %f\n", cc);
    }
#endif

    delete [] cc_sums;
    delete [] lcc;
    delete [] lsize;
  }
#endif


  cudaMemcpy(volmap, devdensity, volmemsz, cudaMemcpyDeviceToHost);
  cudaFree(devdensity);
#if 1
  if (devrefmap && devdiffmap) {
    cudaMemcpy(diffmap, devdiffmap, volmemsz, cudaMemcpyDeviceToHost);
    cudaFree(devrefmap);
    cudaFree(devdiffmap);
  }
#endif

  // free uniform grid acceleration structure
  vmd_cuda_destroy_accel(xyzr_d, sorted_xyzr_d, cellStartEnd_d);

  err = cudaGetLastError();

  wkf_timer_stop(globaltimer);
  double totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  if (err != cudaSuccess) { 
    printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    return -1;
  }

  if (verbose) {
    printf("MDFF CC GPU runtime: %.3f [sort: %.3f density %.3f diff %.3f copy: %.3f]\n", 
           totalruntime, sorttime, densitykerneltime-sorttime, 
           diffkerneltime-densitykerneltime, totalruntime-densitykerneltime);
  }

  return 0;
}


static int vmd_cuda_cc_calc(long int natoms,
                            float3 acoriginoffset,
                            int3 acvolsz,
                            const float *xyzr_f, 
                            int3 volsz, float maxrad,
                            float radscale, float gridspacing,
                            float gausslim,
                            int3 refvolsz,
                            int3 refoffset,
                            const float *devrefmap,
                            float ccthreshdensity,
                            float &result_cc,
                            const float *origin,
                            VolumetricData **spatialccvol,
                            int verbose) {
  float4 *xyzr_d = NULL;
  float4 *sorted_xyzr_d = NULL;
  uint2 *cellStartEnd_d = NULL;

  wkf_timerhandle globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

  if (!devrefmap)
    return -1;

  cudaError_t err;
  if (check_gpu_compute20(err) != 0) 
    return -1;

  double refcopytime = wkf_timer_timenow(globaltimer);

  int3 accelcells = make_int3(1, 1, 1);
  float acgridspacing = 0.0f;
  if (vmd_cuda_build_accel(err, natoms, acvolsz, gausslim, radscale, maxrad,
                           gridspacing, xyzr_f, xyzr_d, sorted_xyzr_d,
                           cellStartEnd_d, acgridspacing, accelcells) != 0) {
    return -1;
  }

  double sorttime = wkf_timer_timenow(globaltimer);

  dim3 Bsz, Gsz;
  vmd_cuda_kernel_launch_dims(volsz, Bsz, Gsz);

  //
  // cross correlation
  //
  unsigned int blockcount = Gsz.x * Gsz.y * Gsz.z;
  float4 *dev_cc_sums = NULL;
  float *dev_lcc = NULL;
  int *dev_lsize = NULL;
  float *dev_tbCC = NULL;

  cudaMalloc((void**)&dev_cc_sums, (blockcount+1) * sizeof(float4));
  cudaMalloc((void**)&dev_lcc, (blockcount+1) * sizeof(float));

  if (spatialccvol != NULL) {
    cudaMalloc((void**)&dev_tbCC, (blockcount+1) * sizeof(float));
  }

  if (cudaMalloc((void**)&dev_lsize, (blockcount+1) * sizeof(int)) != cudaSuccess) {
    err = cudaGetLastError(); // eat error so next CUDA op succeeds
    return -1;
  }

  gaussdensity_cc<<<Gsz, Bsz, 0>>>(blockcount,
                                   natoms, 
                                   sorted_xyzr_d,
                                   volsz,
                                   accelcells,
                                   acoriginoffset,
                                   acgridspacing,
                                   1.0f / acgridspacing,
                                   cellStartEnd_d,
                                   gridspacing, 0,
                                   ccthreshdensity,
                                   refvolsz,
                                   refoffset,
                                   devrefmap,
                                   dev_cc_sums,
                                   dev_lcc,
                                   dev_lsize,
                                   dev_tbCC);
  cudaDeviceSynchronize(); 
  double cctime = wkf_timer_timenow(globaltimer);

  float4 *cc_sums = new float4[blockcount+1];
  cudaMemcpy(cc_sums, dev_cc_sums, (blockcount+1)*sizeof(float4), cudaMemcpyDeviceToHost);
  cudaFree(dev_cc_sums);

  float *lcc = new float[blockcount+1];
  cudaMemcpy(lcc, dev_lcc, (blockcount+1)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dev_lcc);

  int *lsize = new int[blockcount+1];
  cudaMemcpy(lsize, dev_lsize, (blockcount+1)*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_lsize);

  float mean_ref     = cc_sums[blockcount].x / lsize[blockcount];
  float mean_synth   = cc_sums[blockcount].y / lsize[blockcount];
  float stddev_ref   = sqrtf(cc_sums[blockcount].z / lsize[blockcount] - mean_ref*mean_ref);
  float stddev_synth = sqrtf(cc_sums[blockcount].w / lsize[blockcount] - mean_synth*mean_synth);

  float cc = 0.0f;

  // detect and prevent cases that would cause division by zero
  // compute CC if at least one 1 pair of voxels was compared...
  if (lsize[blockcount] > 0) {
    if (stddev_ref == 0.0f || stddev_synth == 0.0f) {
      printf("WARNING: Ill-conditioned CC calc. due to zero std. deviation:\n");
      printf("WARNING: stddev_ref: %f stddev_synth: %f\n", stddev_ref, stddev_synth);
      cc = 0.0f; 
    } else {
      cc = (lcc[blockcount] - lsize[blockcount]*mean_ref*mean_synth) / 
           (lsize[blockcount] * stddev_ref * stddev_synth);
    }
  }

  if (spatialccvol != NULL) {
    char mapname[64] = { "mdff spatial CC map" };
    float spxaxis[3] = {1.0f, 0.0f, 0.0f};
    float spyaxis[3] = {0.0f, 1.0f, 0.0f};
    float spzaxis[3] = {0.0f, 0.0f, 1.0f}; 
    spxaxis[0] = Bsz.x *  Gsz.x * gridspacing;
    spyaxis[1] = Bsz.y *  Gsz.y * gridspacing;
    spzaxis[2] = Bsz.z *  Gsz.z * gridspacing * GUNROLL;

    if (verbose) {
      printf("Spatial CC map: blockcount[%d]  x: %d y: %d z: %d  cnt: %d\n",
             blockcount, Gsz.x, Gsz.y, Gsz.z, (Gsz.x*Gsz.y*Gsz.z));
    }

    float *spatialccmap = new float[blockcount];
    cudaMemcpy(spatialccmap, dev_tbCC, blockcount*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_tbCC);
     
    *spatialccvol = new VolumetricData(mapname, origin, 
                                       spxaxis, spyaxis, spzaxis,
                                       Gsz.x, Gsz.y, Gsz.z, spatialccmap);
  }

#if 0
  if (verbose) {
    printf("Final MDFF Kernel CC results:\n");
    printf("  GPU CC sum data:\n");
    printf("    mean_ref: %f\n", cc_sums[blockcount].x);
    printf("    mean_synth: %f\n", cc_sums[blockcount].y);
    printf("    squares_ref: %f\n", cc_sums[blockcount].z);
    printf("    squares_synth: %f\n", cc_sums[blockcount].w);
    printf("    lcc: %f\n", lcc[blockcount]);
    printf("    lsize: %d\n", lsize[blockcount]);
    printf("  Reduced/finalized GPU CC data:\n");
    printf("    mean_ref: %f\n", mean_ref);
    printf("    mean_synth: %f\n", mean_synth);
    printf("    stdev_ref: %f\n", stddev_ref);
    printf("    stdev_synth: %f\n", stddev_synth);
    printf("    voxels used: %d (total %d)\n", 
           lsize[blockcount], volsz.x*volsz.y*volsz.z);
    printf("  CC: %f\n", cc);
  }
#endif

  result_cc = cc; // assign final result

  delete [] cc_sums;
  delete [] lcc;
  delete [] lsize;

  // free uniform grid acceleration structure
  vmd_cuda_destroy_accel(xyzr_d, sorted_xyzr_d, cellStartEnd_d);

  err = cudaGetLastError();

  wkf_timer_stop(globaltimer);
  double totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  if (err != cudaSuccess) { 
    printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    return -1;
  }

  if (verbose) {
    printf("MDFF CC GPU runtime: %.3f [refcopy: %.3f sort: %.3f cc: %.3f copy: %.3f]\n", 
           totalruntime, refcopytime, sorttime-refcopytime, cctime-sorttime, totalruntime-cctime);
  }

  return 0;
}


#if 0
static int vmd_cuda_gaussdensity_sumabsdiff(long int natoms,
                                      const float *xyzr_f, 
                                      float *volmap, int3 volsz, float maxrad,
                                      float radscale, float gridspacing,
                                      float gausslim) {
  float4 *xyzr_d = NULL;
  float4 *sorted_xyzr_d = NULL;
  uint2 *cellStartEnd_d = NULL;

  wkf_timerhandle globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

  cudaError_t err;
  if (check_gpu_compute20(err) != 0) 
    return -1;

  // Allocate output arrays for the gaussian density map and 3-D texture map
  // We test for errors carefully here since this is the most likely place
  // for a memory allocation failure due to the size of the grid.
  unsigned long volmemsz = volsz.x * volsz.y * volsz.z * sizeof(float);
  float *devdensity = NULL;
  if (cudaMalloc((void**)&devdensity, volmemsz) != cudaSuccess) {
    err = cudaGetLastError(); // eat error so next CUDA op succeeds
    return -1;
  }

  int3 accelcells = make_int3(1, 1, 1);
  float acgridspacing = 0.0f;
  if (vmd_cuda_build_accel(err, natoms, volsz, gausslim, radscale, maxrad,
                           gridspacing, xyzr_f, xyzr_d, sorted_xyzr_d,
                           cellStartEnd_d, acgridspacing, accelcells) != 0) {
    return -1;
  }

  double sorttime = wkf_timer_timenow(globaltimer);

  dim3 Bsz, Gsz;
  vmd_cuda_kernel_launch_dims(volsz, Bsz, Gsz);

#if 0
  int3 refoffset = make_int3(0, 0, 0);
  gaussdensity_sumabsdiff<<<Gsz, Bsz, 0>>>(natoms,
                                     sorted_xyzr_d,
                                     volsz,
                                     accelcells,
                                     acgridspacing,
                                     1.0f / acgridspacing,
                                     cellStartEnd_d,
                                     gridspacing, 0,
                                     refoffset,
#if 1
                                     NULL, NULL);
#else
                                     densitygrid,
                                     densitysumabsdiff);
#endif
#endif

  cudaDeviceSynchronize(); 
  double densitykerneltime = wkf_timer_timenow(globaltimer);
  cudaMemcpy(volmap, devdensity, volmemsz, cudaMemcpyDeviceToHost);
  cudaFree(devdensity);

  // free uniform grid acceleration structure
  vmd_cuda_destroy_accel(xyzr_d, sorted_xyzr_d, cellStartEnd_d);

  err = cudaGetLastError();

  wkf_timer_stop(globaltimer);
  double totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  if (err != cudaSuccess) { 
    printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    return -1;
  }

  printf("MDFF absdiff GPU runtime: %.3f [sort: %.3f density %.3f copy: %.3f]\n", 
         totalruntime, sorttime, densitykerneltime-sorttime, 
         totalruntime-densitykerneltime);

  return 0;
}
#endif


static float gausslim_quality(int quality) {
  // set gaussian window size based on user-specified quality parameter
  float gausslim = 2.0f;
  switch (quality) {
    case 3: gausslim = 4.0f; break; // max quality

    case 2: gausslim = 3.0f; break; // high quality

    case 1: gausslim = 2.5f; break; // medium quality

    case 0:
    default: gausslim = 2.0f; // low quality
      break;
  }

  return gausslim;
}



static int calc_density_bounds(const AtomSel *sel, MoleculeList *mlist,
                               int verbose,
                               int quality, float radscale, float gridspacing,
                               float &maxrad, float *origin, 
                               float *xaxis, float *yaxis, float *zaxis,
                               int3 & volsz) {
  Molecule *m = mlist->mol_from_id(sel->molid());
  const float *atompos = sel->coordinates(mlist);
  const float *atomradii = m->extraflt.data("radius");

  vec_zero(origin);
  vec_zero(xaxis);
  vec_zero(yaxis);
  vec_zero(zaxis);
 
  // Query min/max atom radii for the entire molecule
  float minrad = 1.0f;
  maxrad = 1.0f;
  m->get_radii_minmax(minrad, maxrad);

  float fmin[3], fmax[3];
  minmax_selected_3fv_aligned(atompos, sel->on, sel->num_atoms,
                              sel->firstsel, sel->lastsel, fmin, fmax);

  float minx, miny, minz, maxx, maxy, maxz;
  minx = fmin[0]; miny = fmin[1]; minz = fmin[2];
  maxx = fmax[0]; maxy = fmax[1]; maxz = fmax[2];

  if (verbose) {
    printf("DenBBox: rminmax %f %f  min: %f %f %f  max: %f %f %f\n",
           minrad, maxrad, minx, miny, minz, maxx, maxy, maxz);
  }

  float mincoord[3], maxcoord[3];
  mincoord[0] = minx; mincoord[1] = miny; mincoord[2] = minz;
  maxcoord[0] = maxx; maxcoord[1] = maxy; maxcoord[2] = maxz;

  // crude estimate of the grid padding we require to prevent the
  // resulting isosurface from being clipped
  float gridpadding = radscale * maxrad * 1.70f;
  float padrad = gridpadding;
  padrad = 0.65f * sqrtf(4.0f/3.0f*((float) VMD_PI)*padrad*padrad*padrad);
  gridpadding = MAX(gridpadding, padrad);

  if (verbose) {
    printf("MDFFden: R*%.1f, H=%.1f Pad: %.1f minR: %.1f maxR: %.1f)\n",
           radscale, gridspacing, gridpadding, minrad, maxrad);
  }

  mincoord[0] -= gridpadding;
  mincoord[1] -= gridpadding;
  mincoord[2] -= gridpadding;
  maxcoord[0] += gridpadding;
  maxcoord[1] += gridpadding;
  maxcoord[2] += gridpadding;

  // compute the real grid dimensions from the selected atoms
  volsz.x = (int) ceil((maxcoord[0]-mincoord[0]) / gridspacing);
  volsz.y = (int) ceil((maxcoord[1]-mincoord[1]) / gridspacing);
  volsz.z = (int) ceil((maxcoord[2]-mincoord[2]) / gridspacing);

  // recalc the grid dimensions from rounded/padded voxel counts
  xaxis[0] = (volsz.x-1) * gridspacing;
  yaxis[1] = (volsz.y-1) * gridspacing;
  zaxis[2] = (volsz.z-1) * gridspacing;
  maxcoord[0] = mincoord[0] + xaxis[0];
  maxcoord[1] = mincoord[1] + yaxis[1];
  maxcoord[2] = mincoord[2] + zaxis[2];

  if (verbose) {
    printf("  GridSZ: (%4d %4d %4d)  BBox: (%.1f %.1f %.1f)->(%.1f %.1f %.1f)\n",
           volsz.x, volsz.y, volsz.z,
           mincoord[0], mincoord[1], mincoord[2],
           maxcoord[0], maxcoord[1], maxcoord[2]);
  }

  vec_copy(origin, mincoord);
  return 0;
}


static int map_uniform_spacing(double xax, double yax, double zax,
                               int szx, int szy, int szz) {
  // compute grid spacing values
  double xdelta = xax / (szx - 1);
  double ydelta = yax / (szy - 1);
  double zdelta = zax / (szz - 1);

  if (xdelta == ydelta == zdelta)
    return 1;

  // if we have less than 0.00001% relative error, that seems okay to treat
  // as "uniform" spacing too.
  const double relerrlim = 1e-7;
  double dxydelta = fabs(xdelta-ydelta); 
  double dyzdelta = fabs(ydelta-zdelta); 
  double dxzdelta = fabs(xdelta-zdelta); 
  if (((dxydelta / xdelta) < relerrlim) && ((dxydelta / xdelta) < relerrlim) &&
      ((dyzdelta / ydelta) < relerrlim) && ((dyzdelta / zdelta) < relerrlim) &&
      ((dxzdelta / xdelta) < relerrlim) && ((dxzdelta / zdelta) < relerrlim))
    return 1;
 
  return 0;
}
                           

static int calc_density_bounds_overlapping_map(int verbose, float &gridspacing,
                      float *origin, float *xaxis, float *yaxis, float *zaxis,
                      int3 &volsz, int3 &refoffset, 
                      const VolumetricData * refmap) {
  // compute grid spacing values
  float xdelta = xaxis[0] / (volsz.x - 1);
  float ydelta = yaxis[1] / (volsz.y - 1);
  float zdelta = zaxis[2] / (volsz.z - 1);

  if (verbose) {
    printf("calc_overlap: delta %f %f %f, gs: %f vsz: %d %d %d\n",
           xdelta, ydelta, zdelta, gridspacing, volsz.x, volsz.y, volsz.z);

    if (gridspacing != xdelta) {
      printf("calc_overlap: WARNING grid spacing != ref map spacing: (%f != %f)\n", gridspacing, xdelta);
    }
  }

  float refxdelta = refmap->xaxis[0] / (refmap->xsize - 1);
  float refydelta = refmap->yaxis[1] / (refmap->ysize - 1);
  float refzdelta = refmap->zaxis[2] / (refmap->zsize - 1);
  if (verbose) {
    printf("calc_overlap: refdelta %f %f %f, gs: %f vsz: %d %d %d\n",
           refxdelta, refydelta, refzdelta, gridspacing, 
           refmap->xsize, refmap->ysize, refmap->zsize);
  }

  // compute new origin, floor of the refmap nearest voxel coord
  float fvoxorigin[3];
  fvoxorigin[0] = (origin[0] - refmap->origin[0]) / refxdelta; 
  fvoxorigin[1] = (origin[1] - refmap->origin[1]) / refydelta; 
  fvoxorigin[2] = (origin[2] - refmap->origin[2]) / refzdelta; 

  refoffset.x = int((fvoxorigin[0] < 0) ? 0 : floor(fvoxorigin[0])); 
  refoffset.y = int((fvoxorigin[1] < 0) ? 0 : floor(fvoxorigin[1])); 
  refoffset.z = int((fvoxorigin[2] < 0) ? 0 : floor(fvoxorigin[2])); 
  if (verbose) {
    printf("calc_overlap: refoffset: %d %d %d\n", 
           refoffset.x, refoffset.y, refoffset.z);
  }

  float maxcorner[3];
  maxcorner[0] = origin[0] + xaxis[0];
  maxcorner[1] = origin[1] + yaxis[1];
  maxcorner[2] = origin[2] + zaxis[2];

  float refmaxcorner[3];
  refmaxcorner[0] = refmap->origin[0] + refmap->xaxis[0];
  refmaxcorner[1] = refmap->origin[1] + refmap->yaxis[1];
  refmaxcorner[2] = refmap->origin[2] + refmap->zaxis[2];

  maxcorner[0] = (maxcorner[0] > refmaxcorner[0]) ? refmaxcorner[0] : maxcorner[0]; 
  maxcorner[1] = (maxcorner[1] > refmaxcorner[1]) ? refmaxcorner[1] : maxcorner[1]; 
  maxcorner[2] = (maxcorner[2] > refmaxcorner[2]) ? refmaxcorner[2] : maxcorner[2]; 

  origin[0] = refmap->origin[0] + refoffset.x * refxdelta;
  origin[1] = refmap->origin[1] + refoffset.y * refydelta;
  origin[2] = refmap->origin[2] + refoffset.z * refzdelta;

  volsz.x = (int) round((maxcorner[0] - origin[0]) / refxdelta)+1;
  volsz.y = (int) round((maxcorner[1] - origin[1]) / refydelta)+1;
  volsz.z = (int) round((maxcorner[2] - origin[2]) / refzdelta)+1;

  // compute new corner
  xaxis[0] = ((volsz.x-1) * refxdelta);
  yaxis[1] = ((volsz.y-1) * refydelta);
  zaxis[2] = ((volsz.z-1) * refzdelta);

  return 0;
}



static float *build_xyzr_from_coords(const AtomSel *sel, const float *atompos,
                                     const float *atomradii, const float *origin) {
  // build compacted lists of bead coordinates, radii, and colors
  int ind = sel->firstsel * 3;
  int ind4=0;

  float *xyzr = (float *) malloc(sel->selected * sizeof(float) * 4);

  // build compacted lists of atom coordinates and radii only
  int i;
  for (i=sel->firstsel; i <= sel->lastsel; i++) {
    if (sel->on[i]) {
      const float *fp = atompos + ind;
      xyzr[ind4    ] = fp[0]-origin[0];
      xyzr[ind4 + 1] = fp[1]-origin[1];
      xyzr[ind4 + 2] = fp[2]-origin[2];
      xyzr[ind4 + 3] = atomradii[i];
      ind4 += 4;
    }
    ind += 3;
  }

  return xyzr;
}
                               


static float *build_xyzr_from_sel(const AtomSel *sel, MoleculeList *mlist,
                                  const float *origin) {
  Molecule *m = mlist->mol_from_id(sel->molid());
  const float *atompos = sel->coordinates(mlist);
  const float *atomradii = m->extraflt.data("radius");

  float *xyzr = build_xyzr_from_coords(sel, atompos, atomradii, origin);

  return xyzr;
}


//
// Compute a simulated density map for a given atom selection and
// reference density map.
//
int vmd_cuda_calc_density(const AtomSel *sel, MoleculeList *mlist, 
                          int quality, float radscale, float gridspacing, 
                          VolumetricData ** synthvol, 
                          const VolumetricData * refmap,
                          VolumetricData ** diffvol, 
                          int verbose) {
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

  float maxrad, acorigin[3], origin[3], xaxis[3], yaxis[3], zaxis[3];
  int3 volsz   = make_int3(1, 1, 1);
  int3 refsz   = make_int3(1, 1, 1);
  int3 acvolsz = make_int3(1, 1, 1);
  int3 refoffset = make_int3(0, 0, 0);

  if (refmap) {
    gridspacing = refmap->xaxis[0] / (refmap->xsize - 1);
    if (verbose) {
      printf("refmap gridspacing: %f\n", gridspacing);
    }
  } 

  calc_density_bounds(sel, mlist, verbose, quality, radscale, gridspacing,
                      maxrad, origin, xaxis, yaxis, zaxis, volsz);
  vec_copy(acorigin, origin);
  acvolsz = volsz;

  if (verbose) {
    printf("dmap init orig: %f %f %f  axes: %f %f %f  sz: %d %d %d\n",
           origin[0], origin[1], origin[2],
           xaxis[0], yaxis[1], zaxis[2], 
           volsz.x, volsz.y, volsz.z);
  }

  if (refmap) {
    calc_density_bounds_overlapping_map(verbose, gridspacing, origin, 
                                        xaxis, yaxis, zaxis, volsz, 
                                        refoffset, refmap);

    if (verbose) {
      printf("ref  dvol orig: %f %f %f  axes: %f %f %f  sz: %d %d %d\n",
             refmap->origin[0], refmap->origin[1], refmap->origin[2],
             refmap->xaxis[0], refmap->yaxis[1], refmap->zaxis[2], 
             refmap->xsize, refmap->ysize, refmap->zsize);

      printf("dmap rmap orig: %f %f %f  axes: %f %f %f  sz: %d %d %d\n",
             origin[0], origin[1], origin[2],
             xaxis[0], yaxis[1], zaxis[2], 
             volsz.x, volsz.y, volsz.z);
    }
  }

  float3 acoriginoffset = make_float3(origin[0] - acorigin[0],
                                      origin[1] - acorigin[1],
                                      origin[2] - acorigin[2]);

  float *xyzr = build_xyzr_from_sel(sel, mlist, acorigin);
  float gausslim = gausslim_quality(quality); // set gaussian cutoff radius

  double pretime = wkf_timer_timenow(timer);
  float *volmap = new float[volsz.x*volsz.y*volsz.z];

  if (!diffvol) {
    vmd_cuda_gaussdensity_calc(sel->selected, acoriginoffset, acvolsz,
                               xyzr, volmap, volsz, 
                               maxrad, radscale, gridspacing, gausslim,
                               refsz, refoffset, NULL, NULL,
                               verbose);
  } else {
    // emit a difference map
    float *diffmap = new float[volsz.x*volsz.y*volsz.z];

    refsz = make_int3(refmap->xsize, refmap->ysize, refmap->zsize);
    vmd_cuda_gaussdensity_calc(sel->selected, acoriginoffset, acvolsz,
                               xyzr, volmap, volsz, 
                               maxrad, radscale, gridspacing, gausslim,
                               refsz, refoffset, refmap->data, diffmap,
                               verbose);


    char diffdataname[64] = { "mdff density difference map" };
    *diffvol = new VolumetricData(diffdataname, origin, xaxis, yaxis, zaxis,
                                  volsz.x, volsz.y, volsz.z, diffmap);
  }

  char dataname[64] = { "mdff synthetic density map" };
  *synthvol = new VolumetricData(dataname, origin, xaxis, yaxis, zaxis,
                              volsz.x, volsz.y, volsz.z, volmap);

  double voltime = wkf_timer_timenow(timer);
  free(xyzr);

  if (verbose) {
    char strmsg[1024];
    sprintf(strmsg, "MDFFcc: %.3f [pre:%.3f vol:%.3f]",
            voltime, pretime, voltime-pretime);
    msgInfo << strmsg << sendmsg;
  }
  wkf_timer_destroy(timer);

  return 0; 
}



#if 0
static int vmd_calc_cc(const AtomSel *sel, MoleculeList *mlist, 
                       int quality, float radscale, float gridspacing, 
                       const VolumetricData * refmap,
                       int verbose, float ccthreshdensity, float &result_cc) {
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return -1;

  if (!refmap)
    return -1;

  float maxrad, acorigin[3], origin[3], xaxis[3], yaxis[3], zaxis[3];
  int3 volsz   = make_int3(1, 1, 1);
  int3 refsz   = make_int3(1, 1, 1);
  int3 acvolsz = make_int3(1, 1, 1);
  int3 refoffset = make_int3(0, 0, 0);

  gridspacing = refmap->xaxis[0] / (refmap->xsize - 1);
  calc_density_bounds(sel, mlist, verbose, quality, radscale, gridspacing,
                      maxrad, origin, xaxis, yaxis, zaxis, volsz);
  vec_copy(acorigin, origin);
  acvolsz = volsz;

  calc_density_bounds_overlapping_map(verbose, gridspacing, origin, 
                                      xaxis, yaxis, zaxis, volsz, 
                                      refoffset, refmap);

  float3 acoriginoffset = make_float3(origin[0] - acorigin[0],
                                      origin[1] - acorigin[1],
                                      origin[2] - acorigin[2]);

  float *xyzr = build_xyzr_from_sel(sel, mlist, acorigin);
  float gausslim = gausslim_quality(quality); // set gaussian cutoff radius

  double pretime = wkf_timer_timenow(timer);
  refsz = make_int3(refmap->xsize, refmap->ysize, refmap->zsize);
  
  if (refoffset.x >= refsz.x || refoffset.y >= refsz.y || refoffset.z >= refsz.z) {
    printf("MDFF CC: no overlap between synthetic map and reference map!\n");
    return -1;
  }

  // copy the reference density map to the GPU before we begin using it for
  // subsequent calculations...
  float *devrefmap = NULL;
  unsigned long refmemsz = refsz.x * refsz.y * refsz.z * sizeof(float);
  if (cudaMalloc((void**)&devrefmap, refmemsz) != cudaSuccess) {
    err = cudaGetLastError(); // eat error so next CUDA op succeeds
    return -1;
  }
  cudaMemcpy(devrefmap, refmap->data, refmemsz, cudaMemcpyHostToDevice);

  result_cc = 0.0f;
  vmd_cuda_cc_calc(sel->selected, acoriginoffset, acvolsz, xyzr, 
                   volsz, maxrad, radscale, gridspacing, 
                   gausslim, refsz, refoffset, devrefmap, 
                   ccthreshdensity, result_cc, origin, NULL,
                   verbose);
  cudaFree(devrefmap);

  double cctime = wkf_timer_timenow(timer);
  free(xyzr);

  if (verbose) {
    char strmsg[1024];
    sprintf(strmsg, "MDFFcc: %.3f [pre:%.3f cc:%.3f]",
            cctime, pretime, cctime-pretime);
    msgInfo << strmsg << sendmsg;
  }
  wkf_timer_destroy(timer);

  return 0;
}

#endif


//
// Fast single-pass algorithm for computing a synthetic density map
// for a given atom selection and reference map, and compute the 
// cross correlation, optionally saving a spatially localized
// cross correlation map, difference map, and simulated density map.
//
int vmd_cuda_compare_sel_refmap(const AtomSel *sel, MoleculeList *mlist, 
                                int quality, float radscale, float gridspacing, 
                                const VolumetricData * refmap,
                                VolumetricData **synthvol, 
                                VolumetricData **diffvol, 
                                VolumetricData **spatialccvol, 
                                float *CC, float ccthreshdensity,
                                int verbose) {
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

  if (!map_uniform_spacing(refmap->xaxis[0], refmap->yaxis[1], refmap->zaxis[2],
                           refmap->xsize, refmap->ysize, refmap->zsize)) {
    if (verbose)
      printf("mdffi cc: non-uniform grid spacing unimplemented on GPU, falling back to CPU!\n");

    return -1;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return -1;

  if (!refmap)
    return -1;

  float maxrad, acorigin[3], origin[3], xaxis[3], yaxis[3], zaxis[3];
  int3 volsz   = make_int3(1, 1, 1);
  int3 refsz   = make_int3(1, 1, 1);
  int3 acvolsz = make_int3(1, 1, 1);
  int3 refoffset = make_int3(0, 0, 0);

  if (verbose) {
    printf("vmd_cuda_compare_sel_refmap():\n");
    printf("  refmap xaxis: %f %f %f   ",
           refmap->xaxis[0], 
           refmap->xaxis[1], 
           refmap->xaxis[2]);
    printf("  refmap size: %d %d %d\n",
           refmap->xsize, refmap->ysize, refmap->zsize);
    printf("  gridspacing (orig): %f  ", gridspacing);
  } 
  gridspacing = refmap->xaxis[0] / (refmap->xsize - 1);
  if (verbose) {
    printf("(new): %f\n", gridspacing);
  }

  // if we get a bad computed grid spacing, we have to bail out or
  // we will end up segfaulting...
  if (gridspacing == 0.0) {
    if (verbose)
      printf("GPU gridspacing is zero! bailing out!\n");
    return -1;
  }

  calc_density_bounds(sel, mlist, verbose, quality, radscale, gridspacing,
                      maxrad, origin, xaxis, yaxis, zaxis, volsz);
  vec_copy(acorigin, origin);
  acvolsz = volsz;

  calc_density_bounds_overlapping_map(verbose, gridspacing, origin, 
                                      xaxis, yaxis, zaxis, volsz, 
                                      refoffset, refmap);

  float3 acoriginoffset = make_float3(origin[0] - acorigin[0],
                                      origin[1] - acorigin[1],
                                      origin[2] - acorigin[2]);

  float *xyzr = build_xyzr_from_sel(sel, mlist, acorigin);
  float gausslim = gausslim_quality(quality); // set gaussian cutoff radius

  double pretime = wkf_timer_timenow(timer);
  refsz = make_int3(refmap->xsize, refmap->ysize, refmap->zsize);

  if (refoffset.x >= refsz.x || refoffset.y >= refsz.y || refoffset.z >= refsz.z) {
    printf("MDFF CC: no overlap between synthetic map and reference map!\n");
    return -1;
  }

  // copy the reference density map to the GPU before we begin using it for
  // subsequent calculations...
  float *devrefmap = NULL;
  unsigned long refmemsz = refsz.x * refsz.y * refsz.z * sizeof(float);
  if (cudaMalloc((void**)&devrefmap, refmemsz) != cudaSuccess) {
    err = cudaGetLastError(); // eat error so next CUDA op succeeds
    return -1;
  }
  cudaMemcpy(devrefmap, refmap->data, refmemsz, cudaMemcpyHostToDevice);

  if (CC != NULL) {
    vmd_cuda_cc_calc(sel->selected, acoriginoffset, acvolsz, xyzr, 
                     volsz, maxrad, radscale, gridspacing, 
                     gausslim, refsz, refoffset, devrefmap,
                     ccthreshdensity, *CC, origin, spatialccvol,
                     verbose);
  }

  // generate/save synthetic density map if requested
  float *synthmap = NULL;
  float *diffmap = NULL;
  if (synthvol != NULL || diffvol != NULL) {
    if (synthvol != NULL)
      synthmap = new float[volsz.x*volsz.y*volsz.z];

    if (diffvol != NULL)
      diffmap = new float[volsz.x*volsz.y*volsz.z];

    vmd_cuda_gaussdensity_calc(sel->selected, acoriginoffset, acvolsz, xyzr,
                               synthmap, volsz, maxrad, radscale, gridspacing,
                               gausslim, refsz, refoffset, refmap->data, diffmap,
                               verbose);
  }


  if (synthmap != NULL) {
    char mapname[64] = { "mdff synthetic density map" };
    *synthvol = new VolumetricData(mapname, origin, xaxis, yaxis, zaxis,
                                   volsz.x, volsz.y, volsz.z, synthmap);
  }

  if (diffmap != NULL) {
    char mapname[64] = { "MDFF density difference map" };
    *diffvol = new VolumetricData(mapname, origin, xaxis, yaxis, zaxis,
                                  volsz.x, volsz.y, volsz.z, diffmap);
  }

  cudaFree(devrefmap);

  double cctime = wkf_timer_timenow(timer);
  free(xyzr);

  if (verbose) {
    char strmsg[1024];
    sprintf(strmsg, "MDFF comp: %.3f [pre:%.3f cc:%.3f]",
            cctime, pretime, cctime-pretime);
    msgInfo << strmsg << sendmsg;
  }
  wkf_timer_destroy(timer);

  return 0; 
}


#if 0

static int vmd_test_sumabsdiff(const AtomSel *sel, MoleculeList *mlist, 
                               int quality, float radscale, float gridspacing, 
                               const VolumetricData * refmap, int verbose) {
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

  float maxrad, origin[3], xaxis[3], yaxis[3], zaxis[3];
  int3 volsz = make_int3(1, 1, 1);
  int3 refoffset = make_int3(0, 0, 0);

  calc_density_bounds(sel, mlist, verbose, quality, radscale, gridspacing,
                      maxrad, origin, xaxis, yaxis, zaxis, volsz);

  if (refmap) {
    printf("dmap init orig: %f %f %f  axes: %f %f %f  sz: %d %d %d\n",
           origin[0], origin[1], origin[2],
           xaxis[0], yaxis[1], zaxis[2], 
           volsz.x, volsz.y, volsz.z);

    calc_density_bounds_overlapping_map(1, gridspacing, origin, 
                                        xaxis, yaxis, zaxis, volsz, 
                                        refoffset, refmap);

    printf("ref  dvol orig: %f %f %f  axes: %f %f %f  sz: %d %d %d\n",
           refmap->origin[0], refmap->origin[1], refmap->origin[2],
           refmap->xaxis[0], refmap->yaxis[1], refmap->zaxis[2], 
           refmap->xsize, refmap->ysize, refmap->zsize);

    printf("dmap rmap orig: %f %f %f  axes: %f %f %f  sz: %d %d %d\n",
           origin[0], origin[1], origin[2],
           xaxis[0], yaxis[1], zaxis[2], 
           volsz.x, volsz.y, volsz.z);
  }

  float *xyzr = build_xyzr_from_sel(sel, mlist, origin);
  float gausslim = gausslim_quality(quality); // set gaussian cutoff radius

  double pretime = wkf_timer_timenow(timer);

#if 0
  float *volmap = new float[volsz.x*volsz.y*volsz.z];
  char dataname[64] = { "mdff synthetic density map" };

  vmd_cuda_gaussdensity_calc(sel->selected, xyzr, volmap, volsz, 
                             maxrad, radscale, gridspacing, gausslim, verbose);

  *synthvol = new VolumetricData(dataname, origin, xaxis, yaxis, zaxis,
                               volsz.x, volsz.y, volsz.z, volmap);
#endif

  double voltime = wkf_timer_timenow(timer);
  free(xyzr);

  if (verbose) {
    char strmsg[1024];
    sprintf(strmsg, "MDFFcc: %.3f [pre:%.3f vol:%.3f]",
            voltime, pretime, voltime-pretime);
    msgInfo << strmsg << sendmsg;
  }
  wkf_timer_destroy(timer);

  return 0; 
}
#endif




