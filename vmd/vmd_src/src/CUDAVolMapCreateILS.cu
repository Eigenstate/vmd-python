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
 *      $RCSfile: CUDAVolMapCreateILS.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.52 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This source file contains CUDA kernels for calculating the
 * occupancy map for implicit ligand sampling.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h"
#include "utilities.h"

//#define DEBUG 1

#if defined(VMDTHREADS)
#define USE_CUDADEVPOOL 1
#endif

#if 1
#define CUERR \
  do { \
    cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
      printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      printf("Thread aborting...\n"); \
      return NULL; \
    } \
  } while (0)
// to be terminated with semi-colon
#else
#define CUERR
#endif

//#define DEBUG

typedef union flint_t {
  float f;
  int i;
} flint;


//
// algorithm optimizations
//

// Computing distance-based exclusions either along with or instead of
// energy-based exclusions turns out to decrease performance.
//#define USE_DISTANCE_EXCL  // use the distance-based exclusion kernel
#define USE_ENERGY_EXCL    // use the energy-based exclusion kernel

#ifdef USE_DISTANCE_EXCL
#define TEST_BLOCK_EXCL    // early exit if entire thread block is excluded
// don't enable unless we have distance-based exclusions
#endif

#if defined(USE_ENERGY_EXCL) || defined(USE_DISTANCE_EXCL)
#define TEST_MBLOCK_EXCL     // early exit from multiatom kernel
#define TEST_SHIFT_EXCL
#endif

#ifdef TEST_SHIFT_EXCL
#undef TEST_MBLOCK_EXCL
#endif


#define MAX_THREADBLOCKS  65536

#define BIN_DEPTH     8  // number of slots per bin
#define BIN_SLOTSIZE  4  // slot permits x, y, z, vdwtype  (4 elements)
#define BIN_SIZE      (BIN_DEPTH * BIN_SLOTSIZE)  // size given in "flints"
#define BIN_SHIFT     5  // log2(BIN_SIZE)
#define BIN_SLOTSHIFT 2  // log2(BIN_SLOTSIZE)

typedef struct occthrparms_t {
  int errcode;                   // 0 for success, -1 for failure

  int mx, my, mz;                // map dimensions
  float *map;                    // buffer space for occupancy map
                                 // (length mx*my*mz floats)

  float max_energy;              // max energy threshold
  float cutoff;                  // vdw cutoff distance
  float min_excldist;            // min exclusion distance
  float hx, hy, hz;              // map lattice spacing
  float x0, y0, z0;              // map origin
  float bx_1, by_1, bz_1;        // inverse of atom bin lengths

  int nbx, nby, nbz;             // bin dimensions
  const flint *bin;              // atom bins
                                 // (length BIN_SIZE*nbx*nby*nbz)
  const flint *bin_zero;         // bin pointer shifted to origin

  int num_binoffsets;            // number of offsets
  const char *binoffsets;        // bin neighborhood index offsets
                                 // (length 3*num_bin_offsets)

  int num_extras;                // number of extra atoms
  const flint *extra;            // extra atoms from overfilled bins
                                 // (length BIN_SLOTSIZE*num_extras)

  int num_vdwparms;              // number of vdw parameter types
  const float *vdwparms;         // vdw parameters
                                 // (length 2*num_vdw_params)

  int num_probes;                // number of probe atoms
  const float *probevdwparms;    // vdw parameters of probe atoms
                                 // (length 2*num_probes)

  int num_conformers;            // number of conformers
  const float *conformers;       // probe atom offsets for conformers
                                 // (length 3*num_probes*num_conformers)
} occthrparms;


// constant memory, page 1 (8K)
//
__constant__ static float const_max_energy;     // energy threshold
__constant__ static float const_min_occupancy;  // occupancy threshold
__constant__ static float const_cutoff;         // cutoff distance
__constant__ static float const_min_excldist;   // min exclusion distance
__constant__ static float const_hx, const_hy, const_hz;  // map spacings
__constant__ static float const_x0, const_y0, const_z0;  // map origin
__constant__ static float const_bx_1, const_by_1, const_bz_1;
                                                // inverse of bin lengths
__constant__ static float const_inv_numconf;    // inverse number conformers

__constant__ static int const_num_binoffsets;   // need use lengths of arrays
__constant__ static int const_num_extras;
__constant__ static int const_num_probes;
__constant__ static int const_num_conformers;

#define MAX_VDWPARMS  160
__constant__ static float const_vdwparms[2 * MAX_VDWPARMS];

#define MAX_PROBES  8
__constant__ static float const_probevdwparms[2 * MAX_PROBES];

#define MAX_EXTRAS  50
__constant__ static flint const_extra[BIN_SLOTSIZE * MAX_EXTRAS];

// each offset is a single flat index offset into atom bin array
#define MAX_BINOFFSETS  1467
__constant__ static int const_binoffsets[MAX_BINOFFSETS];

// nearest neighbor list for distance-based exclusions
#define MAX_EXCLOFFSETS  27
__constant__ static int const_excloffsets[MAX_EXCLOFFSETS];

// constant memory, pages 2 - 7
//
// the max number of conformers is chosen to leave 1K available
// to the runtime library
//
#define MAX_CONFORMERS 586
__constant__ static float const_conformers[3*MAX_PROBES*MAX_CONFORMERS];


#define DO_ONE_PLANE         // optimze when z-slabs are slices of thickness 1
#undef DO_ONE_PLANE

#define THBLEN   4           // thread block length of cube
#define NTHREADSPERBLOCK  (THBLEN*THBLEN*THBLEN)

#define BSHIFT   2           // shift for map points to blocks = log2(THBLEN)

#define NTPBSHIFT  6         // do bit shifting instead of int multiplies
                             //  = log2(NTHREADPERBLOCK)

// Performance is improved for exclusion optimizations
// if we reduce shared memory use to < 4K
#if defined(TEST_MBLOCK_EXCL) || defined(TEST_SHIFT_EXCL)
#define NCONFPERLOOP       5
#else
#define NCONFPERLOOP       8
#endif

#define NTPB_TIMES_NCPL  (NTHREADSPERBLOCK * NCONFPERLOOP)

#define PADMASK  (THBLEN-1)  // bit mask for padding up to multiple of THBLEN

#define MAX_ALLOWABLE_ENERGY  87.f  // otherwise expf(-u) gives 0


///////////////////////////////////////////////////////////////////////////////
//
//     ROUTINES FOR EXCLUSIONS
//
///////////////////////////////////////////////////////////////////////////////

#ifdef USE_DISTANCE_EXCL
__global__ static void cuda_find_distance_exclusions(
    int *excl,              // exclusions, stored in contiguous thread blocks
    const flint *bin_zero,  // points to bin containing map origin
    int nbx, int nby,       // number of bins along x and y dimensions
#ifndef DO_ONE_PLANE
    int mzblocks,           // number of thread blocks in z dimension
#endif
    int mzblockoffset       // offset (number of planes) to starting z
    ) {

  // space for 1 atom bin in shared memory
  __shared__ flint atombin[BIN_SIZE];

#ifndef DO_ONE_PLANE
  const int nblockx = gridDim.x;
  const int nblocky = gridDim.y / mzblocks;

  // 3D index for this thread block
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y % nblocky;
  const int bidz = blockIdx.y / nblocky + mzblockoffset;

  // block ID, flat index
  const int bid = (bidz * nblocky + bidy)*nblockx + bidx;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (bidx<<BSHIFT) * const_hx;
  float py = (bidy<<BSHIFT) * const_hy;
  float pz = (bidz<<BSHIFT) * const_hz;
#else
  // can optimize from above for case when mzblocks = 1

  // block ID, flat index
  const int bid
    = (mzblockoffset * gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (blockIdx.x<<BSHIFT) * const_hx;
  float py = (blockIdx.y<<BSHIFT) * const_hy;
  float pz = (mzblockoffset<<BSHIFT) * const_hz;
#endif

  const int mindex = (bid<<NTPBSHIFT) + tid;  // index into map and excl
  int isexcluded = 0;  // assume it isn't excluded

  const float mindis2 = const_min_excldist * const_min_excldist;

  const int ib = (int) floorf(px * const_bx_1);  // bin index for entire
  const int jb = (int) floorf(py * const_by_1);  // thread block
  const int kb = (int) floorf(pz * const_bz_1);

  const int binid = (kb * nby + jb)*nbx + ib;    // flat index of my bin

  int n, nb;

  px += threadIdx.x * const_hx;  // adjust position for this thread
  py += threadIdx.y * const_hy;
  pz += threadIdx.z * const_hz;

  px += const_x0;
  py += const_y0;
  pz += const_z0;  // translate to absolute position

  const int num_extra_slots = const_num_extras << BIN_SLOTSHIFT;

  for (nb = 0;  nb < MAX_EXCLOFFSETS;  nb++) {

    if (tid < BIN_SIZE) {  // cache next atom bin
      const flint *p_bin = bin_zero
        + ((binid + const_excloffsets[nb])<<BIN_SHIFT);
      atombin[tid] = p_bin[tid];  // copy from global to shared memory
    }
    __syncthreads();  // now we have atom bin loaded

    for (n = 0;  n < BIN_SIZE;  n += BIN_SLOTSIZE) { // loop over atoms in bin
      
      if (-1 == atombin[n+3].i) break;  // no more atoms in bin
      const float dx = px - atombin[n  ].f;
      const float dy = py - atombin[n+1].f;
      const float dz = pz - atombin[n+2].f;
      const float r2 = dx*dx + dy*dy + dz*dz;

      if (r2 <= mindis2) isexcluded = 1;

    } // end loop over atoms in bin

    __syncthreads();  // wait for thread block before loading next bin
  } // end loop over bin neighborhood

  for (n = 0;  n < num_extra_slots;  n += BIN_SLOTSIZE) {  // extra atoms
    const float dx = px - const_extra[n  ].f;
    const float dy = py - const_extra[n+1].f;
    const float dz = pz - const_extra[n+2].f;
    const float r2 = dx*dx + dy*dy + dz*dz;

    if (r2 <= mindis2) isexcluded = 1;

  } // end loop over extra atoms

  excl[mindex] = isexcluded;
}
#endif // USE_DISTANCE_EXCL


#ifdef USE_ENERGY_EXCL
__global__ static void cuda_find_energy_exclusions(
    int *excl,              // exclusions, stored in contiguous thread blocks
    const flint *bin_zero,  // points to bin containing map origin
    int nbx, int nby,       // number of bins along x and y dimensions
#ifndef DO_ONE_PLANE
    int mzblocks,           // number of thread blocks in z dimension
#endif
    int mzblockoffset       // offset (number of planes) to starting z
    ) {

  // space for 1 atom bin in shared memory
  __shared__ flint atombin[BIN_SIZE];

#ifdef TEST_BLOCK_EXCL
#if 0
  __shared__ int sh_isall_upper_warp;
#endif
  __shared__ int sh_sum[NTHREADSPERBLOCK];
#endif

#ifndef DO_ONE_PLANE
  const int nblockx = gridDim.x;
  const int nblocky = gridDim.y / mzblocks;

  // 3D index for this thread block
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y % nblocky;
  const int bidz = blockIdx.y / nblocky + mzblockoffset;

  // block ID, flat index
  const int bid = (bidz * nblocky + bidy)*nblockx + bidx;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (bidx<<BSHIFT) * const_hx;
  float py = (bidy<<BSHIFT) * const_hy;
  float pz = (bidz<<BSHIFT) * const_hz;
#else
  // can optimize from above for case when mzblocks = 1

  // block ID, flat index
  const int bid
    = (mzblockoffset * gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (blockIdx.x<<BSHIFT) * const_hx;
  float py = (blockIdx.y<<BSHIFT) * const_hy;
  float pz = (mzblockoffset<<BSHIFT) * const_hz;
#endif

  const int mindex = (bid<<NTPBSHIFT) + tid;  // index into map and excl
#ifdef USE_DISTANCE_EXCL
  int isexcluded = excl[mindex];  // read exclusion from global memory
#else
  int isexcluded = 0;  // assume it isn't excluded
#endif

#ifdef TEST_BLOCK_EXCL
#if 0
  // warp vote is available only in devices 1.2 and later
  // warp vote to see if we continue
  int isall = __all(isexcluded);
  if (32==tid) sh_isall_upper_warp = isall;
  if (0==isany+sh_isall_upper_warp) return; // all points excluded, exit
#endif
  sh_sum[tid] = !(isexcluded);  // count inclusions
  int nbrsum = sh_sum[tid^1];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^2];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^4];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^8];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^16];
  sh_sum[tid] += nbrsum;
  __syncthreads();
  nbrsum = sh_sum[tid^32];
  __syncthreads();
  sh_sum[tid] += nbrsum;

  if (0==sh_sum[tid]) return;  // all points excluded, exit
#endif

  const float cutoff2 = const_cutoff * const_cutoff;  // cutoff2 in register
  const float probe_vdweps = const_probevdwparms[0];
  const float probe_vdwrmin = const_probevdwparms[1];

  const int ib = (int) floorf(px * const_bx_1);  // bin index for entire
  const int jb = (int) floorf(py * const_by_1);  // thread block
  const int kb = (int) floorf(pz * const_bz_1);

  const int binid = (kb * nby + jb)*nbx + ib;    // flat index of my bin

  int n, nb;

  px += threadIdx.x * const_hx;  // adjust position for this thread
  py += threadIdx.y * const_hy;
  pz += threadIdx.z * const_hz;

  px += const_x0;
  py += const_y0;
  pz += const_z0;  // translate to absolute position

  float u = 0.f;  // sum potential energy

  const int num_extra_slots = const_num_extras << BIN_SLOTSHIFT;

  for (nb = 0;  nb < const_num_binoffsets;  nb++) {

    if (tid < BIN_SIZE) {  // cache next atom bin
      const flint *p_bin = bin_zero
        + ((binid + const_binoffsets[nb])<<BIN_SHIFT);
      atombin[tid] = p_bin[tid];  // copy from global to shared memory
    }
    __syncthreads();  // now we have atom bin loaded

    for (n = 0;  n < BIN_SIZE;  n += BIN_SLOTSIZE) { // loop over atoms in bin
      int vdwtype = atombin[n+3].i;
      if (-1 == vdwtype) break;  // no more atoms in bin
      vdwtype <<= 1;  // multiply by 2 for indexing VDW parms

      const float dx = px - atombin[n  ].f;
      const float dy = py - atombin[n+1].f;
      const float dz = pz - atombin[n+2].f;
      const float r2 = dx*dx + dy*dy + dz*dz;
      if (r2 < cutoff2) {  // atom within cutoff, accumulate energy
        const float epsilon = const_vdwparms[vdwtype] * probe_vdweps;
        const float rmin = const_vdwparms[vdwtype+1] + probe_vdwrmin;
        float rm6 = rmin*rmin / r2;
        rm6 = rm6 * rm6 * rm6;
        u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
      }
    } // end loop over atoms in bin

    __syncthreads();  // wait for thread block before loading next bin
  } // end loop over bin neighborhood

  for (n = 0;  n < num_extra_slots;  n += BIN_SLOTSIZE) {  // extra atoms
    const int vdwtype = const_extra[n+3].i << 1; // mult by 2 index VDW parms
    const float dx = px - const_extra[n  ].f;
    const float dy = py - const_extra[n+1].f;
    const float dz = pz - const_extra[n+2].f;
    const float r2 = dx*dx + dy*dy + dz*dz;
    if (r2 < cutoff2) {  // atom within cutoff, accumulate energy
      const float epsilon = const_vdwparms[vdwtype] * probe_vdweps;
      const float rmin = const_vdwparms[vdwtype + 1] + probe_vdwrmin;
      float rm6 = rmin*rmin / r2;
      rm6 = rm6 * rm6 * rm6;
      u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
    }
  } // end loop over extra atoms

  if (u >= const_max_energy) isexcluded = 1;  // exclude this map point

  excl[mindex] = isexcluded;
}
#endif // USE_ENERGY_EXCL


// For a monoatomic probe compute the occupancy rho
// (probability of finding the probe)
//
// For each map point the occupancy is computed as
//
//   rho = exp(-U)
//
// where U is the interaction energy of the probe with the system
// due to the VDW force field.
//
__global__ static void cuda_occupancy_monoatom(
    float *map,             // map buffer, stored in contiguous thread blocks
    const int *excl,        // exclusion buffer, stored same
    const flint *bin_zero,  // points to bin containing map origin
    int nbx, int nby,       // number of bins along x and y dimensions
#ifndef DO_ONE_PLANE
    int mzblocks,           // number of thread blocks in z dimension
#endif
    int mzblockoffset       // offset (number of planes) to starting z
    ) {

  // space for 1 atom bin in shared memory
  __shared__ flint atombin[BIN_SIZE];

#ifdef TEST_BLOCK_EXCL
#if 0
  __shared__ int sh_isall_upper_warp;
#endif
  __shared__ int sh_sum[NTHREADSPERBLOCK];
#endif

#ifndef DO_ONE_PLANE
  const int nblockx = gridDim.x;
  const int nblocky = gridDim.y / mzblocks;

  // 3D index for this thread block
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y % nblocky;
  const int bidz = blockIdx.y / nblocky + mzblockoffset;

  // block ID, flat index
  const int bid = (bidz * nblocky + bidy)*nblockx + bidx;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (bidx<<BSHIFT) * const_hx;
  float py = (bidy<<BSHIFT) * const_hy;
  float pz = (bidz<<BSHIFT) * const_hz;
#else
  // can optimize from above for case when mzblocks = 1

  // block ID, flat index
  const int bid
    = (mzblockoffset * gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (blockIdx.x<<BSHIFT) * const_hx;
  float py = (blockIdx.y<<BSHIFT) * const_hy;
  float pz = (mzblockoffset<<BSHIFT) * const_hz;
#endif

  const int mindex = (bid<<NTPBSHIFT) + tid;  // index into map and excl
  const int isexcluded = excl[mindex];  // read exclusion from global memory

#ifdef TEST_BLOCK_EXCL
#if 0
  // warp vote is available only in devices 1.2 and later
  // warp vote to see if we continue
  int isall = __all(isexcluded);
  if (32==tid) sh_isall_upper_warp = isall;
  if (0==isany+sh_isall_upper_warp) return; // all points excluded, exit
#endif
  sh_sum[tid] = !(isexcluded);  // count inclusions
  int nbrsum = sh_sum[tid^1];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^2];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^4];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^8];
  sh_sum[tid] += nbrsum;
  nbrsum = sh_sum[tid^16];
  sh_sum[tid] += nbrsum;
  __syncthreads();
  nbrsum = sh_sum[tid^32];
  __syncthreads();
  sh_sum[tid] += nbrsum;

  if (0==sh_sum[tid]) return;  // all points excluded, exit
#endif

  const float cutoff2 = const_cutoff * const_cutoff;  // cutoff2 in register
  const float probe_vdweps = const_probevdwparms[0];
  const float probe_vdwrmin = const_probevdwparms[1];

  const int ib = (int) floorf(px * const_bx_1);  // bin index for entire
  const int jb = (int) floorf(py * const_by_1);  // thread block
  const int kb = (int) floorf(pz * const_bz_1);

  const int binid = (kb * nby + jb)*nbx + ib;    // flat index of my bin

  int n, nb;

  px += threadIdx.x * const_hx;  // adjust position for this thread
  py += threadIdx.y * const_hy;
  pz += threadIdx.z * const_hz;

  px += const_x0;
  py += const_y0;
  pz += const_z0;  // translate to absolute position

  float u = 0.f;  // sum potential energy

  const int num_extra_slots = const_num_extras << BIN_SLOTSHIFT;

  for (nb = 0;  nb < const_num_binoffsets;  nb++) {  // bin neighborhood

    if (tid < BIN_SIZE) {  // cache next atom bin
      const flint *p_bin = bin_zero
        + ((binid + const_binoffsets[nb])<<BIN_SHIFT);
      atombin[tid] = p_bin[tid];  // copy from global to shared memory
    }
    __syncthreads();  // now we have atom bin loaded

    for (n = 0;  n < BIN_SIZE;  n += BIN_SLOTSIZE) { // loop over atoms in bin
      int vdwtype = atombin[n+3].i;
      if (-1 == vdwtype) break;  // no more atoms in bin
      vdwtype <<= 1;  // multiply by 2 for indexing VDW parms

      const float dx = px - atombin[n  ].f;
      const float dy = py - atombin[n+1].f;
      const float dz = pz - atombin[n+2].f;
      const float r2 = dx*dx + dy*dy + dz*dz;
      if (r2 < cutoff2) {  // atom within cutoff, accumulate energy
        const float epsilon = const_vdwparms[vdwtype] * probe_vdweps;
        const float rmin = const_vdwparms[vdwtype+1] + probe_vdwrmin;
        float rm6 = rmin*rmin / r2;
        rm6 = rm6 * rm6 * rm6;
        u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
      }
    } // end loop atoms in bin

    __syncthreads();  // wait for entire thread block before loading next bin
  } // end loop over bin neighborhood

  for (n = 0;  n < num_extra_slots;  n += BIN_SLOTSIZE) {  // extra atoms
    const int vdwtype = const_extra[n+3].i << 1; // mult by 2 index VDW parms
    const float dx = px - const_extra[n  ].f;
    const float dy = py - const_extra[n+1].f;
    const float dz = pz - const_extra[n+2].f;
    const float r2 = dx*dx + dy*dy + dz*dz;
    if (r2 < cutoff2) {  // atom within cutoff, accumulate energy
      const float epsilon = const_vdwparms[vdwtype] * probe_vdweps;
      const float rmin = const_vdwparms[vdwtype + 1] + probe_vdwrmin;
      float rm6 = rmin*rmin / r2;
      rm6 = rm6 * rm6 * rm6;
      u += epsilon * rm6 * (rm6 - 2.f);  // sum vdw contribution
    }
  } // end loop over extra atoms

  float occ = 0.f;
  if (!isexcluded && u < const_max_energy) {
    occ = expf(-u);  // the occupancy
  }
  map[mindex] = occ;
} // cuda_occupancy_monoatom()


// For a multiatom probe compute the occupancy rho
// (probability of finding the probe)
//
// For each map point the occupancy is computed as
//
//   rho = (1/m) sum_i exp(-U[i]), i = 1..m
//
// where U[i] is the potential energy of the i-th conformer.
//
__global__ static void cuda_occupancy_multiatom(
    float *map,             // map buffer, stored in contiguous thread blocks
    const int *excl,        // exclusion buffer, stored same
    const flint *bin_zero,  // points to bin containing map origin
    int nbx, int nby,       // number of bins along x and y dimensions
#ifndef DO_ONE_PLANE
    int mzblocks,           // number of thread blocks in z dimension
#endif
    int mzblockoffset       // offset (number of planes) to starting z
    ) {

  // space for 1 atom bin in shared memory
  __shared__ flint atombin[BIN_SIZE];

  // accumulate into shared memory the energy for NCONFPERLOOP conformers
  __shared__ float usum[NTPB_TIMES_NCPL];

  // cache const mem pages 2+ into shared memory for faster access
  // since we have to keep accessing const mem page 1 (8K)
  __shared__ float conf[3*MAX_PROBES * NCONFPERLOOP];

#if defined(TEST_MBLOCK_EXCL) || defined(TEST_SHIFT_EXCL)
#if 0
  __shared__ int sh_isany_upper_warp;
#endif
  // we reuse this buffer:
  // (1) reduce total sum across thread block for prefix sum
  // (2) to accumulate occupancy
  __shared__ flint sh_buffer[NTHREADSPERBLOCK];
#endif

#ifndef DO_ONE_PLANE
  const int nblockx = gridDim.x;
  const int nblocky = gridDim.y / mzblocks;

  // 3D index for this thread block
  const int bidx = blockIdx.x;
  const int bidy = blockIdx.y % nblocky;
  const int bidz = blockIdx.y / nblocky + mzblockoffset;

  // block ID, flat index
  const int bid = (bidz * nblocky + bidy)*nblockx + bidx;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (bidx<<BSHIFT) * const_hx;
  float py = (bidy<<BSHIFT) * const_hy;
  float pz = (bidz<<BSHIFT) * const_hz;
#else
  // can optimize from above for case when mzblocks = 1

  // block ID, flat index
  const int bid
    = (mzblockoffset * gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;

  // thread ID, flat index
  const int tid = (((threadIdx.z<<BSHIFT) + threadIdx.y)<<BSHIFT) + threadIdx.x;

  // position of lower left corner of this thread block
  float px = (blockIdx.x<<BSHIFT) * const_hx;
  float py = (blockIdx.y<<BSHIFT) * const_hy;
  float pz = (mzblockoffset<<BSHIFT) * const_hz;
#endif

  const int mindex = (bid<<NTPBSHIFT) + tid;  // index into map and excl
  const int isincluded = !(excl[mindex]);  // read exclusion from global memory

#ifdef TEST_MBLOCK_EXCL
#if 0
  // warp vote is available only in devices 1.2 and later
  // warp vote to see if we continue
  int isany = __any(isincluded);
  if (32==tid) sh_isany_upper_warp = isany;
  if (0==isany+sh_isany_upper_warp) return; // all points excluded, exit
#endif

  sh_buffer[tid].i = isincluded;
  int nbrsum = sh_buffer[tid^1].i;
  sh_buffer[tid].i += nbrsum;
  nbrsum = sh_buffer[tid^2].i;
  sh_buffer[tid].i += nbrsum;
  nbrsum = sh_buffer[tid^4].i;
  sh_buffer[tid].i += nbrsum;
  nbrsum = sh_buffer[tid^8].i;
  sh_buffer[tid].i += nbrsum;
  nbrsum = sh_buffer[tid^16].i;
  sh_buffer[tid].i += nbrsum;
  __syncthreads();
  nbrsum = sh_buffer[tid^32].i;
  __syncthreads();
  sh_buffer[tid].i += nbrsum;

  if (0==sh_buffer[tid].i) {
    map[mindex] = 0.f;
    return;
  }
#endif

#ifdef TEST_SHIFT_EXCL
  int numincl = 0;  // number of included points in thread block
  {
    int psum = isincluded;  // will have prefix sum over isincluded for tid
    sh_buffer[tid].i = isincluded;

    int nbrid = tid^1;
    int nbrsum = sh_buffer[nbrid].i;
    if (nbrid < tid) psum += nbrsum;
    sh_buffer[tid].i += nbrsum;

    nbrid = tid^2;
    nbrsum = sh_buffer[nbrid].i;
    if (nbrid < tid) psum += nbrsum;
    sh_buffer[tid].i += nbrsum;

    nbrid = tid^4;
    nbrsum = sh_buffer[nbrid].i;
    if (nbrid < tid) psum += nbrsum;
    sh_buffer[tid].i += nbrsum;

    nbrid = tid^8;
    nbrsum = sh_buffer[nbrid].i;
    if (nbrid < tid) psum += nbrsum;
    sh_buffer[tid].i += nbrsum;

    nbrid = tid^16;
    nbrsum = sh_buffer[nbrid].i;
    if (nbrid < tid) psum += nbrsum;
    sh_buffer[tid].i += nbrsum;

    __syncthreads();
    nbrid = tid^32;
    nbrsum = sh_buffer[nbrid].i;
    if (nbrid < tid) psum += nbrsum;
    __syncthreads();
    sh_buffer[tid].i += nbrsum;

    numincl = sh_buffer[tid].i;  // number of included points
    if (0==numincl) {
      map[mindex] = 0.f;
      return;  // all points in thread block excluded so exit early
    }
    __syncthreads();
    if (isincluded) {
      sh_buffer[psum-1].i = tid;  // assign my work to thread #(psum-1)
    }
  }
  __syncthreads();
  const int sid = sh_buffer[tid].i;  // shifted thread id for whom i calculate
#endif

  const float cutoff2 = const_cutoff * const_cutoff;  // cutoff2 in register

  const int ib = (int) floorf(px * const_bx_1);  // bin index for entire
  const int jb = (int) floorf(py * const_by_1);  // thread block
  const int kb = (int) floorf(pz * const_bz_1);

  const int binid = (kb * nby + jb)*nbx + ib;    // flat index of my bin

  int n, nb;
  int m;
  int ma;
  int nc;

#ifdef TEST_SHIFT_EXCL
  if (tid < numincl) {
    int ssid = sid;
    const int sid_x = ssid & PADMASK;
    ssid >>= BSHIFT;
    const int sid_y = ssid & PADMASK;
    ssid >>= BSHIFT;
    const int sid_z = ssid;
    px += sid_x * const_hx;  // adjust position for shifted thread
    py += sid_y * const_hy;
    pz += sid_z * const_hz;
  }
#else
  px += threadIdx.x * const_hx;  // adjust position for this thread
  py += threadIdx.y * const_hy;
  pz += threadIdx.z * const_hz;
#endif

  px += const_x0;
  py += const_y0;
  pz += const_z0;  // translate to absolute position

#ifdef TEST_SHIFT_EXCL
  sh_buffer[tid].f = 0.f;  // for summing occupancy
#else
  float occ = 0.f;  // sum occupancy
#endif

  const int twice_num_probes = const_num_probes << 1;
  const int num_extra_slots = const_num_extras << BIN_SLOTSHIFT;

  // number of floats from conformers to move into shared memory
  const int total_shmem_copy = 3*NCONFPERLOOP * const_num_probes;

  // loop over all conformers
  for (m = 0;  m < const_num_conformers;  m += NCONFPERLOOP) {

    // reset shared memory energy accumulators
    for (nc = 0;  nc < NTPB_TIMES_NCPL;  nc += NTHREADSPERBLOCK) {
      usum[nc + tid] = 0.f;  // these are NOT shared
    }

    // cooperatively copy conformers from constant memory page 2+
    // into shared memory for improved performance
    // while we continue to access constant memory page 1 (8K)
    __syncthreads();
    if (tid < 32) {  // use only first warp

      // number of loop iterations for all threads in warp
      const int npass = (total_shmem_copy >> 5);

      // remaining number of floats to move
      const int nleft = (total_shmem_copy & 31);

      // skip over previously computed conformers (3*const_num_probes*m)
      const int skip = (const_num_probes + twice_num_probes) * m;

      int off = 0;
      for (nc = 0;  nc < npass;  nc++, off += 32) {
        conf[off + tid] = const_conformers[skip + off + tid];
      }
      if (tid < nleft) {
        conf[off + tid] = const_conformers[skip + off + tid];
      }
    }
    __syncthreads();  // wait for entire thread block

    for (nb = 0;  nb < const_num_binoffsets;  nb++) {  // bin neighborhood

      __syncthreads();  // wait for entire thread block before loading next bin
      if (tid < BIN_SIZE) {  // cache next atom bin
        const flint *p_bin = bin_zero
          + ((binid + const_binoffsets[nb]) << BIN_SHIFT);
        atombin[tid] = p_bin[tid];  // copy from global to shared memory
      }
      __syncthreads();  // now we have atom bin loaded

#ifdef TEST_SHIFT_EXCL
    if (tid < numincl) {
#endif

      for (n = 0;  n < BIN_SIZE;  n += BIN_SLOTSIZE) { // loop over atoms in bin
        int vdwtype = atombin[n+3].i;
        if (-1 == vdwtype) break;  // no more atoms in bin
        vdwtype <<= 1;  // multiply by 2 for indexing VDW parms
        const float epsilon_atom = const_vdwparms[vdwtype];
        const float rmin_atom = const_vdwparms[vdwtype+1];
        const float ax = px - atombin[n  ].f;
        const float ay = py - atombin[n+1].f;
        const float az = pz - atombin[n+2].f;

        int sindex = 0;  // index conformers
       
        // loop shared memory conformers
        for (nc = 0;  nc < NTPB_TIMES_NCPL;  nc += NTHREADSPERBLOCK) {

          for (ma = 0;  ma < twice_num_probes;  ma += 2) {  // probe atoms
            const float dx = conf[sindex++] + ax;
            const float dy = conf[sindex++] + ay;
            const float dz = conf[sindex++] + az;
            const float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < cutoff2) {  // atom within cutoff, accumulate energy
              const float epsilon = epsilon_atom * const_probevdwparms[ma];
              const float rmin = rmin_atom + const_probevdwparms[ma+1];
              float rm6 = rmin*rmin / r2;
              rm6 = rm6 * rm6 * rm6;

              // XXX rather than reading/writing to shared memory here,
              //     we should be working with a register.  Only when we
              //     complete the two innermost loops should we be storing
              //     back to shared memory.  Shared memory limits performance
              //     to 66% of what's possible with a register...
              //     The stores to usum[] should be moved out to the 'nc' loop.
              // XXX Accumulating usum in 'nc' loop is actually a bit slower.
              //     Note that we only access shared memory if the system atom
              //     is within the cutoff.  Even with small atom bins we still
              //     have a large percentage of atoms outside of cutoff.
              usum[nc + tid] += epsilon * rm6 * (rm6 - 2.f);
            }
          } // end loop over probe atoms

        } // end loop over shared memory conformers

      } // end loop over atoms in bin
#ifdef TEST_SHIFT_EXCL
    } // end if (tid < numincl)
#endif

    } // end loop over bin neighborhood

#ifdef TEST_SHIFT_EXCL
  if (tid < numincl) {
#endif

    for (n = 0;  n < num_extra_slots;  n += BIN_SLOTSIZE) {  // extra atoms
      const int vdwtype = const_extra[n+3].i << 1; // mult by 2 index VDW parms
      const float epsilon_atom = const_vdwparms[vdwtype];
      const float rmin_atom = const_vdwparms[vdwtype+1];
      const float ax = px - const_extra[n  ].f;
      const float ay = py - const_extra[n+1].f;
      const float az = pz - const_extra[n+2].f;
      int sindex = 0;
    
      // loop over shared memory conformers
      for (nc = 0;  nc < NTPB_TIMES_NCPL;  nc += NTHREADSPERBLOCK) {

        for (ma = 0;  ma < twice_num_probes;  ma += 2) {  // probe atoms
          const float dx = conf[sindex++] + ax;
          const float dy = conf[sindex++] + ay;
          const float dz = conf[sindex++] + az;
          const float r2 = dx*dx + dy*dy + dz*dz;
          if (r2 < cutoff2) {  // atom within cutoff, accumulate energy
            const float epsilon = epsilon_atom * const_probevdwparms[ma];
            const float rmin = rmin_atom + const_probevdwparms[ma+1];
            float rm6 = rmin*rmin / r2;
            rm6 = rm6 * rm6 * rm6;
            usum[nc + tid] += epsilon * rm6 * (rm6 - 2.f);
          }
        } // end loop over probe atoms

      } // end loop over shared memory conformers

    } // end loop over extra atoms

    for (nc = 0;  nc < NCONFPERLOOP;  nc++) {  // loop over some conformers

      if (m + nc < const_num_conformers) {  // ensure this 'nc' is meaningful
        float u = usum[(nc << NTPBSHIFT) + tid];

#ifdef TEST_SHIFT_EXCL
        if (u < MAX_ALLOWABLE_ENERGY) {  // XXX should we test?
          sh_buffer[sid].f += expf(-u);  // the occupancy
        }
#else
        if (isincluded && u < MAX_ALLOWABLE_ENERGY) {  // XXX should we test?
          occ += expf(-u);  // the occupancy
        }
#endif
      }

    } // end loop over some conformers
#ifdef TEST_SHIFT_EXCL
  } // end if (tid < numincl)
#endif

  } // end loop over all conformers

#ifdef TEST_SHIFT_EXCL
  __syncthreads();  // must wait for thread block before finishing
  sh_buffer[tid].f *= const_inv_numconf;  // averaged occupancy
  if (sh_buffer[tid].f <= const_min_occupancy) sh_buffer[tid].f = 0.f;

  map[mindex] = sh_buffer[tid].f;
#else
  occ = occ * const_inv_numconf;  // averaged occupancy
  if (occ <= const_min_occupancy) occ = 0.f;

  map[mindex] = occ;
#endif

} // cuda_occupancy_multiatom()



static void *cuda_occupancy_thread(void *voidparms) {
  occthrparms *parms = NULL;
  int mxpad, mypad, mzpad, mappad;
  float *h_map;       // padded map buffer on host
  float *d_map;       // padded map buffer on device
  int *d_excl;        // padded exclusion map buffer on device
  int binsz;          // size of bin buffer
  flint *d_bin;       // atom bins stored on device
  flint *d_bin_zero;  // shifted bin pointer on device
  int gpuid = -1;     // default: let VMD choose.
  int numgpus = 0;
  int i, j, k;

  // override device choice from environment for backward compatibility.
  if (getenv("VMDILSCUDADEVICE")) {
    gpuid = atoi(getenv("VMDILSCUDADEVICE"));
  } 

  if (gpuid < 0) {
    if (vmd_cuda_num_devices(&numgpus) != VMDCUDA_ERR_NONE) {
      printf("ILS CUDA device init error\n");
      return NULL;
    }
    if (numgpus > 0) {
      // XXX: here we should pick the first available GPU
      //      and skip over device GPUs. but how to do this 
      //      cleanly from in here? 
      //      change the code to use the device pool?
      // AK 2010/03/15
      gpuid = 0;
    } else {
      return NULL;
    }    
  }

  if (gpuid >= 0) {
    printf("Using CUDA device: %d\n", gpuid);
  } else {
    // no suitable GPU.
    return NULL;
  }
  
  wkf_timerhandle timer = wkf_timer_create();

  parms = (occthrparms *) voidparms;
  parms->errcode = -1;  // be pessimistic until the very end

  if (getenv("VMDILSVERBOSE")) {
    // XXX look at parms
    printf("*****************************************************************\n");
    printf("mx = %d  my = %d  mz = %d\n", parms->mx, parms->my, parms->mz);
    printf("hx = %g  hy = %g  hz = %g\n", parms->hx, parms->hy, parms->hz);
    printf("x0 = %g  y0 = %g  z0 = %g\n", parms->x0, parms->y0, parms->z0);
    printf("bx_1 = %g  by_1 = %g  bz_1 = %g\n",
           parms->bx_1, parms->by_1, parms->bz_1);
    printf("nbx = %d  nby = %d  nbz = %d\n", parms->nbx, parms->nby, parms->nbz);
    printf("num_binoffsets = %d\n", parms->num_binoffsets);
    printf("num_extras = %d\n", parms->num_extras);
    printf("num_vdwparms = %d\n", parms->num_vdwparms);
    printf("num_probes = %d\n", parms->num_probes);
    printf("num_conformers = %d\n", parms->num_conformers);
    printf("*****************************************************************\n");
  }

  // check that data will fit constant memory
  if (parms->num_binoffsets > MAX_BINOFFSETS) {
    printf("***** ERROR: Exceeded MAX_BINOFFSETS for CUDA kernel\n");
    return NULL;  // XXX How do I raise an error / exception?
  }
  if (parms->num_extras > MAX_EXTRAS) {
    printf("***** ERROR: Exceeded MAX_EXTRAS for CUDA kernel\n");
    return NULL;  // XXX How do I raise an error / exception?
  }
  if (parms->num_vdwparms > MAX_VDWPARMS) {
    printf("***** ERROR: Exceeded MAX_VDWPARMS for CUDA kernel\n");
    return NULL;  // XXX How do I raise an error / exception?
  }
  if (parms->num_probes > MAX_PROBES) {
    printf("***** ERROR: Exceeded MAX_PROBES for CUDA kernel\n");
    return NULL;  // XXX How do I raise an error / exception?
  }
  if (parms->num_conformers > MAX_CONFORMERS) {
    printf("***** ERROR: Exceeded MAX_CONFORMERS for CUDA kernel\n");
    return NULL;  // XXX How do I raise an error / exception?
  }

  // attach to GPU and check for errors
  cudaError_t rc;
  rc = cudaSetDevice(gpuid);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError();  // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess) {
      return NULL;  // abort and return an error
    }
#else
    cudaGetLastError();  // just ignore and reset error state, since older CUDA
                         // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }

  // pad the map for CUDA thread blocks
  mxpad = (parms->mx + PADMASK) & ~PADMASK;
  mypad = (parms->my + PADMASK) & ~PADMASK;
  mzpad = (parms->mz + PADMASK) & ~PADMASK;
  mappad = mxpad * mypad * mzpad;

  if (getenv("VMDILSVERBOSE")) {
    printf("mxpad=%d  mypad=%d  mzpad=%d  mappad=%d\n",
           mxpad, mypad, mzpad, mappad);
  }

  h_map = (float *) malloc(mappad * sizeof(float));
  if (getenv("VMDILSVERBOSE")) {
    printf("Allocate %g MB for map\n", mappad*sizeof(float)/(1024*1024.f));
  }
  cudaMalloc((void **) &d_map, mappad * sizeof(float));
  CUERR;
  if (getenv("VMDILSVERBOSE")) {
    printf("Allocate %g MB for exclusions\n", mappad*sizeof(int)/(1024*1024.f));
  }
  cudaMalloc((void **) &d_excl, mappad * sizeof(int));
  CUERR;

#if !defined(USE_DISTANCE_EXCL) && !defined(USE_ENERGY_EXCL)
  // set all points to be excluded by default
  cudaMemset(d_excl, 0, mappad * sizeof(int));
  CUERR;
#endif

  binsz = BIN_SIZE * parms->nbx * parms->nby * parms->nbz;
  if (getenv("VMDILSVERBOSE")) {
    printf("nbx=%d  nby=%d  nbz=%d  binsz=%d\n",
           parms->nbx, parms->nby, parms->nbz, binsz);
    printf("Allocate %g MB for atom bins\n", binsz*sizeof(flint)/(1024*1024.f));
  }
  cudaMalloc((void **) &d_bin, binsz * sizeof(flint));
  CUERR;
  cudaMemcpy(d_bin, parms->bin, binsz * sizeof(flint), cudaMemcpyHostToDevice);
  CUERR;
  d_bin_zero = d_bin + (parms->bin_zero - parms->bin);

  if (getenv("VMDILSVERBOSE")) {
    printf("parms delta bin = %d   delta bin = %d\n",
           (parms->bin_zero - parms->bin), (d_bin_zero - d_bin));
    
    printf("probe epsilon=%g rmin=%g\n",
           parms->probevdwparms[0], parms->probevdwparms[1]);
    printf("max_energy=%g\n", parms->max_energy);
    printf("cutoff=%g\n", parms->cutoff);
    printf("hx=%g hy=%g hz=%g\n", parms->hx, parms->hy, parms->hz);
    printf("x0=%g y0=%g z0=%g\n", parms->x0, parms->y0, parms->z0);
    printf("bx_1=%g by_1=%g bz_1=%g\n", parms->bx_1, parms->by_1, parms->bz_1);
  }

  cudaMemcpyToSymbol(const_max_energy, &(parms->max_energy), sizeof(float), 0);
  CUERR;
  float min_occ = expf(-parms->max_energy);  // occupancy threshold
  cudaMemcpyToSymbol(const_min_occupancy, &min_occ, sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_cutoff, &(parms->cutoff), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_min_excldist, &(parms->min_excldist),
      sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_hx, &(parms->hx), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_hy, &(parms->hy), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_hz, &(parms->hz), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_x0, &(parms->x0), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_y0, &(parms->y0), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_z0, &(parms->z0), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_bx_1, &(parms->bx_1), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_by_1, &(parms->by_1), sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_bz_1, &(parms->bz_1), sizeof(float), 0);
  CUERR;
  float inv_numconf = 1.f / parms->num_conformers;
  cudaMemcpyToSymbol(const_inv_numconf, &inv_numconf, sizeof(float), 0);

  if (getenv("VMDILSVERBOSE")) {
    printf("num_binoffsets = %d\n", parms->num_binoffsets);
  }
  cudaMemcpyToSymbol(const_num_binoffsets, &(parms->num_binoffsets),
      sizeof(int), 0);
  CUERR;
  if (getenv("VMDILSVERBOSE")) {
    printf("num_extras = %d\n", parms->num_extras);
  }
  cudaMemcpyToSymbol(const_num_extras, &(parms->num_extras), sizeof(int), 0);
  CUERR;
  if (getenv("VMDILSVERBOSE")) {
    printf("num_probes = %d\n", parms->num_probes);
  }
  cudaMemcpyToSymbol(const_num_probes, &(parms->num_probes), sizeof(int), 0);
  CUERR;
  if (getenv("VMDILSVERBOSE")) {
    printf("num_conformers = %d\n", parms->num_conformers);
  }
  cudaMemcpyToSymbol(const_num_conformers, &(parms->num_conformers),
      sizeof(int), 0);
  CUERR;

  if (getenv("VMDILSVERBOSE")) {
    printf("num_vdwparms = %d\n", parms->num_vdwparms);
    for (i = 0;  i < parms->num_vdwparms;  i++) {
      printf("  %2d:  epsilon=%g  rmin=%g\n", i,
             parms->vdwparms[2*i], parms->vdwparms[2*i+1]);
    }
  }

  cudaMemcpyToSymbol(const_vdwparms, parms->vdwparms,
      2 * parms->num_vdwparms * sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_probevdwparms, parms->probevdwparms,
      2 * (0==parms->num_probes ? 1 : parms->num_probes) * sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(const_extra, parms->extra,
      BIN_SLOTSIZE * parms->num_extras * sizeof(flint), 0);
  CUERR;

  // take 3D bin offset and calculate array of flat bin offset
  {
    const int nbx = parms->nbx;
    const int nby = parms->nby;
    int n;
    int *h_flatbinoffset = (int *) calloc(parms->num_binoffsets, sizeof(int));
    for (n = 0;  n < parms->num_binoffsets;  n++) {
      int i = (int) (parms->binoffsets[3*n  ]);
      int j = (int) (parms->binoffsets[3*n+1]);
      int k = (int) (parms->binoffsets[3*n+2]);
      int index = (k*nby + j)*nbx + i;
      h_flatbinoffset[n] = index;
    }
    cudaMemcpyToSymbol(const_binoffsets, h_flatbinoffset,
        parms->num_binoffsets * sizeof(int), 0);
    CUERR;
    free(h_flatbinoffset);
  }

  // calculate array of flat bin offsets for distance-based exclusions
  {
    const int nbx = parms->nbx;
    const int nby = parms->nby;
    int i, j, k, n=0;
    int h_flatexcloffset[27];  // 3x3x3 cube
    for (k = -1;  k <= 1;  k++) {
      for (j = -1;  j <= 1;  j++) {
        for (i = -1;  i <= 1;  i++) {
          int index = (k*nby + j)*nbx + i;
          h_flatexcloffset[n++] = index;
        }
      }
    }
    cudaMemcpyToSymbol(const_excloffsets, h_flatexcloffset, 27*sizeof(int), 0);
    CUERR;
  }

  cudaMemcpyToSymbol(const_conformers, parms->conformers,
      3 * parms->num_probes * parms->num_conformers * sizeof(float), 0);
  CUERR;

  // cluster slabs based on time per work unit;
  // the following constants empirically derived from
  // oxygen testcase run on GT200
#define TIME_PER_MONOATOM_WORKUNIT   1.24553913519431e-10
#define TIME_PER_MULTIATOM_WORKUNIT  5.24559386973180e-09
//#define TIME_PER_MONOATOM_WORKUNIT   3.73661740558292e-09
//#define TIME_PER_MULTIATOM_WORKUNIT  1.57367816091954e-07

  int mxblocks = (mxpad >> BSHIFT);
  int myblocks = (mypad >> BSHIFT);
  int mzblocks = (mzpad >> BSHIFT);
  int nblocksperplane = mxblocks * myblocks;
  int maxplanes = MAX_THREADBLOCKS / nblocksperplane;
  int numconf = (0==parms->num_conformers ? 1 : parms->num_conformers);

  double monoatom_time_per_plane = (double)(parms->num_probes * numconf *
      parms->num_binoffsets * nblocksperplane) * TIME_PER_MONOATOM_WORKUNIT;

  double multiatom_time_per_plane = (double)(parms->num_probes * numconf *
      parms->num_binoffsets * nblocksperplane) * TIME_PER_MULTIATOM_WORKUNIT;

  // number of planes to schedule per slab
#define MAX_TIME_PER_KERNEL  1.f  // schedule up to 1 second per kernel call
  int max_mono_planes = (int)(MAX_TIME_PER_KERNEL / monoatom_time_per_plane);
  int max_multi_planes = (int)(MAX_TIME_PER_KERNEL / multiatom_time_per_plane);

  if (max_mono_planes > maxplanes) max_mono_planes = maxplanes;
  else if (max_mono_planes < 1)    max_mono_planes = 1;

  if (max_multi_planes > maxplanes) max_multi_planes = maxplanes;
  else if (max_multi_planes < 1)    max_multi_planes = 1;

  if (getenv("VMDILSVERBOSE")) {
    printf("mxblocks = %d  myblocks = %d  mzblocks = %d\n",
           mxblocks, myblocks, mzblocks);
  }
  if (nblocksperplane > MAX_THREADBLOCKS) {
    printf("***** ERROR: Too many blocks in X-Y plane for CUDA kernel\n");
    return NULL;  // XXX raise exception
  }

  if (0 == parms->num_conformers) {  // monoatom
    if (getenv("VMDILSVERBOSE")) {
      printf("Scheduling up to %d planes (%d thread blocks) per monoatom "
        "kernel call\n", max_mono_planes, nblocksperplane*max_mono_planes);
    }
    int mzblockoffset = 0;
    int nplanes_remaining = mzblocks;
    while (nplanes_remaining > 0) {
#ifndef DO_ONE_PLANE
      // do as many planes as possible given constraints
      int nplanes = nplanes_remaining;
      if (nplanes > max_mono_planes) nplanes = max_mono_planes;
#else
      // XXX do only one slice at a time
      int nplanes = 1;
#endif
      dim3 gsz, bsz;
      gsz.x = mxblocks;
      gsz.y = myblocks * nplanes;
      gsz.z = 1;
      bsz.x = THBLEN;
      bsz.y = THBLEN;
      bsz.z = THBLEN;

      if ( ! getenv("VMDILSNOEXCL") ) {
#ifdef USE_DISTANCE_EXCL
      // distance exclusions
#ifdef DEBUG
      printf("*** VMD CUDA: cuda_find_distance_exclusions() "
          "on %2d planes   ",
          nplanes);
#endif
#if 1
      cudaDeviceSynchronize();
#endif
      wkf_timer_start(timer);
      cuda_find_distance_exclusions<<<gsz, bsz, 0>>>(
          d_excl, d_bin_zero,
          parms->nbx, parms->nby,
#ifndef DO_ONE_PLANE
          nplanes,
#endif
          mzblockoffset);
#if 1
      cudaDeviceSynchronize();
#endif
#ifdef DEBUG
      printf("%f s\n", wkf_timer_timenow(timer));
#endif
      CUERR;
#endif // USE_DISTANCE_EXCL
      }

      // monoatom occupancy
#ifdef DEBUG
     printf("*** VMD CUDA: cuda_occupancy_monoatom()       "
          "on %2d planes   ",
          nplanes);
#endif
#if 1
      cudaDeviceSynchronize();
#endif
      wkf_timer_start(timer);
      cuda_occupancy_monoatom<<<gsz, bsz, 0>>>(
          d_map, d_excl, d_bin_zero,
          parms->nbx, parms->nby,
#ifndef DO_ONE_PLANE
          nplanes,
#endif
          mzblockoffset);
#if 1
      cudaDeviceSynchronize();
#endif
#ifdef DEBUG
      printf("%f s\n", wkf_timer_timenow(timer));
#endif
      CUERR;

      nplanes_remaining -= nplanes;
      mzblockoffset += nplanes;
    }
  } // if monoatom

  else { // multiatom
    if (getenv("VMDILSVERBOSE")) {
      printf("Scheduling up to %d planes (%d thread blocks) per monoatom "
        "kernel call\n", max_mono_planes, nblocksperplane*max_mono_planes);
      printf("Scheduling up to %d planes (%d thread blocks) per multiatom "
        "kernel call\n", max_multi_planes, nblocksperplane*max_multi_planes);
    }

    int mzblockoffset;
    int nplanes_remaining;

    if ( ! getenv("VMDILSNOEXCL") ) {
    // first do "monoatom" slabs for exclusions
    mzblockoffset = 0;
    nplanes_remaining = mzblocks;
    while (nplanes_remaining > 0) {
#ifndef DO_ONE_PLANE
      // do as many planes as possible given constraints
      int nplanes = nplanes_remaining;
      if (nplanes > max_mono_planes) nplanes = max_mono_planes;
#else
      // XXX do only one slice at a time
      int nplanes = 1;
#endif
      dim3 gsz, bsz;
      gsz.x = mxblocks;
      gsz.y = myblocks * nplanes;
      gsz.z = 1;
      bsz.x = THBLEN;
      bsz.y = THBLEN;
      bsz.z = THBLEN;

#ifdef USE_DISTANCE_EXCL
      // distance exclusions
#ifdef DEBUG
      printf("*** VMD CUDA: cuda_find_distance_exclusions() "
          "on %2d planes   ",
          nplanes);
#endif
#if 1
      cudaDeviceSynchronize();
#endif
      wkf_timer_start(timer);
      cuda_find_distance_exclusions<<<gsz, bsz, 0>>>(
          d_excl, d_bin_zero,
          parms->nbx, parms->nby,
#ifndef DO_ONE_PLANE
          nplanes,
#endif
          mzblockoffset);
#if 1
      cudaDeviceSynchronize();
#endif
#ifdef DEBUG
      printf("%f s\n", wkf_timer_timenow(timer));
#endif
      CUERR;
#endif // USE_DISTANCE_EXCL

#ifdef USE_ENERGY_EXCL
      // energy exclusions
#ifdef DEBUG
      printf("*** VMD CUDA: cuda_find_energy_exclusions()   "
          "on %2d planes   ",
          nplanes);
#endif
#if 1
      cudaDeviceSynchronize();
#endif
      wkf_timer_start(timer);
      cuda_find_energy_exclusions<<<gsz, bsz, 0>>>(
          d_excl, d_bin_zero,
          parms->nbx, parms->nby,
#ifndef DO_ONE_PLANE
          nplanes,
#endif
          mzblockoffset);
#if 1
      cudaDeviceSynchronize();
#endif
#ifdef DEBUG
      printf("%f s\n", wkf_timer_timenow(timer));
#endif
      CUERR;
#endif // USE_ENERGY_EXCL

      nplanes_remaining -= nplanes;
      mzblockoffset += nplanes;
    }
    } // ( ! getenv("VMDILSNOEXCL") )

    // next do "multiatom" slabs for exclusions
    mzblockoffset = 0;
    nplanes_remaining = mzblocks;
    while (nplanes_remaining > 0) {
#ifndef DO_ONE_PLANE
      // do as many planes as possible given constraints
      int nplanes = nplanes_remaining;
      if (nplanes > max_multi_planes) nplanes = max_multi_planes;
#else
      // XXX do only one slice at a time
      int nplanes = 1;
#endif
      dim3 gsz, bsz;
      gsz.x = mxblocks;
      gsz.y = myblocks * nplanes;
      gsz.z = 1;
      bsz.x = THBLEN;
      bsz.y = THBLEN;
      bsz.z = THBLEN;

      // multiatom occupancy
#ifdef DEBUG
      printf("*** VMD CUDA: cuda_occupancy_multiatom()      "
          "on %2d planes   ",
          nplanes);
#endif
#if 1
      cudaDeviceSynchronize();
#endif
      wkf_timer_start(timer);
      cuda_occupancy_multiatom<<<gsz, bsz, 0>>>(
          d_map, d_excl, d_bin_zero,
          parms->nbx, parms->nby,
#ifndef DO_ONE_PLANE
          nplanes,
#endif
          mzblockoffset);
#if 1
      cudaDeviceSynchronize();
#endif
#ifdef DEBUG
      printf("%f s\n", wkf_timer_timenow(timer));
#endif
      CUERR;

      nplanes_remaining -= nplanes;
      mzblockoffset += nplanes;
    }
  } // else multiatom

  // retrieve padded map from GPU
  cudaMemcpy(h_map, d_map, mappad * sizeof(float), cudaMemcpyDeviceToHost);
  CUERR;

  // copy padded map into user map buffer
  // must transpose the thread blocks
  for (k = 0;  k < parms->mz;  k++) {
    for (j = 0;  j < parms->my;  j++) {
      for (i = 0;  i < parms->mx;  i++) {
        int mindex = (k*parms->my + j)*parms->mx + i;
        int block = ((k >> BSHIFT)*myblocks + (j >> BSHIFT))*mxblocks
          + (i >> BSHIFT);
        int tid = ((k & PADMASK)*THBLEN + (j & PADMASK))*THBLEN + (i & PADMASK);
        int pindex = block*(THBLEN*THBLEN*THBLEN) + tid;
        parms->map[mindex] = h_map[pindex];
      }
    }
  }

  cudaFree(d_excl);
  cudaFree(d_bin);
  cudaFree(d_map);
  CUERR;

  free(h_map);
  wkf_timer_destroy(timer);

  parms->errcode = 0;  // success!
  return NULL;
}


extern "C" int vmd_cuda_evaluate_occupancy_map(
    int mx, int my, int mz,             // map dimensions
    float *map,                         // buffer space for occupancy map
                                        // (length mx*my*mz floats)

    float max_energy,                   // max energy threshold
    float cutoff,                       // vdw cutoff distance
    float hx, float hy, float hz,       // map lattice spacing
    float x0, float y0, float z0,       // map origin
    float bx_1, float by_1, float bz_1, // inverse of atom bin lengths

    int nbx, int nby, int nbz,          // bin dimensions
    const float *bin,                   // atom bins XXX typecast to flint
                                        // (length BIN_SIZE*nbx*nby*nbz)
    const float *bin_zero,              // bin pointer shifted to origin

    int num_binoffsets,                 // number of offsets
    const char *binoffsets,             // bin neighborhood index offsets
                                        // (length 3*num_bin_offsets)

    int num_extras,                     // number of extra atoms
    const float *extra,                 // extra atoms from overfilled bins
                                        // XXX typecast to flint
                                        // (length BIN_SLOTSIZE*num_extras)

    int num_vdwparms,                   // number of vdw parameter types
    const float *vdwparms,              // vdw parameters
                                        // (length 2*num_vdw_params)

    int num_probes,                     // number of probe atoms
    const float *probevdwparms,         // vdw parameters of probe atoms
                                        // (length 2*num_probes)

    int num_conformers,                 // number of conformers
    const float *conformers             // probe atom offsets for conformers
                                        // (length 3*num_probes*num_conformers)
    ) {
  occthrparms parms;

  parms.mx = mx;
  parms.my = my;
  parms.mz = mz;
  parms.map = map;
  parms.max_energy = max_energy;
  parms.cutoff = cutoff;
#define DEFAULT_EXCL_DIST  1.f
  parms.min_excldist = DEFAULT_EXCL_DIST;
  parms.hx = hx;
  parms.hy = hy;
  parms.hz = hz;
  parms.x0 = x0;
  parms.y0 = y0;
  parms.z0 = z0;
  parms.bx_1 = bx_1;
  parms.by_1 = by_1;
  parms.bz_1 = bz_1;
  parms.nbx = nbx;
  parms.nby = nby;
  parms.nbz = nbz;
  parms.bin = (flint *) bin;
  parms.bin_zero = (flint *) bin_zero;
  parms.num_binoffsets = num_binoffsets;
  parms.binoffsets = binoffsets;
  parms.num_extras = num_extras;
  parms.extra = (flint *) extra;
  parms.num_vdwparms = num_vdwparms;
  parms.vdwparms = vdwparms;
  parms.num_probes = num_probes;
  parms.probevdwparms = probevdwparms;
  parms.num_conformers = num_conformers;
  parms.conformers = conformers;

#ifdef DEBUG
  printf("*** calling cuda_occupancy_thread()\n");
#endif
  cuda_occupancy_thread(&parms);

  return parms.errcode;
}
