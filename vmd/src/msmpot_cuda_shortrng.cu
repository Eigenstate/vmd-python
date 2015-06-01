/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: msmpot_cuda_shortrng.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $      $Date: 2014/05/27 15:31:50 $
 *
 ***************************************************************************/
/*
 * msmcuda_shortrng.cu
 */

#include "msmpot_cuda.h"


#define BLOCK_DIM_X  8
#define BLOCK_DIM_Y  2
#define UNROLL_Y     4
#define BLOCK_DIM_Z  8

#define REGSIZE_X    BLOCK_DIM_X
#define REGSIZE_Y    (BLOCK_DIM_Y * UNROLL_Y)
#define REGSIZE_Z    BLOCK_DIM_Z

#define REGION_SIZE  (REGSIZE_X * REGSIZE_Y * REGSIZE_Z)

#define LOG_BDIM_X   3
#define LOG_BDIM_Y   1
#define LOG_UNRL_Y   2
#define LOG_BDIM_Z   3

#define LOG_REGS_X   LOG_BDIM_X
#define LOG_REGS_Y   (LOG_BDIM_Y + LOG_UNRL_Y)
#define LOG_REGS_Z   LOG_BDIM_Z

#define LOG_REGSIZE  (LOG_REGS_X + LOG_REGS_Y + LOG_REGS_Z)

#define MASK_REGS_X  (REGSIZE_X - 1)
#define MASK_REGS_Y  (REGSIZE_Y - 1)
#define MASK_REGS_Z  (REGSIZE_Z - 1)


/* XXX what should this be? */
#define MAX_BIN_DEPTH  8


/*
 * neighbor list storage uses 64000 bytes
 */
__constant__ static int NbrListLen;
__constant__ static int3 NbrList[NBRLIST_MAXLEN];


/*
 * The following code is adapted from kernel
 *   cuda_cutoff_potential_lattice10overlap()
 * from source file mgpot_cuda_binsmall.cu from VMD cionize plugin.
 *
 * potential lattice is decomposed into size 8^3 lattice point "regions"
 *
 * THIS IMPLEMENTATION:  one thread per lattice point
 * thread block size 128 gives 4 thread blocks per region
 * kernel is invoked for each x-y plane of regions,
 * where gridDim.x is 4*(x region dimension) so that blockIdx.x 
 * can absorb the z sub-region index in its 2 lowest order bits
 *
 * Regions are stored contiguously in memory in row-major order
 *
 * The bins cover the atom domain.  For nonperiodic dimensions,
 * the domain length must be just long enough to include the atoms,
 * and the neighborhood of bins is truncated to the domain edges.
 * For periodic dimensions, the domain length is preset by the caller,
 * and the neighborhood of bins wraps around the domain edges.
 *
 * The atom coordinates are stored in bins relative to the domain origin.
 * The (rx0,ry0,rz0) offset for map points takes this into account.
 */


/*
 * cuda_shortrange_nonperiodic()  - all dimensions must be nonperiodic
 */
__global__ static void cuda_shortrange_nonperiodic(
    float *regionBase,  /* address of map regions */
    int zRegionDim,     /* z region dimension (x, y given by gridDim) */
    float dx,           /* map spacing along x-axis */
    float dy,           /* map spacing along y-axis */
    float dz,           /* map spacing along z-axis */
    float rx0,          /* x offset of map points, rx0 = px0 - lx0 */
    float ry0,          /* y offset of map points, ry0 = py0 - ly0 */
    float rz0,          /* z offset of map points, rz0 = pz0 - lz0 */
    float4 *binBase,    /* address of bins */
    int xBinDim,        /* x bin dimension */
    int yBinDim,        /* y bin dimension */
    int zBinDim,        /* z bin dimension */
    int bindepth,       /* number of atom slots per bin */
    float invbx,        /* inverse of bin length in x */
    float invby,        /* inverse of bin length in y */
    float invbz,        /* inverse of bin length in z */
    float cutoff2,      /* square of cutoff distance */
    float invcut        /* inverse of cutoff distance */
    ) {

  __shared__ float4 binCache[MAX_BIN_DEPTH];
  __shared__ float *myRegionAddr;
  __shared__ int3 myBinIndex;

  const int xRegionIndex = blockIdx.x;
  const int yRegionIndex = blockIdx.y;

  /* thread id */
  const int tid = (threadIdx.z*BLOCK_DIM_Y + threadIdx.y)*BLOCK_DIM_X
    + threadIdx.x;

  /* neighbor index */
  int nbrid;

  /* constants for TAYLOR2 softening */
  /* XXX is it more efficient to read these values from const memory? */
  float gc0, gc1, gc2;
  gc1 = invcut * invcut;
  gc2 = gc1 * gc1;
  gc0 = 1.875f * invcut;
  gc1 *= -1.25f * invcut;
  gc2 *= 0.375f * invcut;

  int zRegionIndex;
  for (zRegionIndex=0; zRegionIndex < zRegionDim; zRegionIndex++) {

    /* this is the start of the sub-region indexed by tid */
    myRegionAddr = regionBase + ((zRegionIndex*gridDim.y
          + yRegionIndex)*gridDim.x + xRegionIndex)*REGION_SIZE;
      
    /* spatial coordinate of this lattice point */
    float x = (REGSIZE_X * xRegionIndex + threadIdx.x) * dx - rx0;
    float y = (REGSIZE_Y * yRegionIndex + threadIdx.y) * dy - ry0;
    float z = (REGSIZE_Z * zRegionIndex + threadIdx.z) * dz - rz0;

    /* bin number determined by center of region */
    myBinIndex.x = (int) floorf((REGSIZE_X * xRegionIndex
          + REGSIZE_X / 2) * dx * invbx);
    myBinIndex.y = (int) floorf((REGSIZE_Y * yRegionIndex
          + REGSIZE_Y / 2) * dy * invby);
    myBinIndex.z = (int) floorf((REGSIZE_Z * zRegionIndex
          + REGSIZE_Z / 2) * dz * invbz);

    float energy0 = 0.f;
#if UNROLL_Y >= 2
    float energy1 = 0.f;
#if UNROLL_Y >= 3
    float energy2 = 0.f;
#if UNROLL_Y >= 4
    float energy3 = 0.f;
#endif
#endif
#endif

    for (nbrid = 0;  nbrid < NbrListLen;  nbrid++) {

      int ib = myBinIndex.x + NbrList[nbrid].x;
      int jb = myBinIndex.y + NbrList[nbrid].y;
      int kb = myBinIndex.z + NbrList[nbrid].z;

      if (ib >= 0 && ib < xBinDim &&
          jb >= 0 && jb < yBinDim &&
          kb >= 0 && kb < zBinDim) {

        int n;

        /* thread block caches one bin */
        __syncthreads();
        if (tid < bindepth) {

          /* determine global memory location of atom bin */
          float4 *bin = binBase
            + (((__mul24(kb, yBinDim) + jb)*xBinDim + ib) * bindepth);

          binCache[tid] = bin[tid];
        }
        __syncthreads();

        for (n = 0;  n < bindepth;  n++) {

          float aq = binCache[n].w;
          if (0.f == aq) break;  /* no more atoms in bin */

          float rx = binCache[n].x - x;
          float rz = binCache[n].z - z;
          float rxrz2 = rx*rx + rz*rz;
#ifdef CHECK_CYLINDER
          if (rxrz2 >= cutoff2) continue;
#endif
          float ry = binCache[n].y - y;
          float r2 = ry*ry + rxrz2;

          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy0 += aq * (rsqrtf(r2) - gr2);
          }
#if UNROLL_Y >= 2
          ry -= BLOCK_DIM_Y*dy;
          r2 = ry*ry + rxrz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy1 += aq * (rsqrtf(r2) - gr2);
          }
#if UNROLL_Y >= 3
          ry -= BLOCK_DIM_Y*dy;
          r2 = ry*ry + rxrz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy2 += aq * (rsqrtf(r2) - gr2);
          }
#if UNROLL_Y >= 4
          ry -= BLOCK_DIM_Y*dy;
          r2 = ry*ry + rxrz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy3 += aq * (rsqrtf(r2) - gr2);
          }
#endif
#endif
#endif
        } /* end loop over atoms in bin */
      } /* end if bin in domain */

    } /* end loop over neighbor list */

    /* store into global memory */
#define DSHIFT  (LOG_BDIM_X + LOG_BDIM_Y)
#define REGSXY  (REGSIZE_X * REGSIZE_Y)
#define BDIMXY  (BLOCK_DIM_X * BLOCK_DIM_Y)
#define MASK    (BDIMXY - 1)

    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK)             ] = energy0;
#if UNROLL_Y >= 2
    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK) + (  BDIMXY)] = energy1;
#if UNROLL_Y >= 3
    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK) + (2*BDIMXY)] = energy2;
#if UNROLL_Y >= 4
    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK) + (3*BDIMXY)] = energy3;
#endif
#endif
#endif

  } /* end loop over zRegionIndex */

}
/* end cuda_shortrange_nonperiodic() */


/*
 * cuda_shortrange()  - periodic dimensions wrap bin neighborhood
 */
__global__ static void cuda_shortrange(
    float *regionBase,  /* address of map regions */
    int zRegionDim,     /* z region dimension (x, y given by gridDim) */
    float dx,           /* map spacing along x-axis */
    float dy,           /* map spacing along y-axis */
    float dz,           /* map spacing along z-axis */
    float rx0,          /* x offset of map points, rx0 = px0 - lx0 */
    float ry0,          /* y offset of map points, ry0 = py0 - ly0 */
    float rz0,          /* z offset of map points, rz0 = pz0 - lz0 */
    float px,           /* domain length along x dimension */
    float py,           /* domain length along y dimension */
    float pz,           /* domain length along z dimension */
    int isperiodic,     /* bit flags for periodicity in each dimension */
    float4 *binBase,    /* address of bins */
    int xBinDim,        /* x bin dimension */
    int yBinDim,        /* y bin dimension */
    int zBinDim,        /* z bin dimension */
    int bindepth,       /* number of atom slots per bin */
    float invbx,        /* inverse of bin length in x */
    float invby,        /* inverse of bin length in y */
    float invbz,        /* inverse of bin length in z */
    float cutoff2,      /* square of cutoff distance */
    float invcut        /* inverse of cutoff distance */
    ) {

  __shared__ float4 binCache[MAX_BIN_DEPTH];
  __shared__ float *myRegionAddr;
  __shared__ int3 myBinIndex;

  const int xRegionIndex = blockIdx.x;
  const int yRegionIndex = blockIdx.y;

  /* thread id */
  const int tid = (threadIdx.z*BLOCK_DIM_Y + threadIdx.y)*BLOCK_DIM_X
    + threadIdx.x;

  /* neighbor index */
  int nbrid;

  /* constants for TAYLOR2 softening */
  /* XXX is it more efficient to read these values from const memory? */
  float gc0, gc1, gc2;
  gc1 = invcut * invcut;
  gc2 = gc1 * gc1;
  gc0 = 1.875f * invcut;
  gc1 *= -1.25f * invcut;
  gc2 *= 0.375f * invcut;

  int zRegionIndex;
  for (zRegionIndex=0; zRegionIndex < zRegionDim; zRegionIndex++) {

    /* this is the start of the sub-region indexed by tid */
    myRegionAddr = regionBase + ((zRegionIndex*gridDim.y
          + yRegionIndex)*gridDim.x + xRegionIndex)*REGION_SIZE;
      
    /* spatial coordinate of this lattice point */
    float x = (REGSIZE_X * xRegionIndex + threadIdx.x) * dx - rx0;
    float y = (REGSIZE_Y * yRegionIndex + threadIdx.y) * dy - ry0;
    float z = (REGSIZE_Z * zRegionIndex + threadIdx.z) * dz - rz0;

    /* bin number determined by center of region */
    myBinIndex.x = (int) floorf((REGSIZE_X * xRegionIndex
          + REGSIZE_X / 2) * dx * invbx);
    myBinIndex.y = (int) floorf((REGSIZE_Y * yRegionIndex
          + REGSIZE_Y / 2) * dy * invby);
    myBinIndex.z = (int) floorf((REGSIZE_Z * zRegionIndex
          + REGSIZE_Z / 2) * dz * invbz);

    float energy0 = 0.f;
#if UNROLL_Y >= 2
    float energy1 = 0.f;
#if UNROLL_Y >= 3
    float energy2 = 0.f;
#if UNROLL_Y >= 4
    float energy3 = 0.f;
#endif
#endif
#endif

    for (nbrid = 0;  nbrid < NbrListLen;  nbrid++) {

      int ib = myBinIndex.x + NbrList[nbrid].x;
      int jb = myBinIndex.y + NbrList[nbrid].y;
      int kb = myBinIndex.z + NbrList[nbrid].z;

      float xw = 0;  /* periodic offset for wrapping coordinates */
      float yw = 0;
      float zw = 0;

      if (IS_SET_X(isperiodic)) {
#if 1
        if (ib < 0) {
          ib += xBinDim;
          xw -= px;
        }
        else if (ib >= xBinDim) {
          ib -= xBinDim;
          xw += px;
        }
#else
        while (ib < 0)        { ib += xBinDim;  xw -= px; }
        while (ib >= xBinDim) { ib -= xBinDim;  xw += px; }
#endif
      }
      else if (ib < 0 || ib >= xBinDim) continue;

      if (IS_SET_Y(isperiodic)) {
#if 1
        if (jb < 0) {
          jb += yBinDim;
          yw -= py;
        }
        else if (jb >= yBinDim) {
          jb -= yBinDim;
          yw += py;
        }
#else
        while (jb < 0)        { jb += yBinDim;  yw -= py; }
        while (jb >= yBinDim) { jb -= yBinDim;  yw += py; }
#endif
      }
      else if (jb < 0 || jb >= yBinDim) continue;

      if (IS_SET_Z(isperiodic)) {
#if 1
        if (kb < 0) {
          kb += zBinDim;
          zw -= pz;
        }
        else if (kb >= zBinDim) {
          kb -= zBinDim;
          zw += pz;
        }
#else
        while (kb < 0)        { kb += zBinDim;  zw -= pz; }
        while (kb >= zBinDim) { kb -= zBinDim;  zw += pz; }
#endif
      }
      else if (kb < 0 || kb >= zBinDim) continue;

      {
        int n;

        /* thread block caches one bin */
        __syncthreads();
        if (tid < bindepth) {

          /* determine global memory location of atom bin */
          float4 *bin = binBase
            + (((__mul24(kb, yBinDim) + jb)*xBinDim + ib) * bindepth);

          binCache[tid] = bin[tid];
        }
        __syncthreads();

        for (n = 0;  n < bindepth;  n++) {

          float aq = binCache[n].w;
          if (0.f == aq) break;  /* no more atoms in bin */

          float rx = (binCache[n].x+xw) - x;
          float rz = (binCache[n].z+zw) - z;
          float rxrz2 = rx*rx + rz*rz;
#ifdef CHECK_CYLINDER
          if (rxrz2 >= cutoff2) continue;
#endif
          float ry = (binCache[n].y+yw) - y;
          float r2 = ry*ry + rxrz2;

          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy0 += aq * (rsqrtf(r2) - gr2);
          }
#if UNROLL_Y >= 2
          ry -= BLOCK_DIM_Y*dy;
          r2 = ry*ry + rxrz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy1 += aq * (rsqrtf(r2) - gr2);
          }
#if UNROLL_Y >= 3
          ry -= BLOCK_DIM_Y*dy;
          r2 = ry*ry + rxrz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy2 += aq * (rsqrtf(r2) - gr2);
          }
#if UNROLL_Y >= 4
          ry -= BLOCK_DIM_Y*dy;
          r2 = ry*ry + rxrz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy3 += aq * (rsqrtf(r2) - gr2);
          }
#endif
#endif
#endif
        } /* end loop over atoms in bin */
      } /* end if bin in domain */

    } /* end loop over neighbor list */

    /* store into global memory */
#define DSHIFT  (LOG_BDIM_X + LOG_BDIM_Y)
#define REGSXY  (REGSIZE_X * REGSIZE_Y)
#define BDIMXY  (BLOCK_DIM_X * BLOCK_DIM_Y)
#define MASK    (BDIMXY - 1)

    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK)             ] = energy0;
#if UNROLL_Y >= 2
    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK) + (  BDIMXY)] = energy1;
#if UNROLL_Y >= 3
    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK) + (2*BDIMXY)] = energy2;
#if UNROLL_Y >= 4
    myRegionAddr[(tid>>DSHIFT)*REGSXY + (tid&MASK) + (3*BDIMXY)] = energy3;
#endif
#endif
#endif

  } /* end loop over zRegionIndex */

}
/* end cuda_shortrange() */



/*
 * call when finished
 */
void Msmpot_cuda_cleanup_shortrng(MsmpotCuda *mc) {
  cudaFree(mc->dev_bin);
  cudaFree(mc->dev_padmap);
  free(mc->padmap);
}


/*
 * call once or whenever parameters are changed
 */
int Msmpot_cuda_setup_shortrng(MsmpotCuda *mc) {
  Msmpot *msm = mc->msmpot;
  const int mx = msm->mx;
  const int my = msm->my;
  const int mz = msm->mz;
  int rmx, rmy, rmz;
  int pmx, pmy, pmz;
  long pmall;
  const int nbins = (msm->nbx * msm->nby * msm->nbz);
  int rc;

  /* count "regions" of map points in each dimension, rounding up */
  rmx = (mx >> LOG_REGS_X) + ((mx & MASK_REGS_X) ? 1 : 0);
  rmy = (my >> LOG_REGS_Y) + ((my & MASK_REGS_Y) ? 1 : 0);
  rmz = (mz >> LOG_REGS_Z) + ((mz & MASK_REGS_Z) ? 1 : 0);

  /* padded epotmap dimensions */
  pmx = (rmx << LOG_REGS_X);
  pmy = (rmy << LOG_REGS_Y);
  pmz = (rmz << LOG_REGS_Z);

  /* allocate space for padded epotmap */
  pmall = (pmx * pmy) * (long)pmz;
  if (mc->maxpm < pmall) {
    void *v = realloc(mc->padmap, pmall * sizeof(float));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    mc->padmap = (float *) v;
    mc->maxpm = pmall;
  }
  mc->pmx = pmx;
  mc->pmy = pmy;
  mc->pmz = pmz;

  REPORT("Determine bin neighborhood for CUDA short-range part.");
  rc = Msmpot_compute_shortrng_bin_neighborhood(msm,
      REGSIZE_X * msm->dx, REGSIZE_Y * msm->dy, REGSIZE_Z * msm->dz);
  if (rc != MSMPOT_SUCCESS) return ERROR(rc);

  /*
   * protect CUDA periodic case against having to wrap multiple times
   * around the domain for an atom bin
   */
  if (msm->isperiodic) {
    int n;
    int ibmax = 0, jbmax = 0, kbmax = 0;
    for (n = 0;  n < msm->nboff;  n++) {
      int ib = msm->boff[3*n  ];
      int jb = msm->boff[3*n+1];
      int kb = msm->boff[3*n+2];
      if (ib < 0)      ib = -ib;
      if (ibmax < ib)  ibmax = ib;
      if (jb < 0)      jb = -jb;
      if (jbmax < jb)  jbmax = jb;
      if (kb < 0)      kb = -kb;
      if (kbmax < kb)  kbmax = kb;
    }
    if (ibmax >= msm->nbx || jbmax >= msm->nby || kbmax >= msm->nbz) {
      REPORT("Bin neighborhood is too big for wrapping with CUDA.");
      return ERROR(MSMPOT_ERROR_CUDA_SUPPORT);
    }
  }

  /*
   * allocate CUDA device memory
   * (for now make CUDA arrays same length as host arrays)
   */
  if (mc->dev_maxpm < mc->maxpm) {
    void *v = NULL;
    cudaFree(mc->dev_padmap);
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    cudaMalloc(&v, mc->maxpm * sizeof(float));
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    mc->dev_padmap = (float *) v;
    mc->dev_maxpm = mc->maxpm;
  }

  if (mc->dev_nbins < nbins) {
    void *v = NULL;
    cudaFree(mc->dev_bin);
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    cudaMalloc(&v, nbins * msm->bindepth * sizeof(float4));
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    mc->dev_bin = (float4 *) v;
    mc->dev_nbins = nbins;
  }

  /*
   * copy region neighborhood atom bin index offsets
   * to device constant memory
   */
  cudaMemcpyToSymbol(NbrListLen, &(msm->nboff), sizeof(int), 0);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);
  cudaMemcpyToSymbol(NbrList, msm->boff, msm->nboff * sizeof(int3), 0);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);

  return MSMPOT_SUCCESS;
}


int Msmpot_cuda_compute_shortrng(MsmpotCuda *mc) {
  Msmpot *msm = mc->msmpot;
  float *epotmap = msm->epotmap;
  float *padmap = mc->padmap;
  const int nxbins = msm->nbx;
  const int nybins = msm->nby;
  const int nzbins = msm->nbz;
  const int nbins = nxbins * nybins * nzbins;
  const int mx = msm->mx;
  const int my = msm->my;
  const int mz = msm->mz;
  const int mxRegions = (mc->pmx >> LOG_REGS_X);
  const int myRegions = (mc->pmy >> LOG_REGS_Y);
  const int mzRegions = (mc->pmz >> LOG_REGS_Z);
  const long pmall = (mc->pmx * mc->pmy) * (long) mc->pmz;
  const float cutoff2 = msm->a * msm->a;
  const float invcut = 1.f / msm->a;
  const float rx0 = msm->px0 - msm->lx0;  /* translation for grid points */
  const float ry0 = msm->py0 - msm->ly0;
  const float rz0 = msm->pz0 - msm->lz0;
  int i, j, k;
  int mxRegionIndex, myRegionIndex, mzRegionIndex;
  int mxOffset, myOffset, mzOffset;
  long indexRegion, index;
  float *thisRegion;
  dim3 gridDim, blockDim;
  cudaStream_t shortrng_stream;
  int rc = MSMPOT_SUCCESS;

  REPORT("Perform atom hashing for CUDA short-range part.");
  rc = Msmpot_compute_shortrng_bin_hashing(msm);
  if (rc != MSMPOT_SUCCESS) return ERROR(rc);

#ifdef MSMPOT_REPORT
  if (msm->isperiodic) {
    REPORT("Computing periodic short-range part with CUDA.");
  }
  else {
    REPORT("Computing short-range part with CUDA.");
  }
#endif

  /* copy atom bins to device */
  cudaMemcpy(mc->dev_bin, msm->bin, nbins * msm->bindepth * sizeof(float4),
      cudaMemcpyHostToDevice);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);

  gridDim.x = mxRegions;
  gridDim.y = myRegions;
  gridDim.z = 1;
  blockDim.x = BLOCK_DIM_X;
  blockDim.y = BLOCK_DIM_Y;
  blockDim.z = BLOCK_DIM_Z;

  cudaStreamCreate(&shortrng_stream);  /* asynchronously invoke CUDA kernel */
  if (msm->isperiodic) {
    cuda_shortrange<<<gridDim, blockDim, 0>>>(
        mc->dev_padmap, mzRegions,
        msm->dx, msm->dy, msm->dz,
        rx0, ry0, rz0,
        msm->px, msm->py, msm->pz, msm->isperiodic,
        mc->dev_bin, nxbins, nybins, nzbins, msm->bindepth,
        msm->invbx, msm->invby, msm->invbz,
        cutoff2, invcut);
  }
  else {
    cuda_shortrange_nonperiodic<<<gridDim, blockDim, 0>>>(
        mc->dev_padmap, mzRegions,
        msm->dx, msm->dy, msm->dz,
        rx0, ry0, rz0,
        mc->dev_bin, nxbins, nybins, nzbins, msm->bindepth,
        msm->invbx, msm->invby, msm->invbz,
        cutoff2, invcut);
  }
  if (msm->nover > 0) {  /* call CPU to concurrently compute extra atoms */
#ifdef MSMPOT_REPORT
    char msg[120];
    sprintf(msg, "Extra atoms (%d) from overflowed bins "
        "must also be evaluated.", msm->nover);
    REPORT(msg);
#endif
    rc = Msmpot_compute_shortrng_linklist(mc->msmpot, msm->over, msm->nover);
    if (rc) return ERROR(rc);
  }
  cudaStreamSynchronize(shortrng_stream);
  CUERR(MSMPOT_ERROR_CUDA_KERNEL);
  cudaDeviceSynchronize();
  cudaStreamDestroy(shortrng_stream);

  /* copy result regions from CUDA device */
  cudaMemcpy(padmap, mc->dev_padmap, pmall * sizeof(float),
      cudaMemcpyDeviceToHost);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);

  /* transpose regions from padEpotmap and add into result epotmap */
  for (k = 0;  k < mz;  k++) {
    mzRegionIndex = (k >> LOG_REGS_Z);
    mzOffset = (k & MASK_REGS_Z);

    for (j = 0;  j < my;  j++) {
      myRegionIndex = (j >> LOG_REGS_Y);
      myOffset = (j & MASK_REGS_Y);

      for (i = 0;  i < mx;  i++) {
        mxRegionIndex = (i >> LOG_REGS_X);
        mxOffset = (i & MASK_REGS_X);

        thisRegion = padmap
          + ((mzRegionIndex * myRegions + myRegionIndex) * (long) mxRegions
              + mxRegionIndex) * REGION_SIZE;

        indexRegion = (mzOffset * REGSIZE_Y + myOffset) * REGSIZE_X + mxOffset;
        index = (k * my + j) * mx + i;

        epotmap[index] += thisRegion[indexRegion];
      }
    }
  }

  return MSMPOT_SUCCESS;
}
