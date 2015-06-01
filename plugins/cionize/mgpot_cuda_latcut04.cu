/* mgpot_cuda_latcut02.cu */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mgpot_cuda.h"

#undef PRECOMP_1
#define PRECOMP_1

#undef UNROLL_1
#define UNROLL_1

#undef SHMEMCOPY
#define SHMEMCOPY

#undef UNROLL_2
#define UNROLL_2

__constant__ int4 sinfo[MAXLEVELS];
/* subcube info: { {nx, ny, nz, bid_start}, ... } lookup table for each level,
 * where bid_start is starting index for level's threadblocks in flat array,
 * region of subcubes corresponding to threadblocks is surrounded by
 * padding layer of subcubes
 */

__constant__ float lfac[MAXLEVELS];
/* constant factor for each level: {1, 1/2, 1/4, ... } */

__constant__ float wt[24*24*24];


__global__ static void latcut_kernel(
    unsigned int nsubcubes,  /* total number of subcubes */
    int nlevels,     /* number of levels */
    int srad,        /* subcube radius */
    int padding,     /* how many subcubes padding around boundary */
    float *qgrids,   /* grids of charge, collapsed into flat array */
    float *egrids    /* grids of potential, collapsed into flat array */
    )
{
  /* assert:  nlevels <= MAXLEVELS */
  /* assert:  srad <= 3 */
  /* assert:  1 <= padding <= srad */
  /* blockIdx.x is the flat index from start of inner subcube array */
  int soff = 0;          /* subcube offset to this level */
  int level = 0;         /* what level am I? */

  __shared__ float cq[8*8*8];

  unsigned int block_index = gridDim.x * blockIdx.y + blockIdx.x;

  if (block_index >= nsubcubes) return;

  /* find my soff and level in sinfo lookup table */
  while (block_index >= sinfo[level].w) {
    soff += (sinfo[level].x * sinfo[level].y * sinfo[level].z);
    level++;
  }

  /* start of charge grid for this level */
  float *qlevel = qgrids + soff * SUBCUBESZ;

  /* calculate my 3D subcube index relative to my level */
  int nx = sinfo[level].x;  /* number of subcubes in x-dim */
  int ny = sinfo[level].y;  /* number of subcubes in y-dim */
  int nz = sinfo[level].z;  /* number of subcubes in z-dim */

  int boff = block_index - (level > 0 ? sinfo[level-1].w : 0);
    /* threadblock offset */
  int sbx = nx - 2*padding;  /* dimensions of inner subcube array */
  int sby = ny - 2*padding;
  /* int sbz = nz - 2*padding; */
  /* assert:  0 <= boff < sbx*sby*sbz */
  /* need to find (sx,sy,sz) subcube index (inner array + padding) */
  int nrow = boff / sbx;
  int sx = (boff % sbx) + padding;  /* x subcube index (pad <= sx < nx-pad) */
  int sz = (nrow / sby) + padding;  /* z subcube index (pad <= sz < nz-pad) */
  int sy = (nrow % sby) + padding;  /* y subcube index (pad <= sy < ny-pad) */

  int tx = threadIdx.x;     /* thread index within subcube */
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  /* find index boundaries of subcube neighbors */
  int ax = sx - srad;
  if (ax < 0) ax = 0;
  int bx = sx + srad;
  if (bx >= nx) bx = nx-1;
  int ay = sy - srad;
  if (ay < 0) ay = 0;
  int by = sy + srad;
  if (by >= ny) by = ny-1;
  int az = sz - srad;
  if (az < 0) az = 0;
  int bz = sz + srad;
  if (bz >= nz) bz = nz-1;
  /* assert:  ax < bx, ay < by, az < bz */
  int mx, my, mz;

  /* flat thread ID */
  int tid = (tz*4 + ty)*4 + tx;

  /* loop over neighboring subcubes */
  float e = 0;
  for (mz = az;  mz < bz;  mz++) {
    for (my = ay;  my < by;  my++) {
#ifdef SHMEMCOPY
      int addr;
      float *q;

#ifdef UNROLL_2
      q = qlevel + (((mz+0)*ny + (my+0))*nx + ax) * SUBCUBESZ;
      addr = ((tz + 4*0)*8 + (ty + 4*0))*8 + tx;
      cq[addr+4] = q[tid];

      q = qlevel + (((mz+0)*ny + (my+1))*nx + ax) * SUBCUBESZ;
      addr = ((tz + 4*0)*8 + (ty + 4*1))*8 + tx;
      cq[addr+4] = q[tid];

      q = qlevel + (((mz+1)*ny + (my+0))*nx + ax) * SUBCUBESZ;
      addr = ((tz + 4*1)*8 + (ty + 4*0))*8 + tx;
      cq[addr+4] = q[tid];

      q = qlevel + (((mz+1)*ny + (my+1))*nx + ax) * SUBCUBESZ;
      addr = ((tz + 4*1)*8 + (ty + 4*1))*8 + tx;
      cq[addr+4] = q[tid];
#else
      int j, k;
      /* copy 4 more subcubes from global memory */
      for (k = 0;  k < 2;  k++) {
        for (j = 0;  j < 2;  j++) {
          int qsid = ((mz+k)*ny + (my+j))*nx + ax;
          float *q = qlevel + qsid * SUBCUBESZ;
          int addr = ((tz + 4*k)*8 + (ty + 4*j))*8 + tx;
          cq[addr+4] = q[tid];
        }
      }
#endif /* UNROLL_2 */

#endif
      for (mx = ax;  mx < bx;  mx++) {

#if !defined(SHMEMCOPY)
        int i, j, k;

        /* read block of 8 subcubes starting at (mx,my,mz) into shared mem */
        for (k = 0;  k < 2;  k++) {
          for (j = 0;  j < 2;  j++) {
            for (i = 0;  i < 2;  i++) {

              int qsid = ((mz+k)*ny + (my+j))*nx + (mx+i);
              float *q = qlevel + qsid * SUBCUBESZ;

              int idest = ((tz + 4*k)*8 + (ty + 4*j))*8 + (tx + 4*i);

              cq[idest] = q[tid];  /* coalesced read from global mem */
            }
          }
        }
#else

#ifdef UNROLL_2
        /* shift 4 subcubes in shared memory */
        addr = ((tz + 4*0)*8 + (ty + 4*0))*8 + tx;
        cq[addr] = cq[addr+4];

        addr = ((tz + 4*0)*8 + (ty + 4*1))*8 + tx;
        cq[addr] = cq[addr+4];

        addr = ((tz + 4*1)*8 + (ty + 4*0))*8 + tx;
        cq[addr] = cq[addr+4];

        addr = ((tz + 4*1)*8 + (ty + 4*1))*8 + tx;
        cq[addr] = cq[addr+4];

        /* copy 4 more subcubes from global memory */
        q = qlevel + (((mz+0)*ny + (my+0))*nx + (mx+1)) * SUBCUBESZ;
        addr = ((tz + 4*0)*8 + (ty + 4*0))*8 + tx;
        cq[addr+4] = q[tid];

        q = qlevel + (((mz+0)*ny + (my+1))*nx + (mx+1)) * SUBCUBESZ;
        addr = ((tz + 4*0)*8 + (ty + 4*1))*8 + tx;
        cq[addr+4] = q[tid];

        q = qlevel + (((mz+1)*ny + (my+0))*nx + (mx+1)) * SUBCUBESZ;
        addr = ((tz + 4*1)*8 + (ty + 4*0))*8 + tx;
        cq[addr+4] = q[tid];

        q = qlevel + (((mz+1)*ny + (my+1))*nx + (mx+1)) * SUBCUBESZ;
        addr = ((tz + 4*1)*8 + (ty + 4*1))*8 + tx;
        cq[addr+4] = q[tid];
#else
        /* shift 4 subcubes in shared memory */
        for (k = 0;  k < 2;  k++) {
          for (j = 0;  j < 2;  j++) {
            int addr = ((tz + 4*k)*8 + (ty + 4*j))*8 + tx;
            cq[addr] = cq[addr+4];
          }
        }
        /* copy 4 more subcubes from global memory */
        for (k = 0;  k < 2;  k++) {
          for (j = 0;  j < 2;  j++) {
            int qsid = ((mz+k)*ny + (my+j))*nx + (mx+1);
            q = qlevel + qsid * SUBCUBESZ;
            addr = ((tz + 4*k)*8 + (ty + 4*j))*8 + tx;
            cq[addr+4] = q[tid];
          }
        }
#endif /* UNROLL_2 */

#endif
        __syncthreads();

        /* lowest corner of weight subcube to apply */
        int wx = (mx-sx+srad)*4;
        int wy = (my-sy+srad)*4;
        int wz = (mz-sz+srad)*4;

#if defined(SHMEMCOPY) && defined(UNROLL_2) && defined(UNROLL_1)
        int j, k;
#elif defined(SHMEMCOPY) && defined(UNROLL_2)
        int i, j, k;
#elif defined(SHMEMCOPY)
        int i;
#endif

        for (k = 0;  k < 4;  k++) {
          for (j = 0;  j < 4;  j++) {
#ifdef PRECOMP_1
            int cq_index_jk = ((tz+k)*8 + (ty+j))*8;
            int wt_index_jk = ((wz+k)*(8*srad) + (wy+j))*(8*srad);
#endif

#if !defined(UNROLL_1)
            for (i = 0;  i < 4;  i++) {
#if !defined(PRECOMP_1)
              int cq_index = ((tz+k)*8 + (ty+j))*8 + (tx+i);
              int wt_index = ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+i);
#else
              int cq_index = cq_index_jk + (tx+i);
              int wt_index = wt_index_jk + (wx+i);
#endif
              e += cq[cq_index] * wt[wt_index];

            }
#else
            int cq_index;
            int wt_index;

#if !defined(PRECOMP_1)
            cq_index = ((tz+k)*8 + (ty+j))*8 + (tx+0);
            wt_index = ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+0);
#else
            cq_index = cq_index_jk + (tx+0);
            wt_index = wt_index_jk + (wx+0);
#endif
            e += cq[cq_index] * wt[wt_index];

#if !defined(PRECOMP_1)
            cq_index = ((tz+k)*8 + (ty+j))*8 + (tx+1);
            wt_index = ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+1);
#else
            cq_index = cq_index_jk + (tx+1);
            wt_index = wt_index_jk + (wx+1);
#endif
            e += cq[cq_index] * wt[wt_index];

#if !defined(PRECOMP_1)
            cq_index = ((tz+k)*8 + (ty+j))*8 + (tx+2);
            wt_index = ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+2);
#else
            cq_index = cq_index_jk + (tx+2);
            wt_index = wt_index_jk + (wx+2);
#endif
            e += cq[cq_index] * wt[wt_index];

#if !defined(PRECOMP_1)
            cq_index = ((tz+k)*8 + (ty+j))*8 + (tx+3);
            wt_index = ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+3);
#else
            cq_index = cq_index_jk + (tx+3);
            wt_index = wt_index_jk + (wx+3);
#endif
            e += cq[cq_index] * wt[wt_index];
#endif /* UNROLL_1 */

          }
        }
        __syncthreads();

      }
    }
  }
  e *= lfac[level];  /* scale by weighting factor for this level */

  float *eout = egrids + (soff + (sz*ny + sy)*nx + sx) * SUBCUBESZ;
  eout[ (tz*4 + ty)*4 + tx ] = e;
}


extern "C" int mgpot_cuda_latcut04(Mgpot *mg) {
  const int nlevels = mg->lk_nlevels;
  const int srad = mg->lk_srad;
  const int padding = mg->lk_padding;
  const int wt_total = (8*srad) * (8*srad) * (8*srad);
  const long memsz = mg->subcube_total * SUBCUBESZ * sizeof(float);

  dim3 gridDim, blockDim;

  unsigned int bx = mg->block_total;
  unsigned int by = 1;
#define MAX_GRID_DIM  65536u

  while (bx > MAX_GRID_DIM) {
    bx >>= 1;  /* divide by 2 */
    by <<= 1;  /* multiply by 2 */
  }
  if (bx * by < (unsigned int)(mg->block_total)) bx++;
  if (bx > MAX_GRID_DIM || by > MAX_GRID_DIM) {
    return ERROR("sub-cube array length %lu is too large to launch kernel\n",
        (unsigned long)(mg->block_total));
  }
 
  gridDim.x = (int) bx;
  gridDim.y = (int) by;
  gridDim.z = 1;

  blockDim.x = 4;
  blockDim.y = 4;
  blockDim.z = 4;

  /* copy some host memory data into device constant memory */
  cudaMemcpyToSymbol(sinfo, mg->host_sinfo, nlevels * sizeof(int4), 0);
  CUERR;
  cudaMemcpyToSymbol(lfac, mg->host_lfac, nlevels * sizeof(float), 0);
  CUERR;
  cudaMemcpyToSymbol(wt, mg->host_wt, wt_total * sizeof(float), 0);
  CUERR;

  /* copy qgrids from host to device */
  cudaMemcpy(mg->device_qgrids, mg->host_qgrids, memsz, cudaMemcpyHostToDevice);
  CUERR;

  /* invoke kernel */
  printf("gridDim.x=%d\n", gridDim.x);
  printf("gridDim.y=%d\n", gridDim.y);
  printf("nsubcubes=%u  (using %u extra thread blocks)\n",
      (unsigned int)(mg->block_total),
      (gridDim.x*gridDim.y - (unsigned int)(mg->block_total)));
  printf("nlevels=%d\n", nlevels);
  printf("srad=%d\n", srad);
  printf("padding=%d\n", padding);
  printf("address of qgrids=%lx\n", (long) (mg->device_qgrids));
  printf("address of egrids=%lx\n", (long) (mg->device_egrids));
  latcut_kernel<<<gridDim, blockDim, 0>>>((unsigned int)(mg->block_total),
      nlevels, srad, padding, mg->device_qgrids, mg->device_egrids);
  CUERR;

  /* copy egrids from device to host */
  cudaMemcpy(mg->host_egrids, mg->device_egrids, memsz, cudaMemcpyDeviceToHost);
  CUERR;

  return 0;
}
