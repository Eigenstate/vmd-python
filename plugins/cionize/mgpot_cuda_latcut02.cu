/* mgpot_cuda_latcut02.cu */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mgpot_cuda.h"


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

  /* find my soff and level in sinfo lookup table */
  while (blockIdx.x >= sinfo[level].w) {
    soff += (sinfo[level].x * sinfo[level].y * sinfo[level].z);
    level++;
  }

  /* start of charge grid for this level */
  float *qlevel = qgrids + soff * SUBCUBESZ;

  /* calculate my 3D subcube index relative to my level */
  int nx = sinfo[level].x;  /* number of subcubes in x-dim */
  int ny = sinfo[level].y;  /* number of subcubes in y-dim */
  int nz = sinfo[level].z;  /* number of subcubes in z-dim */

  int boff = blockIdx.x - (level > 0 ? sinfo[level-1].w : 0);
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

  /* loop over neighboring subcubes */
  float e = 0;
  for (mz = az;  mz < bz;  mz++) {
    for (my = ay;  my < by;  my++) {
      for (mx = ax;  mx < bx;  mx++) {
        int i, j, k;

        /* flat thread ID */
        int isrc = (tz*4 + ty)*4 + tx;

        /* read block of 8 subcubes starting at (mx,my,mz) into shared mem */
        for (k = 0;  k < 2;  k++) {
          for (j = 0;  j < 2;  j++) {
            for (i = 0;  i < 2;  i++) {

              int qsid = ((mz+k)*ny + (my+j))*nx + (mx+i);
              float *q = qlevel + qsid * SUBCUBESZ;

              int idest = ((tz + 4*k)*8 + (ty + 4*j))*8 + (tx + 4*i);

              cq[idest] = q[isrc];  /* coalesced read from global mem */
            }
          }
        }
        __syncthreads();

        /* lowest corner of weight subcube to apply */
        int wx = (mx-sx+srad)*4;
        int wy = (my-sy+srad)*4;
        int wz = (mz-sz+srad)*4;

        for (k = 0;  k < 4;  k++) {
          for (j = 0;  j < 4;  j++) {
            for (i = 0;  i < 4;  i++) {

              int cq_index = ((tz+k)*8 + (ty+j))*8 + (tx+i);
              int wt_index = ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+i);
              e += cq[cq_index] * wt[wt_index];

            }
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


extern "C" int mgpot_cuda_latcut02(Mgpot *mg) {
  const int nlevels = mg->lk_nlevels;
  const int srad = mg->lk_srad;
  const int padding = mg->lk_padding;
  const int wt_total = (8*srad) * (8*srad) * (8*srad);
  const long memsz = mg->subcube_total * SUBCUBESZ * sizeof(float);

  dim3 gridDim, blockDim;
 
  gridDim.x = mg->block_total;
  gridDim.y = 1;
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
  latcut_kernel<<<gridDim, blockDim, 0>>>(nlevels, srad, padding,
      mg->device_qgrids, mg->device_egrids);
  CUERR;

  /* copy egrids from device to host */
  cudaMemcpy(mg->host_egrids, mg->device_egrids, memsz, cudaMemcpyDeviceToHost);
  CUERR;

  return 0;
}
