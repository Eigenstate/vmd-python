#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mgpot_cuda.h"


float splitting(float s) {
  return (1 + (s-1)*(-1.f/2 + (s-1)*(3.f/8)));
}


__constant__ int4 sinfo[MAXLEVELS];
/* subcube info: { {nx, ny, nz, bid_start}, ... } lookup table for each level,
 * where bid_start is starting index for level's threadblocks in flat array,
 * region of subcubes corresponding to threadblocks is surrounded by
 * padding layer of subcubes
 */

__constant__ float lfac[MAXLEVELS];
/* constant factor for each level: {1, 1/2, 1/4, ... } */

__constant__ float wt[24*24*24];
/* __constant__ float topwt[8*16*16]; (not used) */


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

#if 0
  if (0==blockIdx.x && 0==threadIdx.x && 0==threadIdx.y && 0==threadIdx.z) {
    printf("lk:  nlevels=%d  srad=%d\n", nlevels, srad);
  }
#endif

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

#undef FOLLOW
#ifdef FOLLOW
  int fb = 0;
  int fx = 0;
  int fy = 0;
  int fz = 0;
  int follow = 0;
  if (fb==blockIdx.x && fx==tx && fy==ty && fz==tz) {
    follow = 1;
    printf("following subcube=%d tx=%d ty=%d tz=%d\n", fb, fx, fy, fz);
  }
  int once = 1;
#endif

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

#if 0
  if (0==blockIdx.x && 0==threadIdx.x && 0==threadIdx.y && 0==threadIdx.z) {
    printf("sid=%d\n", sid);
    printf("level=%d\n", level);
    printf("sinfo[level].w=%d\n", sinfo[level].w);
    printf("nx=%d  ny=%d  nz=%d\n", nx, ny, nz);
    printf("sx=%d  sy=%d  sz=%d\n", sx, sy, sz);
    printf("tx=%d  ty=%d  tz=%d\n", tx, ty, tz);
    printf("ax=%d  ay=%d  az=%d\n", ax, ay, az);
    printf("bx=%d  by=%d  bz=%d\n", bx, by, bz);
  }
#endif

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

#if 0
        if (follow && once) {
          once = 0;
        }
#endif

        /* lowest corner of weight subcube to apply */
        int wx = (mx-sx+srad)*4;
        int wy = (my-sy+srad)*4;
        int wz = (mz-sz+srad)*4;

        for (k = 0;  k < 4;  k++) {
          for (j = 0;  j < 4;  j++) {
            for (i = 0;  i < 4;  i++) {

#if 0
              float h = 2.f;
              float a = 12.f;

              int dx = 4*(mx-sx) + i;
              int dy = 4*(my-sy) + j;
              int dz = 4*(mz-sz) + k;
              if (0==blockIdx.x && 0==threadIdx.x && 0==threadIdx.y &&
                  0==threadIdx.z) {
                printf("mx=%d  my=%d  mz=%d\n", mx, my, mz);
                printf("sx=%d  sy=%d  sz=%d\n", sx, sy, sz);
                printf("dx=%d  dy=%d  dz=%d\n", dx, dy, dz);
              }

              float s = (dx*dx + dy*dy + dz*dz) * h*h/(a*a);
              float t = 0.25f * s;

              float gd;
              float gs, gt;

              if (t >= 1) {
                gd = 0;
              }
              else if (s >= 1) {
                gs = 1/sqrtf(s);
                /* mgpot_split(&gt, t, TAYLOR2); */
                gt = splitting(t);
                //gd = lfac[level] * (gs - 0.5f * gt)/a;
                gd = (gs - 0.5f * gt)/a;
              }
              else {
                /* mgpot_split(&gs, s, split); */
                /* mgpot_split(&gt, t, split); */
                gs = splitting(s);
                gt = splitting(t);
                //gd = lfac[level] * (gs - 0.5f * gt)/a;
                gd = (gs - 0.5f * gt)/a;
              }
#endif

              /* e += cq[tx+i, ty+j, tz+k] * wt[wx+i, wy+j, wz+k] */

              /*
              e += cq[ ((tz+k)*8 + (ty+j))*8 + (tx+i) ] *
                wt[ ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+i) ];
                */

              int cq_index = ((tz+k)*8 + (ty+j))*8 + (tx+i);
              int wt_index = ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+i);
              e += cq[cq_index] * wt[wt_index];

#ifdef FOLLOW
              if (follow && cq[cq_index]!=0 && wt[wt_index]!=0) {
                printf("cq[%d]=%g  wt[%d]=%g  e+=%g\n", cq_index, cq[cq_index],
                    wt_index, wt[wt_index], cq[cq_index]*wt[wt_index]);
              }
#endif

#if 0
              if (0==blockIdx.x && 0==threadIdx.x && 0==threadIdx.y &&
                  0==threadIdx.z) {
                printf("calc wt = %g    lookup wt = %g\n", gd,
                    wt[ ((wz+k)*(8*srad) + (wy+j))*(8*srad) + (wx+i) ]);
              }
#endif

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


extern "C" int mgpot_cuda_latcut01(Mgpot *mg) {
  const int nlevels = mg->lk_nlevels;
  const int srad = mg->lk_srad;
  const int padding = mg->lk_padding;
  const int wt_total = (8*srad) * (8*srad) * (8*srad);
  const long memsz = mg->subcube_total * SUBCUBESZ * sizeof(float);

  dim3 gridDim, blockDim;
 
  //printf("lk_nlevels = %d\n", nlevels);
  //printf("subcube_total = %d\n", mg->subcube_total);

  gridDim.x = mg->block_total;
  printf("gridDim.x=%d\n", gridDim.x);
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
