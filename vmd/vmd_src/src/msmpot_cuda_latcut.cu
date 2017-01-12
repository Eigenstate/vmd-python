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
 *      $RCSfile: msmpot_cuda_latcut.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $      $Date: 2010/06/03 20:07:10 $
 *
 ***************************************************************************/

#include "msmpot_cuda.h"

/* constants for lattice cutoff kernel */
#define MAXLEVELS     28
#define SUBCUBESZ     64
#define LG_SUBCUBESZ   6

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


__global__ static void cuda_latcut(
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


/*
 * call when finished
 */
void Msmpot_cuda_cleanup_latcut(MsmpotCuda *mc) {
  cudaFree(mc->device_qgrids);  /* free device memory allocations */
  cudaFree(mc->device_egrids);
  free(mc->host_qgrids);
  free(mc->host_egrids);
  free(mc->host_wt);
  free(mc->host_sinfo);
  free(mc->host_lfac);  /* free host memory allocations */
}


int Msmpot_cuda_setup_latcut(MsmpotCuda *mc) {
  Msmpot *msm = mc->msmpot;
  const float hx = msm->hx;
  const float hy = msm->hy;
  const float hz = msm->hz;
  float hmin;  /* minimum of hx, hy, hz */
  const float a = msm->a;
  const int split = msm->split;
  const int maxlevels = msm->maxlevels;
  int nlevels = msm->nlevels - 1;  /* don't do top level on GPU */
  int nrad;
  int srad;
  int pad;
  int i, j, k, ii, jj, kk;
  int index;
  long btotal, stotal, maxwts, maxgridpts, memsz;
  float s, t, gs, gt;
  float lfac;
  float *wt;

  if (nlevels > MAXLEVELS) return ERROR(MSMPOT_ERROR_CUDA_SUPPORT);
  mc->lk_nlevels = nlevels;
  hmin = hx;
  if (hmin < hy)  hmin = hy;
  if (hmin < hz)  hmin = hz;
  nrad = (int) ceilf(2*a/hmin) - 1;
  srad = (int) ceilf((nrad + 1) / 4.f);
  if (srad > 3) return ERROR(MSMPOT_ERROR_CUDA_SUPPORT);
  mc->lk_srad = srad;
  if (msm->isperiodic) {
    pad = srad;  /* if periodic in ANY dimension */
  }
  else {
    pad = 1;  /* for non-periodic systems */
  }
  mc->lk_padding = pad;
#ifdef MSMPOT_DEBUG
  printf("a=%g  h=%g\n", a, h);
  printf("nrad=%d\n", nrad);
  printf("srad=%d\n", srad);
  printf("pad=%d\n", pad);
#endif

  if (mc->maxlevels < maxlevels) {
    void *v;
    v = realloc(mc->host_lfac, maxlevels * sizeof(float));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    mc->host_lfac = (float *) v;
    v = realloc(mc->host_sinfo, maxlevels * 4 * sizeof(int));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    mc->host_sinfo = (int *) v;
    mc->maxlevels = maxlevels;
  }

  lfac = 1.f;
  for (i = 0;  i < nlevels;  i++) {
    mc->host_lfac[i] = lfac;
    lfac *= 0.5f;
  }

  stotal = 0;
  btotal = 0;
  for (i = 0;  i < nlevels;  i++) {
    /* determine lattice dimensions measured in subcubes */
    const floatGrid *f = &(mc->msmpot->qh[i]);
    int nx = mc->host_sinfo[4*i    ] = (int) ceilf(f->ni * 0.25f) + 2*pad;
    int ny = mc->host_sinfo[4*i + 1] = (int) ceilf(f->nj * 0.25f) + 2*pad;
    int nz = mc->host_sinfo[4*i + 2] = (int) ceilf(f->nk * 0.25f) + 2*pad;
    stotal += nx * ny * nz;
    btotal += (nx - 2*pad) * (ny - 2*pad) * (nz - 2*pad);
    mc->host_sinfo[4* i + 3] = btotal;
#ifdef MSMPOT_DEBUG
    printf("\nlevel %d:  ni=%2d  nj=%2d  nk=%2d\n", i, f->ni, f->nj, f->nk);
    printf("          nx=%2d  ny=%2d  nz=%2d  stotal=%d\n",
        nx, ny, nz, stotal);
    printf("          bx=%2d  by=%2d  bz=%2d  btotal=%d\n",
        nx-2*pad, ny-2*pad, nz-2*pad, btotal);
#endif
  }
#ifdef MSMPOT_DEBUG
  printf("\n");
#endif
  /* stotal counts total number of subcubes for collapsed grid hierarchy */
  mc->subcube_total = stotal;
  mc->block_total = btotal;

#ifdef MSMPOT_DEBUG
  printf("nlevels=%d\n", nlevels);
  for (i = 0;  i < nlevels;  i++) {
    printf("ni=%d  nj=%d  nk=%d\n",
        msm->qh[i].ni, msm->qh[i].nj, msm->qh[i].nk);
    printf("nx=%d  ny=%d  nz=%d  nw=%d\n",
        mc->host_sinfo[4*i    ],
        mc->host_sinfo[4*i + 1],
        mc->host_sinfo[4*i + 2],
        mc->host_sinfo[4*i + 3]);
  }
#endif

  /* allocate and calculate weights for lattice cutoff */
  maxwts = (8*srad) * (8*srad) * (8*srad);
  if (mc->maxwts < maxwts) {
    void *v = realloc(mc->host_wt, maxwts * sizeof(float));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    mc->host_wt = (float *) v;
    mc->maxwts = maxwts;
  }
  wt = mc->host_wt;
  for (kk = 0;  kk < 8*srad;  kk++) {
    for (jj = 0;  jj < 8*srad;  jj++) {
      for (ii = 0;  ii < 8*srad;  ii++) {
        index = (kk*(8*srad) + jj)*(8*srad) + ii;
        i = ii - 4*srad;  /* distance (in grid points) from center */
        j = jj - 4*srad;
        k = kk - 4*srad;
        s = (i*i*hx*hx + j*j*hy*hy + k*k*hz*hz) / (a*a);
        t = 0.25f * s;
        if (t >= 1) {
          wt[index] = 0;
        }
        else if (s >= 1) {
          gs = 1/sqrtf(s);
          SPOLY(&gt, t, split);
          wt[index] = (gs - 0.5f * gt) / a;
        }
        else {
          SPOLY(&gs, s, split);
          SPOLY(&gt, t, split);
          wt[index] = (gs - 0.5f * gt) / a;
        }
      }
    }
  }

  /* allocate host memory flat array of subcubes */
  maxgridpts = stotal * SUBCUBESZ;
  memsz = maxgridpts * sizeof(float);
  if (mc->maxgridpts < memsz) {
    void *v;
    v = realloc(mc->host_qgrids, memsz);
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    mc->host_qgrids = (float *) v;
    v = realloc(mc->host_egrids, memsz);
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    mc->host_egrids = (float *) v;
    cudaFree(mc->device_qgrids);
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    cudaMalloc(&v, memsz);
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    mc->device_qgrids = (float *) v;
    cudaFree(mc->device_egrids);
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    cudaMalloc(&v, memsz);
    CUERR(MSMPOT_ERROR_CUDA_MALLOC);
    mc->device_egrids = (float *) v;
    mc->maxgridpts = maxgridpts;
  }

  return MSMPOT_SUCCESS;
}


/* condense q grid hierarchy into flat array of subcubes */
int Msmpot_cuda_condense_qgrids(MsmpotCuda *mc) {
  const int *host_sinfo = mc->host_sinfo;
  float *host_qgrids = mc->host_qgrids;

  const long memsz = mc->subcube_total * SUBCUBESZ * sizeof(float);
  const int nlevels = mc->lk_nlevels;
  const int pad = mc->lk_padding;
  int level, in, jn, kn, i, j, k;
  int isrc, jsrc, ksrc, subcube_index, grid_index, off;

  const int ispx = (IS_SET_X(mc->msmpot->isperiodic) != 0);
  const int ispy = (IS_SET_Y(mc->msmpot->isperiodic) != 0);
  const int ispz = (IS_SET_Z(mc->msmpot->isperiodic) != 0);

  memset(host_qgrids, 0, memsz);  /* zero the qgrids subcubes */

  off = 0;
  for (level = 0;  level < nlevels;  level++) {
    const floatGrid *qgrid = &(mc->msmpot->qh[level]);
    const float *qbuffer = qgrid->buffer;

    const int ni = (int) (qgrid->ni);
    const int nj = (int) (qgrid->nj);
    const int nk = (int) (qgrid->nk);

    const int nx = host_sinfo[4*level    ];
    const int ny = host_sinfo[4*level + 1];
    const int nz = host_sinfo[4*level + 2];

#ifdef MSMPOT_DEBUG
    printf("level=%d\n", level);
    printf("  nx=%d  ny=%d  nz=%d\n", nx, ny, nz);
    printf("  ni=%d  nj=%d  nk=%d\n", ni, nj, nk);
#endif

    for (kn = 0;  kn < nz;  kn++) {
      for (jn = 0;  jn < ny;  jn++) {
        for (in = 0;  in < nx;  in++) {

          for (k = 0;  k < 4;  k++) {
            ksrc = (kn-pad)*4 + k;
            if (ispz) {
              while (ksrc < 0)    ksrc += nk;
              while (ksrc >= nk)  ksrc -= nk;
            }
            else if (ksrc < 0 || ksrc >= nk) break;

            for (j = 0;  j < 4;  j++) {
              jsrc = (jn-pad)*4 + j;
              if (ispy) {
                while (jsrc < 0)    jsrc += nj;
                while (jsrc >= nj)  jsrc -= nj;
              }
              else if (jsrc < 0 || jsrc >= nj) break;

              for (i = 0;  i < 4;  i++) {
                isrc = (in-pad)*4 + i;
                if (ispx) {
                  while (isrc < 0)    isrc += ni;
                  while (isrc >= ni)  isrc -= ni;
                }
                else if (isrc < 0 || isrc >= ni) break;

                grid_index = (ksrc * nj + jsrc) * ni + isrc;
                subcube_index = (((kn*ny + jn)*nx + in) + off) * SUBCUBESZ
                  + (k*4 + j)*4 + i;

                host_qgrids[subcube_index] = qbuffer[grid_index];
              }
            }
          } /* loop over points in a subcube */

        }
      }
    } /* loop over subcubes in a level */

    off += nx * ny * nz;  /* offset to next level */

  } /* loop over levels */

  return 0;
}


/* expand flat array of subcubes into e grid hierarchy */
int Msmpot_cuda_expand_egrids(MsmpotCuda *mc) {
  const int *host_sinfo = mc->host_sinfo;
  const float *host_egrids = mc->host_egrids;

  const int nlevels = mc->lk_nlevels;
  const int pad = mc->lk_padding;
  int level, in, jn, kn, i, j, k;
  int isrc, jsrc, ksrc, subcube_index, grid_index, off;

  off = 0;
  for (level = 0;  level < nlevels;  level++) {
    floatGrid *egrid = &(mc->msmpot->eh[level]);
    float *ebuffer = egrid->buffer;

    const int ni = (int) (egrid->ni);
    const int nj = (int) (egrid->nj);
    const int nk = (int) (egrid->nk);

    const int nx = host_sinfo[4*level    ];
    const int ny = host_sinfo[4*level + 1];
    const int nz = host_sinfo[4*level + 2];

    for (kn = pad;  kn < nz-pad;  kn++) {
      for (jn = pad;  jn < ny-pad;  jn++) {
        for (in = pad;  in < nx-pad;  in++) {

          for (k = 0;  k < 4;  k++) {
            ksrc = (kn-pad)*4 + k;
            if (ksrc >= nk) break;

            for (j = 0;  j < 4;  j++) {
              jsrc = (jn-pad)*4 + j;
              if (jsrc >= nj) break;

              for (i = 0;  i < 4;  i++) {
                isrc = (in-pad)*4 + i;
                if (isrc >= ni) break;

                grid_index = (ksrc * nj + jsrc) * ni + isrc;
                subcube_index = (((kn*ny + jn)*nx + in) + off) * SUBCUBESZ
                  + (k*4 + j)*4 + i;

                ebuffer[grid_index] = host_egrids[subcube_index];
              }
            }
          } /* loop over points in a subcube */

        }
      }
    } /* loop over subcubes in a level */

    off += nx * ny * nz;  /* offset to level */

  } /* loop over levels */

  return 0;
}


int Msmpot_cuda_compute_latcut(MsmpotCuda *mc) {
  const int nlevels = mc->lk_nlevels;
  const int srad = mc->lk_srad;
  const int padding = mc->lk_padding;
  const int wt_total = (8*srad) * (8*srad) * (8*srad);
  const long memsz = mc->subcube_total * SUBCUBESZ * sizeof(float);

  dim3 gridDim, blockDim;

  unsigned int bx = mc->block_total;
  unsigned int by = 1;
#define MAX_GRID_DIM  65536u

  while (bx > MAX_GRID_DIM) {
    bx >>= 1;  /* divide by 2 */
    by <<= 1;  /* multiply by 2 */
  }
  if (bx * by < (unsigned int)(mc->block_total)) bx++;
  if (bx > MAX_GRID_DIM || by > MAX_GRID_DIM) {
    /* subcube array length is too large to launch kernel */
    return ERROR(MSMPOT_ERROR_CUDA_SUPPORT);
  }
 
  gridDim.x = (int) bx;
  gridDim.y = (int) by;
  gridDim.z = 1;

  blockDim.x = 4;
  blockDim.y = 4;
  blockDim.z = 4;

  /* copy some host memory data into device constant memory */
  cudaMemcpyToSymbol(sinfo, mc->host_sinfo, nlevels * sizeof(int4), 0);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);
  cudaMemcpyToSymbol(lfac, mc->host_lfac, nlevels * sizeof(float), 0);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);
  cudaMemcpyToSymbol(wt, mc->host_wt, wt_total * sizeof(float), 0);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);

  /* copy qgrids from host to device */
  cudaMemcpy(mc->device_qgrids, mc->host_qgrids, memsz, cudaMemcpyHostToDevice);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);

  /* invoke kernel */
#ifdef MSMPOT_DEBUG
  printf("gridDim.x=%d\n", gridDim.x);
  printf("gridDim.y=%d\n", gridDim.y);
  printf("nsubcubes=%u  (using %u extra thread blocks)\n",
      (uint)(mc->block_total),
      (gridDim.x*gridDim.y - (uint)(mc->block_total)));
  printf("nlevels=%d\n", nlevels);
  printf("srad=%d\n", srad);
  printf("padding=%d\n", padding);
  printf("address of qgrids=%lx\n", (long) (mc->device_qgrids));
  printf("address of egrids=%lx\n", (long) (mc->device_egrids));
#endif
  cuda_latcut<<<gridDim, blockDim, 0>>>((unsigned int)(mc->block_total),
      nlevels, srad, padding, mc->device_qgrids, mc->device_egrids);
  CUERR(MSMPOT_ERROR_CUDA_KERNEL);

  /* copy egrids from device to host */
  cudaMemcpy(mc->host_egrids, mc->device_egrids, memsz, cudaMemcpyDeviceToHost);
  CUERR(MSMPOT_ERROR_CUDA_MEMCPY);

  return 0;
}
