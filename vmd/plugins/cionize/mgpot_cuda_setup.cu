#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mgpot_cuda.h"

#undef DEBUGGING

#ifdef __cplusplus
extern "C" {
#endif


int mgpot_cuda_device_list(void) {
  int deviceCount;
  int dev;

  deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Detected %d CUDA accelerators:\n", deviceCount);
  for (dev = 0;  dev < deviceCount;  dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("  CUDA device[%d]: '%s'  Mem: %dMB  Rev: %d.%d\n"
        "    maxGridSize:  %d, %d, %d\n",
        dev, deviceProp.name, deviceProp.totalGlobalMem / (1024*1024),
        deviceProp.major, deviceProp.minor,
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]
        );
  }
  return deviceCount;
}


int mgpot_cuda_device_set(int devnum) {
  cudaError_t rc;

if (getenv("CUDADEV")) 
  sscanf(getenv("CUDADEV"), "%d", &devnum);

  printf("Opening CUDA device %d...\n", devnum);

  rc = cudaSetDevice(devnum);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess)
      return NULL; // abort and return an error
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }

  CUERR;  /* check and clear any existing errors */
  return 0;
}


int mgpot_cuda_setup_shortrng(Mgpot *mg) {
  switch (mg->use_cuda & MBINMASK) {
    case MBINLARGE:
      if (mgpot_cuda_setup_binlarge(mg)) {
        return ERROR("mgpot_cuda_setup_binlarge() failed\n");
      }
      break;
    case MBINSMALL:
      //return ERROR("MBINSMALL not yet supported\n");
      break;
  }
  return 0;
}


int mgpot_cuda_cleanup_shortrng(Mgpot *mg) {
  switch (mg->use_cuda & MBINMASK) {
    case MBINLARGE:
      if (mgpot_cuda_cleanup_binlarge(mg)) {
        return ERROR("mgpot_cuda_cleanup_binlarge() failed\n");
      }
      break;
    case MBINSMALL:
      //return ERROR("MBINSMALL not yet supported\n");
      break;
  }
  return 0;
}


int mgpot_cuda_setup_longrng(Mgpot *mg) {
  if (mg->use_cuda & MLATCUTMASK) {
    if (mgpot_cuda_setup_latcut(mg)) {
      return ERROR("mgpot_cuda_setup_latcut() failed\n");
    }
  }
  return 0;
}


int mgpot_cuda_cleanup_longrng(Mgpot *mg) {
  if (mg->use_cuda & MLATCUTMASK) {
    if (mgpot_cuda_cleanup_latcut(mg)) {
      return ERROR("mgpot_cuda_cleanup_latcut() failed\n");
    }
  }
  return 0;
}


int mgpot_cuda_setup_latcut(Mgpot *mg) {
  const float h = mg->h;
  const float a = mg->a;
  const int split = mg->split;
  int nlevels = mg->nlevels - 1;  /* don't do top level on GPU */
  int nrad;
  int srad;
  int pad;
  int i, j, k, ii, jj, kk;
  int index;
  long btotal, stotal, memsz;
  float s, t, gs, gt;
  float lfac;
  float *wt;

  if (nlevels > MAXLEVELS) {
    return ERROR("number of levels %d exceeds maximum %d\n",
        nlevels, MAXLEVELS);
  }
  mg->lk_nlevels = nlevels;
  nrad = (int) ceilf(2*a/h) - 1;
  srad = (int) ceilf((nrad + 1) / 4.f);
  if (srad > 3) {
    return ERROR("subcube radius %d exceeds maximum radius %d\n",
        srad, 3);
  }
  mg->lk_srad = srad;
  pad = 1;  /* for non-periodic systems */
  mg->lk_padding = pad;
#ifdef DEBUGGING
  printf("a=%g  h=%g\n", a, h);
  printf("nrad=%d\n", nrad);
  printf("srad=%d\n", srad);
  printf("padding=%d\n", padding);
#endif

  mg->host_lfac = (float *) calloc(nlevels, sizeof(float));
  if (NULL==mg->host_lfac) return FAIL;
  lfac = 1.f;
  for (i = 0;  i < nlevels;  i++) {
    mg->host_lfac[i] = lfac;
    lfac *= 0.5f;
  }

  mg->host_sinfo = (int *) calloc(4 * nlevels, sizeof(int));
  if (NULL==mg->host_sinfo) return FAIL;
  stotal = 0;
  btotal = 0;
  for (i = 0;  i < nlevels;  i++) {
    /* determine lattice dimensions measured in subcubes */
    const floatLattice *f = mg->qgrid[i];
    int nx = mg->host_sinfo[ INDEX_X(i) ] = (int) ceilf(f->ni / 4.f) + 2*pad;
    int ny = mg->host_sinfo[ INDEX_Y(i) ] = (int) ceilf(f->nj / 4.f) + 2*pad;
    int nz = mg->host_sinfo[ INDEX_Z(i) ] = (int) ceilf(f->nk / 4.f) + 2*pad;
    stotal += nx * ny * nz;
    btotal += (nx - 2*pad) * (ny - 2*pad) * (nz - 2*pad);
    mg->host_sinfo[ INDEX_Q(i) ] = btotal;

    printf("\nlevel %d:  ni=%2d  nj=%2d  nk=%2d\n", i, f->ni, f->nj, f->nk);
    printf("          nx=%2d  ny=%2d  nz=%2d  stotal=%d\n",
        nx, ny, nz, stotal);
    printf("          bx=%2d  by=%2d  bz=%2d  btotal=%d\n",
        nx-2*pad, ny-2*pad, nz-2*pad, btotal);
  }
  printf("\n");
  /* stotal counts total number of subcubes for collapsed grid hierarchy */
  mg->subcube_total = stotal;
  mg->block_total = btotal;
  //printf("stotal=%d\n", stotal);
  //printf("btotal=%d\n", btotal);

#ifdef DEBUGGING
  printf("nlevels=%d\n", nlevels);
  for (i = 0;  i < nlevels;  i++) {
    printf("ni=%d  nj=%d  nk=%d\n",
        mg->qgrid[i]->ni, mg->qgrid[i]->nj, mg->qgrid[i]->nk);
    printf("nx=%d  ny=%d  nz=%d  nw=%d\n",
        mg->host_sinfo[ INDEX_X(i) ],
        mg->host_sinfo[ INDEX_Y(i) ],
        mg->host_sinfo[ INDEX_Z(i) ],
        mg->host_sinfo[ INDEX_Q(i) ]);
  }
#endif

  /* allocate and calculate weights for lattice cutoff */
  mg->host_wt = (float *) calloc((8*srad) * (8*srad) * (8*srad), sizeof(float));
  if (NULL==mg->host_wt) return FAIL;
  wt = mg->host_wt;
  for (kk = 0;  kk < 8*srad;  kk++) {
    for (jj = 0;  jj < 8*srad;  jj++) {
      for (ii = 0;  ii < 8*srad;  ii++) {
        index = (kk*(8*srad) + jj)*(8*srad) + ii;
        i = ii - 4*srad;  /* distance (in grid points) from center */
        j = jj - 4*srad;
        k = kk - 4*srad;
        s = (i*i + j*j + k*k) * h*h / (a*a);
        t = 0.25f * s;
        if (t >= 1) {
          wt[index] = 0;
        }
        else if (s >= 1) {
          gs = 1/sqrtf(s);
          mgpot_split(&gt, t, split);
          wt[index] = (gs - 0.5f * gt) / a;
        }
        else {
          mgpot_split(&gs, s, split);
          mgpot_split(&gt, t, split);
          wt[index] = (gs - 0.5f * gt) / a;
        }
      }
    }
  }

  /* allocate host memory flat array of subcubes */
  memsz = stotal * SUBCUBESZ * sizeof(float);
  mg->host_qgrids = (float *) malloc(memsz);
  if (NULL==mg->host_qgrids) return FAIL;
  mg->host_egrids = (float *) malloc(memsz);
  if (NULL==mg->host_egrids) return FAIL;

  /* allocate device global memory flat array of subcubes */
  printf("Allocating %.2fMB of device memory for grid hierarchy...\n",
      (2.f * memsz) / (1024.f * 1024.f));
  cudaMalloc((void **) &(mg->device_qgrids), memsz);
  CUERR;  /* check and clear any existing errors */
  printf("device q grid addr:  %lx\n", (long)(mg->device_qgrids));
  cudaMalloc((void **) &(mg->device_egrids), memsz);
  CUERR;  /* check and clear any existing errors */
  printf("device e grid addr:  %lx\n", (long)(mg->device_egrids));

  return 0;
}


int mgpot_cuda_cleanup_latcut(Mgpot *mg) {
  free(mg->host_lfac);  /* free host memory allocations */
  free(mg->host_sinfo);
  free(mg->host_wt);
  free(mg->host_qgrids);
  free(mg->host_egrids);
  cudaFree(mg->device_qgrids);  /* free device memory allocations */
  cudaFree(mg->device_egrids);
  return 0;
}


/* condense q grid hierarchy into flat array of subcubes */
int mgpot_cuda_condense_qgrids(Mgpot *mg) {
  const int *host_sinfo = mg->host_sinfo;
  float *host_qgrids = mg->host_qgrids;

  const long memsz = mg->subcube_total * SUBCUBESZ * sizeof(float);
  const int nlevels = mg->lk_nlevels;
  const int pad = mg->lk_padding;
  int level, in, jn, kn, i, j, k;
  int isrc, jsrc, ksrc, subcube_index, grid_index, off;

  //printf("4\n");
  memset(host_qgrids, 0, memsz);  /* zero the qgrids subcubes */

  //printf("5\n");
  off = 0;
  for (level = 0;  level < nlevels;  level++) {
    const floatLattice *qgrid = mg->qgrid[level];
    const float *qbuffer = qgrid->buffer;

    const int nx = host_sinfo[ INDEX_X(level) ];
    const int ny = host_sinfo[ INDEX_Y(level) ];
    const int nz = host_sinfo[ INDEX_Z(level) ];
    //const int nw = host_sinfo[ INDEX_Q(level) ] - nx*ny*nz;

#ifdef DEBUGGING
    printf("level=%d\n", level);
    printf("  nx=%d  ny=%d  nz=%d\n", nx, ny, nz);
    printf("  ni=%d  nj=%d  nk=%d\n", qgrid->ni, qgrid->nj, qgrid->nk);
#endif

    for (kn = pad;  kn < nz-pad;  kn++) {
      for (jn = pad;  jn < ny-pad;  jn++) {
        for (in = pad;  in < nx-pad;  in++) {

          for (k = 0;  k < 4;  k++) {
            ksrc = (kn-pad)*4 + k;
            if (ksrc >= qgrid->nk) break;

            for (j = 0;  j < 4;  j++) {
              jsrc = (jn-pad)*4 + j;
              if (jsrc >= qgrid->nj) break;

              for (i = 0;  i < 4;  i++) {
                isrc = (in-pad)*4 + i;
                if (isrc >= qgrid->ni) break;

                grid_index = (ksrc * qgrid->nj + jsrc) * qgrid->ni + isrc;
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
int mgpot_cuda_expand_egrids(Mgpot *mg) {
  const int *host_sinfo = mg->host_sinfo;
  const float *host_egrids = mg->host_egrids;

  const int nlevels = mg->lk_nlevels;
  const int pad = mg->lk_padding;
  int level, in, jn, kn, i, j, k;
  int isrc, jsrc, ksrc, subcube_index, grid_index, off;

  off = 0;
  for (level = 0;  level < nlevels;  level++) {
    floatLattice *egrid = mg->egrid[level];
    float *ebuffer = egrid->buffer;

    const int nx = host_sinfo[ INDEX_X(level) ];
    const int ny = host_sinfo[ INDEX_Y(level) ];
    const int nz = host_sinfo[ INDEX_Z(level) ];
    //const int nw = host_sinfo[ INDEX_Q(level) ] - nx*ny*nz;

    for (kn = pad;  kn < nz-pad;  kn++) {
      for (jn = pad;  jn < ny-pad;  jn++) {
        for (in = pad;  in < nx-pad;  in++) {

          for (k = 0;  k < 4;  k++) {
            ksrc = (kn-pad)*4 + k;
            if (ksrc >= egrid->nk) break;

            for (j = 0;  j < 4;  j++) {
              jsrc = (jn-pad)*4 + j;
              if (jsrc >= egrid->nj) break;

              for (i = 0;  i < 4;  i++) {
                isrc = (in-pad)*4 + i;
                if (isrc >= egrid->ni) break;

                grid_index = (ksrc * egrid->nj + jsrc) * egrid->ni + isrc;
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


int mgpot_cuda_setup_binlarge(Mgpot *mg) {
  long memsz;
  float xlen, ylen, zlen;
  float xmin, ymin, zmin;
  float cellen;
  int nx, ny, nz;
  int i, j, k, index;
  MgpotLargeBin *b;

#if defined(MGPOT_SPACING_1_0)
  if (mg->gridspacing != 1.f || mg->a != 12.f) {
    return ERROR("MSM parameters do not conform to MBINLARGE\n");
  }
  mg->slabsz = 24;
#elif defined(MGPOT_SPACING_0_2)
  if (mg->gridspacing != 0.2f || mg->a != 9.6f) {
    return ERROR("MSM parameters do not conform to MBINLARGE\n");
  }
  mg->slabsz = 96;
#else
  if (mg->gridspacing != 0.5f || mg->a != 12.f) {
    return ERROR("MSM parameters do not conform to MBINLARGE\n");
  }
  mg->slabsz = 48;
#endif

  /* have to round up lattice dimensions to next multiple of slabz */
  mg->nxsub = (int) ceilf(mg->numpt / (float) mg->slabsz);
  mg->nysub = (int) ceilf(mg->numcol / (float) mg->slabsz);
  mg->nzsub = (int) ceilf(mg->numplane / (float) mg->slabsz);

  mg->allnx = mg->slabsz * mg->nxsub;
  mg->allny = mg->slabsz * mg->nysub;
  mg->allnz = mg->slabsz * mg->nzsub;

  /* allocate memory on host to receive all epot map results */
  memsz = sizeof(float) * mg->allnx * mg->allny * mg->allnz;
  printf("Allocating %.2fMB of memory on host for buffering epot map...\n",
      memsz / (1024.f * 1024.f));
  mg->host_epot = (float *) malloc(memsz);
  if (NULL==mg->host_epot) {
    return ERROR("malloc() failed\n");
  }

  /* allocate one slab of epot map on device */
  memsz = sizeof(float) * mg->allnx * mg->allny * mg->slabsz;
  printf("Allocating %.2fMB of memory on device for epot map slab...\n",
      memsz / (1024.f * 1024.f));
  cudaMalloc((void **) &(mg->device_epot_slab), memsz);
  CUERR;

  printf("Setting up CUDA large bin kernel\n");
  printf("CUDA subgrid size: %ld x %ld x %ld\n",
      mg->slabsz, mg->slabsz, mg->slabsz);
  printf("Entire grid size:  %ld x %ld x %ld\n",
      mg->allnx, mg->allny, mg->allnz);

  /*
   * bins and subcubes are both supposed to have side length
   *   2*cutoff = slabz*gridspacing
   * but bins are offset by cutoff, so there are expected to be
   * one extra bin than subcubes in each dimension
   */

  cellen = 2 * mg->a;
  xlen = mg->allnx * mg->gridspacing + mg->a;
  ylen = mg->allny * mg->gridspacing + mg->a;
  zlen = mg->allnz * mg->gridspacing + mg->a;
  xmin = -mg->a;
  ymin = -mg->a;
  zmin = -mg->a;

  /* allocate large bins for geometric hashing */
  mg->nxbin = nx = (int) ceilf(xlen / cellen);
  mg->nybin = ny = (int) ceilf(ylen / cellen);
  mg->nzbin = nz = (int) ceilf(zlen / cellen);
  if (nx != mg->nxsub+1 || ny != mg->nysub+1 || nz != mg->nzsub+1) {
    return ERROR("number of bins not as expected\n");
  }
  printf("Allocating %.2fMB of memory on host for the large bins...\n",
      (nx*ny*nz*sizeof(MgpotLargeBin)) / (1024.f * 1024.f));
  mg->largebin = b = (MgpotLargeBin *) calloc(nx*ny*nz, sizeof(MgpotLargeBin));
  if (NULL==mg->largebin) {
    return ERROR("calloc() failed\n");
  }

  /* prepare bins */
  for (k = 0;  k < nz;  k++) {
    for (j = 0;  j < ny;  j++) {
      for (i = 0;  i < nx;  i++) {
        index = (k*ny + j)*nx + i;
        b[index].x0 = xmin + i*cellen;
        b[index].y0 = ymin + j*cellen;
        b[index].z0 = zmin + k*cellen;
      }
    }
  }

  return 0;
}


int mgpot_cuda_cleanup_binlarge(Mgpot *mg) {
  free(mg->host_epot);
  cudaFree(mg->device_epot_slab);
  free(mg->largebin);
  return 0;
}


/*
 * mgpot_cuda_binlarge_pre():
 *
 * used for very coarse hashing of 24 A^3 regions of space;
 * tile these in space, offset by 12 A cutoff along each dimension
 * from the grid point blocks being updated by the GPU,
 * so that blocks of 8 grid cells completely cover the grid
 * point block plus a 12 A cutoff margin in each direction
 *
 * note that 24^3/10 = 1382.4 expected atoms;
 * the extra memory below, although seeming excessive, is still
 * less than 10% of the total memory required for a large grid,
 * e.g. the virus test problem
 */
int mgpot_cuda_binlarge_pre(Mgpot *mg) {
  const long natoms = mg->numatoms;
  const float *atom = mg->atoms;
  long n;
  int i, j, k;
  const int nx = mg->nxbin;
  const int ny = mg->nybin;
  const int nz = mg->nzbin;
  long index, aindex;
  const float xmin = -mg->a;    /* lattice orgin at (0,0,0) */
  const float ymin = -mg->a;    /* min of bins is offset by the cutoff */
  const float zmin = -mg->a;
  const float inv_cellen = 1.f / (2 * mg->a);
  MgpotLargeBin *bin = mg->largebin;

  /* hash atoms */
  for (n = 0;  n < 4*natoms;  n += 4) {
    i = (int) floorf((atom[n  ] - xmin) * inv_cellen);
    j = (int) floorf((atom[n+1] - ymin) * inv_cellen);
    k = (int) floorf((atom[n+2] - zmin) * inv_cellen);
    if (i < 0) i = 0;
    else if (i >= nx) i = nx-1;
    if (j < 0) j = 0;
    else if (j >= ny) j = ny-1;
    if (k < 0) k = 0;
    else if (k >= nz) k = nz-1;
    index = (k*ny + j)*nx + i;
    aindex = 4*bin[index].atomcnt;
    bin[index].atom[aindex  ] = atom[n  ];
    bin[index].atom[aindex+1] = atom[n+1];
    bin[index].atom[aindex+2] = atom[n+2];
    bin[index].atom[aindex+3] = atom[n+3];
    bin[index].atomcnt++;
  }

  return 0;
}


int mgpot_cuda_binlarge_post(Mgpot *mg) {
  const long numplane = mg->numplane;
  const long numcol = mg->numcol;
  const long numpt = mg->numpt;
  const long n = numpt * numcol * numplane;
  const long allnx = mg->allnx;
  const long allny = mg->allny;
  long i, j, k;
  const float *host_epot = mg->host_epot;
  const unsigned char *excl = mg->excludepos;
  float *grideners = mg->grideners;

  /* copy host buffer into (likely smaller) grideners buffer */
  for (k = 0;  k < numplane;  k++) {
    for (j = 0;  j < numcol;  j++) {
      for (i = 0;  i < numpt;  i++) {
        grideners[(k*numcol+j)*numpt+i] += host_epot[(k*allny+j)*allnx+i];
      }
    }
  }

  /* set excluded points to zero */
  for (i = 0;  i < n;  i++) {
    if (excl[i]) grideners[i] = 0;
  }

  return 0;
}


#ifdef __cplusplus
}
#endif
