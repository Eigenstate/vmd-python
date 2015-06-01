#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEBUG
#undef DEBUG
#include "mgpot_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "util.h"


#ifdef __DEVICE_EMULATION__
#define DEBUG
/* define which grid block and which thread to examine */
#define BX  0
#define BY  0
#define TX  0
#define TY  0
#define TZ  0
#define EMU(code) do { \
  if (blockIdx.x==BX && blockIdx.y==BY && \
      threadIdx.x==TX && threadIdx.y==TY && threadIdx.z==TZ) { \
    code; \
  } \
} while (0)
#define INT(n)    printf("%s = %d\n", #n, n)
#define FLOAT(f)  printf("%s = %g\n", #f, (double)(f))
#define INT3(n)   printf("%s = %d %d %d\n", #n, (n).x, (n).y, (n).z)
#define FLOAT4(f) printf("%s = %g %g %g %g\n", #f, (double)(f).x, \
    (double)(f).y, (double)(f).z, (double)(f).w)
#else
#define EMU(code)
#define INT(n)
#define FLOAT(f)
#define INT3(n)
#define FLOAT4(f)
#endif


/*
 * neighbor list:
 * stored in constant memory as table of offsets
 * flat index addressing is computed by kernel
 *
 * reserve enough memory for 17^3 stencil of grid cells
 * this fits within 64K of constant memory
 */
#define NBRLIST_DIM  17
#define NBRLIST_MAXLEN (NBRLIST_DIM * NBRLIST_DIM * NBRLIST_DIM)
static __constant__ int NbrListLen;
static __constant__ int3 NbrList[NBRLIST_MAXLEN];

/*
 * atom bins cached into shared memory for processing
 *
 * this reserves 4K of shared memory for 32 atom bins each containing 8 atoms,
 * should permit scheduling of up to 3 thread blocks per SM
 */
#define BIN_DEPTH         8  /* max number of atoms per bin */
#define BIN_SIZE         32  /* size of bin in floats */
#define BIN_SHIFT         5  /* # of bits to shift for mul/div by BIN_SIZE */
#define BIN_CACHE_MAXLEN  1  /* max number of atom bins to cache */

#define BIN_LENGTH      4.f  /* spatial length in Angstroms */
#define BIN_INVLEN  (1.f / BIN_LENGTH)
/* assuming density of 1 atom / 10 A^3, expectation is 6.4 atoms per bin
 * so that bin fill should be 80% (for non-empty regions of space) */

#define REGION_SIZE     512  /* number of floats in lattice region */

/*
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
 * The bins have to not only cover the region, but they need to surround
 * the outer edges so that region sides and corners can still use
 * neighbor list stencil.  The binZeroAddr is actually a shifted pointer into
 * the bin array (binZeroAddr = binBaseAddr + (c*binDim_y + c)*binDim_x + c)
 * where c = ceil(cutoff / binsize).  This allows for negative offsets to
 * be added to myBinIndex.
 *
 * The (0,0,0) spatial origin corresponds to lower left corner of both
 * regionZeroAddr and binZeroAddr.  The atom coordinates are translated
 * during binning to enforce this assumption.
 */
__global__ static void cuda_cutoff_potential_lattice10overlap(
    int binDim_x,
    int binDim_y,
    float4 *binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float invcut,           /* 1/cutoff */
    float *regionZeroAddr,  /* address of lattice regions starting at origin */
    int zRegionDim
    )
{
  __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
  __shared__ float *myRegionAddr;
  __shared__ int3 myBinIndex;

  const int xRegionIndex = blockIdx.x;
  const int yRegionIndex = blockIdx.y;

  /* thread id */
  const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x
    + threadIdx.x;
  /* blockDim.x == 8, blockDim.y == 2, blockDim.z == 8 */

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
  myRegionAddr = regionZeroAddr + ((zRegionIndex*gridDim.y
        + yRegionIndex)*gridDim.x + xRegionIndex)*REGION_SIZE;
    
  /* spatial coordinate of this lattice point */
  float x = (8 * xRegionIndex + threadIdx.x) * h;
  float y = (8 * yRegionIndex + threadIdx.y) * h;
  float z = (8 * zRegionIndex + threadIdx.z) * h;

  /* bin number determined by center of region */
  myBinIndex.x = (int) floorf((8 * xRegionIndex + 4) * h * BIN_INVLEN);
  myBinIndex.y = (int) floorf((8 * yRegionIndex + 4) * h * BIN_INVLEN);
  myBinIndex.z = (int) floorf((8 * zRegionIndex + 4) * h * BIN_INVLEN);

  float energy0 = 0.f;
  float energy1 = 0.f;
  float energy2 = 0.f;
  float energy3 = 0.f;

  for (nbrid = 0;  nbrid < NbrListLen;  nbrid++) {

    /* thread block caches one bin */
    if (tid < 32) {
      int i = myBinIndex.x + NbrList[nbrid].x;
      int j = myBinIndex.y + NbrList[nbrid].y;
      int k = myBinIndex.z + NbrList[nbrid].z;

      /* determine global memory location of atom bin */
      float *p_global = ((float *) binZeroAddr)
        + (((__mul24(k, binDim_y) + j)*binDim_x + i) << BIN_SHIFT);

      AtomBinCache[tid] = p_global[tid];
    }
    __syncthreads();

    {
      int i;

      for (i = 0;  i < BIN_DEPTH;  i++) {
        int off = (i << 2);

        float aq = AtomBinCache[off + 3];
        if (0.f == aq) 
          break;  /* no more atoms in bin */

        float dx = AtomBinCache[off    ] - x;
        float dz = AtomBinCache[off + 2] - z;
        float dxdz2 = dx*dx + dz*dz;
#ifdef CHECK_CYLINDER
        if (dxdz2 >= cutoff2) continue;
#endif
        float dy = AtomBinCache[off + 1] - y;
        float r2 = dy*dy + dxdz2;

        if (r2 < cutoff2) {
          float gr2 = gc0 + r2*(gc1 + r2*gc2);
          energy0 += aq * (rsqrtf(r2) - gr2);
        }
        dy -= 2.0f*h;
        r2 = dy*dy + dxdz2;
        if (r2 < cutoff2) {
          float gr2 = gc0 + r2*(gc1 + r2*gc2);
          energy1 += aq * (rsqrtf(r2) - gr2);
        }
        dy -= 2.0f*h;
        r2 = dy*dy + dxdz2;
        if (r2 < cutoff2) {
          float gr2 = gc0 + r2*(gc1 + r2*gc2);
          energy2 += aq * (rsqrtf(r2) - gr2);
        }
        dy -= 2.0f*h;
        r2 = dy*dy + dxdz2;
        if (r2 < cutoff2) {
          float gr2 = gc0 + r2*(gc1 + r2*gc2);
          energy3 += aq * (rsqrtf(r2) - gr2);
        }
      } /* end loop over atoms in bin */
    } /* end loop over cached atom bins */
    __syncthreads();

  } /* end loop over neighbor list */

  /* store into global memory */
  myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
  myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
  myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
  myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;

 } // zRegionIndex...

}




int gpu_compute_cutoff_potential_lattice10overlap(
    float *lattice,                    /* the lattice */
    int nx, int ny, int nz,            /* its dimensions, length nx*ny*nz */
    float xlo, float ylo, float zlo,   /* lowest corner of lattice */
    float h,                           /* lattice spacing */
    float cutoff,                      /* cutoff distance */
    Atom *atom,                        /* array of atoms */
    int natoms,                        /* number of atoms */
    int verbose                        /* print info/debug messages */
    )
{
  int3 nbrlist[NBRLIST_MAXLEN];
  int nbrlistlen = 0;

  int binHistoFull[BIN_DEPTH+1] = { 0 };   /* clear every array element */
  int binHistoCover[BIN_DEPTH+1] = { 0 };  /* clear every array element */
  int num_excluded = 0;

  int xRegionDim, yRegionDim, zRegionDim;
  int xRegionIndex, yRegionIndex, zRegionIndex;
  int xOffset, yOffset, zOffset;
  int lnx, lny, lnz, lnall;
  float *regionZeroAddr, *thisRegion;
  float *regionZeroCuda;
  int index, indexRegion;

  int c;
  int3 binDim;
  int nbins;
  float4 *binBaseAddr, *binZeroAddr;
  float4 *binBaseCuda, *binZeroCuda;
  int *bincntBaseAddr, *bincntZeroAddr;
  Atom *extra;
  int extralen = 0;

  int i, j, k, n;
  int sum, total;

  float avgFillFull, avgFillCover;
  const float cutoff2 = cutoff * cutoff;
  const float invcut = 1.f / cutoff;

  dim3 gridDim, blockDim;

#define MGPOT_GEN_NBRLIST
#define MGPOT_GEN_NBRLIST_IMPROVED

  verbose =1;

#ifdef MGPOT_GEN_NBRLIST
  float r, r2;
  int cnt, d;

#else

  /* pad lattice to be factor of 8 in each dimension */
  xRegionDim = (int) ceilf(nx/8.f);
  yRegionDim = (int) ceilf(ny/8.f);
  zRegionDim = (int) ceilf(nz/8.f);

  lnx = 8 * xRegionDim;
  lny = 8 * yRegionDim;
  lnz = 8 * zRegionDim;
  lnall = lnx * lny * lnz;

  /* will receive energies from CUDA */
  regionZeroAddr = (float *) malloc(lnall * sizeof(float));

  /* create bins */
  c = (int) ceil(cutoff * BIN_INVLEN);  /* count extra bins around lattice */
  binDim.x = (int) ceil(lnx * h * BIN_INVLEN) + 2*c;
  binDim.y = (int) ceil(lny * h * BIN_INVLEN) + 2*c;
  binDim.z = (int) ceil(lnz * h * BIN_INVLEN) + 2*c;
  nbins = binDim.x * binDim.y * binDim.z;
  binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
  binZeroAddr = binBaseAddr + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;

  bincntBaseAddr = (int *) calloc(nbins, sizeof(int));
  bincntZeroAddr = bincntBaseAddr + (c * binDim.y + c) * binDim.x + c;

  /* array of extra atoms, permit average of one extra per bin */
  extra = (Atom *) calloc(nbins, sizeof(Atom));
#endif

#ifdef MGPOT_GEN_NBRLIST
#ifdef MGPOT_GEN_NBRLIST_IMPROVED
  printf("IMPROVED neighborlist creation\n");
  {
    int bpr0 = (int) floorf((8*h)/BIN_LENGTH);
    int bpr1 = (int) ceilf((8*h)/BIN_LENGTH);

    c = (int) ceilf(cutoff * BIN_INVLEN);

    if (bpr0 == bpr1) {
      d = c + (bpr0 / 2);
      if (bpr0 & 1) {  /* is odd number */
        r = cutoff + (8*h + BIN_LENGTH) * 0.5f*sqrt(3);
      }
      else {
        r = cutoff + (8*h + 2*BIN_LENGTH) * 0.5f*sqrt(3);
        c++;
      }
    }
    else {
      r = cutoff + (8*h + 2*BIN_LENGTH) * 0.5f*sqrt(3);
      d = (int) ceil((cutoff + 4*h + BIN_LENGTH) * BIN_INVLEN);
      c++;
    }
    r2 = r * r;
  }
#else
  printf("using generic neighborlist creation\n");
  if (ceilf((8*h) / BIN_LENGTH) == floorf((8*h) / BIN_LENGTH)) {
    if ((int)(ceilf((8*h)/BIN_LENGTH)) & 1) {  /* is odd number */
      r = cutoff + (8*h + BIN_LENGTH) * 0.5f*sqrtf(3);
    }
    else {
      r = cutoff + (8*h + 2*BIN_LENGTH) * 0.5f*sqrtf(3);
    }
  }
  else {
    r = cutoff + (8*h + 2*BIN_LENGTH) * 0.5f*sqrtf(3);
  }
  r2 = r * r;
  d = (int) ceilf(r / BIN_LENGTH);
  c = d;
#endif
  cnt = 0;
  for (k = -d;  k <= d;  k++) {
    for (j = -d;  j <= d;  j++) {
      for (i = -d;  i <= d;  i++) {
        if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
        if (NBRLIST_MAXLEN == cnt) {
          fprintf(stderr, "exceeded max neighborhood size %d\n", cnt);
        }
        nbrlist[cnt].x = i;
        nbrlist[cnt].y = j;
        nbrlist[cnt].z = k;
        cnt++;
      }
    }
  }
  nbrlistlen = cnt;
  if (verbose) {
    printf("padding c=%d\n", c);
    printf("nbrlist bin radius d=%d\n", d);
    printf("nbrlist cutoff r=%g\n", r);
    printf("nbrlistlen=%d\n", nbrlistlen);
  }

  /* now we can create lattice and bins */

  /* pad lattice to be factor of 8 in each dimension */
  xRegionDim = (int) ceilf(nx/8.f);
  yRegionDim = (int) ceilf(ny/8.f);
  zRegionDim = (int) ceilf(nz/8.f);

  lnx = 8 * xRegionDim;
  lny = 8 * yRegionDim;
  lnz = 8 * zRegionDim;
  lnall = lnx * lny * lnz;

  /* will receive energies from CUDA */
  regionZeroAddr = (float *) malloc(lnall * sizeof(float));

  /* create bins */
//  c = (int) ceil(cutoff * BIN_INVLEN);  /* count extra bins around lattice */
  binDim.x = (int) ceil(lnx * h * BIN_INVLEN) + 2*c;
  binDim.y = (int) ceil(lny * h * BIN_INVLEN) + 2*c;
  binDim.z = (int) ceil(lnz * h * BIN_INVLEN) + 2*c;
  nbins = binDim.x * binDim.y * binDim.z;
  binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
  binZeroAddr = binBaseAddr + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;

  bincntBaseAddr = (int *) calloc(nbins, sizeof(int));
  bincntZeroAddr = bincntBaseAddr + (c * binDim.y + c) * binDim.x + c;

  /* array of extra atoms, permit average of one extra per bin */
  extra = (Atom *) calloc(nbins, sizeof(Atom));

#else
  /* create neighbor list */
  if (ceilf(BIN_LENGTH / (8*h)) == floorf(BIN_LENGTH / (8*h))) {
    float s = sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 1 cell */
    if (2*c + 1 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-1)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else if (8*h <= 2*BIN_LENGTH) {
    float s = 2.f*sqrtf(3);
    float r2 = (cutoff + s*BIN_LENGTH) * (cutoff + s*BIN_LENGTH);
    int cnt = 0;
    /* develop neighbor list around 3-cube of cells */
    if (2*c + 3 > NBRLIST_DIM) {
      fprintf(stderr, "must have cutoff <= %f\n",
          (NBRLIST_DIM-3)/2 * BIN_LENGTH);
      return -1;
    }
    for (k = -c;  k <= c;  k++) {
      for (j = -c;  j <= c;  j++) {
        for (i = -c;  i <= c;  i++) {
          if ((i*i + j*j + k*k)*BIN_LENGTH*BIN_LENGTH >= r2) continue;
          nbrlist[cnt].x = i;
          nbrlist[cnt].y = j;
          nbrlist[cnt].z = k;
          cnt++;
        }
      }
    }
    nbrlistlen = cnt;
  }
  else {
    fprintf(stderr, "must have h <= %f\n", 0.25 * BIN_LENGTH);
    return -1;
  }
#endif /* MGPOT_GEN_NBRLIST */

  /* perform geometric hashing of atoms into bins */
  for (n = 0;  n < natoms;  n++) {
    float4 p;
    p.x = atom[n].x - xlo;
    p.y = atom[n].y - ylo;
    p.z = atom[n].z - zlo;
    p.w = atom[n].q;
    i = (int) floorf(p.x * BIN_INVLEN);
    j = (int) floorf(p.y * BIN_INVLEN);
    k = (int) floorf(p.z * BIN_INVLEN);
    if (i >= -c && i < binDim.x - c &&
        j >= -c && j < binDim.y - c &&
        k >= -c && k < binDim.z - c &&
        atom[n].q != 0) {
      int index = (k * binDim.y + j) * binDim.x + i;
      float4 *bin = binZeroAddr + index * BIN_DEPTH;
      int bindex = bincntZeroAddr[index];
      if (bindex < BIN_DEPTH) {
        /* copy atom into bin and increase counter for this bin */
        bin[bindex] = p;
        bincntZeroAddr[index]++;
      }
      else {
        /* add index to array of extra atoms to be computed with CPU */
        if (extralen >= nbins) {
          fprintf(stderr, "exceeded space for storing extra atoms\n");
          return -1;
        }
        extra[extralen] = atom[n];
        extralen++;
      }
    }
    else {
      /* excluded atoms are either outside bins or neutrally charged */
      num_excluded++;
    }
  }

  /* bin stats */
  sum = total = 0;
  for (n = 0;  n < nbins;  n++) {
    binHistoFull[ bincntBaseAddr[n] ]++;
    sum += bincntBaseAddr[n];
    total += BIN_DEPTH;
  }
  avgFillFull = sum / (float) total;
  sum = total = 0;
  for (k = 0;  k < binDim.z - 2*c;  k++) {
    for (j = 0;  j < binDim.y - 2*c;  j++) {
      for (i = 0;  i < binDim.x - 2*c;  i++) {
        int index = (k * binDim.y + j) * binDim.x + i;
        binHistoCover[ bincntZeroAddr[index] ]++;
        sum += bincntZeroAddr[index];
        total += BIN_DEPTH;
      }
    }
  }
  avgFillCover = sum / (float) total;

  if (verbose) {
    /* report */
    printf("number of atoms = %d\n", natoms);
    printf("lattice spacing = %g\n", h);
    printf("cutoff distance = %g\n", cutoff);
    printf("\n");
    printf("requested lattice dimensions = %d %d %d\n", nx, ny, nz);
    printf("requested space dimensions = %g %g %g\n", nx*h, ny*h, nz*h);
    printf("expanded lattice dimensions = %d %d %d\n", lnx, lny, lnz);
    printf("expanded space dimensions = %g %g %g\n", lnx*h, lny*h, lnz*h);
    printf("number of bytes for lattice data = %u\n", lnall*sizeof(float));
    printf("\n");
    printf("bin padding thickness = %d\n", c);
    printf("bin cover dimensions = %d %d %d\n",
        binDim.x - 2*c, binDim.y - 2*c, binDim.z - 2*c);
    printf("bin full dimensions = %d %d %d\n", binDim.x, binDim.y, binDim.z);
    printf("number of bins = %d\n", nbins);
    printf("total number of atom slots = %d\n", nbins * BIN_DEPTH);
    printf("%% overhead space = %g\n",
        (natoms / (double) (nbins * BIN_DEPTH)) * 100);
    printf("number of bytes for bin data = %u\n",
        nbins * BIN_DEPTH * sizeof(float4));
    printf("\n");
    printf("bin histogram with padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoFull[n]);
      sum += binHistoFull[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillFull * 100);
    printf("\n");
    printf("bin histogram excluding padding:\n");
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      printf("     number of bins with %d atoms:  %d\n", n, binHistoCover[n]);
      sum += binHistoCover[n];
    }
    printf("     total number of bins:  %d\n", sum);
    printf("     %% average fill:  %g\n", avgFillCover * 100);
    printf("\n");
    printf("number of extra atoms = %d\n", extralen);
    printf("%% atoms that are extra = %g\n", (extralen / (double) natoms) * 100);
    printf("\n");

    /* sanity check on bins */
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoFull[n];
    }
    sum += extralen + num_excluded;
    printf("sanity check on bin histogram with edges:  "
        "sum + others = %d\n", sum);
    sum = 0;
    for (n = 0;  n <= BIN_DEPTH;  n++) {
      sum += n * binHistoCover[n];
    }
    sum += extralen + num_excluded;
    printf("sanity check on bin histogram excluding edges:  "
        "sum + others = %d\n", sum);
    printf("\n");

    /* neighbor list */
    printf("neighbor list length = %d\n", nbrlistlen);
    printf("\n");
  }

  /* setup CUDA kernel parameters */
  gridDim.x = xRegionDim;
  gridDim.y = yRegionDim;
  gridDim.z = 1;
  blockDim.x = 8;
  blockDim.y = 2;
  blockDim.z = 8;

  /* time CUDA operations */
  rt_timerhandle gputm = rt_timer_create();
  rt_timer_start(gputm);

  /* allocate and initialize memory on CUDA device */
  if (verbose) {
    printf("Allocating %.2fMB on CUDA device for potentials\n",
           lnall * sizeof(float) / (double) (1024*1024));
  }
  cudaMalloc((void **) &regionZeroCuda, lnall * sizeof(float));
  CUERR;
  cudaMemset(regionZeroCuda, 0, lnall * sizeof(float));
  CUERR;
  if (verbose) {
    printf("Allocating %.2fMB on CUDA device for atom bins\n",
           nbins * BIN_DEPTH * sizeof(float4) / (double) (1024*1024));
  }
  cudaMalloc((void **) &binBaseCuda, nbins * BIN_DEPTH * sizeof(float4));
  CUERR;
  cudaMemcpy(binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
      cudaMemcpyHostToDevice);
  CUERR;
  binZeroCuda = binBaseCuda + ((c * binDim.y + c) * binDim.x + c) * BIN_DEPTH;
  cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0);
  CUERR;
  cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0);
  CUERR;

  if (verbose) 
    printf("\n");


  cudaStream_t cutoffstream;
  cudaStreamCreate(&cutoffstream);

printf("GPU setup time: %.3f\n", rt_timer_timenow(gputm));

  /* loop over z-dimension, invoke CUDA kernel for each x-y plane */
  printf("Invoking CUDA kernel on %d region planes...\n", zRegionDim);
#if 0
  printf("gridDim=(%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z);
  printf("blockDim=(%d,%d,%d)\n", blockDim.x, blockDim.y, blockDim.z);
#endif
  cuda_cutoff_potential_lattice10overlap<<<gridDim, blockDim, 0>>>(binDim.x, binDim.y,
  binZeroCuda, h, cutoff2, invcut, regionZeroCuda, zRegionDim);

printf("Time to start of async CPU kernel: %.3f\n", rt_timer_timenow(gputm));

  /* 
   * handle extra atoms on the CPU, concurrently with the GPU calculations
   */
  if (extralen > 0) {
    rt_timerhandle tm = rt_timer_create();
    printf("computing extra atoms on CPU\n");
    rt_timer_start(tm);
    if (cpu_compute_cutoff_potential_lattice(lattice, nx, ny, nz,
          xlo, ylo, zlo, h, cutoff, extra, extralen)) {
      fprintf(stderr, "cpu_compute_cutoff_potential_lattice() failed "
          "for extra atoms\n");
      return -1;
    }
    rt_timer_stop(tm);
    printf("Runtime consumed for extra atom handling on the CPU: %.3f\n", rt_timer_time(tm));
    printf("\n");
  }
printf("Time to completion of async CPU kernel: %.3f\n", rt_timer_timenow(gputm));

  cudaStreamSynchronize(cutoffstream);
  CUERR;
  cudaThreadSynchronize();
  cudaStreamDestroy(cutoffstream);
  printf("Finished CUDA kernel calls                        \n");

  /* copy result regions from CUDA device */
  cudaMemcpy(regionZeroAddr, regionZeroCuda, lnall * sizeof(float),
      cudaMemcpyDeviceToHost);
  CUERR;

  /* free CUDA memory allocations */
  cudaFree(regionZeroCuda);
  cudaFree(binBaseCuda);

  rt_timer_stop(gputm);
  printf("Total GPU prep/kernel runtime: %.3f\n", rt_timer_time(gputm));



  /*
   * transpose on CPU, updating, producing the final lattice
   */
  rt_timerhandle tptm = rt_timer_create();
  rt_timer_start(tptm);
  /* transpose regions back into lattice */
  for (k = 0;  k < nz;  k++) {
    zRegionIndex = (k >> 3);
    zOffset = (k & 7);

    for (j = 0;  j < ny;  j++) {
      yRegionIndex = (j >> 3);
      yOffset = (j & 7);

      for (i = 0;  i < nx;  i++) {
        xRegionIndex = (i >> 3);
        xOffset = (i & 7);

        thisRegion = regionZeroAddr
          + ((zRegionIndex * yRegionDim + yRegionIndex) * xRegionDim
              + xRegionIndex) * REGION_SIZE;

        indexRegion = (zOffset * 8 + yOffset) * 8 + xOffset;
        index = (k * ny + j) * nx + i;

        lattice[index] += thisRegion[indexRegion];
      }
    }
  }
  rt_timer_stop(tptm);
  printf("Runtime consumed for lattice transpose on CPU: %.3f\n", rt_timer_time(tptm));


  /* cleanup memory allocations */
  free(regionZeroAddr);
  free(binBaseAddr);
  free(bincntBaseAddr);
  free(extra);

  return 0;
}

#ifdef __cplusplus
}
#endif
