/***************************************************************************
 *cr
 *cr            (C) Copyright 2007-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAVolCPotential.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.42 $      $Date: 2011/01/13 18:39:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated coulombic potential grid calculation
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "Inform.h"
#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 

typedef struct {
  float* atoms;
  float* grideners;
  long int numplane;
  long int numcol;
  long int numpt;
  long int natoms;
  float gridspacing;
} enthrparms;

/* thread prototype */
static void * cudaenergythread(void *);

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#else
#define CUERR
#endif

// max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about
// At 16 bytes/atom, 4000 atoms is about the max we can store in
// the constant buffer.
#define MAXATOMS 4000
__constant__ static float4 atominfo[MAXATOMS];


// 
// The CUDA kernels calculate coulombic potential at each grid point and
// store the results in the output array.
//
// These versions of the code use the 64KB constant buffer area reloaded
// for each group of MAXATOMS atoms, until the contributions from all
// atoms have been summed into the potential grid.
//
// These implementations use precomputed and unrolled loops of 
// (dy^2 + dz^2) values for increased FP arithmetic intensity.
// The X coordinate portion of the loop is unrolled by four or eight,
// allowing the same dy^2 + dz^2 values to be reused multiple times,
// increasing the ratio of FP arithmetic relative to FP loads, and
// eliminating some redundant calculations.
//

//
// Tuned global memory coalescing version, unrolled in X
//
// Benchmark for this version: 291 GFLOPS, 39.5 billion atom evals/sec
//  (Test system: GeForce 8800GTX)
// 

#if 1

//
// Tunings for large potential map dimensions (e.g. 384x384x...)
//
// NVCC -cubin says this implementation uses 20 regs
//
#define UNROLLX     8
#define UNROLLY     1
#define BLOCKSIZEX  8  // make large enough to allow coalesced global mem ops
#define BLOCKSIZEY  8  // make as small as possible for finer granularity

#else

//
// Tunings for small potential map dimensions (e.g. 128x128x...)
//
// NVCC -cubin says this implementation uses 16 regs
//
#define UNROLLX     4
#define UNROLLY     1
#define BLOCKSIZEX 16  // make large enough to allow coalesced global mem ops
#define BLOCKSIZEY 16  // make as small as possible for finer granularity

#endif

#define BLOCKSIZE  (BLOCKSIZEX*BLOCKSIZEY)

// FLOP counting
#if UNROLLX == 8
#define FLOPSPERATOMEVAL (59.0/8.0)
#elif UNROLLX == 4 
#define FLOPSPERATOMEVAL (31.0/4.0)
#endif

__global__ static void cenergy(int numatoms, float gridspacing, float * energygrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
                         + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
                         + xindex;

  float coory = gridspacing * yindex;
  float coorx = gridspacing * xindex;

  float energyvalx1=0.0f;
  float energyvalx2=0.0f;
  float energyvalx3=0.0f;
  float energyvalx4=0.0f;
#if UNROLLX > 4
  float energyvalx5=0.0f;
  float energyvalx6=0.0f;
  float energyvalx7=0.0f;
  float energyvalx8=0.0f;
#endif

  float gridspacing_coalesce = gridspacing * BLOCKSIZEX;

  // NOTE: One beneficial side effect of summing groups of 4,000 atoms
  // at a time in multiple passes is that these smaller summation 
  // groupings are likely to result in higher precision, since structures
  // are typically stored with atoms in chain order leading to high 
  // spatial locality within the groups of 4,000 atoms.  Kernels that 
  // sum these subgroups independently of the global sum for each voxel
  // should achieve relatively good floating point precision since large values
  // will tend to be summed with large values, and small values summed with
  // small values, etc. 
  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dyz2 = (dy * dy) + atominfo[atomid].z;

    float dx1 = coorx - atominfo[atomid].x;
    float dx2 = dx1 + gridspacing_coalesce;
    float dx3 = dx2 + gridspacing_coalesce;
    float dx4 = dx3 + gridspacing_coalesce;
#if UNROLLX > 4
    float dx5 = dx4 + gridspacing_coalesce;
    float dx6 = dx5 + gridspacing_coalesce;
    float dx7 = dx6 + gridspacing_coalesce;
    float dx8 = dx7 + gridspacing_coalesce;
#endif

    energyvalx1 += atominfo[atomid].w * rsqrtf(dx1*dx1 + dyz2);
    energyvalx2 += atominfo[atomid].w * rsqrtf(dx2*dx2 + dyz2);
    energyvalx3 += atominfo[atomid].w * rsqrtf(dx3*dx3 + dyz2);
    energyvalx4 += atominfo[atomid].w * rsqrtf(dx4*dx4 + dyz2);
#if UNROLLX > 4
    energyvalx5 += atominfo[atomid].w * rsqrtf(dx5*dx5 + dyz2);
    energyvalx6 += atominfo[atomid].w * rsqrtf(dx6*dx6 + dyz2);
    energyvalx7 += atominfo[atomid].w * rsqrtf(dx7*dx7 + dyz2);
    energyvalx8 += atominfo[atomid].w * rsqrtf(dx8*dx8 + dyz2);
#endif
  }

  energygrid[outaddr             ] += energyvalx1;
  energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
  energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
  energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
#if UNROLLX > 4
  energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
  energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
  energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
  energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
#endif
}


// required GPU array padding to match thread block size
// XXX note: this code requires block size dimensions to be a power of two
#define TILESIZEX BLOCKSIZEX*UNROLLX
#define TILESIZEY BLOCKSIZEY*UNROLLY
#define GPU_X_ALIGNMASK (TILESIZEX - 1)
#define GPU_Y_ALIGNMASK (TILESIZEY - 1)

static int copyatomstoconstbuf(const float *atoms, float *atompre, 
                        int count, float zplane) {
  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);

  return 0;
}

int vmd_cuda_vol_cpotential(long int natoms, float* atoms, float* grideners,
                            long int numplane, long int numcol, long int numpt, 
                            float gridspacing) {
  enthrparms parms;
  wkf_timerhandle globaltimer;
  double totalruntime;
  int rc=0;

  int numprocs = wkf_thread_numprocessors();
  int deviceCount = 0;
  if (vmd_cuda_num_devices(&deviceCount))
    return -1;
  if (deviceCount < 1)
    return -1;

  /* take the lesser of the number of CPUs and GPUs */
  /* and execute that many threads                  */
  if (deviceCount < numprocs) {
    numprocs = deviceCount;
  }

  printf("Using %d CUDA GPUs\n", numprocs);
  printf("GPU padded grid size: %d x %d x %d\n", 
    (numpt  + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK),
    (numcol + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK),
    numplane);

  parms.atoms = atoms;
  parms.grideners = grideners;
  parms.numplane = numplane;
  parms.numcol = numcol;
  parms.numpt = numpt;
  parms.natoms = natoms;
  parms.gridspacing = gridspacing;

  globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

  /* spawn child threads to do the work */
  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=numplane;
  rc = wkf_threadlaunch(numprocs, &parms, cudaenergythread, &tile);

  // Measure GFLOPS
  wkf_timer_stop(globaltimer);
  totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  if (!rc) {
    double atomevalssec = ((double) numplane * numcol * numpt * natoms) / (totalruntime * 1000000000.0);
    printf("  %g billion atom evals/second, %g GFLOPS\n",
           atomevalssec, atomevalssec * FLOPSPERATOMEVAL);
  } else {
    msgWarn << "A GPU encountered an unrecoverable error." << sendmsg;
    msgWarn << "Calculation will continue using the main CPU." << sendmsg;
  }
  return rc;
}





static void * cudaenergythread(void *voidparms) {
  dim3 volsize, Gsz, Bsz;
  float *devenergy = NULL;
  float *hostenergy = NULL;
  float *atomprebuf = NULL;
  enthrparms *parms = NULL;
  int threadid=0;
  int atomprebufpinned = 0; // try to use pinned/page-locked atom buffer

  wkf_threadlaunch_getid(voidparms, &threadid, NULL);
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);

  /* 
   * copy in per-thread parameters 
   */
  const float *atoms = parms->atoms;
  float* grideners = parms->grideners;
  const long int numplane = parms->numplane;
  const long int numcol = parms->numcol;
  const long int numpt = parms->numpt;
  const long int natoms = parms->natoms;
  const float gridspacing = parms->gridspacing;
  double lasttime, totaltime;

  cudaError_t rc;
  rc = cudaSetDevice(threadid);
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

  // setup energy grid size, padding out arrays for peak GPU memory performance
  volsize.x = (numpt  + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK);
  volsize.y = (numcol + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK);
  volsize.z = 1;      // we only do one plane at a time

  // setup CUDA grid and block sizes
  Bsz.x = BLOCKSIZEX;
  Bsz.y = BLOCKSIZEY;
  Bsz.z = 1;
  Gsz.x = volsize.x / (Bsz.x * UNROLLX);
  Gsz.y = volsize.y / (Bsz.y * UNROLLY); 
  Gsz.z = 1;
  int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

  printf("Thread %d started for CUDA device %d...\n", threadid, threadid);
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);
  wkfmsgtimer * msgt = wkf_msg_timer_create(5);

  // Allocate DMA buffers with some extra padding at the end so that 
  // multiple GPUs aren't DMAing too close to each other, for NUMA..
#define DMABUFPADSIZE (32 * 1024)

  // allocate atom pre-computation and copy buffer in pinned memory
  // for better host/GPU transfer bandwidth, when enabled
  if (atomprebufpinned) {
    // allocate GPU DMA buffer (with padding)
    if (cudaMallocHost((void**) &atomprebuf, MAXATOMS*4*sizeof(float) + DMABUFPADSIZE) != cudaSuccess) {
      printf("Pinned atom copy buffer allocation failed!\n");
      atomprebufpinned=0;
    }
  }

  // if a pinned allocation failed or we choose not to use 
  // pinned memory, fall back to a normal malloc here.
  if (!atomprebufpinned) {
    // allocate GPU DMA buffer (with padding)
    atomprebuf = (float *) malloc(MAXATOMS * 4 * sizeof(float) + DMABUFPADSIZE);
    if (atomprebuf == NULL) {
      printf("Atom copy buffer allocation failed!\n");
      return NULL;
    }
  }


  // allocate and initialize the GPU output array
  cudaMalloc((void**)&devenergy, volmemsz);
  CUERR // check and clear any existing errors

  hostenergy = (float *) malloc(volmemsz); // allocate working buffer

  // For each point in the cube...
  int iterations=0;
  int computedplanes=0;
  wkf_tasktile_t tile;
  while (wkf_threadlaunch_next_tile(voidparms, 1, &tile) != WKF_SCHED_DONE) {
    int k;
    for (k=tile.start; k<tile.end; k++) {
      int y;
      int atomstart;
      float zplane = k * (float) gridspacing;
      computedplanes++; // track work done by this GPU for progress reporting
 
      // Copy energy grid into GPU 16-element padded input
      for (y=0; y<numcol; y++) {
        long eneraddr = k*numcol*numpt + y*numpt;
        memcpy(&hostenergy[y*volsize.x], &grideners[eneraddr], numpt * sizeof(float));
      }

      // Copy the Host input data to the GPU..
      cudaMemcpy(devenergy, hostenergy, volmemsz,  cudaMemcpyHostToDevice);
      CUERR // check and clear any existing errors

      lasttime = wkf_timer_timenow(timer);
      for (atomstart=0; atomstart<natoms; atomstart+=MAXATOMS) {
        iterations++;
        int runatoms;
        int atomsremaining = natoms - atomstart;
        if (atomsremaining > MAXATOMS)
          runatoms = MAXATOMS;
        else
          runatoms = atomsremaining;

        // copy the next group of atoms to the GPU
        if (copyatomstoconstbuf(atoms + 4*atomstart, atomprebuf, runatoms, zplane))
          return NULL;

        // RUN the kernel...
        cenergy<<<Gsz, Bsz, 0>>>(runatoms, gridspacing, devenergy);
        CUERR // check and clear any existing errors
      }

      // Copy the GPU output data back to the host and use/store it..
      cudaMemcpy(hostenergy, devenergy, volmemsz,  cudaMemcpyDeviceToHost);
      CUERR // check and clear any existing errors

      // Copy GPU blocksize padded array back down to the original size
      for (y=0; y<numcol; y++) {
        long eneraddr = k*numcol*numpt + y*numpt;
        memcpy(&grideners[eneraddr], &hostenergy[y*volsize.x], numpt * sizeof(float));
      }
 
      totaltime = wkf_timer_timenow(timer);
      if (wkf_msg_timer_timeout(msgt)) {
        // XXX: we have to use printf here as msgInfo is not thread-safe yet.
        printf("thread[%d] plane %d/%ld (%d computed) time %.2f, elapsed %.1f, est. total: %.1f\n",
               threadid, k, numplane, computedplanes,
               totaltime - lasttime, totaltime,
               totaltime * numplane / (k+1));
      }
    }
  }

  wkf_timer_destroy(timer); // free timer
  wkf_msg_timer_destroy(msgt); // free timer
  free(hostenergy);    // free working buffer
  cudaFree(devenergy); // free CUDA memory buffer
  CUERR // check and clear any existing errors

  // free pinned GPU copy buffer
  if (atomprebufpinned) {
    if (cudaFreeHost(atomprebuf) != cudaSuccess) {
      printf("Pinned atom buffer deallocation failed!\n");
      return NULL;
    }
  } else {
    free(atomprebuf);
  }
  return NULL;
}




