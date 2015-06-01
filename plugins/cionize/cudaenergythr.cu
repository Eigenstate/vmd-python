/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: cudaenergythr.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.13 $      $Date: 2009/08/17 18:57:35 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated coulombic potential grid calculation
 *     John E. Stone <johns@ks.uiuc.edu>
 *     http://www.ks.uiuc.edu/~johns/
 *
 */


#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "util.h"    /* timer code taken from Tachyon */
#include "threads.h" /* threads code taken from Tachyon */
}

#include "cudaenergythr.h" 

typedef struct {
  int threadid;
  int threadcount;
  float* atoms;
  float* grideners;
  long int numplane;
  long int numcol;
  long int numpt;
  long int natoms;
  float gridspacing;
  unsigned char* excludepos;
} enthrparms;

/* thread prototype */
static void * cudaenergythread(void *);

// required GPU array size alignment in bytes, 16 elements is ideal
#define GPU_ALIGNMENT 16
#define GPU_ALIGNMASK (GPU_ALIGNMENT - 1)

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#define CUERR_INT { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  printf("Thread aborting...\n"); \
  return -1; }}
#else
#define CUERR
#define CUERR_INT
#endif

// max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about
// At 16 bytes for atom, for this program 4000 atoms is about the max
// we can store in the constant buffer.
#define MAXATOMS 4000
static __constant__ float4 atominfo[MAXATOMS];

// select which version of the CUDA potential grid kernel to use
#define UNROLLGRIDLOOP 1

#if !defined(UNROLLGRIDLOOP)

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
//
// This version of the code uses the 64KB constant buffer area reloaded
// for each group of MAXATOMS atoms, until the contributions from all
// atoms have been summed into the potential grid.
//
// NVCC -cubin says this implementation uses 10 regs, 28 smem
//
// Benchmark for this version: 150 GFLOPS, 16.7 billion atom evals/sec
//  (Test system: GeForce 8800GTX)
//
// Unroll macro for more readable code
#define UNROLLX 1
#define FLOPSPERATOMEVAL 10.0

__global__ static void cenergy(int numatoms, float gridspacing, float * energygrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = __umul24(gridDim.x, blockDim.x) * yindex + xindex;

  // query current energy value in the grid, start the read early
  // so the fetch occurs while we're summing the new energy values
  float curenergy = energygrid[outaddr];

  float coorx = gridspacing * xindex;
  float coory = gridspacing * yindex;

  int atomid;

  // NOTE: One beneficial side effect of summing groups of 4,000 atoms
  // at a time in multiple passes is that these smaller summation
  // groupings are likely to result in higher precision, since structures
  // are typically stored with atoms in chain order leading to high
  // spatial locality within the groups of 4,000 atoms.  Kernels that
  // sum these subgroups independently of the global sum for each voxel
  // should achieve relatively good floating point precision since large values
  // will tend to be summed with large values, and small values summed with
  // small values, etc.
  float energyval=0.0f;

  /* Main loop: 9 floating point ops, 4 FP loads per iteration */
  for (atomid=0; atomid<numatoms; atomid++) {
    float dx = coorx - atominfo[atomid].x;
    float dy = coory - atominfo[atomid].y;

    // explicitly dividing 1.0f / sqrt() helps the compiler clue in
    // that we really wanted rsqrt() to begin with, it drops 30% performance
    // otherwise.
    // XXX this version uses precomputed dz*dz values
    float r_1 = 1.0f / sqrtf(dx*dx + dy*dy + atominfo[atomid].z);
    energyval += atominfo[atomid].w * r_1;
  }

  // accumulate energy value with the existing value
  energygrid[outaddr] = curenergy + energyval;
}

#else

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
//
// This version of the code uses the 64KB constant buffer area reloaded
// for each group of MAXATOMS atoms, until the contributions from all
// atoms have been summed into the potential grid.
//
// This implementation uses precomputed and unrolled loops of 
// (dy^2 + dz^2) values for increased FP arithmetic intensity.
// The X coordinate portion of the loop is unrolled by four, 
// allowing the same dy^2 + dz^2 values to be reused four times,
// increasing the ratio of FP arithmetic relative to FP loads, and
// eliminating some redundant calculations.
//
// NVCC -cubin says this implementation uses 20 regs, 28 smem
// Profiler output says this code gets 50% warp occupancy
//
// Benchmark for this version: 269 GFLOPS, 34.7 billion atom evals/sec
//  (Test system: GeForce 8800GTX)
//
// Unroll macro for more readable code
#define UNROLLX 4
#define FLOPSPERATOMEVAL (31.0/4.0)

__global__ static void cenergy(int numatoms, float gridspacing, float * energygrid) {
  unsigned int xindex  = (__umul24(blockIdx.x, blockDim.x) + threadIdx.x) * UNROLLX;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex + xindex;

  // query current energy value in the grid, start the read early
  // so the fetch occurs while we're summing the new energy values
  float curenergyx1 = energygrid[outaddr    ];
  float curenergyx2 = energygrid[outaddr + 1];
  float curenergyx3 = energygrid[outaddr + 2];
  float curenergyx4 = energygrid[outaddr + 3];

  float coory = gridspacing * yindex;

  float coorx1 = gridspacing * (xindex    );
  float coorx2 = gridspacing * (xindex + 1);
  float coorx3 = gridspacing * (xindex + 2);
  float coorx4 = gridspacing * (xindex + 3);

  // NOTE: One beneficial side effect of summing groups of 4,000 atoms
  // at a time in multiple passes is that these smaller summation
  // groupings are likely to result in higher precision, since structures
  // are typically stored with atoms in chain order leading to high
  // spatial locality within the groups of 4,000 atoms.  Kernels that
  // sum these subgroups independently of the global sum for each voxel
  // should achieve relatively good floating point precision since large values
  // will tend to be summed with large values, and small values summed with
  // small values, etc.
  float energyvalx1=0.0f;
  float energyvalx2=0.0f;
  float energyvalx3=0.0f;
  float energyvalx4=0.0f;

  // Atom loop: 4 voxels, 31 floating point ops, 4 FP loads per iteration
  //            31/4 = 7.75 floating point ops per voxel
  //
  // Note: this implementation uses precomputed and unrolled
  // loops of dy*dy + dz*dz values for increased FP arithmetic intensity
  // per FP load.
  // XXX explicitly dividing 1.0f / sqrt() helps the compiler clue in
  //     that we really wanted rsqrt() to begin with, it drops 30%
  //     performance otherwise.
  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dysqpdzsq = (dy * dy) + atominfo[atomid].z;

    float dx1 = coorx1 - atominfo[atomid].x;
    float dx2 = coorx2 - atominfo[atomid].x;
    float dx3 = coorx3 - atominfo[atomid].x;
    float dx4 = coorx4 - atominfo[atomid].x;

    energyvalx1 += atominfo[atomid].w * (1.0f / sqrtf(dx1*dx1 + dysqpdzsq));
    energyvalx2 += atominfo[atomid].w * (1.0f / sqrtf(dx2*dx2 + dysqpdzsq));
    energyvalx3 += atominfo[atomid].w * (1.0f / sqrtf(dx3*dx3 + dysqpdzsq));
    energyvalx4 += atominfo[atomid].w * (1.0f / sqrtf(dx4*dx4 + dysqpdzsq));
  }

  // accumulate energy value with the existing value
  energygrid[outaddr    ] = curenergyx1 + energyvalx1;
  energygrid[outaddr + 1] = curenergyx2 + energyvalx2;
  energygrid[outaddr + 2] = curenergyx3 + energyvalx3;
  energygrid[outaddr + 3] = curenergyx4 + energyvalx4;
}

#endif



int copyatomstoconstbuf(const float *atoms, int count, float zplane) {
  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  float atompre[4*MAXATOMS];
  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  cudaMemcpyToSymbol(atominfo, atompre, count * 4 * sizeof(float), 0);
  CUERR_INT // check and clear any existing errors

  return 0;
}

int calc_grid_energies_cuda_thr(float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char* excludepos, int maxnumprocs) {
  int i;
  enthrparms *parms;
  rt_thread_t * threads;
  rt_timerhandle globaltimer;
  double totalruntime;

#if defined(THR)
  int numprocs;
  int availprocs = rt_thread_numprocessors();
  if (maxnumprocs <= availprocs) {
    numprocs = maxnumprocs;
  } else {
    numprocs = availprocs;
  }

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  // don't print startup messages when we go into ion-placement phase
  if (natoms > 1) {
    printf("Detected %d CUDA accelerators:\n", deviceCount);
  }
  int dev;
  for (dev=0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // don't print startup messages when we go into ion-placement phase
    if (natoms > 1) {
      printf("  CUDA device[%d]: '%s'  Mem: %dMB  Rev: %d.%d\n", 
             dev, deviceProp.name, deviceProp.totalGlobalMem / (1024*1024), 
             deviceProp.major, deviceProp.minor);
    }
  }

  /* take the lesser of the number of CPUs and GPUs */
  /* and execute that many threads                  */
  if (deviceCount < numprocs) {
    numprocs = deviceCount;
  }

#else
  int numprocs = 1;
#endif

  printf("calc_grid_energies_cuda_thr()\n");
  printf("  using %d processors\n", numprocs);

  /* allocate array of threads */
  threads = (rt_thread_t *) calloc(numprocs * sizeof(rt_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (enthrparms *) malloc(numprocs * sizeof(enthrparms));
  for (i=0; i<numprocs; i++) {
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;
    parms[i].atoms = atoms;
    parms[i].grideners = grideners;
    parms[i].numplane = numplane;
    parms[i].numcol = numcol;
    parms[i].numpt = numpt;
    parms[i].natoms = natoms;
    parms[i].gridspacing = gridspacing;
    parms[i].excludepos = excludepos;
  }

  // don't print info messages in ion placement phase
  if (natoms > 1) {
    printf("GPU padded grid size: %d x %d x %d\n", numpt, numcol, numplane);
  }

  globaltimer = rt_timer_create();
  rt_timer_start(globaltimer);

#if defined(THR)
  /* spawn child threads to do the work */
  for (i=0; i<numprocs; i++) {
    rt_thread_create(&threads[i], cudaenergythread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numprocs; i++) {
    rt_thread_join(threads[i], NULL);
  }
#else
  /* single thread does all of the work */
  cudaenergythread((void *) &parms[0]);
#endif

  // Measure GFLOPS
  rt_timer_stop(globaltimer);
  totalruntime = rt_timer_time(globaltimer);
  rt_timer_destroy(globaltimer);

  double atomevalssec = ((double) numplane * numcol * numpt * natoms) / (totalruntime * 1000000000.0);
  printf("  %g billion atom evals/second, %g GFLOPS\n",
           atomevalssec, atomevalssec * FLOPSPERATOMEVAL);


  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}





static void * cudaenergythread(void *voidparms) {
  dim3 volsize, Gsz, Bsz;
  float *devenergy = NULL;
  float *hostenergy = NULL;

  enthrparms *parms = (enthrparms *) voidparms;
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
  const unsigned char* excludepos = parms->excludepos;
  const int threadid = parms->threadid;
  const int threadcount = parms->threadcount;
  rt_timerhandle runtimer, copytimer, timer;
  float copytotal, runtotal;
  cudaError_t rc;

  if (natoms > 1)
    printf("Thread %d opening CUDA device %d...\n", threadid, threadid);

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


  CUERR // check and clear any existing errors

  // setup energy grid size, padding out arrays for peak GPU memory performance
  volsize.x = (numpt  + GPU_ALIGNMASK) & ~(GPU_ALIGNMASK);
  volsize.y = (numcol + GPU_ALIGNMASK) & ~(GPU_ALIGNMASK);
  volsize.z = 1;      // we only do one plane at a time

  // setup CUDA grid and block sizes
  Bsz.x = 16 / UNROLLX;                  // each thread does multiple Xs
  Bsz.y = 16;
  Bsz.z = 1;
  Gsz.x = volsize.x / (Bsz.x * UNROLLX); // each thread does multiple Xs
  Gsz.y = volsize.y / Bsz.y;
  Gsz.z = 1;
  int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

  float lasttime, totaltime;
  copytimer = rt_timer_create();
  runtimer = rt_timer_create();
  timer = rt_timer_create();

  rt_timer_start(timer);
  if (natoms > 1)
    printf("thread %d started...\n", threadid);

  // allocate and initialize the GPU output array
  cudaMalloc((void**)&devenergy, volmemsz);
  CUERR // check and clear any existing errors

  hostenergy = (float *) malloc(volmemsz); // allocate working buffer

  // For each point in the cube...
  int iterations=0;
  int k;
  for (k=threadid; k<numplane; k+= threadcount) {
    int x, y;
    int atomstart;
    float zplane = k * (float) gridspacing;
  
    // Copy energy grid into GPU 16-element padded input
    for (y=0; y<numcol; y++) {
      long eneraddr = k*numcol*numpt + y*numpt;
      for (x=0; x<numpt; x++) {
        long addr = eneraddr + x;
        hostenergy[y*volsize.x + x] = grideners[addr];
      }
    }

    // Copy the Host input data to the GPU..
    cudaMemcpy(devenergy, hostenergy, volmemsz,  cudaMemcpyHostToDevice);
    CUERR // check and clear any existing errors

    lasttime = rt_timer_timenow(timer);
    for (atomstart=0; atomstart<natoms; atomstart+=MAXATOMS) {
      iterations++;
      int runatoms;
      int atomsremaining = natoms - atomstart;
      if (atomsremaining > MAXATOMS)
        runatoms = MAXATOMS;
      else
        runatoms = atomsremaining;

      // copy the next group of atoms to the GPU
      rt_timer_start(copytimer);
      if (copyatomstoconstbuf(atoms + 4*atomstart, runatoms, zplane))
        return NULL;
      rt_timer_stop(copytimer);
      copytotal += rt_timer_time(copytimer);

      // RUN the kernel...
      rt_timer_start(runtimer);
      cenergy<<<Gsz, Bsz, 0>>>(runatoms, gridspacing, devenergy);
      CUERR // check and clear any existing errors
      rt_timer_stop(runtimer);
      runtotal += rt_timer_time(runtimer);
    }

    // Copy the GPU output data back to the host and use/store it..
    cudaMemcpy(hostenergy, devenergy, volmemsz,  cudaMemcpyDeviceToHost);
    CUERR // check and clear any existing errors

    // Copy GPU 16-byte padded output back down to the original grid size,
    // and check for exclusions
    for (y=0; y<numcol; y++) {
      long eneraddr = k*numcol*numpt + y*numpt;
      for (x=0; x<numpt; x++) {
        long addr = eneraddr + x;
        grideners[addr] = hostenergy[y*volsize.x + x];
        if (excludepos[addr] != 0) {
          grideners[addr] = 0; 
        }
      }
    }
    totaltime = rt_timer_timenow(timer);
 
    // only print per-thread status messages when we're calculating a 
    // full grid, don't bother when we're just adding a single ion. 
    if (natoms > 1) {
      printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n",
           threadid, k, numplane,
           totaltime - lasttime, totaltime, 
           totaltime * numplane / (k+1));
    }
  }

  free(hostenergy);        // free working buffer
  cudaFree(devenergy); // free CUDA memory buffer

  rt_timer_destroy(timer);
  rt_timer_destroy(runtimer);
  rt_timer_destroy(copytimer);

  return NULL;
}




