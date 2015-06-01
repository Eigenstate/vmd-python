/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
 * CUDA accelerated coulombic potential grid test code
 *   John E. Stone <johns@ks.uiuc.edu>
 *   http://www.ks.uiuc.edu/~johns/
 *
 * Coulombic potential grid calculation microbenchmark based on the time
 * consuming portions of the 'cionize' ion placement tool.
 *
 * Benchmark for this version: xxx GFLOPS, xxxx billion atom evals/sec
 *   (Test system: GeForce 8800GTX)
 */

#include <stdio.h>
#include <stdlib.h>
#include "util.h"

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
// Note: this implementation uses precomputed and unrolled
// loops of dy*dy + dz*dz values for increased FP arithmetic intensity
// per FP load.  The X coordinate portion of the loop is unrolled by 
// four, allowing the same dy^2 + dz^2 values to be reused four times,
// increasing the ratio of FP arithmetic relative to FP loads, and 
// eliminating some redundant calculations.
//
// NVCC -cubin says this implementation uses xx regs, yy smem
// Profiler output says this code gets xx% warp occupancy
//
// Best benchmark to date: xxx GFLOPS
#define UNROLLX 4
__global__ static void cenergy(float4 atominfo, float gridspacing, float * energygrid) {
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
  float dy = coory - atominfo.y;
  float dysqpdzsq = (dy * dy) + atominfo.z;

  float dx1 = coorx1 - atominfo.x;
  float dx2 = coorx2 - atominfo.x;
  float dx3 = coorx3 - atominfo.x;
  float dx4 = coorx4 - atominfo.x;

  energyvalx1 += atominfo.w * (1.0f / sqrtf(dx1*dx1 + dysqpdzsq));
  energyvalx2 += atominfo.w * (1.0f / sqrtf(dx2*dx2 + dysqpdzsq));
  energyvalx3 += atominfo.w * (1.0f / sqrtf(dx3*dx3 + dysqpdzsq));
  energyvalx4 += atominfo.w * (1.0f / sqrtf(dx4*dx4 + dysqpdzsq));

  // accumulate energy value with the existing value
  energygrid[outaddr    ] = curenergyx1 + energyvalx1;
  energygrid[outaddr + 1] = curenergyx2 + energyvalx2;
  energygrid[outaddr + 2] = curenergyx3 + energyvalx3;
  energygrid[outaddr + 3] = curenergyx4 + energyvalx4;
}


int main(int argc, char** argv) {
  float *doutput = NULL;
  float *energy = NULL;
  float *atoms = NULL;
  dim3 volsize, Gsz, Bsz;
  rt_timerhandle runtimer, mastertimer, copytimer, hostcopytimer;
  float copytotal, runtotal, mastertotal, hostcopytotal;
  const char *statestr = "|/-\\.";
  int state=0;

  printf("CUDA accelerated coulombic potential microbenchmark V2.0\n");
  printf("John E. Stone <johns@ks.uiuc.edu>\n");
  printf("http://www.ks.uiuc.edu/~johns/\n");
  printf("--------------------------------------------------------\n");
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Detected %d CUDA accelerators:\n", deviceCount);
  int dev;
  for (dev=0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("  CUDA device[%d]: '%s'  Mem: %dMB  Rev: %d.%d\n", 
           dev, deviceProp.name, deviceProp.totalGlobalMem / (1024*1024), 
           deviceProp.major, deviceProp.minor);
  }

  int cudadev = 0;
  if (argc == 2) {
    sscanf(argv[1], "%d", &cudadev);
    if (cudadev < 0 || cudadev >= deviceCount) {
      cudadev = 0; 
    }    
  }
  printf("  Single-threaded single-GPU test run.\n");
  printf("  Opening CUDA device %d...\n", cudadev);

  cudaError_t rc;
  rc = cudaSetDevice(cudadev);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess)
      return -1; // abort and return an error
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }
  CUERR // check and clear any existing errors

  // number of atoms to simulate
  int atomcount = 1;

  // setup energy grid size
  // XXX this is a large test case to clearly illustrate that even while
  //     the CUDA kernel is running entirely on the GPU, the CUDA runtime
  //     library is soaking up the entire host CPU for some reason.
  volsize.x = 384;
  volsize.y = 384;
  volsize.z = 512;

  // set voxel spacing
  float gridspacing = 0.1;

  // setup CUDA grid and block sizes
  // XXX we have to make a trade-off between the number of threads per
  //     block and the resulting padding size we'll end up with since
  //     each thread will do 4 consecutive grid cells in this version,
  //     we're using up some of our available parallelism to reduce overhead.
  Bsz.x =  4;                            // each thread does multiple Xs
  Bsz.y = 16;
  Bsz.z = 1;
  Gsz.x = volsize.x / (Bsz.x * UNROLLX); // each thread does multiple Xs
  Gsz.y = volsize.y / Bsz.y; 
  Gsz.z = 1; 

  // initialize the wall clock timers
  runtimer = rt_timer_create();
  mastertimer = rt_timer_create();
  copytimer = rt_timer_create();
  hostcopytimer = rt_timer_create();
  copytotal = 0;
  runtotal = 0;
  hostcopytotal = 0;

  printf("Grid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);
  printf("Running kernel(atoms:%d, gridspacing %g, z %d)\n", 1, gridspacing, 0);

  // allocate and initialize atom coordinates and charges
  if (initatoms(&atoms, 1, volsize, gridspacing))
    return -1;

  // allocate and initialize the GPU output array
  int volmemsz = sizeof(float) * volsize.x * volsize.y;
  printf("Allocating %.2fMB of memory for output buffer...\n", volmemsz / (1024.0 * 1024.0));

  cudaMalloc((void**)&doutput, volmemsz);
  CUERR // check and clear any existing errors
  cudaMemset(doutput, 0, volmemsz);
  CUERR // check and clear any existing errors

  rt_timer_start(mastertimer);

  printf("%c\r", statestr[state]);
  fflush(stdout);
  state = (state+1) & 3;

  // RUN the kernel...
  rt_timer_start(runtimer);
  int plane;
  for (plane=0; plane<volsize.z; plane++) {
    float zplane = plane*0.1;

    float4 atominfo;
    atominfo.x = atoms[0];   // X
    atominfo.y = atoms[1];   // Y 
    atominfo.z = (0 - atoms[2])*(0 - atoms[2]); // dz^2
    atominfo.w = atoms[3];   // Q

    cenergy<<<Gsz, Bsz, 0>>>(atominfo, 0.1, doutput);
    cudaThreadSynchronize(); // wait for kernel to complete
    CUERR // check and clear any existing errors
  }

  rt_timer_stop(runtimer);
  runtotal += rt_timer_time(runtimer);
  printf("Done\n");

  rt_timer_stop(mastertimer);
  mastertotal = rt_timer_time(mastertimer);

  // Copy the GPU output data back to the host and use/store it..
  energy = (float *) malloc(volmemsz);
  rt_timer_start(hostcopytimer);
  cudaMemcpy(energy, doutput, volmemsz,  cudaMemcpyDeviceToHost);
  CUERR // check and clear any existing errors
  rt_timer_stop(hostcopytimer);
  hostcopytotal=rt_timer_time(hostcopytimer);

#if 0
  int x, y;
  for (y=0; y<16; y++) {
    for (x=0; x<16; x++) {
      int addr = y * volsize.x + x;
      printf("out[%d]: %f\n", addr, energy[addr]);
    }
  }
#endif

  printf("Total time: %f seconds\n", mastertotal);
  printf("GPU to host copy bandwidth: %gMB/sec, %f seconds total\n",
         (volmemsz / (1024.0 * 1024.0)) / hostcopytotal, hostcopytotal);

  double atomevalssec = ((double) volsize.x * volsize.y * volsize.z * atomcount) / (mastertotal * 1000000000.0);
  printf("Efficiency metric, %g billion atom evals per second\n", atomevalssec);

  /* 31/4 FLOPS per atom eval */
  printf("FP performance: %g GFLOPS\n", atomevalssec * (31.0/4.0));
  printf(" (1xADD=1 + 5xSUB=5 + 1xMUL=1 + 8xMADD=16 + 4xRSQRT=8 = 31 per iteration)\n");
  free(atoms);
  free(energy);
  cudaFree(doutput);

  return 0;
}



