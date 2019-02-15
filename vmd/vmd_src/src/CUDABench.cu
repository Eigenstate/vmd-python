/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDABench.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.35 $      $Date: 2019/01/17 21:38:54 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Short benchmark kernels to measure GPU performance
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  return -1; }}


//
// Benchmark peak Multiply-Add instruction performance, in GFLOPS
//

// FMADD16 macro contains a sequence of operations that the compiler
// won't optimize out, and will translate into a densely packed block
// of multiply-add instructions with no intervening register copies/moves
// or other instructions. 
#define FMADD16 \
    tmp0  = tmp0*tmp4+tmp7;     \
    tmp1  = tmp1*tmp5+tmp0;     \
    tmp2  = tmp2*tmp6+tmp1;     \
    tmp3  = tmp3*tmp7+tmp2;     \
    tmp4  = tmp4*tmp0+tmp3;     \
    tmp5  = tmp5*tmp1+tmp4;     \
    tmp6  = tmp6*tmp2+tmp5;     \
    tmp7  = tmp7*tmp3+tmp6;     \
    tmp8  = tmp8*tmp12+tmp15;   \
    tmp9  = tmp9*tmp13+tmp8;    \
    tmp10 = tmp10*tmp14+tmp9;   \
    tmp11 = tmp11*tmp15+tmp10;  \
    tmp12 = tmp12*tmp8+tmp11;   \
    tmp13 = tmp13*tmp9+tmp12;   \
    tmp14 = tmp14*tmp10+tmp13;  \
    tmp15 = tmp15*tmp11+tmp14;

// CUDA grid, thread block, loop, and MADD operation counts
#define GRIDSIZEX   6144
#define BLOCKSIZEX  64
#define GLOOPS      2000
#define MADDCOUNT   64

// FLOP counting
#define FLOPSPERLOOP (MADDCOUNT * 16)

__global__ static void madd_kernel(float *doutput) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7;
  float tmp8,tmp9,tmp10,tmp11,tmp12,tmp13,tmp14,tmp15;
  tmp0=tmp1=tmp2=tmp3=tmp4=tmp5=tmp6=tmp7=0.0f;
  tmp8=tmp9=tmp10=tmp11=tmp12=tmp13=tmp14=tmp15 = 0.0f;

  tmp15=tmp7 = blockIdx.x * 0.001f; // prevent compiler from optimizing out
  tmp1 = blockIdx.y * 0.001f;       // the body of the loop...

  int loop;
  for(loop=0; loop<GLOOPS; loop++){
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
  }

  doutput[tid] = tmp0+tmp1+tmp2+tmp3+tmp4+tmp5+tmp6+tmp7
                 +tmp8+tmp9+tmp10+tmp11+tmp12+tmp13+tmp14+tmp15;
}


static int cudamaddgflops(int cudadev, double *gflops, int testloops) {
  float *doutput = NULL;
  dim3 Bsz, Gsz;
  wkf_timerhandle timer;
  int i;

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


  // setup CUDA grid and block sizes
  Bsz.x = BLOCKSIZEX;
  Bsz.y = 1;
  Bsz.z = 1;
  Gsz.x = GRIDSIZEX;
  Gsz.y = 1;
  Gsz.z = 1;

  // allocate output array
  cudaMalloc((void**)&doutput, BLOCKSIZEX * GRIDSIZEX * sizeof(float));
  CUERR // check and clear any existing errors

  timer=wkf_timer_create();
  wkf_timer_start(timer);
  for (i=0; i<testloops; i++) { 
    madd_kernel<<<Gsz, Bsz>>>(doutput);
  }
  cudaDeviceSynchronize(); // wait for kernel to finish
  CUERR // check and clear any existing errors
  wkf_timer_stop(timer);

  double runtime = wkf_timer_time(timer);
  double gflop = ((double) GLOOPS) * ((double) FLOPSPERLOOP) *
                  ((double) BLOCKSIZEX) * ((double) GRIDSIZEX) * (1.0e-9) * testloops;
  
  *gflops = gflop / runtime;

  cudaFree(doutput);
  CUERR // check and clear any existing errors

  wkf_timer_destroy(timer);

  return 0;
}

typedef struct {
  int deviceid;
  int testloops;
  double gflops;
} maddthrparms;

static void * cudamaddthread(void *voidparms) {
  maddthrparms *parms = (maddthrparms *) voidparms;
  cudamaddgflops(parms->deviceid, &parms->gflops, parms->testloops);
  return NULL;
}

int vmd_cuda_madd_gflops(int numdevs, int *devlist, double *gflops,
                         int testloops) {
  maddthrparms *parms;
  wkf_thread_t * threads;
  int i;

  /* allocate array of threads */
  threads = (wkf_thread_t *) calloc(numdevs * sizeof(wkf_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (maddthrparms *) malloc(numdevs * sizeof(maddthrparms));
  for (i=0; i<numdevs; i++) {
    if (devlist != NULL)
      parms[i].deviceid = devlist[i];
    else
      parms[i].deviceid = i;

    parms[i].testloops = testloops;
    parms[i].gflops = 0.0;
  }

#if defined(VMDTHREADS)
  /* spawn child threads to do the work */
  /* thread 0 must also be processed this way otherwise    */
  /* we'll permanently bind the main thread to some device */
  for (i=0; i<numdevs; i++) {
    wkf_thread_create(&threads[i], cudamaddthread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numdevs; i++) {
    wkf_thread_join(threads[i], NULL);
  }
#else
  /* single thread does all of the work */
  cudamaddthread((void *) &parms[0]);
#endif

  for (i=0; i<numdevs; i++) {
    gflops[i] = parms[i].gflops; 
  }

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}






//
// Host-GPU memcpy I/O bandwidth benchmark
//

#define BWITER 500
#define LATENCYITER 50000

static int cudabusbw(int cudadev, 
                     double *hdmbsec, double *hdlatusec, 
                     double *phdmbsec, double *phdlatusec, 
                     double *dhmbsec, double *dhlatusec,
                     double *pdhmbsec, double *pdhlatusec) {
  float *hdata = NULL;   // non-pinned DMA buffer
  float *phdata = NULL;  // pinned DMA buffer
  float *ddata = NULL;
  int i;
  double runtime;
  wkf_timerhandle timer;
  int memsz = 1024 * 1024 * sizeof(float);

  *hdmbsec = 0.0;
  *hdlatusec = 0.0;
  *dhmbsec = 0.0;
  *dhlatusec = 0.0;
  *phdmbsec = 0.0;
  *phdlatusec = 0.0;
  *pdhmbsec = 0.0;
  *pdhlatusec = 0.0;

  // attach to the selected device
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

  // allocate non-pinned output array
  hdata = (float *) malloc(memsz); 

  // allocate pinned output array
  cudaMallocHost((void**) &phdata, memsz);
  CUERR // check and clear any existing errors

  // allocate device memory
  cudaMalloc((void**) &ddata, memsz);
  CUERR // check and clear any existing errors

  // create timer
  timer=wkf_timer_create();

  //
  // Host to device timings
  //

  // non-pinned bandwidth
  wkf_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(ddata, hdata, memsz,  cudaMemcpyHostToDevice);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *hdmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);

  // non-pinned latency
  wkf_timer_start(timer);
  for (i=0; i<LATENCYITER; i++) {
    cudaMemcpy(ddata, hdata, 1,  cudaMemcpyHostToDevice);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *hdlatusec = runtime * 1.0e6 / ((double) LATENCYITER);


  // pinned bandwidth
  wkf_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(ddata, phdata, memsz,  cudaMemcpyHostToDevice);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *phdmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);

  // pinned latency
  wkf_timer_start(timer);
  for (i=0; i<LATENCYITER; i++) {
    cudaMemcpy(ddata, phdata, 1,  cudaMemcpyHostToDevice);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *phdlatusec = runtime * 1.0e6 / ((double) LATENCYITER);

 
  //
  // Device to host timings
  //

  // non-pinned bandwidth
  wkf_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(hdata, ddata, memsz,  cudaMemcpyDeviceToHost);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *dhmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);

  // non-pinned latency
  wkf_timer_start(timer);
  for (i=0; i<LATENCYITER; i++) {
    cudaMemcpy(hdata, ddata, 1,  cudaMemcpyDeviceToHost);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *dhlatusec = runtime * 1.0e6 / ((double) LATENCYITER);


  // pinned bandwidth
  wkf_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(phdata, ddata, memsz,  cudaMemcpyDeviceToHost);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *pdhmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);

  // pinned latency
  wkf_timer_start(timer);
  for (i=0; i<LATENCYITER; i++) {
    cudaMemcpy(phdata, ddata, 1,  cudaMemcpyDeviceToHost);
  }
  wkf_timer_stop(timer);
  CUERR // check and clear any existing errors
  runtime = wkf_timer_time(timer);
  *pdhlatusec = runtime * 1.0e6 / ((double) LATENCYITER);
 
 
  cudaFree(ddata);
  CUERR // check and clear any existing errors
  cudaFreeHost(phdata);
  CUERR // check and clear any existing errors
  free(hdata);

  wkf_timer_destroy(timer);

  return 0;
}

typedef struct {
  int deviceid;
  double hdmbsec;
  double hdlatusec;
  double phdmbsec;
  double phdlatusec;
  double dhmbsec;
  double dhlatusec;
  double pdhmbsec;
  double pdhlatusec;
} busbwthrparms;

static void * cudabusbwthread(void *voidparms) {
  busbwthrparms *parms = (busbwthrparms *) voidparms;
  cudabusbw(parms->deviceid, 
            &parms->hdmbsec, &parms->hdlatusec,
            &parms->phdmbsec, &parms->phdlatusec,
            &parms->dhmbsec, &parms->dhlatusec,
            &parms->pdhmbsec, &parms->pdhlatusec);
  return NULL;
}

int vmd_cuda_bus_bw(int numdevs, int *devlist, 
                    double *hdmbsec, double *hdlatusec,
                    double *phdmbsec,double *phdlatusec,
                    double *dhmbsec, double *dhlatusec,
                    double *pdhmbsec, double *pdhlatusec) {
  busbwthrparms *parms;
  wkf_thread_t * threads;
  int i;

  /* allocate array of threads */
  threads = (wkf_thread_t *) calloc(numdevs * sizeof(wkf_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (busbwthrparms *) malloc(numdevs * sizeof(busbwthrparms));
  for (i=0; i<numdevs; i++) {
    if (devlist != NULL)
      parms[i].deviceid = devlist[i];
    else
      parms[i].deviceid = i;
    parms[i].hdmbsec = 0.0;
    parms[i].hdlatusec = 0.0;
    parms[i].phdmbsec = 0.0;
    parms[i].phdlatusec = 0.0;
    parms[i].dhmbsec = 0.0;
    parms[i].dhlatusec = 0.0;
    parms[i].pdhmbsec = 0.0;
    parms[i].pdhlatusec = 0.0;
  }

#if defined(VMDTHREADS)
  /* spawn child threads to do the work */
  /* thread 0 must also be processed this way otherwise    */
  /* we'll permanently bind the main thread to some device */
  for (i=0; i<numdevs; i++) {
    wkf_thread_create(&threads[i], cudabusbwthread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numdevs; i++) {
    wkf_thread_join(threads[i], NULL);
  }
#else
  /* single thread does all of the work */
  cudabusbwthread((void *) &parms[0]);
#endif

  for (i=0; i<numdevs; i++) {
    hdmbsec[i] = parms[i].hdmbsec; 
    hdlatusec[i] = parms[i].hdlatusec; 
    phdmbsec[i] = parms[i].phdmbsec; 
    phdlatusec[i] = parms[i].phdlatusec; 
    dhmbsec[i] = parms[i].dhmbsec; 
    dhlatusec[i] = parms[i].dhlatusec; 
    pdhmbsec[i] = parms[i].pdhmbsec; 
    pdhlatusec[i] = parms[i].pdhlatusec; 
  }

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}



//
// GPU device global memory bandwidth benchmark
//
template <class T>
__global__ void gpuglobmemcpybw(T *dest, const T *src) {
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dest[idx] = src[idx];
}

template <class T>
__global__ void gpuglobmemsetbw(T *dest, const T val) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dest[idx] = val;
}

typedef float4 datatype;

static int cudaglobmembw(int cudadev, double *gpumemsetgbsec, double *gpumemcpygbsec) {
  int i;
  int len = 1 << 22; // one thread per data element
  int loops = 500;
  datatype *src, *dest;
  datatype val=make_float4(1.0f, 1.0f, 1.0f, 1.0f);

  // initialize to zero for starters
  float memsettime = 0.0f;
  float memcpytime = 0.0f;
  *gpumemsetgbsec = 0.0;
  *gpumemcpygbsec = 0.0;

  // attach to the selected device
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

  cudaMalloc((void **) &src, sizeof(datatype)*len);
  CUERR
  cudaMalloc((void **) &dest, sizeof(datatype)*len);
  CUERR

  dim3 BSz(256, 1, 1);
  dim3 GSz(len / (BSz.x * BSz.y * BSz.z), 1, 1); 

  // do a warm-up pass
  gpuglobmemsetbw<datatype><<< GSz, BSz >>>(src, val);
  CUERR
  gpuglobmemsetbw<datatype><<< GSz, BSz >>>(dest, val);
  CUERR
  gpuglobmemcpybw<datatype><<< GSz, BSz >>>(dest, src);
  CUERR

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // execute the memset kernel
  cudaEventRecord(start, 0);
  for (i=0; i<loops; i++) {
    gpuglobmemsetbw<datatype><<< GSz, BSz >>>(dest, val);
  }
  CUERR
  cudaEventRecord(end, 0);
  CUERR
  cudaEventSynchronize(start);
  CUERR
  cudaEventSynchronize(end);
  CUERR
  cudaEventElapsedTime(&memsettime, start, end);
  CUERR

  // execute the memcpy kernel
  cudaEventRecord(start, 0);
  for (i=0; i<loops; i++) {
    gpuglobmemcpybw<datatype><<< GSz, BSz >>>(dest, src);
  }
  cudaEventRecord(end, 0);
  CUERR
  cudaEventSynchronize(start);
  CUERR
  cudaEventSynchronize(end);
  CUERR
  cudaEventElapsedTime(&memcpytime, start, end);
  CUERR

  cudaEventDestroy(start);
  CUERR
  cudaEventDestroy(end);
  CUERR

  *gpumemsetgbsec = (len * sizeof(datatype) / (1024.0 * 1024.0)) / (memsettime / loops);
  *gpumemcpygbsec = (2 * len * sizeof(datatype) / (1024.0 * 1024.0)) / (memcpytime / loops);
  cudaFree(dest);
  cudaFree(src);
  CUERR

  return 0;
}

typedef struct {
  int deviceid;
  double memsetgbsec;
  double memcpygbsec;
} globmembwthrparms;

static void * cudaglobmembwthread(void *voidparms) {
  globmembwthrparms *parms = (globmembwthrparms *) voidparms;
  cudaglobmembw(parms->deviceid, &parms->memsetgbsec, &parms->memcpygbsec);
  return NULL;
}

int vmd_cuda_globmem_bw(int numdevs, int *devlist, 
                        double *memsetgbsec, double *memcpygbsec) {
  globmembwthrparms *parms;
  wkf_thread_t * threads;
  int i;

  /* allocate array of threads */
  threads = (wkf_thread_t *) calloc(numdevs * sizeof(wkf_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (globmembwthrparms *) malloc(numdevs * sizeof(globmembwthrparms));
  for (i=0; i<numdevs; i++) {
    if (devlist != NULL)
      parms[i].deviceid = devlist[i];
    else
      parms[i].deviceid = i;
    parms[i].memsetgbsec = 0.0;
    parms[i].memcpygbsec = 0.0;
  }

#if defined(VMDTHREADS)
  /* spawn child threads to do the work */
  /* thread 0 must also be processed this way otherwise    */
  /* we'll permanently bind the main thread to some device */
  for (i=0; i<numdevs; i++) {
    wkf_thread_create(&threads[i], cudaglobmembwthread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numdevs; i++) {
    wkf_thread_join(threads[i], NULL);
  }
#else
  /* single thread does all of the work */
  cudaglobmembwthread((void *) &parms[0]);
#endif

  for (i=0; i<numdevs; i++) {
    memsetgbsec[i] = parms[i].memsetgbsec;
    memcpygbsec[i] = parms[i].memcpygbsec;
  }

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}


//
// Benchmark latency for complete threadpool barrier wakeup/run/sleep cycle
//
static void * vmddevpoollatencythread(void *voidparms) {
  return NULL;
}

static void * vmddevpooltilelatencythread(void *voidparms) {
  int threadid=-1;
  int tilesize=1;
  void *parms=NULL;
  wkf_threadpool_worker_getid(voidparms, &threadid, NULL);
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);

  // grind through task tiles until none are left
  wkf_tasktile_t tile;
  while (wkf_threadpool_next_tile(voidparms, tilesize, &tile) != WKF_SCHED_DONE) {
    // do nothing but eat work units...
  }

  return NULL;
}


// no-op kernel for timing kernel launches
__global__ static void nopkernel(float * ddata) {
  unsigned int xindex  = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yindex  = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int outaddr = gridDim.x * blockDim.x * yindex + xindex;

  if (ddata != NULL)
    ddata[outaddr] = outaddr;
}

// empty kernel for timing kernel launches
__global__ static void voidkernel(void) {
  return;
}

static void * vmddevpoolcudatilelatencythread(void *voidparms) {
  int threadid=-1;
  int tilesize=1;
  float *parms=NULL;
  wkf_threadpool_worker_getid(voidparms, &threadid, NULL);

  // XXX Note that we expect parms to be set to NULL or a valid CUDA
  //     global memory pointer for correct operation of the NOP kernel below
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);

#if 0
  // scale tile size by device performance
  tilesize=4; // GTX 280, Tesla C1060 starting point tile size
  wkf_threadpool_worker_devscaletile(voidparms, &tilesize);
#endif

  // grind through task tiles until none are left
  wkf_tasktile_t tile;
  dim3 Gsz(1,1,0);
  dim3 Bsz(8,8,1);
  while (wkf_threadpool_next_tile(voidparms, tilesize, &tile) != WKF_SCHED_DONE) {
    // launch a no-op CUDA kernel
    nopkernel<<<Gsz, Bsz, 0>>>(parms);
  }

  // wait for all GPU kernels to complete
  cudaDeviceSynchronize();

  return NULL;
}


int vmd_cuda_devpool_latency(wkf_threadpool_t *devpool, int tilesize,
                             double *kernlaunchlatency,
                             double *barlatency,
                             double *cyclelatency, 
                             double *tilelatency,
                             double *kernellatency) {
  int i;
  wkf_tasktile_t tile;
  wkf_timerhandle timer;
  int loopcount;

  timer=wkf_timer_create();

  // execute just a CUDA kernel launch and measure latency on whatever
  // GPU we get.
  loopcount = 15000;
  dim3 VGsz(1,1,0);
  dim3 VBsz(8,8,1);
  wkf_timer_start(timer);
  for (i=0; i<loopcount; i++) {
    voidkernel<<<VGsz, VBsz, 0>>>();
  }
  // wait for GPU kernels to complete
  cudaDeviceSynchronize();
  wkf_timer_stop(timer);
  *kernlaunchlatency = wkf_timer_time(timer) / ((double) loopcount);

  // execute just a raw barrier sync and measure latency
  loopcount = 15000;
  wkf_timer_start(timer);
  for (i=0; i<loopcount; i++) {
    wkf_threadpool_wait(devpool);
  }
  wkf_timer_stop(timer);
  *barlatency = wkf_timer_time(timer) / ((double) loopcount);

  // time wake-up, launch, and sleep/join of device pool doing a no-op
  loopcount = 5000;
  wkf_timer_start(timer);
  for (i=0; i<loopcount; i++) {
    tile.start=0;
    tile.end=0;
    wkf_threadpool_sched_dynamic(devpool, &tile);
    wkf_threadpool_launch(devpool, vmddevpoollatencythread, NULL, 1);
  }
  wkf_timer_stop(timer);
  *cyclelatency = wkf_timer_time(timer) / ((double) loopcount);

  // time wake-up, launch, and sleep/join of device pool eating tiles
  loopcount = 5000;
  wkf_timer_start(timer);
  for (i=0; i<loopcount; i++) {
    tile.start=0;
    tile.end=tilesize;
    wkf_threadpool_sched_dynamic(devpool, &tile);
    wkf_threadpool_launch(devpool, vmddevpooltilelatencythread, NULL, 1);
  }
  wkf_timer_stop(timer);
  *tilelatency = wkf_timer_time(timer) / ((double) loopcount);

  // time wake-up, launch, and sleep/join of device pool eating tiles
  loopcount = 2000;
  wkf_timer_start(timer);
  for (i=0; i<loopcount; i++) {
    tile.start=0;
    tile.end=tilesize;
    wkf_threadpool_sched_dynamic(devpool, &tile);
    wkf_threadpool_launch(devpool, vmddevpoolcudatilelatencythread, NULL, 1);
  }
  wkf_timer_stop(timer);
  *kernellatency = wkf_timer_time(timer) / ((double) loopcount);

  wkf_timer_destroy(timer);

#if 1
  vmd_cuda_measure_latencies(devpool);
#endif

  return 0;
}


//
// Benchmark CUDA kernel launch and memory copy latencies in isolation
//
typedef struct {
  int deviceid;
  int testloops;
  double kernlatency;
  double bcopylatency;
  double kbseqlatency;
} latthrparms;

static void * vmddevpoolcudalatencythread(void *voidparms) {
  int threadid=-1;
  latthrparms *parms=NULL;

  wkf_threadpool_worker_getid(voidparms, &threadid, NULL);
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);
  if (parms->deviceid == threadid) { 
    wkf_timerhandle timer;
    timer=wkf_timer_create();
    printf("Thread/device %d running...\n", threadid);
    cudaStream_t devstream;
    cudaStreamCreate(&devstream);

    char *hostbuf = (char *) calloc(1, 65536 * sizeof(char));
    char  *gpubuf = NULL;
    cudaMalloc((void**)&gpubuf, 65536 * sizeof(char));

    dim3 Gsz(1,1,0);
    dim3 Bsz(8,8,1);

    // measure back-to-back NULL kernel launches
    wkf_timer_start(timer);
    int i;
    for (i=0; i<parms->testloops; i++) {
      // launch a no-op CUDA kernel
      nopkernel<<<Gsz, Bsz, 0, devstream>>>(NULL);
    }
    // wait for all GPU kernels to complete
    cudaStreamSynchronize(devstream);
    wkf_timer_stop(timer);
    parms->kernlatency =  1000000 * wkf_timer_time(timer) / ((double) parms->testloops);

    // measure back-to-back round-trip 1-byte memcpy latencies
    wkf_timer_start(timer);
    for (i=0; i<parms->testloops; i++) {
      cudaMemcpyAsync(gpubuf, hostbuf, 1, cudaMemcpyHostToDevice, devstream);
      cudaMemcpyAsync(hostbuf, gpubuf, 1, cudaMemcpyDeviceToHost, devstream);
    }
    // wait for all GPU kernels to complete
    cudaStreamSynchronize(devstream);
    wkf_timer_stop(timer);
    parms->kernlatency =  1000000 * wkf_timer_time(timer) / ((double) parms->testloops);

    printf("NULL kernel launch latency (usec): %.2f\n", parms->kernlatency);

    cudaStreamDestroy(devstream);
    cudaFree(gpubuf);
    free(hostbuf);
    wkf_timer_destroy(timer);
  }

  return NULL;
}


int vmd_cuda_measure_latencies(wkf_threadpool_t *devpool) {
  latthrparms thrparms;
  int workers = wkf_threadpool_get_workercount(devpool);
  int i;
printf("vmd_cuda_measure_latencies()...\n");
  for (i=0; i<workers; i++) {
    memset(&thrparms, 0, sizeof(thrparms));
    thrparms.deviceid = i;
    thrparms.testloops = 2500;
    wkf_threadpool_launch(devpool, vmddevpoolcudalatencythread, &thrparms, 1);
  }

  return 0;
}




