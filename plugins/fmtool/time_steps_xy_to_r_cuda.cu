#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "time_steps_xy_to_r_cuda.h"
void printArrayF(const char *str, const float *d, const int x, const int y);

// required GPU array size alignment in bytes, 16 elements is ideal
#define BLOCKSIZE     16
#define BLOCKSIZEP2   (BLOCKSIZE+2)
#define ROWSIZE       19
#define BLOCKARRAY    (ROWSIZE*BLOCKSIZEP2)
#define GPU_ALIGNMENT BLOCKSIZE
#define GPU_ALIGNMASK (GPU_ALIGNMENT - 1)

#if 1
#define CUERR { cudaError_t err; \
  cudaThreadSynchronize(); \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  printf("aborting...\n"); \
  return -1; }}
#else
#define CUERR
#endif

typedef struct {
  dim3 sz;
  dim3 gpusz;
  dim3 Gsz;
  dim3 Bsz;
  float *pold;
  float *pnew;
  float *hI0kh;
} cudadevdata;


// -----------------------------------------------------------------
// Diffusion and bleaching.
//////////////////////////////////////////////////

// testing kernel
__global__ static void null_grid_cudakernel(const float dt,
                      const float rmin,
                      const int kiNrPadded,
                      const int Nr, const int Nz,
                      const float *hI0kh, const float dr,
                      const float *p, float *pnew, const float D,
                      const float odr_o2, const float odr2, const float odz2,
                      const float t2odrz) {
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int k = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int n = k*kiNrPadded + i;
  pnew[n] = 0;
}


// ----------------------------------------------------------------
// real kernel: experimental shared memory version
// Design: 
//   This kernel attempts to reduce the number of global memory reads
//   by storing global data into shared memory, and referencing only the
//   shared memory during subsequent computations.
//   This kernel uses 11 registers, and achieves 66% occupancy.
//
__global__ static void calc_grid_cudakernel_shared(const float dt,
                      const float rmin,
                      const int kiNrPadded,
                      const int Nr, const int Nz,
                      const float *hI0kh, const float dr, 
                      const float *p, float *pnew, const float D,
                      const float odr_o2, const float odr2, const float odz2,
                      const float t2odrz) {
  __shared__ float psh[BLOCKARRAY];
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int k = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int n = k*kiNrPadded + i;

  // read in values from global memory only once
  // grid data is read into shared memory,
  // eliminating redundant loads from sibling threads 

  // first global read
  float hI0khn = hI0kh[n];

  // second global read
  float Pn = p[n];

  int nsh = __umul24(ROWSIZE, (threadIdx.y+1)) + threadIdx.x + 1;

  int edgeidx = -1;
  int readidx = -1;
  if (threadIdx.y == 0) {
    edgeidx = nsh - 1;
    readidx = n + ((i > 0) ? -1 : 1);
  }
  if (threadIdx.y == 4) {
    edgeidx = nsh + 1;
    readidx = n + ((i <(Nr-1)) ? 1 : -1);
  }
  if (threadIdx.y == 8) {
    edgeidx = nsh - ROWSIZE;
    readidx = n + ((k > 0) ?  -kiNrPadded : kiNrPadded);
  }
  if (threadIdx.y == 12) {
    edgeidx = nsh + ROWSIZE;
    readidx = n + ((k < (Nz-1)) ? kiNrPadded : -kiNrPadded);
  }

  // store Pn value into shared memory
  psh[nsh] = Pn;

  // third global read
  if (readidx > 0)
    psh[edgeidx] = p[readidx];

  // wait until shared memory is in a consistent state
  __syncthreads();

  float Pnm1 = psh[nsh - 1];
  float Pnp1 = psh[nsh + 1];
  float Pnmr = psh[nsh - ROWSIZE];
  float Pnpr = psh[nsh + ROWSIZE];

  float r = rmin + (i)*dr;
  float tmpr1;
  if (!i)
    tmpr1 = 0.0f;
  else 
    tmpr1 = (Pnp1 - Pnm1) / r;

  // Get the function p for the new step.
  float tmpr = Pnp1 + Pnm1;
  float tmpz = Pnpr + Pnmr;

  float tmp  = odr2 * tmpr + odz2 * tmpz - t2odrz * Pn + odr_o2 * tmpr1;
  float result = Pn + dt * (D * tmp - hI0khn * Pn);
  pnew[n] = (result < 0.0f) ? 0.0f : result;
}


// ----------------------------------------------------------------
// real kernel: latest stable version
// Design: 
//   This kernel achieves performance solely by running
//   a huge number of threads concurrently, hoping to hide global
//   memory latency by switching to other runnable threads where possible.
//   This kernel uses 10 registers, and achieves 100% occupancy.
//
__global__ static void calc_grid_cudakernel(const float dt,
                      const float rmin,
                      const int kiNrPadded,
                      const int Nr, const int Nz,
                      const float *hI0kh, const float dr, 
                      const float *p, float *pnew, const float D,
                      const float odr_o2, const float odr2, const float odz2,
                      const float t2odrz) {
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int k = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int n = k*kiNrPadded + i;

  //
  // read in values from global memory only once
  // XXX grid data should be read into shared memory,
  //     thereby eliminating redundant loads from sibling threads 
  // Global memory references: 
  //   input: 5 for stencil values from p[], 1 for hI0kh
  //   output: 1 for pnew
  // 18 FLOPS
  // 5 conditional assignments
  //
  float Pn   = p[n    ];
  float hI0khn = hI0kh[n];

  const int np1 = n+1;
  const int nm1 = n-1;
  float Pnm1 = (i>0)      ? p[nm1] : p[np1];
  float Pnp1 = (i<(Nr-1)) ? p[np1] : p[nm1];
  float Pnmr = (k>0)      ? p[n - kiNrPadded] : p[n + kiNrPadded];
  float Pnpr = (k<(Nz-1)) ? p[n + kiNrPadded] : p[n - kiNrPadded];

  float r = rmin + (i)*dr;
  float tmpr1;
  if (!i)
    tmpr1 = 0.0f;
  else 
    tmpr1 = (Pnp1 - Pnm1) / r;

  // Get the function p for the new step.
  float tmpr = Pnp1 + Pnm1;
  float tmpz = Pnpr + Pnmr;
  float tmp  = odr2 * tmpr + odz2 * tmpz - t2odrz * Pn + odr_o2 * tmpr1;
  float result = Pn + dt * (D * tmp - hI0khn * Pn);
  pnew[n] = (result < 0.0f) ? 0.0f : result;
}



// ----------------------------------------------------------------
// real kernel: this is the version published in the 4Pi paper.
// Design: 
//   This kernel achieves performance solely by running
//   a huge number of threads concurrently, hoping to hide global
//   memory latency by switching to other runnable threads where possible.
//   This kernel uses 9 registers, and achieves 100% occupancy.
//
__global__ static void calc_grid_cudakernel_4pi(const float dt,
                      const float rmin,
                      const int kiNrPadded,
                      const int Nr, const int Nz,
                      const float *hI0kh, const float dr, 
                      const float *p, float *pnew, const float D,
                      const float odr_o2, const float odr2, const float odz2,
                      const float t2odrz) {
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int k = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

  int n = k*kiNrPadded + i;
  const int np1 = n+1;
  const int nm1 = n-1;

  // read in values from global memory only once
  // XXX grid data should be read into shared memory,
  // eliminating redundant loads from sibling threads 
  float Pn   = p[n    ];

  float Pnm1;
  if (i>0)
  {
    Pnm1 = p[nm1];
  }
  else 
  {
    Pnm1 = p[np1];
  }
 
  float Pnp1;
  if (i<(Nr-1))
  {
    Pnp1 = p[np1];
  }
  else 
  {
    Pnp1 = p[nm1];
  }
  float tmpr = Pnp1 + Pnm1;

  float Pnmr;
  if (k>0) 
  {
    Pnmr = p[n - kiNrPadded];
  }
  else 
  {
    Pnmr = p[n + kiNrPadded];
  }

  float Pnpr;
  if (k<(Nz-1)) 
  {
    Pnpr = p[n + kiNrPadded];
  }
  else
  {
    Pnpr = p[n - kiNrPadded];
  }

  float tmpz  = Pnpr + Pnmr;

  float r = rmin + (i)*dr;
  float tmpr1;
  if (!i)
    tmpr1 = 0.0f;
  else 
    tmpr1 = (Pnp1 - Pnm1) / r;

  // Get the function p for the new step.
  float tmp = odr2 * tmpr + odz2 * tmpz - t2odrz * Pn + odr_o2 * tmpr1;
  float result = Pn + dt * (D * tmp - hI0kh[n] * Pn);
  pnew[n] = (result < 0.0f) ? 0.0f : result;
}




// -----------------------------------------------------------------
// copy old (unpadded) arrays to GPU
int copy_old_to_cuda_padded(float *dest, dim3 padsz, const float *src, dim3 sz) {
  int y;
  for (y=0; y<(sz.y-2); y++) {
    cudaMemcpy(dest + 1 + ((y+1) * padsz.x), 
               src + (y * (sz.x-2)), 
               (sz.x-2) * sizeof(float), 
               cudaMemcpyHostToDevice);
    CUERR // check and clear any existing errors
  }
  return 0;
}

// -----------------------------------------------------------------
// copy new boundary-padded arrays to GPU
int copy_to_cuda_padded(float *dest, dim3 padsz, const float *src, dim3 sz) {
  int y;
  for (y=0; y<sz.y; y++) {
    cudaMemcpy(dest + (y * padsz.x), 
               src + (y * sz.x), 
               sz.x * sizeof(float), 
               cudaMemcpyHostToDevice);
    CUERR // check and clear any existing errors
  }
  return 0;
}

// -----------------------------------------------------------------
int copy_from_cuda_padded(float *dest, dim3 sz, const float *src, dim3 padsz) {
  int y;
  for (y=0; y<sz.y; y++) {
    cudaMemcpy(dest + (y * sz.x), 
               src + (y * padsz.x), 
               sz.x * sizeof(float), 
               cudaMemcpyDeviceToHost);
    CUERR // check and clear any existing errors
  }
  return 0;
}

// -----------------------------------------------------------------
int init_cuda_devdata(cudadevdata *dev, int Nr, int Nz) {
  dev->pold  = NULL;
  dev->pnew  = NULL;
  dev->hI0kh = NULL;

  // Original array size
  dev->sz.x = Nr;
  dev->sz.y = Nz;
  dev->sz.z = 1;

  // Padded array size
  dev->gpusz.x = (Nr     + GPU_ALIGNMASK) & ~(GPU_ALIGNMASK);
  dev->gpusz.y = (Nz     + GPU_ALIGNMASK) & ~(GPU_ALIGNMASK);
  dev->gpusz.z = 1;

  dev->Bsz.x = 16;
  dev->Bsz.y = 16;
  dev->Bsz.z = 1;
 
  dev->Gsz.x = dev->gpusz.x / dev->Bsz.x;
  dev->Gsz.y = dev->gpusz.y / dev->Bsz.y;
  dev->Gsz.z = dev->gpusz.z / dev->Bsz.z;

  printf("Padded GPU array size: %dx%dx%d\n", 
         dev->gpusz.x, dev->gpusz.y, dev->gpusz.z);
  printf("CUDA Gsz: %dx%dx%d  Bsz: %dx%dx%d\n", 
         dev->Gsz.x, dev->Gsz.y, dev->Gsz.z,
         dev->Bsz.x, dev->Bsz.y, dev->Bsz.z);

  int gpumemsz = sizeof(float) * dev->gpusz.x * dev->gpusz.y * dev->gpusz.z;

  printf("Grid GPU memory allocation size: %.1f MB each, %.1f MB total\n",
         (double) gpumemsz / (1024 * 1024), 
         ((double) gpumemsz / (1024 * 1024)) * 5);
                   

  // allocate and initialize the GPU input/output arrays
  cudaMalloc((void**)&dev->pold,  gpumemsz);
  CUERR // check and clear any existing errors
  cudaMalloc((void**)&dev->hI0kh, gpumemsz);
  CUERR // check and clear any existing errors
  cudaMalloc((void**)&dev->pnew, gpumemsz);
  CUERR // check and clear any existing errors

  return 0;
}

int free_cuda_devdata(cudadevdata *dev) {
  cudaFree(dev->pold);
  cudaFree(dev->hI0kh);
  cudaFree(dev->pnew);
  CUERR // check and clear any existing errors

  return 0;
}


// -----------------------------------------------------------------
int calc_grid_cuda(cudadevdata *dev,
                   const float dt,
                   const int Nr, const int Nz, const float rmin,
                   const float dr, const float dz,
                   const float D, 
                   const float odr2, const float odz2, 
                   const float odr_o2, const float t2odrz) {
#if 0
  // null kernel that just clears the result matrix, for measuring
  // basic kernel call overhead etc.
  null_grid_cudakernel<<<dev->Gsz, dev->Bsz, 0>>>(dt, rmin,
                                        dev->gpusz.x, Nr, Nz,
                                        dev->hI0kh,
                                        dr, dev->pold, dev->pnew,
                                        D, odr_o2, odr2, odz2, t2odrz);
#elif 0
  // shared memory version
  calc_grid_cudakernel_shared<<<dev->Gsz, dev->Bsz, 0>>>(dt, rmin, 
                                        dev->gpusz.x, Nr, Nz,
                                        dev->hI0kh, 
                                        dr, dev->pold, dev->pnew, 
                                        D, odr_o2, odr2, odz2, t2odrz);
#elif 1
  // latest stable version
  calc_grid_cudakernel<<<dev->Gsz, dev->Bsz, 0>>>(dt, rmin, 
                                        dev->gpusz.x, Nr, Nz,
                                        dev->hI0kh, 
                                        dr, dev->pold, dev->pnew, 
                                        D, odr_o2, odr2, odz2, t2odrz);
#else
  // kernel used in 4pi paper
  calc_grid_cudakernel_4pi<<<dev->Gsz, dev->Bsz, 0>>>(dt, rmin, 
                                        dev->gpusz.x, Nr, Nz,
                                        dev->hI0kh, 
                                        dr, dev->pold, dev->pnew, 
                                        D, odr_o2, odr2, odz2, t2odrz);
#endif

  CUERR // check and clear any existing errors

  return 0;
}


// --------------------------------------------------------------------  \\
//                                                                       \\
//                Single Precision Version                               \\
//                                                                       \\
// Obs and Obs1 - arrays of observables, the fluorescent signal (Obs),
// and its RMSD (Obs1). These 2 are the 1Darrays representing functions of time.
// Obs and Obs1 are being calculated within this function.
// M - number of time steps (constant).
// kiOutputFreq - frequency that the observables should be calculated
//                     (every X timesteps)
// dt - time step in seconds (constant).
// N - number of elements in 1D arrays that are functions of position in space.
// h - function of position, 1D array (constant).
// h_det - function of position, 1D array (constant).
// hI0kh - function of position, 1D array (constant).
// dr - grid step in the r-dimension, in micrometers (constant).
// dz - grid step in the z-dimension, in micrometers (constant).
// Nr - number of steps in the r-dimension (constant).
// Nz - number of steps in the z-dimension (constant).
// rmin - minimal value of r (constant).
// p0 - initial value for pnew
// OLD   p - distribution function, the function of r and z; 1D array.
// OLD   pnew - array for the values of p at the next time step.
//      currently, pnew comes in entirely set to 1.0  - kv
// D - diffusion coefficient (constant).

void time_steps_xy_to_r_cuda(float *Obs, float *Obs1, const int M, 
                        const int kiOutputFreq, const float dt, 
                        const int N, const float *h, const float *h_det, 
                        const float *hI0kh, const float dr, const float dz, 
                        const int Nr, const int Nz, const float rmin, 
                        const float p0, const float D)
{
// Counters for loops.
   int l = 0;
   int i = 0;
   int k = 0;
//   int n = 0;
//   int nold = 0;  // used to track location in h/h_det/hI0kh array 
//                  // that we need to grab


// Variable for the values of r.
   float r = 0.0f;

// Auxiliary constants.

// Auxiliary variables.
   const float f2piDrDz =  2.0 * PI * dr * dz;

   float *p = (float *) malloc((Nr)*(Nz) * sizeof(float));
   float *pnew = (float *) malloc((Nr)*(Nz) * sizeof(float));

//   printArrayF("hI0kh is ", hI0kh+1, Nr, Nz);
   // set to initial condition
   for (i=0; i<Nr*Nz;i++) {
      pnew[i]=p0;
//      pnew[i]=i;
//      p[i]=i;
   }

   float *pTemp;

   // define temp array that holds h * h_det for each grid position
   // saves us having to calculate in the inner loop each time
   float *rgTmp1 = (float *) malloc( N*sizeof(float));
   for (i=0; i < N; i++)
   {
      rgTmp1[i] = h[i+1] * h_det[i+1];
   }

#if defined(CUDA)
   // allocate CUDA memory blocks
   cudadevdata *dev = (cudadevdata *) malloc(sizeof(cudadevdata));
   init_cuda_devdata(dev, Nr, Nz);

   // Copy the Host input data to the GPU..
   copy_to_cuda_padded(dev->pnew,  dev->gpusz, pnew, dev->sz);

   // copy non-boundary-padded arrays to GPU
   copy_to_cuda_padded(dev->hI0kh, dev->gpusz, hI0kh+1, dev->sz);
#endif

   float odr2 = 1.0 / (dr * dr);   
   float odz2 = 1.0 / (dz * dz);   
   float odr_o2 = 0.5 / dr;
   float t2odrz = 2.0 * (odr2 + odz2); 


// M time steps
   for (l = 0; l < M; l++)
   {

// Update the arrays at the new time step.
// Array p becomes the same as pnew has become at the previous time step;
// p will be used as an input at this time step, to calculate pnew.
#if defined(CUDA)
      pTemp = dev->pold;
      dev->pold = dev->pnew;
      dev->pnew = pTemp;
#else
      pTemp = p;
      p = pnew;
      pnew = pTemp;
#endif
//      fprintf(stderr,"\ntimestep %d", l); printArray("", p, iNrp2, iNzp2);

      // Diffusion and bleaching.
      //////////////////////////////////////////////////

//      fprintf(stderr, "odr2:%f, odz2:%f, odr_o2:%f, rmin:%f, dr:%f\n", odr2, odz2, odr_o2, rmin, dr);

#if defined(CUDA)
      if (calc_grid_cuda(dev, dt, Nr, Nz, rmin, dr, dz, D, odr2, odz2, odr_o2,
      t2odrz)) {
        printf("Exiting...\n");
        exit(-1); // ugly, but the code has no error trapping presently
      }
#else
      calc_grid_cudacheck(dt, Nr, Nz, rmin,h, h_det, hI0kh, dr, dz, p, pnew, D);
#endif

      if (!(l%kiOutputFreq))
      {
         int iLDivFreq = l/kiOutputFreq;

#if defined(CUDA)
         // Copy the GPU output data back to the host and use it to 
         // calculate observables
//         printf("Copying GPU data back for observable calculations...\n");

         // copy _both_ pold and pnew since both may have been updated,
         // and the observables are presently calculated from p...
         copy_from_cuda_padded(p, dev->sz, dev->pold, dev->gpusz);
//         copy_from_cuda_padded(pnew, dev->sz, dev->pnew, dev->gpusz);
#endif
//         printArrayF("TIMESTEP:P", p, Nr, Nz);
//         printArrayF("timestep:pnew", pnew, Nr, Nz);

         // initial values for observables.  Accumulated over grid
         float obsl  = 0.0f;
         float obsl1 = 0.0f;
         int iLoc=0;
         for (k = 0 ; k < Nz; k++)
         {
            r = rmin /*+ dr */;
            for (i = 0; i < Nr; i++)
            {

               // Accumulate observables
               // Observable; Obs(t) = \int dr h(r) h_det(r) p(r,t).
               // Noise; Obs1(t) = \int dr h(r)^2 h_det(r)^2 p(r,t) - 
               //                         (\int dr h(r) h_det(r) p(r,t))^2.
               float tmp2 = r * rgTmp1[iLoc] * p[iLoc];
               obsl  += tmp2;
               obsl1 += tmp2 * rgTmp1[iLoc];

//            fprintf(stderr,"k:%d,i:%d,n:%d,nold:%d,r:%f,rgTmp:%f,tmp2:%f,obsl:%f,obsl1:%f,p[n]:%f\n",k,i,n,nold,r,rgTmp1[nold],tmp2,obsl,obsl1,p[n]);

               r += dr;
               iLoc++;
            }
         }

         /// store observables 
         Obs[iLDivFreq] = obsl;
         Obs1[iLDivFreq] = obsl1;
         Obs[iLDivFreq]  *= f2piDrDz;
         Obs1[iLDivFreq] = Obs1[iLDivFreq] * f2piDrDz - 
                                    Obs[iLDivFreq] * Obs[iLDivFreq];
//         fprintf(stderr, "end of timestep.  Obs[%d]=%16.12f, Obs1=%16.12f\n", iLDivFreq, Obs[iLDivFreq],Obs1[iLDivFreq]);


        fprintf(stderr,"Time step %d  of %d\n", l, M);
//        printArray("timestep", pnew, iNrp2, iNzp2);
      }
   }

   free_cuda_devdata(dev);
   free(dev);

   free (rgTmp1);
   free (p);
   free (pnew);
}



