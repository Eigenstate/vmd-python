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
 *      $RCSfile: CUDAMeasureRDF.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA accelerated RDF calculation
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "Inform.h"
#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "CUDAKernels.h" 

//
// the below parameters are important to tuning the performance of the code
//
#if 0
// original (slow?) default values for all GPU types
// no hardware support for atomic operations, so the shared memory 
// tagging scheme is used.
#define NBLOCK 128        // size of an RDF data block 
#define MAXBIN 64         // maximum number of bins in a histogram
#elif 1
// optimal values for compute 1.0 (G80/G9x) devices 
// no hardware support for atomic operations, so the shared memory 
// tagging scheme is used.
#define NBLOCK 32         // size of an RDF data block 
#define MAXBIN 1024       // maximum number of bins in a histogram
#elif 1
// optimal values for compute 1.3 (GT200) devices 
// uses atomic operations to increment histogram bins
#define NBLOCK 320        // size of an RDF data block 
#define MAXBIN 3072       // maximum number of bins in a histogram
#define __USEATOMIC  1    // enable atomic increment operations
#elif 1
// optimal values for compute 2.0 (Fermi) devices 
// uses atomic operations to increment histogram bins
// uses Fermi's 3x larger shared memory (48KB) to reduce the number 
// of passes required for computation of very large histograms
#define NBLOCK 896        // size of an RDF data block 
#define MAXBIN 8192       // maximum number of bins in a histogram
#define __USEATOMIC  1    // enable atomic increment operations
#endif

#define NCUDABLOCKS 256   // the number of blocks to divide the shared memory 
                          // dimension into.  Performance increases up to ~8 
                          // and then levels off


#define NBLOCKHIST 64     // the number of threads used in the final histogram
                          // kernel.  The summation of the histograms isn't 
                          // a bottleneck so this isn't a critical choice

#define NCONSTBLOCK 5440  // the number of atoms in constant memory.
                          // Can be 4096 (or larger) up to capacity limits.

#define THREADSPERWARP 32 // this is correct for G80/GT200/Fermi GPUs
#define WARP_LOG_SIZE 5   // this is also correct for G80/GT200/Fermi GPUs

#define BIN_OVERFLOW_LIMIT 2147483648 // maximum value a bin can store before
                                      // we need to accumulate and start over


// This routine prepares data for the calculation of the histogram.
// It zeros out instances of the histograms in global memory.
__global__ void init_hist(unsigned int* llhistg, // histograms
                          int maxbin) // number of bins per histogram
{

#ifndef __USEATOMIC
  int io; // index of atoms in selection
#endif

  int iblock; // index of data blocks 

   unsigned int *llhist; // this pointer will point to the begining of 
  // each thread's instance of the histogram

  int nofblocks; // nofblocks is the number of full data blocks.  
  int nleftover; // nleftover is the dimension of the blocks left over 
  // after all full data blocks
  int maxloop; // the loop condition maximum

  // initialize llhists to point to an instance of the histogram for each
  // warp.  Zero out the integer (llhist) and floating point (histdev) copies
  // of the histogram in global memory
#ifdef __USEATOMIC
  llhist=llhistg+blockIdx.x*maxbin+threadIdx.x;
  nofblocks=maxbin/NBLOCK;
  nleftover=maxbin-nofblocks*NBLOCK;
  maxloop=nofblocks*NBLOCK;

  for (iblock=0; iblock < maxloop; iblock+=NBLOCK) {
    llhist[iblock]=0UL;
  }
  // take care of leftovers
  if (threadIdx.x < nleftover) {
    llhist[iblock]=0UL;
  }
  
#else
  llhist=llhistg+(((blockIdx.x)*NBLOCK)/THREADSPERWARP)*maxbin+threadIdx.x;
  nofblocks=maxbin/NBLOCK;
  nleftover=maxbin-nofblocks*NBLOCK;
  maxloop=nofblocks*NBLOCK;
  int maxloop2=(NBLOCK / THREADSPERWARP)*maxbin;

  for (io=0; io < maxloop2; io+=maxbin) {
    for (iblock=0; iblock < maxloop; iblock+=NBLOCK) {
      llhist[io + iblock]=0UL;
    }
    // take care of leftovers
    if (threadIdx.x < nleftover) {
      llhist[io + iblock]=0UL;
    }
  }
#endif

  return;
}



/* This routine prepares data for the calculation of the histogram.
 * It zeros out the floating point histogram in global memory.
 */
__global__ void init_hist_f(float* histdev) {
  histdev[threadIdx.x]=0.0f;
  return;
}



/* This routine prepares data for the calculation of the histogram.
 * It recenters the atom to a single periodic unit cell.  
 * It also chooses positions for the padding atoms such that 
 * they don't contribute to the histogram
 */
__global__ void reimage_xyz(float* xyz,     // XYZ data in global memory.  
                            int natomsi,    // number of atoms
                            int natomsipad, // # atoms including padding atoms
                            float3 celld,   // the cell size of each frame
                            float rmax)     // dist. used to space padding atoms
{
  int iblock; // index of data blocks 

  __shared__ float xyzi[3*NBLOCK]; // this array hold xyz data for the current 
  // data block.  

  int nofblocks; // ibin is an index associated with histogram 
  // bins.  nofblocks is the number of full data blocks.  nleftover is the 
  // dimension of the blocks left over after all full data blocks

  float *pxi; // these pointers point to the portion of xyz currently
  // being processed

  int threetimesid; // three times threadIdx.x

  float rtmp; // a temporary distance to be used in creating padding atoms

  int ifinalblock; // this is the index of the final block

  int loopmax; // maximum for the loop counter

  // initialize nofblocks, pxi, and threetimesid
  nofblocks=((natomsipad/NBLOCK)+NCUDABLOCKS-blockIdx.x-1)/NCUDABLOCKS;
  loopmax=nofblocks*3*NCUDABLOCKS*NBLOCK;
  ifinalblock=(natomsipad / NBLOCK - 1) % NCUDABLOCKS;
  pxi=xyz+blockIdx.x*3*NBLOCK+threadIdx.x;
  threetimesid=3*threadIdx.x;

  // shift all atoms into the same unit cell centered at the origin
  for (iblock=0; iblock<loopmax; iblock+=3*NCUDABLOCKS*NBLOCK) {
    __syncthreads();
    //these reads from global memory should be coallesced
    xyzi[threadIdx.x         ]=pxi[iblock         ];
    xyzi[threadIdx.x+  NBLOCK]=pxi[iblock+  NBLOCK];
    xyzi[threadIdx.x+2*NBLOCK]=pxi[iblock+2*NBLOCK];
    __syncthreads();

    // Shift the xyz coordinates so that 0 <= xyz < celld.
    // The ?: line is necesary because of the less than convenient behavior
    // of fmod for negative xyz values
    xyzi[threetimesid] = fmodf(xyzi[threetimesid  ], celld.x);
    if (xyzi[threetimesid  ] < 0.0f) {
      xyzi[threetimesid  ] += celld.x;
    }
    xyzi[threetimesid+1]=fmodf(xyzi[threetimesid+1], celld.y);
    if (xyzi[threetimesid+1] < 0.0f) {
      xyzi[threetimesid+1] += celld.y;
    }
    xyzi[threetimesid+2]=fmodf(xyzi[threetimesid+2], celld.z);
    if (xyzi[threetimesid+2] < 0.0f) {
      xyzi[threetimesid+2] += celld.z;
    }

    // if this is the final block then we pad
    if (iblock==loopmax-3*NCUDABLOCKS*NBLOCK && blockIdx.x==ifinalblock) {
      // pad the xyz coordinates with atoms which are spaced out such that they
      // cannot contribute to the histogram.  Note that these atoms are NOT 
      // reimaged
      if ((blockDim.x-threadIdx.x) <= (natomsipad - natomsi)) {
        rtmp = -((float)(threadIdx.x+1))*rmax;
        xyzi[threetimesid  ] = rtmp;
        xyzi[threetimesid+1] = rtmp;
        xyzi[threetimesid+2] = rtmp;
      }
    }

    __syncthreads();

    // store the xyz values back to global memory.  these stores are coallesced
    pxi[iblock         ]=xyzi[threadIdx.x         ];
    pxi[iblock+  NBLOCK]=xyzi[threadIdx.x+  NBLOCK];
    pxi[iblock+2*NBLOCK]=xyzi[threadIdx.x+2*NBLOCK];

    // increment pxi to the next block of global memory to be processed
    //pxi=pxi+3*NCUDABLOCKS*NBLOCK;
  }
}


// shift the "phantom" atoms so they don't interfere in non-periodic
// calculations.
__global__ void phantom_xyz(float* xyz,     // XYZ data in global memory.
                            int natomsi,    // number of atoms
                            int natomsipad, // # atoms including padding atoms
                            float minxyz,
                            float rmax)     // dist. used to space padding atoms
{
  int iblock; // index of data blocks

  __shared__ float xyzi[3*NBLOCK]; // this array hold xyz data for the current
  // data block.

  int nofblocks; // ibin is an index associated with histogram
  // bins.  nofblocks is the number of full data blocks.  nleftover is the
  // dimension of the blocks left over after all full data blocks

  float *pxi; // these pointers point to the portion of xyz currently
  // being processed

  int threetimesid; // three times threadIdx.x

  float rtmp; // a temporary distance to be used in creating padding atoms

  int ifinalblock; // this is the index of the final block

  int loopmax; // maximum for the loop counter

  // initialize nofblocks, pxi, and threetimesid
  nofblocks=((natomsipad/NBLOCK)+NCUDABLOCKS-blockIdx.x-1)/NCUDABLOCKS;
  loopmax=nofblocks*3*NCUDABLOCKS*NBLOCK;
  ifinalblock=(natomsipad / NBLOCK - 1) % NCUDABLOCKS;
  pxi=xyz+blockIdx.x*3*NBLOCK+threadIdx.x;
  threetimesid=3*threadIdx.x;

  // shift all atoms into the same unit cell centered at the origin
  for (iblock=0; iblock<loopmax; iblock+=3*NCUDABLOCKS*NBLOCK) {
    __syncthreads();
    //these reads from global memory should be coallesced
    xyzi[threadIdx.x         ]=pxi[iblock         ];
    xyzi[threadIdx.x+  NBLOCK]=pxi[iblock+  NBLOCK];
    xyzi[threadIdx.x+2*NBLOCK]=pxi[iblock+2*NBLOCK];
    __syncthreads();

    // if this is the final block then we pad
    if (iblock==loopmax-3*NCUDABLOCKS*NBLOCK && blockIdx.x==ifinalblock) {
      // pad the xyz coordinates with atoms which are spaced out such that they
      // cannot contribute to the histogram.  Note that these atoms are NOT
      // reimaged
      if ((blockDim.x-threadIdx.x) <= (natomsipad - natomsi)) {
        rtmp = -((float)(threadIdx.x+1))*rmax-minxyz;
        xyzi[threetimesid  ] = rtmp;
        xyzi[threetimesid+1] = rtmp;
        xyzi[threetimesid+2] = rtmp;
      }
    }

    __syncthreads();

    // store the xyz values back to global memory.  these stores are coallesced
    pxi[iblock         ]=xyzi[threadIdx.x         ];
    pxi[iblock+  NBLOCK]=xyzi[threadIdx.x+  NBLOCK];
    pxi[iblock+2*NBLOCK]=xyzi[threadIdx.x+2*NBLOCK];

    // increment pxi to the next block of global memory to be processed
    //pxi=pxi+3*NCUDABLOCKS*NBLOCK;
  }
}


/* This kernel calculates a radial distribution function between 
 * two selections one selection is stored in its entirety in global memory.
 * The other is stored partially in constant memory.  This routine is 
 * called to calculate the rdf between all of selection 1 and the portion
 * of selection 2 in constant memory.  
 * Each element of selection 2 is associated with it's own thread.  To minimize
 * accesses to global memory selection 1 is divided into chunks of NBLOCK 
 * elements.  These chunks are loaded into shared memory simultaneously, 
 * and then then all possible pairs of these atoms with those in 
 * constant memory are calculated.
 */
__constant__ static float xyzj[3*NCONSTBLOCK]; // this array stores the
// coordinates of selection two in constant memory

// this routine is called from the host to move coordinate data to constant mem
void copycoordstoconstbuff(int natoms, const float* xyzh) {
  cudaMemcpyToSymbol(xyzj, xyzh, 3*natoms*sizeof(float));
}


// This subroutine is based on a similar routine in the CUDA SDK histogram256 
// example.  It adds a count to the histogram in such a way that there are no 
// collisions.  If several routines need to update the same element a tag 
// applied to to the highest bits of that element is used to determine which 
// elements have already updated it.  For devices with compute capability 1.2 
// or higher this is not necessary as atomicAdds can be used.

#ifndef __USEATOMIC
__device__ void addData(volatile unsigned int *s_WarpHist, // the first element
                        // of the histogram to be operated on
                        unsigned int data,      // the bin to add a count to
                        unsigned int threadTag) // tag of the current thread
{
  unsigned int count; // the count in the bin being operated on

  do {  
    count = s_WarpHist[data] & 0x07FFFFFFUL;
    count = threadTag | (count + 1);
    s_WarpHist[data] = count;
  } while(s_WarpHist[data] != count);
}
#endif


// This is the routine that does all the real work.  It takes as input the
// various atomic coordinates and produces one rdf per warp, as described 
// above.  These rdfs will be summed later by calculate_hist.
__global__ void calculate_rdf(int usepbc,     // periodic or non-periodic calc.
                              float* xyz,     // atom XYZ coords, gmem
                              int natomsi,    // # of atoms in sel1, gmem
                              int natomsj,    // # of atoms in sel2, cmem
                              float3 celld,   // cell size of each frame
                              unsigned int* llhistg, // histograms, gmem
                              int nbins,      // # of bins in each histogram
                              float rmin,     // minimum distance for histogram
                              float delr_inv) // recip. width of histogram bins
{
  int io; // indices of the atoms in selection 2
  int iblock; //the block of selection 1 currently being processed.  

#ifdef __USEATOMIC
  unsigned int *llhists1; // a pointer to the beginning of llhists
                          // which is operated on by the warp

  __shared__ unsigned int llhists[MAXBIN];  // array holds 1 histogram per warp.
#else
  volatile unsigned int *llhists1; // a pointer to the beginning of llhists
                                   // which is operated on by the warp

  volatile __shared__ unsigned int llhists[(NBLOCK*MAXBIN)/THREADSPERWARP];  
  // this array will hold 1 histogram per warp.
#endif

  __shared__ float xyzi[3*NBLOCK]; // coords for the portion of sel1 
                                   // being processed in shared memory

  float rxij, rxij2, rij; // registers holding intermediate values in
                          // calculating the distance between two atoms

  int ibin,nofblocksi;    // registers counters and upper limit in some loops
  int nleftover;          // # bins leftover after full blocks are processed
  float *pxi;             // pointer to the portion of sel1 being processed
  unsigned int joffset;           // the current offset in constant memory
  unsigned int loopmax, loopmax2; // the limit for several loops
  float rmin2=.0001/delr_inv;

#ifndef __USEATOMIC
  // this thread tag is needed for the histograming method implemented in
  // addData above.
  const unsigned int threadTag = ((unsigned int)
                                 ( threadIdx.x & (THREADSPERWARP - 1)))
                                 << (32 - WARP_LOG_SIZE);
#endif

  // initialize llhists1.  If atomics are used, then each block has its own 
  // copy of the histogram in shared memory, which will be stored to it's own 
  // place in global memory.  If atomics aren't used then we need 1 histogram 
  // per warp. The number of blocks is also set approprately so that the
  // histograms may be zeroed out.
#ifdef __USEATOMIC
  llhists1=llhists;
  nofblocksi=nbins/NBLOCK;
  nleftover=nbins-nofblocksi*NBLOCK;
#else
  llhists1=llhists+(threadIdx.x/THREADSPERWARP)*MAXBIN;
  nofblocksi=nbins/THREADSPERWARP;
  nleftover=nbins-nofblocksi*THREADSPERWARP;
#endif

  // Here we also zero out the shared memory used in this routine.
#ifdef __USEATOMIC
  loopmax = nofblocksi*NBLOCK;
  for (io=0; io<loopmax; io+=NBLOCK) {
    llhists1[io + threadIdx.x]=0UL;
  } 
  if (threadIdx.x < nleftover) {
    llhists1[io + threadIdx.x]=0UL;
  } 
   
#else
  loopmax = nofblocksi*THREADSPERWARP;
  for (io=0; io<loopmax; io+=THREADSPERWARP) {
    llhists1[io + (threadIdx.x & (THREADSPERWARP - 1))]=0UL;
  }  
  if ((threadIdx.x & (THREADSPERWARP - 1)) < nleftover) {
    llhists1[io + (threadIdx.x & (THREADSPERWARP - 1))]=0UL;
  } 
#endif

  // initialize nofblocks and pxi
  nofblocksi=((natomsi/NBLOCK)+NCUDABLOCKS-blockIdx.x-1)/NCUDABLOCKS;
  pxi=xyz+blockIdx.x*3*NBLOCK+threadIdx.x;

  loopmax = 3 * natomsj;
  int idxt3 = 3 * threadIdx.x;
  int idxt3p1 = idxt3+1;
  int idxt3p2 = idxt3+2;

  loopmax2 = nofblocksi*3*NCUDABLOCKS*NBLOCK;

  // loop over all atoms in constant memory, using either 
  // periodic or non-periodic particle pair distance calculation
  if (usepbc) {
    for (iblock=0; iblock<loopmax2; iblock+=3*NCUDABLOCKS*NBLOCK) {
      __syncthreads();

      // these loads from global memory should be coallesced
      xyzi[threadIdx.x         ]=pxi[iblock         ];
      xyzi[threadIdx.x+  NBLOCK]=pxi[iblock+  NBLOCK];
      xyzi[threadIdx.x+2*NBLOCK]=pxi[iblock+2*NBLOCK];

      __syncthreads();

      for (joffset=0; joffset<loopmax; joffset+=3) {
        // calculate the distance between two atoms.  rxij and rxij2 are reused
        // to minimize the number of registers used
        rxij=fabsf(xyzi[idxt3  ] - xyzj[joffset  ]);
        rxij2=celld.x - rxij;
        rxij=fminf(rxij, rxij2);
        rij=rxij*rxij;
        rxij=fabsf(xyzi[idxt3p1] - xyzj[joffset+1]);
        rxij2=celld.y - rxij;
        rxij=fminf(rxij, rxij2);
        rij+=rxij*rxij;
        rxij=fabsf(xyzi[idxt3p2] - xyzj[joffset+2]);
        rxij2=celld.z - rxij;
        rxij=fminf(rxij, rxij2);
        rij=sqrtf(rij + rxij*rxij);

        // increment the appropriate bin, don't add duplicates to the zeroth bin
        ibin=__float2int_rd((rij-rmin)*delr_inv);
        if (ibin<nbins && ibin>=0 && rij>rmin2) {
#ifdef __USEATOMIC
          atomicAdd(llhists1+ibin, 1U);
#else
          addData(llhists1, ibin, threadTag);
#endif
        }
      } //joffset
    } //iblock
  } else { // non-periodic
    for (iblock=0; iblock<loopmax2; iblock+=3*NCUDABLOCKS*NBLOCK) {
      __syncthreads();

      // these loads from global memory should be coallesced
      xyzi[threadIdx.x         ]=pxi[iblock         ];
      xyzi[threadIdx.x+  NBLOCK]=pxi[iblock+  NBLOCK];
      xyzi[threadIdx.x+2*NBLOCK]=pxi[iblock+2*NBLOCK];

      __syncthreads();

      // loop over all atoms in constant memory
      for (joffset=0; joffset<loopmax; joffset+=3) {
        // calculate the distance between two atoms.  rxij and rxij2 are reused
        // to minimize the number of registers used
        rxij=xyzi[idxt3  ] - xyzj[joffset  ];
        rij=rxij*rxij;
        rxij=xyzi[idxt3p1] - xyzj[joffset+1];
        rij+=rxij*rxij;
        rxij=xyzi[idxt3p2] - xyzj[joffset+2];
        rij=sqrtf(rij + rxij*rxij);

        // increment the appropriate bin, don't add duplicates to the zeroth bin
        ibin=__float2int_rd((rij-rmin)*delr_inv);
        if (ibin<nbins && ibin>=0 && rij>rmin2) {
#ifdef __USEATOMIC
          atomicAdd(llhists1+ibin, 1U);
#else
          addData(llhists1, ibin, threadTag);
#endif
        }
      } //joffset
    } //iblock
  }

  __syncthreads();


  // below we store the histograms to global memory so that they may be summed
  // in another routine.  Setting nofblo
  nofblocksi=nbins/NBLOCK;
  nleftover=nbins-nofblocksi*NBLOCK;
               
#ifdef __USEATOMIC
  // reinitialize llhists1 to point to global memory
  unsigned int *llhistg1=llhistg+blockIdx.x*MAXBIN+threadIdx.x;

  // loop to add all elements to global memory
  loopmax = nofblocksi*NBLOCK;
  for (iblock=0; iblock < loopmax; iblock+=NBLOCK) {
    llhistg1[iblock] += llhists[iblock+threadIdx.x];
  }
  // take care of leftovers
  if (threadIdx.x < nleftover) {
    llhistg1[iblock] += llhists[iblock+threadIdx.x];
  }
#else
  // reinitialize llhists1 to point to global memory
  unsigned int* llhistg1=llhistg+(((blockIdx.x)*NBLOCK)/THREADSPERWARP)*MAXBIN+threadIdx.x;

  // loop to add all elements to global memory
  loopmax = MAXBIN * (NBLOCK / THREADSPERWARP);
  loopmax2 = nofblocksi * NBLOCK;
  for (io=0; io < loopmax; io+=MAXBIN) {
    for (iblock=0; iblock < loopmax2; iblock+=NBLOCK) {
      llhistg1[io+iblock] += llhists[io+iblock+threadIdx.x];
    }
    // take care of leftovers
    if (threadIdx.x < nleftover) {
      llhistg1[iblock] += llhists[io+iblock+threadIdx.x];
    }
  }
#endif

  return;
}


#define ull2float __uint2float_rn

/* This routine takes a series of integer histograms in llhist, 
 * stored in as integers, converts them to floats, multiplies 
 * them by appropriate factors, and sums them.  The result is 
 * stored in histdev.
 */
__global__ void calculate_histogram(float* histdev, // histogram to return
                             unsigned int* llhistg, // the histograms stored
                              // in global memory.  There is one histogram
                              // per block (warp) if atomicAdds are (not) used
                                    int nbins) // stride between geometries in xyz.
{
  int io;      // index for inner loop
  int iblock;  // index for outer loop  
  int maxloop; // maxima for loop conditions

  __shared__ unsigned int llhists[NBLOCKHIST]; // block of int histogram in final summation

  __shared__ float xi[NBLOCKHIST]; // smem for the floating point histograms.
  int nofblocks;                   // number of data blocks per histogram

  nofblocks=nbins/NBLOCKHIST;      // set number of blocks per histogram
  int nleftover=nbins-nofblocks*NBLOCKHIST;
  maxloop = nofblocks*NBLOCKHIST;  // set maxloop and maxloop2

  // loop over all of the blocks in a histogram
  for (iblock=0;iblock<maxloop;iblock+=NBLOCKHIST) {
    xi[threadIdx.x]=0.0f; // zero out xi

    // loop over the histograms created by calculate_rdf
#ifdef __USEATOMIC
    for (io=0; io < MAXBIN*NCUDABLOCKS; io+=MAXBIN) {
#else
    for (io=0; io < MAXBIN * (NCUDABLOCKS*NBLOCK / THREADSPERWARP); io+=MAXBIN) {
#endif
      // load integer histogram data into shared memory (coalesced)
      llhists[threadIdx.x]=llhistg[io+iblock+threadIdx.x];

      // shave off the thread tag that might remain from calculate_rdf
      llhists[threadIdx.x]=llhists[threadIdx.x] & 0x07FFFFFFUL;

      // convert to float
      xi[threadIdx.x]+=ull2float(llhists[threadIdx.x]);
    }

    // ... and store in global memory
    histdev[iblock+threadIdx.x]+=xi[threadIdx.x];
  }

  // take care of leftovers
  if (threadIdx.x < nleftover) {
    xi[threadIdx.x]=0.0f; // zero out xi

    // loop over the histograms created by calculate_rdf
#ifdef __USEATOMIC
    for (io=0; io < MAXBIN*NCUDABLOCKS; io+=MAXBIN) {
#else
    for (io=0; io < MAXBIN *(NCUDABLOCKS*NBLOCK / THREADSPERWARP); io+=MAXBIN) {
#endif
      // load integer histogram data into shared memory (coalesced)
      llhists[threadIdx.x]=llhistg[io+iblock+threadIdx.x];

      // shave off the thread tag that might remain from calculate_rdf
      llhists[threadIdx.x]=llhists[threadIdx.x] & 0x07FFFFFFUL;

      // convert to float
      xi[threadIdx.x]+=ull2float(llhists[threadIdx.x]);
    }

    // ... and store in global memory
    histdev[iblock+threadIdx.x]+=xi[threadIdx.x];
  }

  // calculate_hist out
  return;
}


/*
 * calculate_histogram_block
 * Compute one histogram block 
 */
void calculate_histogram_block(int usepbc,
                               float *xyz, 
                               int natomsi, 
                               int natomsj,
                               float3 celld,
                               unsigned int* llhistg,
                               int nbins,
                               float rmin,
                               float delr_inv,
                               float *histdev,
                               int nblockhist) {
  // zero out the histograms in global memory
  init_hist<<<NCUDABLOCKS, NBLOCK>>>(llhistg, MAXBIN);
      
  // calculate the histograms
  calculate_rdf<<<NCUDABLOCKS, NBLOCK>>>(usepbc, xyz, natomsi, natomsj, celld,
                                         llhistg, nbins, rmin, delr_inv);

  // sum histograms and begin normalization by adjusting for the cell volume
  calculate_histogram<<<1, nblockhist>>>(histdev, llhistg, nbins);
}



/*
 * input parameters for rdf_thread() routine
 */
typedef struct {
  int usepbc;        // periodic or non-periodic calculation
  int natoms1;       // number of atoms in selection 1.
  float* xyz;        // coordinates of first selection, [natoms1][3]
  int natoms2;       // number of atoms in selection 2.
  float** xyz2array; // coordinates of selection 2. [natoms2][3] (per-thread)
  float* cell;       // the cell x y and z dimensions [3]
  float** histarray; // the final histogram in host mem [nbins] (per-thread)
  int nbins;         // the number of bins in the histogram
  int nblockhist;    // the size of a block used in the
                     // reduction of the histogram
  float rmin;        // the minimum value of the first bin
  float delr;        // the width of each bin
  int nblocks;       // number of const blocks to compute all pairs histogram
  int nhblock;       // # NBLOCKHIST-sized blocks needed to calc whole histogram
} rdfthrparms;


/*  
 * rdf_thread -- multithreaded version of subroutine_rdf()
 * This routine is called from the CPU to calculate g(r) for a single frame
 * on the GPU.  Right now it performs a brute force O(N^2) calculations
 * (with all atomic pairs considered).  Future version may reduce the overall
 * work by performing a grid search.
 */
static void * rdf_thread(void *voidparms) {
  rdfthrparms *parms = NULL;
  int threadid=0;

  wkf_threadpool_worker_getid(voidparms, &threadid, NULL);
  wkf_threadpool_worker_getdata(voidparms, (void **) &parms);

#if 0
printf("rdf thread[%d] lanched and running...\n", threadid);
#endif
  cudaSetDevice(threadid);

  /*
   * copy in per-thread parameters
   */
  const int natoms1 = parms->natoms1;
  const float *xyz = parms->xyz;
  const int natoms2 = parms->natoms2;
  const float *cell = parms->cell;
  const int nbins = parms->nbins;
  const int nblockhist = parms->nblockhist;
  const float rmin = parms->rmin;
  const float delr = parms->delr;
  float *xyz2 = parms->xyz2array[threadid];
  float *hist = parms->histarray[threadid];
  const int nhblock = parms->nhblock;
  const int usepbc = parms->usepbc;

  float* xyzdev;         // pointer to xyz data in global memory
  float3 celld;          // cell x, y, and z dimensions
  float* histdev;        // pointer to float histogram in global memory
  unsigned int* llhistg; // pointer to int histograms in global memory

  int nconstblocks; // # full blocks to be sent to constant memory
  int nleftover;    // # elements in leftover block of const mem
  int nbigblocks;   // number of blocks required to prevent histogram overflow
  int ibigblock;    // the index of the current big block
  int nleftoverbig; // the number of elements in the final big block
  int ntmpbig;
  int nbinstmp;     // number of bins currently being processed
  int natoms1pad;   // number of atoms in selection 1 after padding
  int natoms2pad;   // number of atoms in selection 1 after padding
  float rmax;       // cutoff radius for atom contributions to histogram

  // natoms1pad is natoms1 rounded up to the next multiple of NBLOCK
  natoms1pad=natoms1;
  if (natoms1%NBLOCK!=0) {natoms1pad = (natoms1 / NBLOCK + 1) * NBLOCK;}
  natoms2pad=natoms2;
  if (natoms2%NBLOCK!=0) {natoms2pad = (natoms2 / NBLOCK + 1) * NBLOCK;}

  // allocate global memory for:

  // the xyz coordinates of selection 1
  cudaMalloc((void**)&xyzdev, 3*max(natoms1pad,natoms2pad)*sizeof(float));

  // the final copy of the histogram
  cudaMalloc((void**)&histdev, nbins*sizeof(float));

  // zero out floating point histogram
  int ihblock;         // loop counter for histogram blocks
  int maxloop=nbins/nblockhist;
  if (maxloop*nblockhist!=nbins) {maxloop=maxloop+1;}
  for (ihblock=0; ihblock<maxloop; ihblock++) {
    init_hist_f<<<1, nblockhist>>>(histdev+ihblock*nblockhist);
  }

  // the global memory copies of the histogram
#ifdef __USEATOMIC
  cudaMalloc((void**)&llhistg, (NCUDABLOCKS*MAXBIN*sizeof(int)));
#else
  cudaMalloc((void**)&llhistg, (NCUDABLOCKS*NBLOCK*MAXBIN*
             sizeof(int))/THREADSPERWARP);
#endif

  // set the cell dimensions
  celld.x=cell[0];
  celld.y=cell[1];
  celld.z=cell[2];

  // set rmax.  this is the distance that the padding atoms must be from the 
  // unit cell AND that they must be from each other
  rmax = 1.1f * (rmin+((float)nbins)*delr);

  // send the second selection to the gpu for reimaging
  cudaMemcpy(xyzdev, xyz2, 3*natoms2*sizeof(float), cudaMemcpyHostToDevice);

  if (usepbc) {
    // reimage atom coors into a single periodic cell
    reimage_xyz<<<NCUDABLOCKS,NBLOCK>>>(xyzdev,natoms2,natoms2pad,celld,rmax);
  } else {
    // shift phantom atoms so they don't interfere with non-periodic calc.
    phantom_xyz<<<NCUDABLOCKS,NBLOCK>>>(xyzdev,natoms2,natoms2pad,1.0e38f,rmax);
  }

  // now, unfortunately, we have to move it back because this selection 
  // will need to be sent to constant memory (done on the host side)
  cudaMemcpy(xyz2, xyzdev, 3*natoms2*sizeof(float), cudaMemcpyDeviceToHost);

  // send the xyz coords of selection 1 to the gpu
  cudaMemcpy(xyzdev, xyz, 3*natoms1*sizeof(float), cudaMemcpyHostToDevice);

  if (usepbc) {
    // reimage xyz coords back into the periodic box and zero all histograms
    reimage_xyz<<<NCUDABLOCKS,NBLOCK>>>(xyzdev,natoms1,natoms1pad,celld,rmax);
  } else {
    // shift phantom atoms so they don't interfere with non-periodic calc.
    phantom_xyz<<<NCUDABLOCKS,NBLOCK>>>(xyzdev,natoms1,natoms1pad,1.0e38f,rmax);
  }

  // all blocks except for the final one will have MAXBIN elements
  nbinstmp = MAXBIN; 

  // calc # of groups to divide sel2 into to fit it into constant memory
  nconstblocks=natoms2/NCONSTBLOCK;
  nleftover=natoms2-nconstblocks*NCONSTBLOCK;

  // set up the big blocks to avoid bin overflow
  ntmpbig = NBLOCK * ((BIN_OVERFLOW_LIMIT / NCONSTBLOCK) / NBLOCK);
  nbigblocks = natoms1pad / ntmpbig;
  nleftoverbig = natoms1pad - nbigblocks * ntmpbig;

  int nbblocks = (nleftoverbig != 0) ? nbigblocks+1 : nbigblocks;

  // parallel loop over all "constblocks"
  wkf_tasktile_t tile; // next batch of work units
  while (wkf_threadpool_next_tile(voidparms, 1, &tile) != WKF_SCHED_DONE) {
#if 0
printf("rdf thread[%d] fetched tile: s: %d e: %d\n", threadid, tile.start, tile.end);
#endif
    int iconstblock = -1; // loop counter for constblocks, force update 
    ihblock = -1;     // initialize to sentinel value to force immediate update
    int workitem;     // current parallel work item
    for (workitem=tile.start; workitem<tile.end; workitem++) {
      // decode work item into ihblock and iconstblock indices
      ihblock = workitem % nhblock;
      int newiconstblock = workitem / nhblock;
      int blocksz;

      // When moving to the next constblock we must copy in new coordinates
      if (iconstblock != newiconstblock) {
        iconstblock = newiconstblock;

        // take care of the leftovers on the last block
        blocksz = (iconstblock == nconstblocks && nleftover != 0) 
                  ? nleftover : NCONSTBLOCK;

        // send a "constblock" of coords from selection 2 to const mem
        copycoordstoconstbuff(blocksz, xyz2+(3*NCONSTBLOCK*iconstblock));
      } 

#if 0
printf("rdf thread[%d] iconstblock: %d ihblock: %d\n", threadid, 
       iconstblock, ihblock);
#endif

      // loop over blocks of histogram
      // minimum distance for the current histogram block
      // the first block starts at rmin
      float rmintmp=rmin + (MAXBIN * delr * ihblock);

      nbinstmp = MAXBIN;

      // the final block will have nbin%MAXBIN elements
      if (ihblock==(nhblock-1)) { 
        nbinstmp = nbins%MAXBIN;
        // if nbins is an even multiple of MAXBIN, the final block is full
        if (nbinstmp==0) { nbinstmp = MAXBIN;}
      }

      for (ibigblock=0; ibigblock < nbblocks; ibigblock++) {
        // take care of the leftovers on the last block
        int bblocksz = (ibigblock == nbigblocks && nleftoverbig != 0) 
                       ? nleftoverbig : ntmpbig;

        calculate_histogram_block(usepbc, (xyzdev+ibigblock*ntmpbig*3),
                                  bblocksz, blocksz, celld,
                                  llhistg, nbinstmp, rmintmp, (1/delr),
                                  histdev+ihblock*MAXBIN, nblockhist);
      }

      cudaDeviceSynchronize();
    }
  } // end of parallel tile fetch while loop

  // retrieve g of r for this frame
  cudaMemcpy(hist, histdev, nbins*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(xyzdev);  // free gpu global memory
  cudaFree(histdev);
  cudaFree(llhistg);

#if 0
printf("thread[%d] finished, releasing resources\n", threadid); 
#endif

  return NULL;
}


int rdf_gpu(wkf_threadpool_t *devpool, // VMD GPU worker thread pool
            int usepbc,                // periodic or non-periodic calc.
            int natoms1,               // array of the number of atoms in
                                       // selection 1 in each frame.
            float *xyz,                // coordinates of first selection.
                                       // [natoms1][3]
            int natoms2,               // array of the number of atoms in
                                       // selection 2 in each frame.
            float *xyz2,               // coordinates of selection 2.
                                       // [natoms2][3]
            float* cell,               // the cell x y and z dimensions [3]
            float* hist,               // the histograms, 1 per block
                                       // [ncudablocks][maxbin]
            int nbins,                 // the number of bins in the histogram
            float rmin,                // the minimum value of the first bin
            float delr)                // the width of each bin
{
  int i, ibin;

  if (devpool == NULL) {
    return -1; // abort if no device pool exists
  }
  int numprocs = wkf_threadpool_get_workercount(devpool);

  // multi-thread buffers
  float **xyz2array = (float**) calloc(1, sizeof(float *) * numprocs);
  float **histarray = (float**) calloc(1, sizeof(float *) * numprocs);
  for (i=0; i<numprocs; i++) {
    xyz2array[i] = (float*) calloc(1, sizeof(float) * 3 * (natoms2/NBLOCK + 1) * NBLOCK);
    memcpy(xyz2array[i], xyz2, sizeof(float) * 3 * (natoms2/NBLOCK + 1) * NBLOCK);
    histarray[i] = (float*) calloc(1, sizeof(float) * nbins);
  }

  memset(hist, 0, nbins * sizeof(float)); // clear global histogram array

  // setup thread parameters
  rdfthrparms parms;

  parms.usepbc = usepbc;
  parms.natoms1 = natoms1;
  parms.xyz = xyz;
  parms.natoms2 = natoms2;
  parms.cell = cell;
  parms.nbins = nbins;
  parms.nblockhist = NBLOCKHIST; // size of final reduction block 
  parms.rmin = rmin;
  parms.delr = delr;
  parms.xyz2array = xyz2array; // per-thread output data
  parms.histarray = histarray; // per-thread output data

  // calculate the number of blocks to divide the histogram into
  parms.nhblock = (nbins+MAXBIN-1)/MAXBIN;

  // calc # of groups to divide sel2 into to fit it into constant memory
  int nconstblocks=parms.natoms2/NCONSTBLOCK;
  int nleftoverconst=parms.natoms2 - nconstblocks*NCONSTBLOCK;

  // we have to do one extra cleanup pass if there are leftovers
  parms.nblocks = (nleftoverconst != 0) ? nconstblocks+1 : nconstblocks;

  int parblocks = parms.nblocks * parms.nhblock;
#if 0
  printf("master thread: number of parallel work items: %d\n", parblocks);
#endif

  // spawn child threads to do the work
  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=parblocks;
  wkf_threadpool_sched_dynamic(devpool, &tile);
  wkf_threadpool_launch(devpool, rdf_thread, &parms, 1);

  memset(hist, 0, nbins * sizeof(float)); // clear global histogram array
  // collect independent thread results into final histogram buffer
  for (i=0; i<numprocs; i++) {
    for (ibin=0; ibin<nbins; ibin++) {
      hist[ibin] += histarray[i][ibin];
    }      
  }

  return 0;
}



