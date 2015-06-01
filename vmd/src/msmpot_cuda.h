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
 *      $RCSfile: msmpot_cuda.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2010/06/03 20:07:10 $
 *
 ***************************************************************************/

#include "msmpot_internal.h"


#ifndef MSMPOT_MSMCUDA_H
#define MSMPOT_MSMCUDA_H

/*
 * detect and report error from CUDA
 */
#undef  CUERR
#define CUERR(errnum) \
  do { \
    cudaError_t cuerr = cudaGetLastError(); \
    if (cuerr != cudaSuccess) { \
      return ERROR(errnum); \
    } \
  } while (0)


/*
 * Keep NBRLIST_MAXLEN of 3-tuples in GPU const cache memory:
 *   (3 * 5333) ints  +  1 int (giving use length)  ==  64000 bytes
 */
#undef  NBRLIST_MAXLEN
#define NBRLIST_MAXLEN  5333


#ifdef __cplusplus
extern "C" {
#endif

  struct MsmpotCuda_t {
    Msmpot *msmpot;

    /* get CUDA device info */
#if 0
    struct cudaDeviceProp *dev;
    int ndevs;
#endif
    int devnum;           /* device number */

    /* CUDA short-range part ("binsmall") */
    int pmx, pmy, pmz;                 /* dimensions of padded epotmap */
    long maxpm;                        /* allocated points for padded map */ 
    float *padmap;                     /* padded epotmap for CUDA grid */

    float *dev_padmap;                 /* points to device memory */
    long dev_maxpm;                    /* allocated points on device */

    float4 *dev_bin;                   /* points to device memory */
    int dev_nbins;                     /* allocated bins on device */

    /* CUDA lattice cutoff */
    int   lk_nlevels;      /* number of levels for latcut kernel */
    int   lk_srad;         /* subcube radius for latcut kernel */
    int   lk_padding;      /* padding around internal array of subcubes */
    int   subcube_total;   /* total number of subcubes for compressed grids */
    int   block_total;     /* total number of thread blocks */
    /*
     * host_   -->  memory allocated on host
     * device_ -->  global memory allocated on device
     */
    int   *host_sinfo;     /* subcube info copy to device const mem */
    float *host_lfac;      /* level factor copy to device const mem */
    int maxlevels;

    float *host_wt;        /* weights copy to device const mem */
    int maxwts;

    float *host_qgrids;    /* q-grid subcubes copy to device global mem */
    float *host_egrids;    /* e-grid subcubes copy to device global mem */
    float *device_qgrids;  /* q-grid subcubes allocate on device */
    float *device_egrids;  /* e-grid subcubes allocate on device */
    long maxgridpts;

  };

  void Msmpot_cuda_cleanup(MsmpotCuda *);

  int Msmpot_cuda_setup_shortrng(MsmpotCuda *);
  void Msmpot_cuda_cleanup_shortrng(MsmpotCuda *);

  int Msmpot_cuda_setup_latcut(MsmpotCuda *);
  void Msmpot_cuda_cleanup_latcut(MsmpotCuda *);

#ifdef __cplusplus
}
#endif


#endif /* MSMPOT_MSMCUDA_H */
