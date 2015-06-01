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
 *      $RCSfile: msmpot_cuda.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2010/06/03 20:07:09 $
 *
 ***************************************************************************/

#include "msmpot_cuda.h"

#define OK  MSMPOT_SUCCESS

MsmpotCuda *Msmpot_cuda_create(void) {
  MsmpotCuda *mc = (MsmpotCuda *) calloc(1, sizeof(MsmpotCuda));
  return mc;
}


void Msmpot_cuda_destroy(MsmpotCuda *mc) {
  Msmpot_cuda_cleanup(mc);
  free(mc);
}


void Msmpot_cuda_cleanup(MsmpotCuda *mc) {
  REPORT("Cleaning up CUDA lattice cutoff part.");
  Msmpot_cuda_cleanup_latcut(mc);
  REPORT("Cleaning up CUDA short-range part.");
  Msmpot_cuda_cleanup_shortrng(mc);
//  free(mc->dev);
}


//static int list_devices(MsmpotCuda *);
//static int real_devices(MsmpotCuda *);
static int set_device(MsmpotCuda *, int devnum);


int Msmpot_cuda_setup(MsmpotCuda *mc, Msmpot *msm) {
  int rc;  // return code from setup calls
  int is_cuda_optional = msm->cuda_optional;

  int devnum = 0;  // XXX use device 0 for now

  mc->msmpot = msm;  // handle back to Msmpot data

  msm->use_cuda_shortrng = 0;  // be pessimistic
  msm->use_cuda_latcut = 0;

#if 0
  if (msm->isperiodic) {  // XXX can't use CUDA
    REPORT("CUDA version does not support periodic boundaries.");
    if (is_cuda_optional) {
      REPORT("Falling back on CPU for computation.");
      return OK;
    }
    return ERROR(MSMPOT_ERROR_CUDA_SUPPORT);
  }
#endif

#if 0
  err = list_devices(mc);
  if (OK == err) err = set_device(mc, msm->devlist[0]);
                       // for now, use first device given by user
  if (OK == err) mc->devnum = msm->devlist[0];   // set device number
  else  if (msm->cuda_optional && err != MSMPOT_ERROR_MALLOC) {
    return ERRMSG(OK, "falling back on CPU for computation");
  }
  else return ERROR(err);
#endif

  rc = set_device(mc, devnum);     // attempt to use device
  if (rc != OK) {
    if (is_cuda_optional) {
      REPORT("Unable to setup CUDA device, fall back on CPU.");
      return OK;                   // fall back on CPU
    }
    else return ERROR(rc);         // can't keep going without CUDA
  }
  REPORT("Setup CUDA device.");

  rc = Msmpot_cuda_setup_shortrng(mc);
  if (OK == rc) {
    REPORT("Setup CUDA short-range part.");
    msm->use_cuda_shortrng = 1;    // use CUDA for shortrng
  }
  else if ( ! is_cuda_optional) return ERROR(rc);  // can't keep going
  else REPORT("Unable to setup CUDA short-range part, fall back on CPU.");

  rc = Msmpot_cuda_setup_latcut(mc);
  if (OK == rc) {
    REPORT("Setup CUDA lattice cutoff part.");
    msm->use_cuda_latcut = 1;          // use CUDA for latcut
  }
  else if ( ! is_cuda_optional) return ERROR(rc);  // can't keep going
  else REPORT("Unable to setup CUDA lattice cutoff part, fall back on CPU.");

  return OK;
}


#if 0
int list_devices(MsmpotCuda *mc) {
  void *v;
  int ndevs, i;

  if (mc->dev) return real_devices(mc);  // we already have device list

  cudaGetDeviceCount(&ndevs);
  if (ndevs < 1) return ERROR(MSMPOT_ERROR_CUDA_DEVREQ);

  v = realloc(mc->dev, ndevs * sizeof(struct cudaDeviceProp));
  if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
  mc->dev = (struct cudaDeviceProp *) v;
  mc->ndevs = ndevs;

  for (i = 0;  i < ndevs;  i++) {
    cudaError_t cuerr = cudaGetDeviceProperties(mc->dev + i, i);
    if (cuerr != cudaSuccess) return ERROR(MSMPOT_ERROR_CUDA_DEVREQ);
  }
  return real_devices(mc);
}


// verify CUDA devices are real rather than emulation mode
int real_devices(MsmpotCuda *mc) {
  const int ndevs = mc->ndevs;
  int i;

  for (i = 0;  i < ndevs;  i++) {
    if (9999 == mc->dev[i].major && 9999 == mc->dev[i].minor) {
      return ERROR(MSMPOT_ERROR_CUDA_DEVREQ);  // emulation mode
    }
  }
  return OK;
}
#endif


int set_device(MsmpotCuda *mc, int devnum) {
  cudaError_t cuerr = cudaSetDevice(devnum);
  if (cuerr != cudaSuccess) {
#if CUDART_VERSION >= 2010
    cuerr = cudaGetLastError(); // query last error and reset error state
    if (cuerr != cudaErrorSetOnActiveProcess) {
      return ERROR(MSMPOT_ERROR_CUDA_DEVREQ);
    }
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }
  return OK;
}
