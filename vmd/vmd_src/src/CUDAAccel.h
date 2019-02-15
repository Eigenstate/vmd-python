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
 *	$RCSfile: CUDAAccel.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.22 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to store and handle enumeration and initialization of 
 *   CUDA GPU accelerator devices.
 ***************************************************************************/
#ifndef CUDACCEL_H
#define CUDACCEL_H

#include "WKFThreads.h"

typedef struct {
  int deviceid;
  char name[80];
  int major;
  int minor;
  unsigned long membytes;
  int clockratekhz;
  int smcount;
  int integratedgpu;
  int asyncenginecount;
  int kernelexectimeoutenabled;
  int canmaphostmem;
  int computemode;
} cudadevprops;

/// manages enumeration and initialization of CUDA devices
class CUDAAccel {
private:
  int cudaavail;   // whether or not the CUDA runtime is operable for VMD
  int numdevices;  // number of CUDA GPU accelerator devices available
  ResizeArray<cudadevprops> devprops;
  wkf_threadpool_t *cudapool;

  // functions for operating on a pool of CUDA devices
  void devpool_init(void);
  void devpool_fini(void);

  // convenience enum to match CUDA driver APIs
  enum { computeModeDefault=0, 
         computeModeExclusive=1,
         computeModeProhibited=2 }; // computeMode;
 
public:
  CUDAAccel(void);
  virtual ~CUDAAccel(void);

  // functions for enumerating CUDA GPU accelerator devices
  // and their attributes
  void print_cuda_devices(void);
  int num_devices(void);
  int device_index(int dev);
  const char *device_name(int dev);
  int device_version_major(int dev);
  int device_version_minor(int dev);
  unsigned long device_membytes(int dev);
  float device_clock_ghz(int dev);
  int device_sm_count(int dev);
  int device_integratedgpu(int dev);
  int device_asyncenginecount(int dev);
  int device_kerneltimeoutenabled(int dev);
  int device_canmaphostmem(int dev);
  int device_computemode(int dev);

  // functions for operating on an open pool of CUDA devices
  int devpool_launch(void *fctn(void *), void *parms, int blocking);
  int devpool_wait(void);
  wkf_threadpool_t * get_cuda_devpool(void) { return cudapool; }

};

#endif



