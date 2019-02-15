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
 *	$RCSfile: CUDAAccel.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.52 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to store and handle enumeration and initialization of 
 *   CUDA GPU accelerator devices.
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "config.h"     // rebuild on config changes
#include "Inform.h"
#include "ResizeArray.h"
#include "CUDAAccel.h"
#include "CUDAKernels.h"
#include "WKFThreads.h"
#include "ProfileHooks.h"


CUDAAccel::CUDAAccel(void) {
  cudaavail = 0;
  numdevices = 0;
  cudapool=NULL;

  if (getenv("VMDNOCUDA") != NULL) {
    msgInfo << "VMDNOCUDA environment variable is set, CUDA support disabled."
            << sendmsg;
    return; 
  }

#if defined(VMDCUDA)
  PROFILE_PUSH_RANGE("CUDAAccel::CUDAAccel()", 0);

  unsigned int gpumask = 0xffffffff;
  const char *gpumaskstr = getenv("VMDCUDADEVICEMASK");
  if (gpumaskstr != NULL) {
    unsigned int tmp;
    if (sscanf(gpumaskstr, "%x", &tmp) == 1) {
      gpumask = tmp;
      msgInfo << "Using GPU device mask '"
              << gpumaskstr << "'" << sendmsg;
    } else {
      msgInfo << "Failed to parse CUDA GPU device mask string '" 
              << gpumaskstr << "'" << sendmsg;
    }
  }

  int usabledevices = 0;
  int rc = 0;
  if ((rc=vmd_cuda_num_devices(&numdevices)) != VMDCUDA_ERR_NONE) {
    numdevices = 0;

    // Only emit error messages when there are CUDA GPUs on the machine
    // but that they can't be used for some reason
    // XXX turning this off for the time being, as some people have 
    //     NVIDIA drivers installed on machines with no NVIDIA GPU, as can
    //     happen with some distros that package the drivers by default.
    switch (rc) {
      case VMDCUDA_ERR_NODEVICES:
      case VMDCUDA_ERR_SOMEDEVICES:
//        msgInfo << "No CUDA accelerator devices available." << sendmsg;
        break;

#if 0
      case VMDCUDA_ERR_SOMEDEVICES:
        msgWarn << "One or more CUDA accelerators may exist but are not usable." << sendmsg; 
        msgWarn << "Check to make sure that GPU drivers are up to date." << sendmsg;
        break;
#endif

      case VMDCUDA_ERR_DRVMISMATCH:
        msgWarn << "Detected a mismatch between CUDA runtime and GPU driver" << sendmsg; 
        msgWarn << "Check to make sure that GPU drivers are up to date." << sendmsg;
//        msgInfo << "No CUDA accelerator devices available." << sendmsg;
        break;
    }
   
    PROFILE_POP_RANGE();
    return;
  }

  if (numdevices > 0) {
    cudaavail = 1;

    int i;
    for (i=0; i<numdevices; i++) {
      cudadevprops dp;
      memset(&dp, 0, sizeof(dp));
      if (!vmd_cuda_device_props(i, dp.name, sizeof(dp.name),
                                &dp.major, &dp.minor,
                                &dp.membytes, &dp.clockratekhz, 
                                &dp.smcount, &dp.integratedgpu,
                                &dp.asyncenginecount, &dp.kernelexectimeoutenabled,
                                &dp.canmaphostmem, &dp.computemode)) {
        dp.deviceid=i; // save the device index

        // Check that each GPU device has not been excluded by virtue of 
        // being used for display, by a GPU device mask, or by the CUDA
        // device mode being set to a "prohibited" status.
        if (!(dp.kernelexectimeoutenabled && getenv("VMDCUDANODISPLAYGPUS")) &&
            (gpumask & (1 << i)) && 
            (dp.computemode != computeModeProhibited)) {
          devprops.append(dp);
          usabledevices++;
        }
      } else {
        msgWarn << "  Failed to retrieve properties for CUDA accelerator " << i << sendmsg; 
      }
    }
  }
  numdevices=usabledevices;

  devpool_init();

  PROFILE_POP_RANGE();
#endif
}

// destructor
CUDAAccel::~CUDAAccel(void) {
  devpool_fini();
}


void CUDAAccel::devpool_init(void) {
  cudapool=NULL;

#if defined(VMDCUDA)
  PROFILE_PUSH_RANGE("CUDAAccel::devpool_init()", 0);

  if (!cudaavail || numdevices == 0 || getenv("VMDNOCUDA") != NULL)
    return;

  // only use as many GPUs as CPU cores we're allowed to use
  int workercount=numdevices;
  if (workercount > wkf_thread_numprocessors())
    workercount=wkf_thread_numprocessors();

  int *devlist = new int[workercount];
  int i;
  for (i=0; i<workercount; i++) {
    devlist[i]=device_index(i);
  }

  msgInfo << "Creating CUDA device pool and initializing hardware..." << sendmsg;
  cudapool=wkf_threadpool_create(workercount, devlist);
  delete [] devlist;

  // associate each worker thread with a specific GPU
  if (getenv("VMDCUDAVERBOSE") != NULL)
    wkf_threadpool_launch(cudapool, vmd_cuda_devpool_setdevice, (void*)"VMD CUDA Dev Init", 1);
  else
    wkf_threadpool_launch(cudapool, vmd_cuda_devpool_setdevice, NULL, 1);

  if (!getenv("VMDNOCUDA")) {
    // clear all available device memory on each of the GPUs
    wkf_threadpool_launch(cudapool, vmd_cuda_devpool_clear_device_mem, NULL, 1);
  }

  PROFILE_POP_RANGE();
#endif
}

void CUDAAccel::devpool_fini(void) {
  if (!cudapool)
    return;

#if defined(VMDCUDA)
  devpool_wait();
  wkf_threadpool_destroy(cudapool);
#endif
  cudapool=NULL;
}

int CUDAAccel::devpool_launch(void *fctn(void *), void *parms, int blocking) {
  if (!cudapool)
    return -1;

  return wkf_threadpool_launch(cudapool, fctn, parms, blocking); 
}

int CUDAAccel::devpool_wait(void) {
  if (!cudapool)
    return -1;

  return wkf_threadpool_wait(cudapool);
}

void CUDAAccel::print_cuda_devices(void) {
  if (getenv("VMDCUDANODISPLAYGPUS")) {
    msgInfo << "Ignoring CUDA-capable GPUs used for display" << sendmsg;
  }

  if (!cudaavail || numdevices == 0) {
    msgInfo << "No CUDA accelerator devices available." << sendmsg;
    return;
  }

  msgInfo << "Detected " << numdevices << " available CUDA " 
          << ((numdevices > 1) ? "accelerators:" : "accelerator:") << sendmsg;
  int i;
  for (i=0; i<numdevices; i++) {
    char outstr[1024];
    memset(outstr, 0, sizeof(outstr));

    // list primary GPU device attributes
    sprintf(outstr, "[%d] %-20s %2d SM_%d.%d %.2f GHz",
            device_index(i), device_name(i), 
            (device_sm_count(i) > 0) ? device_sm_count(i) : 0,
            device_version_major(i), device_version_minor(i),
            device_clock_ghz(i));
    msgInfo << outstr;

    // list memory capacity 
    int gpumemmb = (device_membytes(i) / (1024 * 1024));
    if (gpumemmb < 1000) {
      sprintf(outstr, ", %4dMB RAM", gpumemmb);
    } else if (gpumemmb < 10240) {
      sprintf(outstr, ", %.1fGB RAM", gpumemmb / 1024.0);
    } else {
      // round up to nearest GB
      sprintf(outstr, ", %dGB RAM", (gpumemmb + 512) / 1024);
    }

    msgInfo << outstr;

    // list optional hardware features and configuration attributes here...
    if (device_computemode(i) == computeModeProhibited) {
      msgInfo << ", Compute Mode: Prohibited";
    } else {
      if (device_integratedgpu(i))
        msgInfo << ", IGPU";

      if (device_kerneltimeoutenabled(i))
        msgInfo << ", KTO";

      if (device_asyncenginecount(i))
        msgInfo << ", AE" << device_asyncenginecount(i);

      if (device_canmaphostmem(i))
        msgInfo << ", ZCP";
    }

    msgInfo << sendmsg; 
  } 
}

int CUDAAccel::num_devices(void) {
  return numdevices;
}

int CUDAAccel::device_index(int dev) {
  return devprops[dev].deviceid;
}

const char *CUDAAccel::device_name(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return NULL;
  return devprops[dev].name; 
}

int CUDAAccel::device_version_major(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return devprops[dev].major;
}

int CUDAAccel::device_version_minor(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return devprops[dev].minor;
}

unsigned long CUDAAccel::device_membytes(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return devprops[dev].membytes;
}

float CUDAAccel::device_clock_ghz(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return (float) (devprops[dev].clockratekhz / 1000000.0);
}

int CUDAAccel::device_sm_count(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].smcount;
}

int CUDAAccel::device_integratedgpu(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].integratedgpu;
}

int CUDAAccel::device_asyncenginecount(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].asyncenginecount;
}

int CUDAAccel::device_kerneltimeoutenabled(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].kernelexectimeoutenabled;
}

int CUDAAccel::device_canmaphostmem(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].canmaphostmem;
}

int CUDAAccel::device_computemode(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].computemode;
}


