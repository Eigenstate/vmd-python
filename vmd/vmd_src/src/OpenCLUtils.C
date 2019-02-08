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
 *      $RCSfile: OpenCLUtils.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   OpenCL utility functions for use in VMD
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if defined(VMDOPENCL)
#include "Inform.h"
#endif

int vmd_cl_print_platform_info(void) {
  cl_int clerr;
  cl_uint numplatforms;
  cl_platform_id *platformlist;
  clerr=clGetPlatformIDs(0, NULL, &numplatforms);
  platformlist = (cl_platform_id *) malloc(sizeof(cl_platform_id)*numplatforms);
  clerr=clGetPlatformIDs(numplatforms, platformlist, NULL);

  cl_uint i;
  for (i=0; i<numplatforms; i++) {
    char platname[80];
    clerr=clGetPlatformInfo(platformlist[i], CL_PLATFORM_NAME, 
                            sizeof(platname), (void *) platname, NULL);

    char platprofile[80];
    clerr=clGetPlatformInfo(platformlist[i], CL_PLATFORM_PROFILE, 
                            sizeof(platprofile), (void *) platprofile, NULL);

#if 0
    char platvendor[80];
    clerr=clGetPlatformInfo(platformlist[i], CL_PLATFORM_VENDOR, 
                            sizeof(platvendor), (void *) platvendor, NULL);
#endif

    cl_uint numdevs;
    clerr=clGetDeviceIDs(platformlist[i], CL_DEVICE_TYPE_ALL,
                         0, NULL, &numdevs);    

    char platforminfo[4096];
#if !defined(VMDOPENCL)
    printf("OpenCL Platform[%d]: %s, %s  Devices: %u\n",
           i, platname, platprofile, numdevs);
#else
    sprintf(platforminfo, "OpenCL Platform[%d]: %s, %s  Devices: %u\n",
           i, platname, platprofile, numdevs);
    msgInfo << platforminfo << sendmsg;
#endif

    int j;
    cl_device_id *devices = new cl_device_id[numdevs];
    clerr = clGetDeviceIDs(platformlist[i], CL_DEVICE_TYPE_ALL,
                           numdevs, devices, NULL);
    for (j=0; j<numdevs; j++) {
      char outstr[1024];
      size_t rsz;
      cl_uint clockratemhz, computeunits;
      cl_ulong globalmemsz;
      cl_char long_device_name[1024] = {0};
      cl_char device_name[1024] = {0};
      cl_device_id dev = devices[j];

      clerr |= clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(long_device_name),
                               long_device_name, &rsz);
      clerr |= clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                               sizeof(clockratemhz), &clockratemhz, &rsz);
      clerr |= clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                               sizeof(computeunits), &computeunits, &rsz);
      clerr |= clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,
                               sizeof(globalmemsz), &globalmemsz, &rsz);

      if (clerr == CL_SUCCESS) {
        // clean up device name string (some contain lots of spaces)
        int len = strlen((const char *) long_device_name);
        int k,l;
        for (k=0,l=0; k<len; k++) {
          if (long_device_name[k] != ' ') {
            device_name[l] = long_device_name[k];
            l++;
            continue;
          }

          device_name[l] = long_device_name[k];
          l++;
          while (k < len) {
            if (long_device_name[k+1] == ' ')
              k++;
            else
              break;
          }
        }

        // list primary GPU device attributes
        sprintf(outstr, "[%d] %-40s %2d CU @ %.2f GHz",
                j, device_name, computeunits, clockratemhz / 1000.0);
        msgInfo << outstr;

        // list memory capacity
        int gpumemmb = globalmemsz / (1024 * 1024);
        if (gpumemmb < 1000)
          sprintf(outstr, ", %4dMB RAM", gpumemmb);
        else if (gpumemmb < 10240)
          sprintf(outstr, ", %.1fGB RAM", gpumemmb / 1024.0);
        else 
          sprintf(outstr, ", %dGB RAM", gpumemmb / 1024);

        msgInfo << outstr;

        // list optional hardware features and configuration attributes here...
        // XXX not implemented yet...

        msgInfo << sendmsg;
      } else {
        sprintf(outstr, "  [%d] Error during OpenCL device query!", j);
        msgInfo << outstr << sendmsg;

        clerr = CL_SUCCESS;
      }
    }

    delete [] devices;
  }

  free(platformlist);

  return 0;
}


cl_platform_id vmd_cl_get_platform_index(int i) {
  cl_int clerr;
  cl_uint numplatforms;
  cl_platform_id *platformlist;
  cl_platform_id plat;    
  clerr=clGetPlatformIDs(0, NULL, &numplatforms);
  if (i >= (int) numplatforms)
    return NULL;

  platformlist = (cl_platform_id *) malloc(sizeof(cl_platform_id)*numplatforms);
  clerr=clGetPlatformIDs(numplatforms, platformlist, NULL);
  if (clerr != CL_SUCCESS) {
    free(platformlist);
    return NULL;
  }

  plat=platformlist[i];
  free(platformlist);

  return plat;
}  


int vmd_cl_context_num_devices(cl_context clctx) {
  size_t parmsz;
  cl_int clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);
  if (clerr != CL_SUCCESS)
    return 0;
  
  return (int) (parmsz / sizeof(size_t));
}


cl_command_queue vmd_cl_create_command_queue(cl_context clctx, int dev) {
  size_t parmsz;
  cl_int clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);
  if (clerr != CL_SUCCESS)
    return NULL;

  cl_device_id* cldevs = (cl_device_id *) malloc(parmsz);
  clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);
  if (clerr != CL_SUCCESS)
    return NULL;

  cl_command_queue clcmdq = clCreateCommandQueue(clctx, cldevs[dev], 0, &clerr);
  free(cldevs);
  if (clerr != CL_SUCCESS)
    return NULL;

  return clcmdq;
}


cl_kernel vmd_cl_compile_kernel(cl_context clctx, const char *kernname,
                                const char *srctext, const char *flags, 
                                cl_int *clerr, int verbose) {
  char buildlog[8192];
  cl_program clpgm = NULL;
  cl_kernel clkern = NULL;

  clpgm = clCreateProgramWithSource(clctx, 1, &srctext, NULL, clerr);
  if (clerr != CL_SUCCESS) {
    if (verbose)
      printf("Failed to compile OpenCL kernel: '%s'\n", kernname);
    return NULL;
  }
  *clerr = clBuildProgram(clpgm, 0, NULL, flags, NULL, NULL);

#if 1
  if (verbose) {
    memset(buildlog, 0, sizeof(buildlog));

    size_t parmsz;
    *clerr |= clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);

    cl_device_id* cldevs = (cl_device_id *) malloc(parmsz);
    *clerr |= clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);

    size_t len=0;
    *clerr = clGetProgramBuildInfo(clpgm, cldevs[0], CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, &len);
    if (len > 1) {
      printf("OpenCL kernel compilation log:\n");
      printf("  '%s'\n", buildlog);
    }
  }   
#endif

  clkern = clCreateKernel(clpgm, kernname, clerr);
  if (clerr != CL_SUCCESS) {
    if (verbose)
      printf("Failed to create OpenCL kernel: '%s'\n", kernname);
    return NULL;
  }

  return clkern;
} 


cl_kernel vmd_cl_compile_kernel_file(cl_context clctx, const char *kernname,
                                const char *filename, const char *flags, 
                                cl_int *clerr, int verbose) {
  FILE *ifp;
  char *src;
  struct stat statbuf;
  cl_kernel clkern = NULL;
   
  if ((ifp = fopen(filename, "r")) == NULL)
    return NULL;
   
  stat(filename, &statbuf);
  src = (char *) calloc(1, statbuf.st_size + 1);
  fread(src, statbuf.st_size, 1, ifp);
  src[statbuf.st_size] = '\0';
  clkern = vmd_cl_compile_kernel(clctx, kernname, src, flags, clerr, verbose);
  free(src);

  return clkern;
}



