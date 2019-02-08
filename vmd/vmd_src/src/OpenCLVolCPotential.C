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
 *      $RCSfile: OpenCLVolCPotential.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.32 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   OpenCL accelerated coulombic potential grid calculation
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "Inform.h"
#include "utilities.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "OpenCLKernels.h"
#include "OpenCLUtils.h"

typedef struct {
  float* atoms;
  float* grideners;
  long int numplane;
  long int numcol;
  long int numpt;
  long int natoms;
  float gridspacing;
} enthrparms;

/* thread prototype */
static void * openclenergythread(void *);

#if 1
#define CLERR \
  if (clerr != CL_SUCCESS) {                     \
    printf("opencl error %d, %s line %d\n", clerr, __FILE__, __LINE__); \
    return NULL;                                   \
  }
#else
#define CLERR
#endif

// max constant buffer size is 64KB, minus whatever
// the OpenCL runtime and compiler are using that we don't know about
// At 16 bytes/atom, 4000 atoms is about the max we can store in
// the constant buffer.
#define MAXATOMS 4000


// 
// The OpenCL kernels calculate coulombic potential at each grid point and
// store the results in the output array.
//
// These versions of the code use the 64KB constant buffer area reloaded
// for each group of MAXATOMS atoms, until the contributions from all
// atoms have been summed into the potential grid.
//
// These implementations use precomputed and unrolled loops of 
// (dy^2 + dz^2) values for increased FP arithmetic intensity.
// The X coordinate portion of the loop is unrolled by four or eight,
// allowing the same dy^2 + dz^2 values to be reused multiple times,
// increasing the ratio of FP arithmetic relative to FP loads, and
// eliminating some redundant calculations.
//

//
// Tuned global memory coalescing version, unrolled in X
//

//
// Tunings for large potential map dimensions (e.g. 384x384x...)
//
#define UNROLLX       8
#define UNROLLY       1
#define BLOCKSIZEX    8  // make large enough to allow coalesced global mem ops
#define BLOCKSIZEY    8  // make as small as possible for finer granularity
#define BLOCKSIZE    (BLOCKSIZEX * BLOCKSIZEY)

#define V4UNROLLX       8
#define V4UNROLLY       1
#define V4BLOCKSIZEX    8
#define V4BLOCKSIZEY    8
#define V4BLOCKSIZE    V4BLOCKSIZEX * V4BLOCKSIZEY

// FLOP counting
#define FLOPSPERATOMEVAL (59.0/8.0)

// OpenCL source code
const char* clenergysrc =
  "__kernel __attribute__((reqd_work_group_size(BLOCKSIZEX, BLOCKSIZEY, 1))) \n"
  "void clenergy(int numatoms, float gridspacing, __global float *energy, __constant float4 *atominfo) {                        \n"
  "  unsigned int xindex  = (get_global_id(0) - get_local_id(0)) * UNROLLX + get_local_id(0); \n"
  "  unsigned int yindex  = get_global_id(1);                              \n"
  "  unsigned int outaddr = get_global_size(0) * UNROLLX * yindex + xindex;\n"
  "                                                                        \n"
  "  float coory = gridspacing * yindex;                                   \n"
  "  float coorx = gridspacing * xindex;                                   \n"
  "                                                                        \n"
  "  float energyvalx1 = 0.0f;                                             \n"
#if UNROLLX >= 4
  "  float energyvalx2 = 0.0f;                                             \n"
  "  float energyvalx3 = 0.0f;                                             \n"
  "  float energyvalx4 = 0.0f;                                             \n"
#endif
#if UNROLLX == 8
  "  float energyvalx5 = 0.0f;                                             \n"
  "  float energyvalx6 = 0.0f;                                             \n"
  "  float energyvalx7 = 0.0f;                                             \n"
  "  float energyvalx8 = 0.0f;                                             \n"
#endif
  "                                                                        \n"
  "  float gridspacing_u = gridspacing * BLOCKSIZEX;                       \n"
  "                                                                        \n"
  "  int atomid;                                                           \n"
  "  for (atomid=0; atomid<numatoms; atomid++) {                           \n"
  "    float dy = coory - atominfo[atomid].y;                              \n"
  "    float dyz2 = (dy * dy) + atominfo[atomid].z;                        \n"
  "                                                                        \n"
  "    float dx1 = coorx - atominfo[atomid].x;                             \n"
#if UNROLLX >= 4
  "    float dx2 = dx1 + gridspacing_u;                                    \n"
  "    float dx3 = dx2 + gridspacing_u;                                    \n"
  "    float dx4 = dx3 + gridspacing_u;                                    \n"
#endif
#if UNROLLX == 8
  "    float dx5 = dx4 + gridspacing_u;                                    \n"
  "    float dx6 = dx5 + gridspacing_u;                                    \n"
  "    float dx7 = dx6 + gridspacing_u;                                    \n"
  "    float dx8 = dx7 + gridspacing_u;                                    \n"
#endif
  "                                                                        \n"
  "    energyvalx1 += atominfo[atomid].w * native_rsqrt(dx1*dx1 + dyz2);   \n"
#if UNROLLX >= 4
  "    energyvalx2 += atominfo[atomid].w * native_rsqrt(dx2*dx2 + dyz2);   \n"
  "    energyvalx3 += atominfo[atomid].w * native_rsqrt(dx3*dx3 + dyz2);   \n"
  "    energyvalx4 += atominfo[atomid].w * native_rsqrt(dx4*dx4 + dyz2);   \n"
#endif
#if UNROLLX == 8
  "    energyvalx5 += atominfo[atomid].w * native_rsqrt(dx5*dx5 + dyz2);   \n"
  "    energyvalx6 += atominfo[atomid].w * native_rsqrt(dx6*dx6 + dyz2);   \n"
  "    energyvalx7 += atominfo[atomid].w * native_rsqrt(dx7*dx7 + dyz2);   \n"
  "    energyvalx8 += atominfo[atomid].w * native_rsqrt(dx8*dx8 + dyz2);   \n"
#endif
  "  }                                                                     \n"
  "                                                                        \n"
  "  energy[outaddr             ] += energyvalx1;                          \n"
#if UNROLLX >= 4
  "  energy[outaddr+1*BLOCKSIZEX] += energyvalx2;                          \n"
  "  energy[outaddr+2*BLOCKSIZEX] += energyvalx3;                          \n"
  "  energy[outaddr+3*BLOCKSIZEX] += energyvalx4;                          \n"
#endif
#if UNROLLX == 8
  "  energy[outaddr+4*BLOCKSIZEX] += energyvalx5;                          \n"
  "  energy[outaddr+5*BLOCKSIZEX] += energyvalx6;                          \n"
  "  energy[outaddr+6*BLOCKSIZEX] += energyvalx7;                          \n"
  "  energy[outaddr+7*BLOCKSIZEX] += energyvalx8;                          \n"
#endif
  "}                                                                       \n"
  "                                                                        \n"
  "                                                                        \n"
  "                                                                        \n"
  "__kernel __attribute__((reqd_work_group_size(V4BLOCKSIZEX, V4BLOCKSIZEY, 1))) \n"
  "void clenergy_vec4(int numatoms, float gridspacing, __global float *energy, __constant float4 *atominfo) {                        \n"
  "  unsigned int xindex  = (get_global_id(0) - get_local_id(0)) * V4UNROLLX + get_local_id(0); \n"
  "  unsigned int yindex  = get_global_id(1);                              \n"
  "  unsigned int outaddr = get_global_size(0) * V4UNROLLX * yindex + xindex;\n"
  "                                                                        \n"
  "  float coory = gridspacing * yindex;                                   \n"
  "  float coorx = gridspacing * xindex;                                   \n"
  "                                                                        \n"
  "  float4 energyvalx = 0.f;                                              \n"
#if V4UNROLLX == 8
  "  float4 energyvalx2 = 0.f;                                             \n"
#endif
  "                                                                        \n"
  "  float4 gridspacing_u4 = { 0.f, 1.f, 2.f, 3.f };                       \n"
  "  gridspacing_u4 *= gridspacing * V4BLOCKSIZEX;                         \n"
  "                                                                        \n"
  "  int atomid;                                                           \n"
  "  for (atomid=0; atomid<numatoms; atomid++) {                           \n"
  "    float dy = coory - atominfo[atomid].y;                              \n"
  "    float dyz2 = (dy * dy) + atominfo[atomid].z;                        \n"
  "                                                                        \n"
  "    float4 dx = gridspacing_u4 + (coorx - atominfo[atomid].x);          \n"
  "    energyvalx += atominfo[atomid].w * native_rsqrt(dx*dx + dyz2);      \n"
#if V4UNROLLX == 8
  "    dx += (4.0f * V4BLOCKSIZEX);                                        \n"
  "    energyvalx2 += atominfo[atomid].w * native_rsqrt(dx*dx + dyz2);     \n"
#endif
  "  }                                                                     \n"
  "                                                                        \n"
  "  energy[outaddr               ] += energyvalx.x;                       \n"
  "  energy[outaddr+1*V4BLOCKSIZEX] += energyvalx.y;                       \n"
  "  energy[outaddr+2*V4BLOCKSIZEX] += energyvalx.z;                       \n"
  "  energy[outaddr+3*V4BLOCKSIZEX] += energyvalx.w;                       \n"
#if V4UNROLLX == 8
  "  energy[outaddr+4*V4BLOCKSIZEX] += energyvalx2.x;                      \n"
  "  energy[outaddr+5*V4BLOCKSIZEX] += energyvalx2.y;                      \n"
  "  energy[outaddr+6*V4BLOCKSIZEX] += energyvalx2.z;                      \n"
  "  energy[outaddr+7*V4BLOCKSIZEX] += energyvalx2.w;                      \n"
#endif
  "}                                                                       \n"
  "                                                                        \n";


// required GPU array padding to match thread block size
// XXX note: this code requires block size dimensions to be a power of two
#define TILESIZEX BLOCKSIZEX*UNROLLX
#define TILESIZEY BLOCKSIZEY*UNROLLY
#define GPU_X_ALIGNMASK (TILESIZEX - 1)
#define GPU_Y_ALIGNMASK (TILESIZEY - 1)

#define V4TILESIZEX V4BLOCKSIZEX*V4UNROLLX
#define V4TILESIZEY V4BLOCKSIZEY*V4UNROLLY
#define V4GPU_X_ALIGNMASK (V4TILESIZEX - 1)
#define V4GPU_Y_ALIGNMASK (V4TILESIZEY - 1)

static int copyatomstoconstbuf(cl_command_queue clcmdq, cl_mem datominfo,
                        const float *atoms, int count, float zplane) {
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

  cl_int clerr = CL_SUCCESS;
  clerr = clEnqueueWriteBuffer(clcmdq, datominfo, CL_TRUE, 0, count * sizeof(cl_float4), (void *) atompre, 0, NULL, NULL);
//  CLERR

  return 0;
}


int vmd_opencl_vol_cpotential(long int natoms, float* atoms, float* grideners,
                            long int numplane, long int numcol, long int numpt, 
                            float gridspacing) {
  enthrparms parms;
  wkf_timerhandle globaltimer;
  double totalruntime;
  int rc=0;
  int deviceCount = 1; // hard coded for now
  int numprocs = 1; // hard coded for now

  /* take the lesser of the number of CPUs and GPUs */
  /* and execute that many threads                  */
  if (deviceCount < numprocs) {
    numprocs = deviceCount;
  }

  printf("Using %d OpenCL devices\n", numprocs);
  int usevec4=0;
  if (getenv("VMDDCSVEC4")!=NULL)
    usevec4=1;

  if (usevec4) {
    printf("OpenCL padded grid size: %ld x %ld x %ld\n", 
      (numpt  + V4GPU_X_ALIGNMASK) & ~(V4GPU_X_ALIGNMASK),
      (numcol + V4GPU_Y_ALIGNMASK) & ~(V4GPU_Y_ALIGNMASK),
      numplane);
  } else {
    printf("OpenCL padded grid size: %ld x %ld x %ld\n", 
      (numpt  + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK),
      (numcol + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK),
      numplane);
  }

  parms.atoms = atoms;
  parms.grideners = grideners;
  parms.numplane = numplane;
  parms.numcol = numcol;
  parms.numpt = numpt;
  parms.natoms = natoms;
  parms.gridspacing = gridspacing;

  globaltimer = wkf_timer_create();
  wkf_timer_start(globaltimer);

  /* spawn child threads to do the work */
  wkf_tasktile_t tile;
  tile.start=0;
  tile.end=numplane;
  rc = wkf_threadlaunch(numprocs, &parms, openclenergythread, &tile);

  // Measure GFLOPS
  wkf_timer_stop(globaltimer);
  totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  if (!rc) {
    double atomevalssec = ((double) numplane * numcol * numpt * natoms) / (totalruntime * 1000000000.0);
    printf("  %g billion atom evals/second, %g GFLOPS\n",
           atomevalssec, atomevalssec * FLOPSPERATOMEVAL);
  } else {
    msgWarn << "An OpenCL device encountered an unrecoverable error." << sendmsg;
    msgWarn << "Calculation will continue using the main CPU." << sendmsg;
  }
  return rc;
}


cl_program vmd_opencl_compile_volcpotential_pgm(cl_context clctx, cl_device_id *cldevs, int &clerr) {
  cl_program clpgm = NULL;

  clpgm = clCreateProgramWithSource(clctx, 1, &clenergysrc, NULL, &clerr);
  CLERR

  char clcompileflags[4096];
  sprintf(clcompileflags,
          "-DUNROLLX=%d -DUNROLLY=%d -DBLOCKSIZEX=%d -DBLOCKSIZEY=%d -DBLOCKSIZE=%d "
          "-DV4UNROLLX=%d -DV4UNROLLY=%d -DV4BLOCKSIZEX=%d -DV4BLOCKSIZEY=%d -DV4BLOCKSIZE=%d "
          "-cl-fast-relaxed-math -cl-single-precision-constant -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros",
          UNROLLX, UNROLLY, BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZE,
          V4UNROLLX, V4UNROLLY, V4BLOCKSIZEX, V4BLOCKSIZEY, V4BLOCKSIZE);

  clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL);
  if (clerr != CL_SUCCESS)
    printf("  compilation failed!\n");

  if (cldevs) {
    char buildlog[8192];
    size_t len=0;
    clerr = clGetProgramBuildInfo(clpgm, cldevs[0], CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, &len);
    if (len > 1) {
      printf("OpenCL compilation log:\n");
      printf("  '%s'\n", buildlog);
    }
    CLERR
  }

  return clpgm;
}



static void * openclenergythread(void *voidparms) {
  size_t volsize[3], Gsz[3], Bsz[3];
  cl_int clerr = CL_SUCCESS;
  cl_mem devenergy = NULL;
  cl_mem datominfo = NULL;
  float *hostenergy = NULL;
  enthrparms *parms = NULL;

  int threadid=0;

  wkf_threadlaunch_getid(voidparms, &threadid, NULL);
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);

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
  double lasttime, totaltime;

printf("OpenCL worker[%d] initializing...\n", threadid);
  cl_platform_id clplatid = vmd_cl_get_platform_index(0);
  cl_context_properties clctxprops[] = {(cl_context_properties) CL_CONTEXT_PLATFORM, (cl_context_properties) clplatid, (cl_context_properties) 0};
#if 0
  // 
  // On the IBM "Blue Drop" Power 775 supercomputer, there are no GPUs, but
  // by using OpenCL on the CPU device type, we can better exploit the 
  // vector units.  The final NSF/NCSA Blue Waters machine ended up being
  // a Cray XE6/XK7, so this code isn't relevant for production use 
  // currently, but there may be other cases where this strategy will 
  // be useful in the future.
  // 
  cl_context clctx = clCreateContextFromType(clctxprops, CL_DEVICE_TYPE_CPU, NULL, NULL, &clerr);
#else
  cl_context clctx = clCreateContextFromType(clctxprops, CL_DEVICE_TYPE_GPU, NULL, NULL, &clerr);
#endif
  CLERR

  size_t parmsz;
  clerr |= clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);
  CLERR

  cl_device_id* cldevs = (cl_device_id *) malloc(parmsz);
  clerr |= clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);
  CLERR

  cl_command_queue clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr);
  CLERR

  cl_program clpgm = vmd_opencl_compile_volcpotential_pgm(clctx, cldevs, clerr);
  CLERR

  cl_kernel clenergy = clCreateKernel(clpgm, "clenergy", &clerr);
  cl_kernel clenergyvec4 = clCreateKernel(clpgm, "clenergy_vec4", &clerr);
  CLERR
printf("OpenCL worker[%d] ready.\n", threadid);

  // setup OpenCL grid and block sizes
  int usevec4=0;
  if (getenv("VMDDCSVEC4")!=NULL)
    usevec4=1;

  if (usevec4) {
    // setup energy grid size, padding arrays for peak GPU memory performance
    volsize[0] = (numpt  + V4GPU_X_ALIGNMASK) & ~(V4GPU_X_ALIGNMASK);
    volsize[1] = (numcol + V4GPU_Y_ALIGNMASK) & ~(V4GPU_Y_ALIGNMASK);
    volsize[2] = 1;      // we only do one plane at a time
    Bsz[0] = V4BLOCKSIZEX;
    Bsz[1] = V4BLOCKSIZEY;
    Bsz[2] = 1;
    Gsz[0] = volsize[0] / V4UNROLLX;
    Gsz[1] = volsize[1] / V4UNROLLY;
    Gsz[2] = volsize[2];
  } else {
    // setup energy grid size, padding arrays for peak GPU memory performance
    volsize[0] = (numpt  + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK);
    volsize[1] = (numcol + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK);
    volsize[2] = 1;      // we only do one plane at a time
    Bsz[0] = BLOCKSIZEX;
    Bsz[1] = BLOCKSIZEY;
    Bsz[2] = 1;
    Gsz[0] = volsize[0] / UNROLLX;
    Gsz[1] = volsize[1] / UNROLLY;
    Gsz[2] = volsize[2];
  }


  int volmemsz = sizeof(float) * volsize[0] * volsize[1] * volsize[2];

  printf("Thread %d started for OpenCL device %d...\n", threadid, threadid);
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);
  wkfmsgtimer * msgt = wkf_msg_timer_create(5);

  // Allocate DMA buffers with some extra padding at the end so that 
  // multiple OpenCL devices aren't DMAing too close to each other, for NUMA..
#define DMABUFPADSIZE (32 * 1024)

  hostenergy = (float *) malloc(volmemsz); // allocate working buffer

  devenergy = clCreateBuffer(clctx, CL_MEM_READ_WRITE, volmemsz, NULL, NULL);
  CLERR

  datominfo = clCreateBuffer(clctx, CL_MEM_READ_ONLY, MAXATOMS * sizeof(cl_float4), NULL, NULL);
  CLERR


  // For each point in the cube...
  int iterations=0;
  int computedplanes=0;
  wkf_tasktile_t tile;
  while (wkf_threadlaunch_next_tile(voidparms, 1, &tile) != WKF_SCHED_DONE) {
    int k;
    for (k=tile.start; k<tile.end; k++) {
      int y;
      int atomstart;
      float zplane = k * (float) gridspacing;
      computedplanes++; // track work done by this GPU for progress reporting
 
      // Copy energy grid into GPU 16-element padded input
      for (y=0; y<numcol; y++) {
        long eneraddr = k*numcol*numpt + y*numpt;
        memcpy(&hostenergy[y*volsize[0]], &grideners[eneraddr], numpt * sizeof(float));
      }

      // Copy the Host input data to the GPU..
      clEnqueueWriteBuffer(clcmdq, devenergy, CL_TRUE, 0, volmemsz, hostenergy, 0, NULL, NULL);
      CLERR // check and clear any existing errors

      lasttime = wkf_timer_timenow(timer);
      for (atomstart=0; atomstart<natoms; atomstart+=MAXATOMS) {
        iterations++;
        int runatoms;
        int atomsremaining = natoms - atomstart;
        if (atomsremaining > MAXATOMS)
          runatoms = MAXATOMS;
        else
          runatoms = atomsremaining;

        // copy the next group of atoms to the GPU
        if (copyatomstoconstbuf(clcmdq, datominfo,
                                atoms + 4*atomstart, runatoms, zplane))
          return NULL;

        cl_kernel clkern;
        if (usevec4)
          clkern = clenergyvec4;
        else
          clkern = clenergy;

        // RUN the kernel...
        clerr |= clSetKernelArg(clkern, 0, sizeof(int), &runatoms);
        clerr |= clSetKernelArg(clkern, 1, sizeof(float), &gridspacing);
        clerr |= clSetKernelArg(clkern, 2, sizeof(cl_mem), &devenergy);
        clerr |= clSetKernelArg(clkern, 3, sizeof(cl_mem), &datominfo);
        CLERR
        cl_event event;
#if 0
printf("Gsz: %ld %ld %ld  Bsz: %ld %ld %ld\n",
  Gsz[0], Gsz[1], Gsz[2], Bsz[0], Bsz[1], Bsz[2]);
#endif
        clerr |= clEnqueueNDRangeKernel(clcmdq, clkern, 2, NULL, Gsz, Bsz, 0, NULL, &event);
        CLERR

        clerr |= clWaitForEvents(1, &event);
        clerr |= clReleaseEvent(event);
        CLERR // check and clear any existing errors
      }
      clFinish(clcmdq);

      // Copy the GPU output data back to the host and use/store it..
      clEnqueueReadBuffer(clcmdq, devenergy, CL_TRUE, 0, volmemsz, hostenergy,
                           0, NULL, NULL);

      CLERR // check and clear any existing errors

      // Copy GPU blocksize padded array back down to the original size
      for (y=0; y<numcol; y++) {
        long eneraddr = k*numcol*numpt + y*numpt;
        memcpy(&grideners[eneraddr], &hostenergy[y*volsize[0]], numpt * sizeof(float));
      }
 
      totaltime = wkf_timer_timenow(timer);
      if (wkf_msg_timer_timeout(msgt)) {
        // XXX: we have to use printf here as msgInfo is not thread-safe yet.
        printf("thread[%d] plane %d/%ld (%d computed) time %.2f, elapsed %.1f, est. total: %.1f\n",
               threadid, k, numplane, computedplanes,
               totaltime - lasttime, totaltime,
               totaltime * numplane / (k+1));
      }
    }
  }

  wkf_timer_destroy(timer); // free timer
  wkf_msg_timer_destroy(msgt); // free timer
  free(hostenergy);    // free working buffer

printf("destroying context, programs, etc\n");
  clReleaseMemObject(devenergy);
  clReleaseMemObject(datominfo);
  clReleaseKernel(clenergy);
  clReleaseKernel(clenergyvec4);
  clReleaseProgram(clpgm);
  clReleaseCommandQueue(clcmdq);
  clReleaseContext(clctx);
printf("done.\n");

  CLERR // check and clear any existing errors

  return NULL;
}




