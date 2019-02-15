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
 *      $RCSfile: VolCPotential.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.30 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************/
/*
 * Calculate a coulombic potential map
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "config.h"         // force recompile when configuration changes
#include "utilities.h"
#include "Inform.h"
#include "WKFThreads.h"
#include "WKFUtils.h"
#include "VolCPotential.h" 
#if defined(VMDCUDA)
#include "CUDAKernels.h"
#endif
#if defined(VMDOPENCL)
#include "OpenCLKernels.h"
#endif

typedef struct {
  float* atoms;
  float* grideners;
  long int numplane;
  long int numcol;
  long int numpt;
  long int natoms;
  float gridspacing;
} enthrparms;

#if 1 || defined(__INTEL_COMPILER)

#define FLOPSPERATOMEVAL 6.0
extern "C" void * energythread(void *voidparms) {
  int threadid;
  enthrparms *parms = NULL;
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);
  wkf_threadlaunch_getid(voidparms, &threadid, NULL);

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
  int i, j, k, n;
  double lasttime, totaltime;

  /* Calculate the coulombic energy at each grid point from each atom
   * This is by far the most time consuming part of the process
   * We iterate over z,y,x, and then atoms
   */

  printf("thread %d started...\n", threadid);
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);
  wkfmsgtimer *msgt = wkf_msg_timer_create(5);

  // Holds atom x, dy**2+dz**2, and atom q as x/r/q/x/r/q...
  float * xrq = (float *) malloc(3*natoms * sizeof(float)); 
  int maxn = natoms * 3;

  // For each point in the cube...
  wkf_tasktile_t tile;
  while (wkf_threadlaunch_next_tile(voidparms, 2, &tile) != WKF_SCHED_DONE) {
    for (k=tile.start; k<tile.end; k++) { 
      const float z = gridspacing * (float) k;
      lasttime = wkf_timer_timenow(timer);
      for (j=0; j<numcol; j++) {
        const float y = gridspacing * (float) j;
        long int voxaddr = numcol*numpt*k + numpt*j;

        // Prebuild a table of dy and dz values on a per atom basis
        for (n=0; n<natoms; n++) {
          int addr3 = n*3;
          int addr4 = n*4;
          float dy = y - atoms[addr4 + 1];
          float dz = z - atoms[addr4 + 2];
          xrq[addr3    ] = atoms[addr4];
          xrq[addr3 + 1] = dz*dz + dy*dy;
          xrq[addr3 + 2] = atoms[addr4 + 3];
        }

// help the vectorizer make reasonable decisions
#if defined(__INTEL_COMPILER)
#pragma vector always 
#endif
        /* walk through more than one grid point at a time */
        for (i=0; i<numpt; i+=8) {
          float energy1 = 0.0f;           /* Energy of first grid point */
          float energy2 = 0.0f;           /* Energy of second grid point */
          float energy3 = 0.0f;           /* Energy of third grid point */
          float energy4 = 0.0f;           /* Energy of fourth grid point */
          float energy5 = 0.0f;           /* Energy of fourth grid point */
          float energy6 = 0.0f;           /* Energy of fourth grid point */
          float energy7 = 0.0f;           /* Energy of fourth grid point */
          float energy8 = 0.0f;           /* Energy of fourth grid point */

          const float x = gridspacing * (float) i;

// help the vectorizer make reasonable decisions
#if defined(__INTEL_COMPILER)
#pragma vector always 
#endif
          /* Calculate the interaction with each atom */
          /* SSE allows simultaneous calculations of  */
          /* multiple iterations                      */
          /* 6 flops per grid point */
          for (n=0; n<maxn; n+=3) {
            float dy2pdz2 = xrq[n + 1];
            float q = xrq[n + 2];
  
            float dx1 = x - xrq[n];
            energy1 += q / sqrtf(dx1*dx1 + dy2pdz2);
  
            float dx2 = dx1 + gridspacing;
            energy2 += q / sqrtf(dx2*dx2 + dy2pdz2);
  
            float dx3 = dx2 + gridspacing;
            energy3 += q / sqrtf(dx3*dx3 + dy2pdz2);
  
            float dx4 = dx3 + gridspacing;
            energy4 += q / sqrtf(dx4*dx4 + dy2pdz2);
  
            float dx5 = dx4 + gridspacing;
            energy5 += q / sqrtf(dx5*dx5 + dy2pdz2);
  
            float dx6 = dx5 + gridspacing;
            energy6 += q / sqrtf(dx6*dx6 + dy2pdz2);
  
            float dx7 = dx6 + gridspacing;
            energy7 += q / sqrtf(dx7*dx7 + dy2pdz2);
  
            float dx8 = dx7 + gridspacing;
            energy8 += q / sqrtf(dx8*dx8 + dy2pdz2);
          }

          grideners[voxaddr + i] = energy1;
          if (i+1 < numpt)
            grideners[voxaddr + i + 1] = energy2;
          if (i+2 < numpt)
            grideners[voxaddr + i + 2] = energy3;
          if (i+3 < numpt)
            grideners[voxaddr + i + 3] = energy4;
          if (i+4 < numpt)
            grideners[voxaddr + i + 4] = energy5;
          if (i+5 < numpt)
            grideners[voxaddr + i + 5] = energy6;
          if (i+6 < numpt)
            grideners[voxaddr + i + 6] = energy7;
          if (i+7 < numpt)
            grideners[voxaddr + i + 7] = energy8;
        }
      }
      totaltime = wkf_timer_timenow(timer);

      if (wkf_msg_timer_timeout(msgt)) {
        // XXX: we have to use printf here as msgInfo is not thread-safe yet.
        printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n",
               threadid, k, numplane,
               totaltime - lasttime, totaltime, 
               totaltime * numplane / (k+1));
      }
    }
  }

  wkf_timer_destroy(timer);
  wkf_msg_timer_destroy(msgt);
  free(xrq);

  return NULL;
}

#else

#define FLOPSPERATOMEVAL 6.0
extern "C" void * energythread(void *voidparms) {
  int threadid;
  enthrparms *parms = NULL;
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);
  wkf_threadlaunch_getid(voidparms, &threadid, NULL);

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
  int i, j, k, n;
  double lasttime, totaltime;

  /* Calculate the coulombic energy at each grid point from each atom
   * This is by far the most time consuming part of the process
   * We iterate over z,y,x, and then atoms
   */

  printf("thread %d started...\n", threadid);
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);
  wkfmsgtimer *msgt = wkf_msg_timer_create(5);

  // Holds atom x, dy**2+dz**2, and atom q as x/r/q/x/r/q...
  float * xrq = (float *) malloc(3*natoms * sizeof(float)); 
  int maxn = natoms * 3;

  // For each point in the cube...
  while (!wkf_threadlaunch_next(voidparms, &k)) {
    const float z = gridspacing * (float) k;
    lasttime = wkf_timer_timenow(timer);
    for (j=0; j<numcol; j++) {
      const float y = gridspacing * (float) j;
      long int voxaddr = numcol*numpt*k + numpt*j;

      // Prebuild a table of dy and dz values on a per atom basis
      for (n=0; n<natoms; n++) {
        int addr3 = n*3;
        int addr4 = n*4;
        float dy = y - atoms[addr4 + 1];
        float dz = z - atoms[addr4 + 2];
        xrq[addr3    ] = atoms[addr4];
        xrq[addr3 + 1] = dz*dz + dy*dy;
        xrq[addr3 + 2] = atoms[addr4 + 3];
      }

#if defined(__INTEL_COMPILER)
// help the vectorizer make reasonable decisions (used prime to keep it honest)
#pragma loop count(1009)
#endif
      for (i=0; i<numpt; i++) {
        float energy = grideners[voxaddr + i]; // Energy at current grid point
        const float x = gridspacing * (float) i;

#if defined(__INTEL_COMPILER)
// help the vectorizer make reasonable decisions
#pragma vector always 
#endif
        // Calculate the interaction with each atom
        for (n=0; n<maxn; n+=3) {
          float dx = x - xrq[n];
          energy += xrq[n + 2] / sqrtf(dx*dx + xrq[n + 1]);
        }
        grideners[voxaddr + i] = energy;
      }
    }
    totaltime = wkf_timer_timenow(timer);

    if (wkf_msg_timer_timeout(msgt)) {
      // XXX: we have to use printf here as msgInfo is not thread-safe yet.
      printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n",
             threadid, k, numplane,
             totaltime - lasttime, totaltime, 
             totaltime * numplane / (k+1));
    }
  }

  wkf_timer_destroy(timer);
  wkf_msg_timer_destroy(msgt);
  free(xrq);

  return NULL;
}

#endif


static int vol_cpotential_cpu(long int natoms, float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, float gridspacing) {
  int rc=0;
  enthrparms parms;
  wkf_timerhandle globaltimer;
  double totalruntime;

#if defined(VMDTHREADS)
  int numprocs = wkf_thread_numprocessors();
#else
  int numprocs = 1;
#endif

  printf("Using %d %s\n", numprocs, ((numprocs > 1) ? "CPUs" : "CPU"));

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
  rc = wkf_threadlaunch(numprocs, &parms, energythread, &tile);

  // Measure GFLOPS
  wkf_timer_stop(globaltimer);
  totalruntime = wkf_timer_time(globaltimer);
  wkf_timer_destroy(globaltimer);

  if (!rc) {
    double atomevalssec = ((double) numplane * numcol * numpt * natoms) / (totalruntime * 1000000000.0);
    printf("  %g billion atom evals/second, %g GFLOPS\n",
           atomevalssec, atomevalssec * FLOPSPERATOMEVAL);
  } else {
    msgWarn << "Encountered an unrecoverable error, calculation terminated." << sendmsg;
  }

  return rc;
}


int vol_cpotential(long int natoms, float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, float gridspacing) {
  int rc = -1; // init rc value to indicate we haven't run yet
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

#if defined(VMDCUDA)
  if (!getenv("VMDNOCUDA")) {
    rc = vmd_cuda_vol_cpotential(natoms, atoms, grideners, 
                                 numplane, numcol, numpt, gridspacing);
  }
#endif
#if defined(VMDOPENCL)
  if ((rc != 0) && !getenv("VMDNOOPENCL")) {
    rc = vmd_opencl_vol_cpotential(natoms, atoms, grideners, 
                                   numplane, numcol, numpt, gridspacing);
  }
#endif

  // if we tried to run on the GPU and failed, or we haven't run yet,
  // run on the CPU
  if (rc != 0)
    rc = vol_cpotential_cpu(natoms, atoms, grideners, 
                            numplane, numcol, numpt, gridspacing);

  double totaltime = wkf_timer_timenow(timer);
  wkf_timer_destroy(timer);

  msgInfo << "Coulombic potential map calculation complete: "
          << totaltime << " seconds" << sendmsg;
   
  return rc;
}



