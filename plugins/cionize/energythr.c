#include "energythr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include "util.h"    /* timer code taken from Tachyon */
#include "threads.h" /* threads code taken from Tachyon */
#include "cionize_enermethods.h" /* definitions for energy calculation methods */

typedef struct {
  int threadid;
  int threadcount;
  float* atoms;
  float* grideners;
  long int numplane;
  long int numcol;
  long int numpt;
  long int natoms;
  float gridspacing;
  unsigned char* excludepos;
  float ddd;
} enthrparms;

/*
 * thread function prototypes
 */
static void * energythread_singleprec(void *); /* standard version */
static void * energythread_doubleprec(void *); /* double precision */
static void * energythread_ddd(void *);        /* dielectric */

int calc_grid_energies(float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, long int natoms, float gridspacing, unsigned char* excludepos, int maxnumprocs, int calctype, float ddd) {
  int i;
  enthrparms *parms;
  rt_thread_t * threads;
  void* (*kernel_thr) (void*);

#if defined(THR)
  int numprocs;
  int availprocs = rt_thread_numprocessors();
  if (maxnumprocs <= availprocs) {
    numprocs = maxnumprocs;
  } else {
    numprocs = availprocs;
  }
#else
  int numprocs = 1;
#endif

  printf("calc_grid_energies_excl_jsthr()\n");
  printf("  using %d processors\n", numprocs);  

  switch(calctype) {
    case STANDARD:
      kernel_thr = energythread_singleprec;
      break;
    case DOUBLEPREC:
      kernel_thr = energythread_doubleprec;
      break;
    case DDD:
      kernel_thr = energythread_ddd;
      break;
    default:
      kernel_thr = energythread_singleprec;
  }

  /* allocate array of threads */
  threads = (rt_thread_t *) calloc(numprocs * sizeof(rt_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (enthrparms *) malloc(numprocs * sizeof(enthrparms));
  for (i=0; i<numprocs; i++) {
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;
    parms[i].atoms = atoms;
    parms[i].grideners = grideners;
    parms[i].numplane = numplane;
    parms[i].numcol = numcol;
    parms[i].numpt = numpt;
    parms[i].natoms = natoms;
    parms[i].gridspacing = gridspacing;
    parms[i].excludepos = excludepos;
    parms[i].ddd = ddd;
  }

#if defined(THR)
  /* spawn child threads to do the work */
  for (i=0; i<numprocs; i++) {
    rt_thread_create(&threads[i], kernel_thr, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numprocs; i++) {
    rt_thread_join(threads[i], NULL);
  } 
#else
  /* single thread does all of the work */
  kernel_thr((void *) &parms[0]);
#endif

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}


/*
 * Standard single-precision version, with loops optimized for 
 * SSE acceleration with the Intel C/C++ compiler, and a version
 * for all other target platforms.
 */
static void * energythread_singleprec(void *voidparms) {
  enthrparms *parms = (enthrparms *) voidparms;
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
  const unsigned char* excludepos = parms->excludepos;
  const int threadid = parms->threadid;
  const int threadcount = parms->threadcount;

  /* Calculate the coulombic energy at each grid point from each atom
   * This is by far the most time consuming part of the process
   * We iterate over z,y,x, and then atoms
   * This function is the same as the original calc_grid_energies, except
   * that it utilizes the exclusion grid
   */
  int i,j,k,n; /*Loop counters */
  float lasttime, totaltime;

  /* Holds atom x, dy**2+dz**2, and atom q as x/r/q/x/r/q... */
  float * xrq = (float *) malloc(3*natoms * sizeof(float)); 
  int maxn = natoms * 3;

  rt_timerhandle timer = rt_timer_create();
  rt_timer_start(timer);

  printf("thread %d started...\n", threadid);

  /* For each point in the cube... */
  for (k=threadid; k<numplane; k+= threadcount) {
    const float z = gridspacing * (float) k;
    lasttime = rt_timer_timenow(timer);
    for (j=0; j<numcol; j++) {
      const float y = gridspacing * (float) j;
      long int voxaddr = numcol*numpt*k + numpt*j;

      /* Prebuild a table of dy and dz values on a per atom basis */
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
      /* unroll the voxel X loop by 8, reusing atom data several times */
      for (i=0; i<numpt; i+=8) {
          /* Check if we're on an excluded point, and skip it if we are 
           *  if (excludepos[voxaddr + i] == 0) { */
          float energy1 = 0.0f;
          float energy2 = 0.0f;
          float energy3 = 0.0f;
          float energy4 = 0.0f;
          float energy5 = 0.0f;
          float energy6 = 0.0f;
          float energy7 = 0.0f;
          float energy8 = 0.0f;

          const float x = gridspacing * (float) i;
/* help the vectorizer make reasonable decisions */
#pragma vector always 
          /* Calculate the interaction with each atom */
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
          grideners[voxaddr + i] += energy1;
          if (i+1 < numpt)
            grideners[voxaddr + i + 1] += energy2;
          if (i+2 < numpt)
            grideners[voxaddr + i + 2] += energy3;
          if (i+3 < numpt)
            grideners[voxaddr + i + 3] += energy4;
          if (i+4 < numpt)
            grideners[voxaddr + i + 4] += energy5;
          if (i+5 < numpt)
            grideners[voxaddr + i + 5] += energy6;
          if (i+6 < numpt)
            grideners[voxaddr + i + 6] += energy7;
          if (i+7 < numpt)
            grideners[voxaddr + i + 7] += energy8;
          /*} */
      }

      for (i=0; i<numpt; i++) {
        if (excludepos[voxaddr + i] != 0) grideners[voxaddr + i] = 0.f;
      }
#else
      for (i=0; i<numpt; i++) {
          /* Check if we're on an excluded point, and skip it if we are */
        if (excludepos[voxaddr + i] == 0) {
          float energy = 0.0f;
          const float x = gridspacing * (float) i;
          /* Calculate the interaction with each atom */
          /* 6 FLOPS per atom evaluation */
          for (n=0; n<maxn; n+=3) {
            float dx = x - xrq[n];
            energy += xrq[n + 2] / sqrtf(dx*dx + xrq[n + 1]);
          }
          grideners[voxaddr + i] += energy; /* Add energy to current grid point */
        }
      }
#endif

    }
    totaltime = rt_timer_timenow(timer);
    printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n",
           threadid, k, numplane,

           totaltime - lasttime, totaltime, 
           totaltime * numplane / (k+1));
  }

  rt_timer_destroy(timer);
  free(xrq);

  return NULL;
}



/*
 * Double precision summation version
 */
static void * energythread_doubleprec(void *voidparms) {
  enthrparms *parms = (enthrparms *) voidparms;
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
  const unsigned char* excludepos = parms->excludepos;
  const int threadid = parms->threadid;
  const int threadcount = parms->threadcount;

  /* Calculate the coulombic energy at each grid point from each atom
   * This is by far the most time consuming part of the process
   * We iterate over z,y,x, and then atoms
   * This function is the same as the original calc_grid_energies, except
   * that it utilizes the exclusion grid
   */
  int i,j,k,n; /*Loop counters */
  float lasttime, totaltime;

  /* Holds atom x, dy**2+dz**2, and atom q as x/r/q/x/r/q... */
  double * xrq = (double *) malloc(3*natoms * sizeof(double)); 
  int maxn = natoms * 3;

  rt_timerhandle timer = rt_timer_create();
  rt_timer_start(timer);

  printf("thread %d started...\n", threadid);

  /* For each point in the cube... */
  for (k=threadid; k<numplane; k+= threadcount) {
    const double z = gridspacing * (double) k;
    lasttime = rt_timer_timenow(timer);
    for (j=0; j<numcol; j++) {
      const double y = gridspacing * (double) j;
      long int voxaddr = numcol*numpt*k + numpt*j;

      /* Prebuild a table of dy and dz values on a per atom basis */
      for (n=0; n<natoms; n++) {
        int addr3 = n*3;
        int addr4 = n*4;
        double dy = y - atoms[addr4 + 1];
        double dz = z - atoms[addr4 + 2];
        xrq[addr3    ] = atoms[addr4];
        xrq[addr3 + 1] = dz*dz + dy*dy;
        xrq[addr3 + 2] = atoms[addr4 + 3];
      }

      for (i=0; i<numpt; i++) {
          /* Check if we're on an excluded point, and skip it if we are */
        if (excludepos[voxaddr + i] == 0) {
          double energy = 0.0f;
          const double x = gridspacing * (double) i;
          /* Calculate the interaction with each atom */
          /* 6 FLOPS per atom evaluation */
          for (n=0; n<maxn; n+=3) {
            double dx = x - xrq[n];
            energy += xrq[n + 2] / sqrtf(dx*dx + xrq[n + 1]);
          }
          grideners[voxaddr + i] += energy; /* Add energy to current grid point */
        }
      }
    }
    totaltime = rt_timer_timenow(timer);
    printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n",
           threadid, k, numplane,

           totaltime - lasttime, totaltime, 
           totaltime * numplane / (k+1));
  }

  rt_timer_destroy(timer);
  free(xrq);

  return NULL;
}


static void * energythread_ddd(void *voidparms) {
  enthrparms *parms = (enthrparms *) voidparms;
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
  const unsigned char* excludepos = parms->excludepos;
  const int threadid = parms->threadid;
  const int threadcount = parms->threadcount;
  const float ddd = parms->ddd;

  /* Calculate the coulombic energy at each grid point from each atom
   * This is by far the most time consuming part of the process
   * We iterate over z,y,x, and then atoms
   * This function is the same as the original calc_grid_energies, except
   * that it utilizes the exclusion grid
   */
  int i,j,k,n; /*Loop counters */
  float lasttime, totaltime;

  /* Holds atom x, dy**2+dz**2, and atom q as x/r/q/x/r/q... */
  float * xrq = (float *) malloc(3*natoms * sizeof(float)); 
  int maxn = natoms * 3;

  rt_timerhandle timer = rt_timer_create();
  rt_timer_start(timer);

  printf("thread %d started...\n", threadid);

  /* For each point in the cube... */
  for (k=threadid; k<numplane; k+= threadcount) {
    const float z = gridspacing * (float) k;
    lasttime = rt_timer_timenow(timer);
    for (j=0; j<numcol; j++) {
      const float y = gridspacing * (float) j;
      long int voxaddr = numcol*numpt*k + numpt*j;

      /* Prebuild a table of dy and dz values on a per atom basis */
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
/* help the vectorizer make reasonable decisions (used prime to keep it honest) */
#pragma loop count(1009)
#endif
      for (i=0; i<numpt; i++) {
          /* Check if we're on an excluded point, and skip it if we are */
        if (excludepos[voxaddr + i] == 0) {
          float energy = grideners[voxaddr + i]; /* Energy at current grid point */
          const float x = gridspacing * (float) i;

#if defined(__INTEL_COMPILER)
/* help the vectorizer make reasonable decisions */
#pragma vector always 
#endif
          /* Calculate the interaction with each atom */
          for (n=0; n<maxn; n+=3) {
            float dx;
            float invr2;
            dx = x - xrq[n];
            invr2 = 1.0 / (ddd * (dx*dx + xrq[n+1]));
            energy += xrq[n + 2] * invr2;
          }
          grideners[voxaddr + i] = energy;
        }
      }
    }
    totaltime = rt_timer_timenow(timer);
    printf("thread[%d] plane %d/%ld time %.2f, elapsed %.1f, est. total: %.1f\n",
           threadid, k, numplane,

           totaltime - lasttime, totaltime, 
           totaltime * numplane / (k+1));
  }

  rt_timer_destroy(timer);
  free(xrq);

  return NULL;
}
