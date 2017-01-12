/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot.c - entry point for computing grid energies using mgpot
 */

#include <stdio.h>
#include "mgpot.h"
#include "mgpot_defn.h"
#if defined(CUDA)
#include "mgpot_cuda.h"
#endif

#include "util.h"    /* timer code taken from Tachyon */
#include "threads.h" /* threads code taken from Tachyon */

#if defined(MGPOT_SPACING_0_2)
#define CUTOFF  9.6f
#else
#define CUTOFF  12.f
#endif


typedef struct mgpot_thr_parms_t {
  int threadid;
  int threadcount;
  int deviceid;
  int error;
  Mgpot *mgpot;
} mgpot_thr_parms;


static void *mgpot_thread(void *vparams);


int calc_grid_energies_excl_mgpot(float* atoms, float* grideners,
    long int numplane, long int numcol, long int numpt, long int natoms,
    float gridspacing, unsigned char* excludepos, int maxnumprocs,
    int emethod) {

  float h;
  float a;
  int interp;
  int split;
  int scalexp, scaling;
  long int nx, ny, nz;
  float xmax, ymax, zmax;
  Mgpot mg;
  long int i;

#if defined(THR)
  int availprocs;
#endif
  int numprocs;
  int deviceid_index;
  rt_thread_t *threads;
  mgpot_thr_parms *parms;

  rt_timerhandle timer = rt_timer_create();
  float totaltime=0, lasttime, elapsedtime;

#if defined(CUDA)
  /* query devices before we start timing so we don't wait for X... */
  int availdevices = 0;
  if (MGPOTUSECUDA(emethod)) 
    availdevices = mgpot_cuda_device_list();
#endif

  /* start timer for setup */
  rt_timer_start(timer);
  lasttime = rt_timer_timenow(timer);

  /* reset grideners */
  memset(grideners, 0, numplane*numcol*numpt*sizeof(float));

  /*
   * grid used by mgpot needs spacing h >= 2,
   * determine grid scaling factor:  scaling * h_ionize = h_mgpot
   */
#if defined(MGPOT_SPACING_0_2)
  printf("setting grid spacing to 0.2\n");
  gridspacing = 0.2;
  h = 1.6;
  scalexp = 3;
#else
  scalexp = 0;
  h = gridspacing;
  while (h < 2) {
    h *= 2;
    scalexp++;
  }
#endif
  scaling = (1 << scalexp);

  DEBUG( printf("h=%g\n" "scalexp=%d\n", h, scalexp); )

  /*
   * need enough mgpot grid points to cover space:
   *   nx = ceiling(numpt / scaling), etc.
   */
  nx = numpt / scaling + (numpt % scaling ? 1 : 0);
  ny = numcol / scaling + (numcol % scaling ? 1 : 0);
  nz = numplane / scaling + (numplane % scaling ? 1 : 0);
  DEBUG( printf("nx=%ld  ny=%ld  nz=%ld\n", nx, ny, nz); );

  /*
   * make sure that atoms are contained within mgpot grid
   * (otherwise mgpot will crash)
   */
  xmax = nx * h;
  ymax = ny * h;
  zmax = nz * h;
  for (i = 0;  i < natoms;  i++) {
    if (atoms[ INDEX_X(i) ] < 0 || atoms[ INDEX_X(i) ] >= xmax
        || atoms[ INDEX_Y(i) ] < 0 || atoms[ INDEX_Y(i) ] >= ymax
        || atoms[ INDEX_Z(i) ] < 0 || atoms[ INDEX_Z(i) ] >= zmax) {
      return ERROR("atom %d is outside of grid\n", i);
    }
  }

  /* use these defaults for other values (for now) */
  a = CUTOFF;
  interp = CUBIC;
  split = TAYLOR2;

#if defined(THR)
  availprocs = rt_thread_numprocessors();
  if (maxnumprocs <= availprocs) {
    numprocs = maxnumprocs;
  }
  else {
    numprocs = availprocs;
  }
#else
  numprocs = 1;
#endif

  if (mgpot_setup(&mg, h, a, nx, ny, nz, scalexp, interp, split,
        atoms, grideners, numplane, numcol, numpt, natoms,
        gridspacing, excludepos, numprocs, emethod)) {
    return ERROR("mgpot_setup() failed\n");
  }

  if (mg.use_cuda) {
#if defined(CUDA)
    if (0==availdevices) {
      return ERROR("no CUDA devices are present\n");
    }
    if (availdevices < numprocs) {
      numprocs = availdevices;
    }
    if (MGPOTDEV(emethod) >= availdevices) {
      return ERROR("unable to use device %d for long-range part\n",
          MGPOTDEV(emethod));
    }
#else
    return ERROR("no compiled support for CUDA\n");
#endif /* CUDA */
  }

  printf("Multilevel summation for electrostatic potential map\n");
  printf("  using %d processors\n", numprocs);

  /* allocate array of threads */
  threads = (rt_thread_t *) calloc(numprocs, sizeof(rt_thread_t));

  /* allocate and initialize array of thread parameters */
  parms = (mgpot_thr_parms *) calloc(numprocs, sizeof(mgpot_thr_parms));
  deviceid_index = 0;
  for (i = 0;  i < numprocs;  i++) {
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;
    if (mg.use_cuda) {
      /* thread 0 always computes long-range part;
       * other device IDs assigned consecutively, skipping over
       * the device ID already assigned to long-range part
       */
      if (0==i) {
        parms[i].deviceid = MGPOTDEV(emethod);  /* for long-range */
      }
      else {
        if (MGPOTDEV(emethod) == deviceid_index) deviceid_index++; /* skip */
        parms[i].deviceid = deviceid_index;  /* for short-range */
        deviceid_index++;
      }
    }
    parms[i].error = 0;
    parms[i].mgpot = &mg;
  }

  elapsedtime = rt_timer_timenow(timer) - lasttime;
  printf("time for setup:             %.2f\n", elapsedtime);
  printf("BENCH_setup:  %.5f\n", elapsedtime);
  totaltime += elapsedtime;

  lasttime = rt_timer_timenow(timer);
#if defined(THR)
  /* spawn child threads to do the work */
  for (i = 0;  i < numprocs;  i++) {
    rt_thread_create(&threads[i], mgpot_thread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i = 0;  i < numprocs;  i++) {
    rt_thread_join(threads[i], NULL);
    if (parms[i].error) {
      return ERROR("thread %d returned an error condition\n", i);
    }
  }
#else
  /* single thread does all of the work */
  mgpot_thread((void *) &parms[0]);
#endif

  /* sum long-range contribution */
  /* XXX this should be timed as well, as it can become noticable */
  /*     for very large grids */
  if (mgpot_longrng_finish(&mg)) {
    return ERROR("mgpot_longrng_finish() failed\n");
  }

  elapsedtime = rt_timer_timenow(timer) - lasttime;
  printf("time for all computation:   %.2f\n", elapsedtime);
  printf("BENCH_computation:  %.5f\n", elapsedtime);
  totaltime += elapsedtime;
  printf("total time:                 %.2f\n", totaltime);
  printf("BENCH_msm_total:  %.5f\n", totaltime);

  if (mgpot_cleanup(&mg)) {
    return ERROR("mgpot_cleanup() failed\n");
  }

  /* free thread parms */
  free(parms);
  free(threads);

  rt_timer_destroy(timer);

  return 0;
}


void *mgpot_thread(void *vparms)
{
  mgpot_thr_parms *parms = (mgpot_thr_parms *) vparms;

  const int threadid = parms->threadid;
  const int threadcount = parms->threadcount;
  Mgpot *mg = parms->mgpot;

  /*
   * For 1 thread, do long-range then short-range.
   *
   * For multiple threads, assign threadid 0 to long-range part
   * and concurrently compute the short-range part equally
   * partitioned between the remaining threads.
   */
  const int do_longrng = (threadid == 0);
  const int do_shortrng = (threadcount == 1 || threadid > 0);
  const int do_other = 0;
#if 0
  const int do_longrng = (threadcount == 1 || threadid == 1);
  const int do_shortrng = (threadid == 0);
  const int do_other = (threadid > 1);
#endif

  rt_timerhandle timer = rt_timer_create();
  float lasttime, elapsedtime;

  rt_timer_start(timer);

#if defined(CUDA)
  if (mg->use_cuda) {
    /* setup cuda devices */
    const int deviceid = parms->deviceid;
    printf("setting up CUDA device %d for %s\n", deviceid,
        ((do_longrng && do_shortrng) ? "long- and short-range parts" :
         (do_longrng ? "long-range part" :
          (do_shortrng ? "short-range part" : "NOTHING"))));
    if (mgpot_cuda_device_set(/* 0 */ parms->deviceid /* 1 */)) {
      ERROR("mgpot_cuda_device_set() failed\n");
      parms->error = FAIL;
      return NULL;
    }
  }
#endif

  /*
   * NOTE:  In multi-threaded mode, the grideners are summed into a
   * separate "Mgpot::grideners_longrng" array and later summed into
   * grideners after the threads join.  So there are no race conditions.
   */
  if (do_longrng) {
#if defined(CUDA)
    if (mg->use_cuda) {  /* setup cuda for long-range part */
      if (mgpot_cuda_setup_longrng(mg)) {
        ERROR("mgpot_cuda_setup_longrng() failed\n");
        parms->error = FAIL;
        return NULL;
      }
    }
#endif
    lasttime = rt_timer_timenow(timer);
    if (mgpot_longrng(mg)) {
      ERROR("mgpot_longrng() failed\n");
      parms->error = FAIL;
      return NULL;
    }
    elapsedtime = rt_timer_timenow(timer) - lasttime;
#if defined(CUDA)
    if (mg->use_cuda) {  /* cleanup cuda for long-range part */
      if (mgpot_cuda_cleanup_longrng(mg)) {
        ERROR("mgpot_cuda_cleanup_longrng() failed\n");
        parms->error = FAIL;
        return NULL;
      }
    }
#endif
    printf("thread[%d]:  time for long-range part:   %.2f\n",
        threadid, elapsedtime);
  }

  if (do_shortrng) {
#if defined(CUDA)
    if (mg->use_cuda) {  /* setup cuda for short-range part */
      /* XXX this is NOT thread safe for "binlarge" */
      if (mgpot_cuda_setup_shortrng(mg)) {
        ERROR("mgpot_cuda_setup_shortrng() failed\n");
        parms->error = FAIL;
        return NULL;
      }
    }
#endif
    lasttime = rt_timer_timenow(timer);
    if (mgpot_shortrng(mg, threadid, threadcount)) {
      ERROR("mgpot_shortrng() failed\n");
      parms->error = FAIL;
      return NULL;
    }
    elapsedtime = rt_timer_timenow(timer) - lasttime;
#if defined(CUDA)
    if (mg->use_cuda) {  /* cleanup cuda for short-range part */
      /* XXX this is NOT thread safe for "binlarge" */
      if (mgpot_cuda_cleanup_shortrng(mg)) {
        ERROR("mgpot_cuda_cleanup_shortrng() failed\n");
        parms->error = FAIL;
        return NULL;
      }
    }
#endif
    printf("thread[%d]:  time for short-range part:  %.2f\n",
        threadid, elapsedtime);
  }

  if (do_other) {
    printf("thread[%d]:  no work available\n", threadid);
  }

  rt_timer_destroy(timer);

  return NULL;
}
