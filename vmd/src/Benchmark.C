/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: Benchmark.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.6 $      $Date: 2010/12/16 04:08:05 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Various CPU/memory subsystem benchmarking routines.
 * The peak performance numbers achieved within a VMD build can be 
 * used to determine how well the VMD build was optimized, the 
 * performance of the host CPU/memory systems, SMP scaling efficiency, etc.
 *
 * The streaming memory bandwidth tests are an alternative implementation 
 * of McCalpin's STREAM benchmark.
 *
 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include "WKFUtils.h"


/*
 * On compilers that accept the C99 'restrict' keyword, we can give
 * the compiler additional help with optimization.  Since the caller is
 * contained within the same source file, this shouldn't be necessary
 * in the current case however. 
 */
#if 0
#define RESTRICT restrict
#else
#define RESTRICT 
#endif

/*
 * If we want, we can create compiler-specific vectorization 
 * helper macros to assist with achieving peak performance, though 
 * this really shouldn't be required.
 */
#if 0
#define VECTORIZEME _Pragma("vector always")
#else
#define VECTORIZEME 
#endif


/*
 * Double precision stream bandwidth tests
 */

void dstream_init(double * RESTRICT a, double * RESTRICT b,
                  double * RESTRICT c, int N) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
}

void dstream_copy(double * RESTRICT a, const double * RESTRICT b, 
                 int N, double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = b[j];

  *mbsize = (2 * sizeof(double) * N) / (1024.0 * 1024.0);
}

void dstream_scale(double * RESTRICT a, const double * RESTRICT b, 
                  double scalar, int N, double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = scalar * b[j];

  *mbsize = (2 * sizeof(double) * N) / (1024.0 * 1024.0);
}

void dstream_add(double * RESTRICT a, const double * RESTRICT b, 
                const double * RESTRICT c, int N, double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = b[j] + c[j];

  *mbsize = (3 * sizeof(double) * N) / (1024.0 * 1024.0);
}

void dstream_triad(double * RESTRICT a, const double * RESTRICT b, 
                  const double * RESTRICT c, double scalar, int N, 
                  double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = b[j] + scalar * c[j];

  *mbsize = (3 * sizeof(double) * N) / (1024.0 * 1024.0);
}



/*
 * Single precision stream bandwidth tests
 */

void fstream_init(float * RESTRICT a, float * RESTRICT b,
                  float * RESTRICT c, int N) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++) {
    a[j] = 1.0f;
    b[j] = 2.0f;
    c[j] = 0.0f;
  }
}

void fstream_copy(float * RESTRICT a, const float * RESTRICT b, 
                 int N, double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = b[j];

  *mbsize = (2 * sizeof(float) * N) / (1024.0 * 1024.0);
}

void fstream_scale(float * RESTRICT a, const float * RESTRICT b, 
                   float scalar, int N, double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = scalar * b[j];

  *mbsize = (2 * sizeof(float) * N) / (1024.0 * 1024.0);
}

void fstream_add(float * RESTRICT a, const float * RESTRICT b, 
                 const float * RESTRICT c, int N, double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = b[j] + c[j];

  *mbsize = (3 * sizeof(float) * N) / (1024.0 * 1024.0);
}

void fstream_triad(float * RESTRICT a, const float * RESTRICT b, 
                  const float * RESTRICT c, float scalar, int N, 
                  double *mbsize) {
  int j;
VECTORIZEME
  for (j=0; j<N; j++)
    a[j] = b[j] + scalar * c[j];

  *mbsize = (3 * sizeof(float) * N) / (1024.0 * 1024.0);
}


/*
 * run the benchmark
 */
int stream_bench(int N, double *time, double *mbsec) {
  double *da, *db, *dc;
  float *fa, *fb, *fc;
  wkf_timerhandle timer;
  int rc = 0;

  timer = wkf_timer_create();

  /*
   * run double precision benchmarks
   */
  da = (double *) malloc(N * sizeof(double));
  db = (double *) malloc(N * sizeof(double));
  dc = (double *) malloc(N * sizeof(double));

  if ((da != NULL) && (db != NULL) && (dc != NULL)) {
    double mbsz;

    dstream_init(da, db, dc, N);

    wkf_timer_start(timer);
    dstream_copy(da, db, N, &mbsz);
    wkf_timer_stop(timer);
    time[0] = wkf_timer_time(timer);
    mbsec[0] = mbsz / time[0];

    wkf_timer_start(timer);
    dstream_scale(da, db, 2.0, N, &mbsz);
    wkf_timer_stop(timer);
    time[1] = wkf_timer_time(timer);
    mbsec[1] = mbsz / time[1];

    wkf_timer_start(timer);
    dstream_add(da, db, dc, N, &mbsz);
    wkf_timer_stop(timer);
    time[2] = wkf_timer_time(timer);
    mbsec[2] = mbsz / time[2];

    wkf_timer_start(timer);
    dstream_triad(da, db, dc, 2.0, N, &mbsz);
    wkf_timer_stop(timer);
    time[3] = wkf_timer_time(timer);
    mbsec[3] = mbsz / time[3];
  } else {
    rc = -1;
  }

  if (da)
    free(da);
  if (db)
    free(db);
  if (dc)
    free(dc);

  if (rc) {
    wkf_timer_destroy(timer);
    return rc;
  }

  /*
   * run float precision benchmarks
   */
  fa = (float *) malloc(N * sizeof(float));
  fb = (float *) malloc(N * sizeof(float));
  fc = (float *) malloc(N * sizeof(float));

  if ((fa != NULL) && (fb != NULL) && (fc != NULL)) {
    double mbsz;

    fstream_init(fa, fb, fc, N);

    wkf_timer_start(timer);
    fstream_copy(fa, fb, N, &mbsz);
    wkf_timer_stop(timer);
    time[4] = wkf_timer_time(timer);
    mbsec[4] = mbsz / time[4];

    wkf_timer_start(timer);
    fstream_scale(fa, fb, 2.0, N, &mbsz);
    wkf_timer_stop(timer);
    time[5] = wkf_timer_time(timer);
    mbsec[5] = mbsz / time[5];

    wkf_timer_start(timer);
    fstream_add(fa, fb, fc, N, &mbsz);
    wkf_timer_stop(timer);
    time[6] = wkf_timer_time(timer);
    mbsec[6] = mbsz / time[6];

    wkf_timer_start(timer);
    fstream_triad(fa, fb, fc, 2.0, N, &mbsz);
    wkf_timer_stop(timer);
    time[7] = wkf_timer_time(timer);
    mbsec[7] = mbsz / time[7];
  } else {
    rc = -1;
  }

  if (fa)
    free(fa);
  if (fb)
    free(fb);
  if (fc)
    free(fc);

  wkf_timer_destroy(timer);

  return rc;
}





