/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_shortrng.c - compute short-range contribution to potentials
 */

#include <math.h>
#include "mgpot_defn.h"
#if defined(CUDA)
#include "mgpot_cuda.h"
#endif

#undef MGPOT_TIMER
#define MGPOT_TIMER

#ifdef MGPOT_TIMER
#include "util.h"    /* timer code taken from Tachyon */
#define TIMING(x)  do { x } while (0)
#else
#define TIMING(x)
#endif


int mgpot_shortrng(Mgpot *mg, int threadid, int threadcount) {
#ifdef MGPOT_TIMER
  rt_timerhandle timer = rt_timer_create();
  float totaltime=0, lasttime, elapsedtime;
#endif

  TIMING( rt_timer_start(timer); );

  TIMING( lasttime = rt_timer_timenow(timer); );

  switch (mg->use_cuda & MBINMASK) {

    case 0:  /* no cuda kernel */
      if (TAYLOR2==mg->split) {
        printf("Calling mgpot_shortrng_optimal()\n");
        if (mgpot_shortrng_optimal(mg)) {
          return ERROR("mgpot_shortrng_optimal() failed\n");
        }
      }
      else {
        printf("Calling mgpot_shortrng_generic()\n");
        if (mgpot_shortrng_generic(mg)) {
          return ERROR("mgpot_shortrng_generic() failed\n");
        }
      }
      break;

    case MBINLARGE:
      if (threadid > 1) break;  /* not thread safe */
#if !defined(CUDA)
      return ERROR("CUDA is not enabled\n");
#else
      printf("Calling mgpot_cuda_binlarge_pre()\n");
      if (mgpot_cuda_binlarge_pre(mg)) {
        return ERROR("mgpot_cuda_binlarge_pre() failed\n");
      }
      printf("Calling mgpot_cuda_binlarge()\n");
      if (mgpot_cuda_binlarge(mg)) {
        return ERROR("mgpot_cuda_binlarge() failed\n");
      }
      printf("Calling mgpot_cuda_binlarge_post()\n");
      if (mgpot_cuda_binlarge_post(mg)) {
        return ERROR("mgpot_cuda_binlarge_post() failed\n");
      }
      break;
#endif

    case MBINSMALL:
#if !defined(CUDA)
      return ERROR("CUDA is not enabled\n");
#else
      printf("Calling gpu_compute_cutoff_potential_lattice10overlap()\n");

#define MULTI_GPU
#define TEST_MULTI_GPU

#if defined(MULTI_GPU)
      {
        const long blocking = (threadcount > 1 ? threadcount - 1 : 1);
        const long planesize = mg->numpt * mg->numcol;
        const long zquotient = mg->numplane / blocking;
        const long zremainder = mg->numplane % blocking;
        long k = (threadid > 0 ? threadid - 1 : 0);  /* slab ID */
        long zdelta;  /* thickness of kth slab */
        long zstart;  /* plane index for start of kth slab */
        if (k < zremainder) {
          zdelta = zquotient + 1;
          zstart = k * (zquotient + 1);
        }
        else {
          zdelta = zquotient;
          zstart = zremainder*(zquotient + 1) + (k - zremainder)*zquotient;
        }
        printf("thread[%d] computing %ld planes of lattice:  %ld..%ld\n",
            threadid, zdelta, zstart, zstart+zdelta-1);
        if (gpu_compute_cutoff_potential_lattice10overlap(
              mg->grideners + planesize * zstart,
              mg->numpt, mg->numcol, zdelta,
              0.f, 0.f, zstart * mg->gridspacing,
              mg->gridspacing, mg->a, 
              (Atom *) (mg->atoms), mg->numatoms, 0)) {
          return ERROR("gpu_compute_cutoff_potential_lattice10overlap() "
              "failed\n");
        }
        else {
          /* set excluded points to zero */
          const long n = planesize * zdelta;
          long i;
          const unsigned char *excl = mg->excludepos + planesize * zstart;
          float *grideners = mg->grideners + planesize * zstart;
          for (i = 0;  i < n;  i++) {
            if (excl[i]) grideners[i] = 0;
          }
        }
      }
#elif defined(TEST_MULTI_GPU)
  #define Z_BLOCKING_FACTOR  5
      {
        long planesize = mg->numpt * mg->numcol;
        long zdelta = mg->numplane / Z_BLOCKING_FACTOR;
        long zremainder = mg->numplane % Z_BLOCKING_FACTOR;
        long zplane = 0;
        long k;
        for (k = 0;  k < Z_BLOCKING_FACTOR;  k++) {
          int isplusone = (k < zremainder);
          printf("computing %ld planes of lattice:  %ld..%ld\n",
              zdelta + isplusone, zplane, zplane + (zdelta+isplusone)-1);
          if (gpu_compute_cutoff_potential_lattice10overlap(
                mg->grideners + planesize * zplane,
                mg->numpt, mg->numcol, zdelta + isplusone,
                0.f, 0.f, zplane * mg->gridspacing,
                mg->gridspacing, mg->a, 
                (Atom *) (mg->atoms), mg->numatoms, 0)) {
            return ERROR("gpu_compute_cutoff_potential_lattice10overlap() "
                "failed\n");
          }
          else {
            /* set excluded points to zero */
            const long n = mg->numpt * mg->numcol * (zdelta + isplusone);
            long i;
            const unsigned char *excl = mg->excludepos + planesize * zplane;
            float *grideners = mg->grideners + planesize * zplane;
            for (i = 0;  i < n;  i++) {
              if (excl[i]) grideners[i] = 0;
            }
          }
          zplane += zdelta + isplusone;
        }
#if 0
        {
          /* set excluded points to zero */

          const long n = mg->numpt * mg->numcol * mg->numplane;
          long i;
          const unsigned char *excl = mg->excludepos;
          float *grideners = mg->grideners;

          for (i = 0;  i < n;  i++) {
            if (excl[i]) grideners[i] = 0;
          }
        }
#endif
      }
#else
      if (threadid > 1) break;  /* this part is not thread safe */
      if (gpu_compute_cutoff_potential_lattice10overlap(
            mg->grideners, mg->numpt, mg->numcol, mg->numplane,
            0.f, 0.f, 0.f, mg->gridspacing, mg->a,
            (Atom *) (mg->atoms), mg->numatoms, 0)) {
        return ERROR("gpu_compute_cutoff_potential_lattice10overlap() "
            "failed\n");
      }
      else {
        /* set excluded points to zero */

        const long n = mg->numpt * mg->numcol * mg->numplane;
        long i;
        const unsigned char *excl = mg->excludepos;
        float *grideners = mg->grideners;

        for (i = 0;  i < n;  i++) {
          if (excl[i]) grideners[i] = 0;
        }
      }
#endif

      break;
#endif

    default:
      return ERROR("unknown short range CUDA kernel\n");
  }

  TIMING(
    elapsedtime = rt_timer_timenow(timer) - lasttime;
    totaltime += elapsedtime;
    printf(  "BENCH_short_range:  %.5f\n", totaltime);
  );

  TIMING( rt_timer_destroy(timer); );

  return 0;
}


int mgpot_shortrng_optimal(Mgpot *mg) {
  const float *atoms = mg->atoms;
  float *grideners = mg->grideners;
  const long int numplane = mg->numplane;
  const long int numcol = mg->numcol;
  const long int numpt = mg->numpt;
  const long int natoms = mg->numatoms;
  const float gridspacing = mg->gridspacing;
  const unsigned char *excludepos = mg->excludepos;

  const int split = mg->split;
  const float a_1 = mg->a_1;
  const float a2 = mg->a * mg->a;
  const float inv_a2 = 1/a2;
  const float inv_gridspacing = 1/gridspacing;
  const long int radius = (long int) ceil(mg->a * inv_gridspacing) - 1;
    /* grid point radius about each atom */

  long int n;
  long int i, j, k;
  long int ia, ib, ic;
  long int ja, jb, jc;
  long int ka, kb, kc;
  long int index;
  long int koff, jkoff;
#if defined(PERFPROF)
  double nsqrts = 0;
  long int gmin, gmax;
  long int *gnum = mg->gnum;
#endif

  float x, y, z, q;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float s, gs, e;
  float xstart, ystart;

  float *pg;
  const unsigned char *excl;

#if defined(MGPOT_GEOMHASH)
  long int *first = mg->first;
  long int *next = mg->next;
  long int nxcell = mg->nxcell;
  long int nycell = mg->nycell;
  long int nzcell = mg->nzcell;
  long int ncell = nxcell * nycell * nzcell;
  long int gindex;
  float inv_cellen = mg->inv_cellen;
#endif

  DEBUG( printf("mg->a=%g  inv_gridspacing=%g  radius=%ld\n",
        mg->a, inv_gridspacing, radius); );

  if (TAYLOR2 != split) {
    return ERROR("must use TAYLOR2 splitting\n");
  }

  /* inline softened potential directly rather than using switch stmt */
  printf("(using inlined softened potential - optimized)\n");

#if defined(MGPOT_GEOMHASH)
  printf("(performing geometric hashing of atoms)\n");

  /* geometric hashing */
  for (n = 0;  n < natoms;  n++) {
    x = atoms[ INDEX_X(n) ];
    y = atoms[ INDEX_Y(n) ];
    z = atoms[ INDEX_Z(n) ];
    i = (long int) floorf(x * inv_cellen);
    j = (long int) floorf(y * inv_cellen);
    k = (long int) floorf(z * inv_cellen);
    gindex = (k*nycell + j)*nxcell + i;
    next[n] = first[gindex];
    first[gindex] = n;
#if defined(PERFPROF)
    gnum[gindex]++;
#endif
  }

#if defined(PERFPROF)
  gmin = natoms;
  gmax = 0;
  for (gindex = 0;  gindex < ncell;  gindex++) {
    if (gmin > gnum[gindex]) gmin = gnum[gindex];
    if (gmax < gnum[gindex]) gmax = gnum[gindex];
  }
  printf("number of atoms: %ld\n", natoms);
  printf("number of gridcells: %ld\n", ncell);
  printf("dimension of gridcells in x: %ld\n", nxcell);
  printf("dimension of gridcells in y: %ld\n", nycell);
  printf("dimension of gridcells in z: %ld\n", nzcell);
  printf("cell length: %f\n", 1/inv_cellen);
  printf("inverse cell length: %f\n", inv_cellen);
  printf("max number of atoms in a gridcell: %ld\n", gmax);
  printf("min number of atoms in a gridcell: %ld\n", gmin);
  printf("avg number of atoms in a gridcell: %.1f\n", (double) natoms/ncell);
#endif

  /* traverse the grid cells */
  for (gindex = 0;  gindex < ncell;  gindex++) {
    for (n = first[gindex];  n != -1;  n = next[n]) {

#else /* not defined MGPOT_GEOMHASH */

  for (n = 0;  n < natoms;  n++) {

#endif /* MGPOT_GEOMHASH */

    q = atoms[ INDEX_Q(n) ];
    if (0==q) continue;

    x = atoms[ INDEX_X(n) ];
    y = atoms[ INDEX_Y(n) ];
    z = atoms[ INDEX_Z(n) ];

    /* find closest grid point with position less than or equal to atom */
    ic = (long int) (x * inv_gridspacing);
    jc = (long int) (y * inv_gridspacing);
    kc = (long int) (z * inv_gridspacing);

    /* find extent of surrounding box of grid points */
    ia = ic - radius;
    ib = ic + radius + 1;
    ja = jc - radius;
    jb = jc + radius + 1;
    ka = kc - radius;
    kb = kc + radius + 1;

    /* trim box edges so that they are within grid point lattice */
    if (ia < 0) ia = 0;
    if (ib >= numpt) ib = numpt-1;
    if (ja < 0) ja = 0;
    if (jb >= numcol) jb = numcol-1;
    if (ka < 0) ka = 0;
    if (kb >= numplane) kb = numplane-1;

    /* loop over surrounding grid points */
    xstart = ia*gridspacing - x;
    ystart = ja*gridspacing - y;
    dz = ka*gridspacing - z;
    for (k = ka;  k <= kb;  k++, dz += gridspacing) {
      koff = k*numcol;
      dz2 = dz*dz;

      dy = ystart;
      for (j = ja;  j <= jb;  j++, dy += gridspacing) {
        jkoff = (koff + j)*numpt;
	dydz2 = dy*dy + dz2;

        dx = xstart;
        index = jkoff + ia;
        pg = grideners + index;

#if defined(__INTEL_COMPILER)
	for (i = ia;  i <= ib;  i++, pg++, dx += gridspacing) {
	  r2 = dx*dx + dydz2;
	  s = r2 * inv_a2;
          gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
          e = q * (1/sqrtf(r2) - a_1 * gs);
	  *pg += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED!! */
#if defined(PERFPROF)
          nsqrts++;
#endif
        }
#else
        excl = excludepos + index;
	for (i = ia;  i <= ib;  i++, pg++, excl++, dx += gridspacing) {
          if (*excl) continue;
	  r2 = dx*dx + dydz2;
          if (r2 >= a2) continue;
	  s = r2 * inv_a2;
          gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
          e = q * (1/sqrtf(r2) - a_1 * gs);
	  *pg += e;
	}
#endif
      }
    } /* end loop over surrounding grid points */

#if defined(MGPOT_GEOMHASH)
    } /* end loop over atoms in a gridcell */
  } /* end loop over gridcells */
#else
  } /* end loop over atoms */
#endif /* MGPOT_GEOMHASH */

#if defined(__INTEL_COMPILER)
  n = numpt * numcol * numplane;
  pg = grideners;
  excl = excludepos;
  for (i = 0;  i < n;  i++, pg++, excl++) {
    *pg = (*excl ? 0 : *pg);
  }
#endif

#if defined(PERFPROF)
  printf("number of square root evaluations: %.0f\n", nsqrts);
#endif

  return 0;
}


int mgpot_shortrng_generic(Mgpot *mg) {
  const float *atoms = mg->atoms;
  float *grideners = mg->grideners;
  const long int numplane = mg->numplane;
  const long int numcol = mg->numcol;
  const long int numpt = mg->numpt;
  const long int natoms = mg->numatoms;
  const float gridspacing = mg->gridspacing;
  const unsigned char *excludepos = mg->excludepos;

  const int split = mg->split;
  const float a_1 = mg->a_1;
  const float a2 = mg->a * mg->a;
  const float inv_a2 = 1/a2;
  const float inv_gridspacing = 1/gridspacing;
  const long int radius = (long int) ceil(mg->a * inv_gridspacing) - 1;
    /* grid point radius about each atom */

  long int n;
  long int i, j, k;
  long int ia, ib, ic;
  long int ja, jb, jc;
  long int ka, kb, kc;
  long int index;

  float x, y, z, q;
  float dx, dy, dz;
  float dz2, dydz2, r2;
  float s, gs;

  /* fall back on mgpot_split() macro for other choices */
  printf("(using macro expanded switch stmt - not optimized)\n");

  for (n = 0;  n < natoms;  n++) {
    q = atoms[ INDEX_Q(n) ];
    if (0==q) continue;

    x = atoms[ INDEX_X(n) ];
    y = atoms[ INDEX_Y(n) ];
    z = atoms[ INDEX_Z(n) ];

    /* find closest grid point with position less than or equal to atom */
    ic = (long int) (x * inv_gridspacing);
    jc = (long int) (y * inv_gridspacing);
    kc = (long int) (z * inv_gridspacing);

    /* find extent of surrounding box of grid points */
    ia = ic - radius;
    ib = ic + radius + 1;
    ja = jc - radius;
    jb = jc + radius + 1;
    ka = kc - radius;
    kb = kc + radius + 1;

    /* trim box edges so that they are within grid point lattice */
    if (ia < 0) ia = 0;
    if (ib >= numpt) ib = numpt-1;
    if (ja < 0) ja = 0;
    if (jb >= numcol) jb = numcol-1;
    if (ka < 0) ka = 0;
    if (kb >= numplane) kb = numplane-1;

    /* loop over surrounding grid points */
    for (k = ka;  k <= kb;  k++) {
      dz = z - k*gridspacing;
      dz2 = dz*dz;
      for (j = ja;  j <= jb;  j++) {
        dy = y - j*gridspacing;
	dydz2 = dy*dy + dz2;
	for (i = ia;  i <= ib;  i++) {
	  index = (k*numcol + j)*numpt + i;
	  if (excludepos[index]) continue;
	  dx = x - i*gridspacing;
	  r2 = dx*dx + dydz2;
	  if (r2 >= a2) continue;
	  s = r2 * inv_a2;
	  mgpot_split(&gs, s, split);  /* macro expands into switch */
	  grideners[index] += q * (1/sqrtf(r2) - a_1 * gs);
	}
      }
    } /* end loop over surrounding grid points */

  } /* end loop over atoms */

  return 0;
}
