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
 *      $RCSfile: msmpot_compute.c,v $
 *      $Author: dhardy $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $      $Date: 2010/06/10 22:36:49 $
 *
 ***************************************************************************/

#include "msmpot_internal.h"

/* macros below for debugging */
/*
#define MSMPOT_LONGRANGE_ONLY
#undef MSMPOT_LONGRANGE_ONLY

#define MSMPOT_SHORTRANGE_ONLY
#undef MSMPOT_SHORTRANGE_ONLY

#define MSMPOT_CHECKMAPINDEX
#undef MSMPOT_CHECKMAPINDEX

#define MAPINDEX  0
*/

#define USE_BIN_HASHING

int Msmpot_compute(Msmpot *msm,
    float *epotmap,               /* electrostatic potential map
                                     assumed to be length mx*my*mz,
                                     stored flat in row-major order, i.e.,
                                     &ep[i,j,k] == ep + ((k*my+j)*mx+i) */
    int mx, int my, int mz,       /* map lattice dimensions */
    float lx, float ly, float lz, /* map lattice lengths */
    float x0, float y0, float z0, /* map origin (lower-left corner) */
    float vx, float vy, float vz, /* periodic cell lengths along x, y, z;
                                     set to 0 for non-periodic direction */
    const float *atom,            /* atoms stored x/y/z/q (length 4*natoms) */
    int natoms                    /* number of atoms */
    ) {
  int err;

  REPORT("Performing MSM calculation of electrostatic potential map.");

  err = Msmpot_check_params(msm, epotmap, mx, my, mz, lx, ly, lz,
      vx, vy, vz, atom, natoms);
  if (err != MSMPOT_SUCCESS) return ERROR(err);

  /* store user parameters */
  msm->atom = atom;
  msm->natoms = natoms;
  msm->epotmap = epotmap;
  msm->mx = mx;
  msm->my = my;
  msm->mz = mz;
  msm->lx = lx;
  msm->ly = ly;
  msm->lz = lz;
  msm->lx0 = x0;
  msm->ly0 = y0;
  msm->lz0 = z0;
  msm->dx = lx / mx;
  msm->dy = ly / my;
  msm->dz = lz / mz;
  msm->px = vx;
  msm->py = vy;
  msm->pz = vz;
  msm->isperiodic = 0;  /* reset flags for periodicity */
  /* zero length indicates nonperiodic direction */
  if (vx > 0) SET_X(msm->isperiodic);
  if (vy > 0) SET_Y(msm->isperiodic);
  if (vz > 0) SET_Z(msm->isperiodic);

  err = Msmpot_setup(msm);
  if (err != MSMPOT_SUCCESS) return ERROR(err);

  memset(epotmap, 0, mx*my*mz*sizeof(float));  /* clear epotmap */


#if !defined(MSMPOT_LONGRANGE_ONLY)
#ifdef MSMPOT_CUDA
  if (msm->use_cuda_shortrng) {
    err = Msmpot_cuda_compute_shortrng(msm->msmcuda);
    if (err && msm->cuda_optional) {  /* fall back on CPU */
#ifdef USE_BIN_HASHING
      err = Msmpot_compute_shortrng_bins(msm);
#else
      err = Msmpot_compute_shortrng_linklist(msm, msm->atom, msm->natoms);
#endif
      if (err) return ERROR(err);
    }
    else if (err) return ERROR(err);
  }
  else {
#ifdef USE_BIN_HASHING
    err = Msmpot_compute_shortrng_bins(msm);
#else
    err = Msmpot_compute_shortrng_linklist(msm, msm->atom, msm->natoms);
#endif /* USE_BIN_HASHING */
    if (err) return ERROR(err);
  }
#else
#ifdef USE_BIN_HASHING
  err = Msmpot_compute_shortrng_bins(msm);
#else
  err = Msmpot_compute_shortrng_linklist(msm, msm->atom, msm->natoms);
#endif /* USE_BIN_HASHING */
  if (err) return ERROR(err);
#endif
#endif

#if !defined(MSMPOT_SHORTRANGE_ONLY)
  err = Msmpot_compute_longrng(msm);
  if (err) return ERROR(err);
#endif

#ifdef MSMPOT_VERBOSE
#ifdef MSMPOT_CHECKMAPINDEX
  printf("epotmap[%d]=%g\n", MAPINDEX, epotmap[MAPINDEX]);
#endif
#endif

  return MSMPOT_SUCCESS;
}



/*** long-range part *********************************************************/


int Msmpot_compute_longrng(Msmpot *msm) {
  int err = 0;

  /* permit only cubic interpolation - for now */
  switch (msm->interp) {
    case MSMPOT_INTERP_CUBIC:
      err = Msmpot_compute_longrng_cubic(msm);
      if (err) return ERROR(err);
      break;
    default:
      return ERRMSG(MSMPOT_ERROR_SUPPORT,
          "interpolation method not implemented");
  }
  return MSMPOT_SUCCESS;
}



/*** short-range part ********************************************************/


static int bin_evaluation(Msmpot *msm);


/*
 * hash atoms into bins, evaluation of grid points loops over neighborhood
 * of bins, any overflowed bins are handled using linked list approach
 */
int Msmpot_compute_shortrng_bins(Msmpot *msm) {
  int err = 0;

  REPORT("Using atom hashing into bins for short-range part.");

  REPORT("Using tight neighborhood for nearby bins.");
  err = Msmpot_compute_shortrng_bin_neighborhood(msm,
      msm->bx, msm->by, msm->bz);
  if (err) return ERROR(err);

  err = Msmpot_compute_shortrng_bin_hashing(msm);
  if (err) return ERROR(err);

  err = bin_evaluation(msm);
  if (err) return ERROR(err);

  if (msm->nover > 0) {
#ifdef MSMPOT_REPORT
    char msg[120];
    sprintf(msg, "Extra atoms (%d) from overflowed bins "
        "must also be evaluated.", msm->nover);
    REPORT(msg);
#endif
    err = Msmpot_compute_shortrng_linklist(msm, msm->over, msm->nover);
    if (err) return ERROR(err);
  }

  return MSMPOT_SUCCESS;
}


/*
 * Determine a tight neighborhood of bins, given a region of map points.
 * Store index offsets from (0,0,0).  Requires rectangular bins.
 *
 * (rx,ry,rz) gives size of region from which we select one or more map points.
 * To setup neighborhood for GPU, the region is the smallest (unrolled)
 * block of space for a thread block.  For CPU, set (rx,ry,rz)=(bx,by,bz).
 *
 * Space allocated as needed for bin offsets.
 */

/*
 * XXX Intel icc 9.0 is choking on this routine.
 * The problem seems to be the triply nested loop. 
 * Although turning off optimizations for this routine is undesirable,
 * it allows the compiler to keep going.
 */
#if defined( __INTEL_COMPILER)
#pragma optimize("",off)
#endif
int Msmpot_compute_shortrng_bin_neighborhood(Msmpot *msm,
    float rx,  /* region length in x-dimension */
    float ry,  /* region length in y-dimension */
    float rz   /* region length in z-dimension */
    ) {

  union {  /* use this to do exact compare of floats */
    float f;
    int i;
  } v0, v1;

  const float cutoff = msm->a;  /* cutoff distance */

  const float bx = msm->bx;  /* bin length along x */
  const float by = msm->by;  /* bin length along y */
  const float bz = msm->bz;  /* bin length along z */

  const float invbx = msm->invbx;  /* 1 / bx */
  const float invby = msm->invby;  /* 1 / by */
  const float invbz = msm->invbz;  /* 1 / bz */

  const float bx2 = bx*bx;
  const float by2 = by*by;
  const float bz2 = bz*bz;

  float sqbindiag = 0.f;
  float r, r2;

  int bpr0, bpr1;

  int cx = (int) ceilf(cutoff * invbx);  /* number of bins on cutoff in x */
  int cy = (int) ceilf(cutoff * invby);  /* number of bins on cutoff in y */
  int cz = (int) ceilf(cutoff * invbz);  /* number of bins on cutoff in z */

  int nbrx, nbry, nbrz;  /* axes of ellipsoid for neighborhood */
  int i, j, k;
  int maxboff;

  int *boff;

  /* x-direction */
  v0.f = bx, v1.f = rx;
  if (v0.i == v1.i) {  /* should be true for CPU code path */
    bpr0 = bpr1 = 1;
  }
  else {
    bpr0 = (int) floorf(rx*invbx);
    bpr1 = (int) ceilf(rx*invbx);
  }

  if (bpr0 == bpr1) {  /* special case:  bins exactly cover region */
    nbrx = cx + (bpr0 >> 1);  /* brp0 / 2 */
    /* if bin cover is odd, use square of half-bin-length */
    sqbindiag += ((bpr0 & 1) ? 0.25f : 1.f) * bx2;
  }
  else {
    nbrx = (int) ceilf((cutoff + 0.5f*rx + bx) * invbx);
    sqbindiag += bx2;
  }

  /* y-direction */
  v0.f = by, v1.f = ry;
  if (v0.i == v1.i) {  /* should be true for CPU code path */
    bpr0 = bpr1 = 1;
  }
  else {
    bpr0 = (int) floorf(ry*invby);
    bpr1 = (int) ceilf(ry*invby);
  }

  if (bpr0 == bpr1) {  /* special case:  bins exactly cover region */
    nbry = cy + (bpr0 >> 1);  /* brp0 / 2 */
    /* if bin cover is odd, use square of half-bin-length */
    sqbindiag += ((bpr0 & 1) ? 0.25f : 1.f) * by2;
  }
  else {
    nbry = (int) ceilf((cutoff + 0.5f*ry + by) * invby);
    sqbindiag += by2;
  }

  /* z-direction */
  v0.f = bz, v1.f = rz;
  if (v0.i == v1.i) {  /* should be true for CPU code path */
    bpr0 = bpr1 = 1;
  }
  else {
    bpr0 = (int) floorf(rz*invbz);
    bpr1 = (int) ceilf(rz*invbz);
  }

  if (bpr0 == bpr1) {  /* special case:  bins exactly cover region */
    nbrz = cz + (bpr0 >> 1);  /* brp0 / 2 */
    /* if bin cover is odd, use square of half-bin-length */
    sqbindiag += ((bpr0 & 1) ? 0.25f : 1.f) * bz2;
  }
  else {
    nbrz = (int) ceilf((cutoff + 0.5f*rz + bz) * invbz);
    sqbindiag += bz2;
  }

  r = cutoff + 0.5f*sqrtf(rx*rx + ry*ry + rz*rz) + sqrtf(sqbindiag);
  r2 = r*r;

  /* upper bound on the size of the neighborhood */
  maxboff = (2*nbrx+1) * (2*nbry+1) * (2*nbrz+1);

  if (msm->maxboff < maxboff) {
    void *v = realloc(msm->boff, 3*maxboff*sizeof(int));
    if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
    msm->boff = (int *) v;
    msm->maxboff = maxboff;
  }

  boff = msm->boff;
  for (k = -nbrz;  k <= nbrz;  k++) {
    for (j = -nbry;  j <= nbry;  j++) {
      for (i = -nbrx;  i <= nbrx;  i++) {
        if ((i*i*bx2 + j*j*by2 + k*k*bz2) >= r2) continue;
        *boff++ = i;
        *boff++ = j;
        *boff++ = k;
      }
    }
  }
  msm->nboff = (boff - msm->boff) / 3;  /* count of the neighborhood */

  return MSMPOT_SUCCESS;
}
#if defined(__INTEL_COMPILER)
#pragma optimize("",on)
#endif


int Msmpot_compute_shortrng_bin_hashing(Msmpot *msm) {

  union {  /* use this to do exact compare of floats */
    float f;
    int i;
  } q;     /* atom charge */

  int i, j, k;
  int n;   /* index atoms */
  int nb;  /* index bins */
  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);
  const float px0 = msm->px0;  /* domain origin */
  const float py0 = msm->py0;
  const float pz0 = msm->pz0;
  const float px = msm->px;    /* domain lengths */
  const float py = msm->py;
  const float pz = msm->pz;
  const float invbx = msm->invbx;
  const float invby = msm->invby;
  const float invbz = msm->invbz;
  float x, y, z;  /* atom position relative to (xmin,ymin,zmin) */

  const float *atom = msm->atom;
  const int natoms = msm->natoms;

  const int nbx = msm->nbx;
  const int nby = msm->nby;
  const int nbz = msm->nbz;
  const int nbins = (nbx * nby * nbz);
  const int bindepth = msm->bindepth;
  float *bin = msm->bin;
  int *bincount = msm->bincount;

  memset(bin, 0, nbins*bindepth*ATOM_SIZE*sizeof(float)); /* clear bins */
  memset(bincount, 0, nbins*sizeof(int));
  msm->nover = 0;  /* clear count of overflowed bins */

  for (n = 0;  n < natoms;  n++) {

    /* atoms with zero charge make no contribution */
    q.f = atom[ATOM_Q(n)];
    if (0==q.i) continue;

    x = atom[ATOM_X(n)] - px0;  /* atom position wrt domain */
    y = atom[ATOM_Y(n)] - py0;
    z = atom[ATOM_Z(n)] - pz0;
    i = (int) floorf(x * invbx);
    j = (int) floorf(y * invby);
    k = (int) floorf(z * invbz);

    /* for periodic directions, wrap bin index and atom coordinate */
    if (ispx) {
      if      (i < 0)    do { i += nbx;  x += px; } while (i < 0);
      else if (i >= nbx) do { i -= nbx;  x -= px; } while (i >= nbx);
    }
    if (ispy) {
      if      (j < 0)    do { j += nby;  y += py; } while (j < 0);
      else if (j >= nby) do { j -= nby;  y -= py; } while (j >= nby);
    }
    if (ispz) {
      if      (k < 0)    do { k += nbz;  z += pz; } while (k < 0);
      else if (k >= nbz) do { k -= nbz;  z -= pz; } while (k >= nbz);
    }

#if 0
    if (i < 0 || i >= nbx ||
        j < 0 || j >= nby ||
        k < 0 || k >= nbz) {
      printf("nbx=%d  nby=%d  nbz=%d\n", nbx, nby, nbz);
      printf("i=%d  j=%d  k=%d\n", i, j, k);
      return ERROR(MSMPOT_ERROR_ASSERT);
    }
#endif

    nb = (k*nby + j)*nbx + i;
    ASSERT(0 <= nb && nb < nbins);
    if (bincount[nb] < bindepth) {
      float *p = bin + (nb*bindepth + bincount[nb])*ATOM_SIZE;
      *p++ = x;
      *p++ = y;
      *p++ = z;
      *p   = q.f;
      bincount[nb]++;
    }
    else {  /* atom must be appended  to overflow bin array */
      int nover = msm->nover;
      float *p;
      if (nover == msm->maxover) {  /* extend length of overflow bin array */
        void *v = realloc(msm->over, 2*msm->maxover*ATOM_SIZE*sizeof(float));
        if (NULL == v) return ERROR(MSMPOT_ERROR_MALLOC);
        msm->over = (float *) v;
        msm->maxover *= 2;
      }
      p = msm->over + nover*ATOM_SIZE;
      *p++ = x;
      *p++ = y;
      *p++ = z;
      *p   = q.f;
      msm->nover++;
    }
  }
  return MSMPOT_SUCCESS;
} /* Msmpot_compute_shortrng_bin_hashing() */


int bin_evaluation(Msmpot *msm) {

  const float lx0 = msm->lx0;  /* epotmap origin */
  const float ly0 = msm->ly0;
  const float lz0 = msm->lz0;

  const float dx = msm->dx;    /* epotmap spacings */
  const float dy = msm->dy;
  const float dz = msm->dz;

  const float px0 = msm->lx0;  /* domain origin */
  const float py0 = msm->ly0;
  const float pz0 = msm->lz0;

  const float px = msm->px;    /* domain length */
  const float py = msm->py;
  const float pz = msm->pz;

  const float invbx = msm->invbx;  /* inverse bin length */
  const float invby = msm->invby;
  const float invbz = msm->invbz;

  const int nbx = msm->nbx;    /* number of bins along each dimension */
  const int nby = msm->nby;
  const int nbz = msm->nbz;

  const int bindepth = msm->bindepth;  /* number of atom slots per bin */

  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);

  const float a = msm->a;      /* cutoff for splitting */
  const float a2 = a*a;
  const float a_1 = 1/a;
  const float inv_a2 = a_1 * a_1;

  const int split = msm->split;

  float *epotmap = msm->epotmap;  /* the map */

  const int mx = msm->mx;  /* lengths of epotmap lattice */
  const int my = msm->my;
  const int mz = msm->mz;

  int n;
  int ic, jc, kc;
  int ig, jg, kg;
  int i, j, k;

  for (kg = 0;  kg < mz;  kg++) { /* loop over map points */
    for (jg = 0;  jg < my;  jg++) {
      for (ig = 0;  ig < mx;  ig++) {

        float e = 0;  /* accumulate potential on each map point */

        float xg = ig * dx;  /* coordinate of map point wrt map origin */
        float yg = jg * dy;
        float zg = kg * dz;

        int index = (kg*my + jg)*mx + ig;  /* map point index */

        xg += lx0;  /* absolute position */
        yg += ly0;
        zg += lz0;

        xg -= px0;  /* position wrt domain origin */
        yg -= py0;
        zg -= pz0;

        ic = (int) floorf(xg * invbx);  /* find bin containing this point */
        jc = (int) floorf(yg * invby);
        kc = (int) floorf(zg * invbz);

        for (n = 0;  n < msm->nboff;  n++) {
          float *pbin;  /* point into this bin */
          int bindex, bincount, m;
          int iw, jw, kw;

          float xw = 0;  /* periodic offset for wrapping coordinates */
          float yw = 0;
          float zw = 0;

          i = ic + msm->boff[3*n    ];
          j = jc + msm->boff[3*n + 1];
          k = kc + msm->boff[3*n + 2];

          if ( ! ispx  &&  (i < 0 || i >= nbx) )  continue;
          if ( ! ispy  &&  (j < 0 || j >= nby) )  continue;
          if ( ! ispz  &&  (k < 0 || k >= nbz) )  continue;

          iw = i;
          jw = j;
          kw = k;

          if (ispx) {  /* wrap bin neighborhood around periodic edges */
            while (iw < 0)    { iw += nbx;  xw -= px; }
            while (iw >= nbx) { iw -= nbx;  xw += px; }
          }
          if (ispy) {
            while (jw < 0)    { jw += nby;  yw -= py; }
            while (jw >= nby) { jw -= nby;  yw += py; }
          }
          if (ispz) {
            while (kw < 0)    { kw += nbz;  zw -= pz; }
            while (kw >= nbz) { kw -= nbz;  zw += pz; }
          }

          bindex = (kw*nby + jw)*nbx + iw;  /* the bin index */
          pbin = msm->bin + bindex*bindepth*ATOM_SIZE;  /* first atom */
          bincount = msm->bincount[bindex];

          for (m = 0;  m < bincount;  m++) {
            float x = *pbin++;  /* get next atom from bin */
            float y = *pbin++;
            float z = *pbin++;
            float q = *pbin++;

            float rx = (x+xw) - xg;  /* positions both relative to domain */
            float ry = (y+yw) - yg;
            float rz = (z+zw) - zg;

            float r2 = rx*rx + ry*ry + rz*rz;
            float s, gs;

            if (r2 >= a2) continue;

            s = r2 * inv_a2;
            SPOLY(&gs, s, split);  /* macro expands into switch */
            e += q * (1/sqrtf(r2) - a_1 * gs);  /* accumulate potential */
          } /* loop over binned atoms */

        } /* loop over bin neighborhood */

        epotmap[index] = e;  /* store entire potential at map point */
      }
    }
  } /* loop over map points */

  return MSMPOT_SUCCESS;
}


static int linklist_hashing(Msmpot *msm, const float *atom, int natoms);
static int linklist_evaluation(Msmpot *msm, const float *atom);


/*
 * explicitly pass atoms, so we can use this for bin overflow array
 */
int Msmpot_compute_shortrng_linklist(Msmpot *msm,
    const float *atom,    /* array of atoms stored x/y/z/q */
    int natoms            /* number of atoms in array */
    ) {
  int err = 0;

  REPORT("Using linked lists of atoms for short-range part.");

  err = linklist_hashing(msm, atom, natoms);
  if (err) return ERROR(err);

  err = linklist_evaluation(msm, atom);
  if (err) return ERROR(err);

  return MSMPOT_SUCCESS;
}


/*
 * perform spatial hashing of atoms, store results using linked list
 */
int linklist_hashing(Msmpot *msm, const float *atom, int natoms) {

  union {  /* use this to do exact compare of floats */
    float f;
    int i;
  } q;     /* atom charge */

  int i, j, k;
  int n;   /* index atoms */
  int nb;  /* index bins */
  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);
  const float x0 = msm->px0;  /* domain origin */
  const float y0 = msm->py0;
  const float z0 = msm->pz0;
  const float invbx = msm->invbx;
  const float invby = msm->invby;
  const float invbz = msm->invbz;
  float x, y, z;  /* atom position relative to (xmin,ymin,zmin) */
  int *first = msm->first_atom_index;  /* first index in grid cell list */
  int *next = msm->next_atom_index;    /* next index in grid cell list */
  const int nbx = msm->nbx;
  const int nby = msm->nby;
  const int nbz = msm->nbz;
  const int nbins = (nbx * nby * nbz);

#ifdef MSMPOT_VERBOSE
  printf("bin array:  %d %d %d\n", nbx, nby, nbz);
  printf("bin lengths:  %f %f %f\n", 1/invbx, 1/invby, 1/invbz);
#endif

  /* must clear first and next links before we hash */
  for (nb = 0;  nb < nbins;  nb++)  first[nb] = -1;
  for (n = 0;  n < natoms;  n++)  next[n] = -1;

  for (n = 0;  n < natoms;  n++) {

    /* atoms with zero charge make no contribution */
    q.f = atom[ATOM_Q(n)];
    if (0==q.i) continue;

    x = atom[ATOM_X(n)] - x0;
    y = atom[ATOM_Y(n)] - y0;
    z = atom[ATOM_Z(n)] - z0;
    i = (int) floorf(x * invbx);
    j = (int) floorf(y * invby);
    k = (int) floorf(z * invbz);

    /* for periodic directions, make sure bin index is wrapped */
    if (ispx) {
      if      (i < 0)     do { i += nbx; } while (i < 0);
      else if (i >= nbx)  do { i -= nbx; } while (i >= nbx);
    }
    if (ispy) {
      if      (j < 0)     do { j += nby; } while (j < 0);
      else if (j >= nby)  do { j -= nby; } while (j >= nby);
    }
    if (ispz) {
      if      (k < 0)     do { k += nbz; } while (k < 0);
      else if (k >= nbz)  do { k -= nbz; } while (k >= nbz);
    }

    nb = (k*nby + j)*nbx + i;  /* flat bin index */
    next[n] = first[nb];       /* insert index n at front of list nb */
    first[nb] = n;
  }
  return MSMPOT_SUCCESS;
} /* linklist_hashing() */


/*
 * evaluate short-range contribution of atoms to mapped potential,
 * must first perform linklist_hashing()
 */
int linklist_evaluation(Msmpot *msm, const float *atom) {

  const float lx0 = msm->lx0;  /* epotmap origin */
  const float ly0 = msm->ly0;
  const float lz0 = msm->lz0;

  const float dx = msm->dx;    /* epotmap spacings */
  const float dy = msm->dy;
  const float dz = msm->dz;
  const float inv_dx = 1/dx;
  const float inv_dy = 1/dy;
  const float inv_dz = 1/dz;

  const float px = msm->px;
  const float py = msm->py;
  const float pz = msm->pz;

  const float plx = px - msm->lx;
  const float ply = py - msm->ly;
  const float plz = pz - msm->lz;

  const float pxd = px * inv_dx;
  const float pyd = py * inv_dy;
  const float pzd = pz * inv_dz;

  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);

  const float a = msm->a;      /* cutoff for splitting */
  const float a2 = a*a;
  const float a_1 = 1/a;
  const float inv_a2 = a_1 * a_1;

  float x, y, z;     /* position of atom relative to epotmap origin */
  float q;           /* charge on atom */

  float xg, yg, zg;  /* positions given in grid spacings */

  int i, j, k;
  int ic, jc, kc;    /* closest map point less than or equal to atom */
  int ia, ib;        /* extent of surrounding box in x-direction */
  int ja, jb;        /* extent of surrounding box in y-direction */
  int ka, kb;        /* extent of surrounding box in z-direction */
  int iw, jw, kw;    /* wrapped epotmap indexes within loops */
  int n;             /* index atoms */
  int nbs;           /* index bins */
  long index;        /* index into epotmap */
  long jkoff;        /* tabulate stride into epotmap */
  int koff;          /* tabulate stride into epotmap */
  float rx, ry, rz;  /* distances between an atom and a map point */
  float rz2, ryrz2;  /* squared circle and cylinder distances */
  float r2;          /* squared pairwise distance */
  float s;           /* normalized distance squared */
  float gs;          /* result of normalized splitting */
  float e;           /* contribution to short-range potential */

  const int split = msm->split;

  const int mx = msm->mx;  /* lengths of epotmap lattice */
  const int my = msm->my;
  const int mz = msm->mz;

  const int mri = (int) ceilf(a * inv_dx) - 1;
  const int mrj = (int) ceilf(a * inv_dy) - 1;
  const int mrk = (int) ceilf(a * inv_dz) - 1;
                     /* lengths (measured in points) of ellipsoid axes */

  const int nbins = (msm->nbx * msm->nby * msm->nbz);
  const int *first = msm->first_atom_index;
  const int *next = msm->next_atom_index;

  float *epotmap = msm->epotmap;
#if 0
  float *pem = NULL;        /* point into epotmap */
#endif

  for (nbs = 0;  nbs < nbins;  nbs++) {
    for (n = first[nbs];  n != -1;  n = next[n]) {

      /* position of atom relative to epotmap origin */
      x = atom[ATOM_X(n)] - lx0;
      y = atom[ATOM_Y(n)] - ly0;
      z = atom[ATOM_Z(n)] - lz0;

      /* charge on atom */
      q = atom[ATOM_Q(n)];

      /*
       * make sure atom is wrapped into cell along periodic directions
       *
       * NOTE: we can avoid this redundancy by storing wrapped
       *   coordinate during geometric hashing
       */ 
      if (ispx) {
        if      (x < 0)   do { x += px; } while (x < 0);
        else if (x >= px) do { x -= px; } while (x >= px);
      }
      if (ispy) {
        if      (y < 0)   do { y += py; } while (y < 0);
        else if (y >= py) do { y -= py; } while (y >= py);
      }
      if (ispz) {
        if      (z < 0)   do { z += pz; } while (z < 0);
        else if (z >= pz) do { z -= pz; } while (z >= pz);
      }

      /* calculate position in units of grid spacings */
      xg = x * inv_dx;
      yg = y * inv_dy;
      zg = z * inv_dz;

      /* find closest map point with position less than or equal to atom */
      ic = (int) floorf(xg);
      jc = (int) floorf(yg);
      kc = (int) floorf(zg);

      /* find extent of surrounding box of map points */
      ia = ic - mri;
      ib = ic + mri + 1;
      ja = jc - mrj;
      jb = jc + mrj + 1;
      ka = kc - mrk;
      kb = kc + mrk + 1;

      /* for nonperiodic directions, trim box edges to be within map */
      if ( ! ispx ) {
        if (ia < 0)   ia = 0;
        if (ib >= mx) ib = mx - 1;
      }
      else {
        if (ia-1 < (mx-1) - pxd) {
          /* atom influence wraps around low end of cell */
          int na = ((int) floorf(xg + pxd)) - mri;
          if (na < 0) na = 0;
          ia = na - mx;
        }
        if (ib+1 > pxd) {
          /* atom influence wraps around high end of cell */
          int nb = ((int) floorf(xg - pxd)) + mri + 1;
          if (nb >= mx) nb = mx - 1;
          ib = nb + mx;
        }
      }

      if ( ! ispy ) {
        if (ja < 0)   ja = 0;
        if (jb >= my) jb = my - 1;
      }
      else {
        if (ja-1 < (my-1) - pyd) {
          /* atom influence wraps around low end of cell */
          int na = ((int) floorf(yg + pyd)) - mrj;
          if (na < 0) na = 0;
          ja = na - my;
        }
        if (jb+1 > pyd) {
          /* atom influence wraps around high end of cell */
          int nb = ((int) floorf(yg - pyd)) + mrj + 1;
          if (nb >= my) nb = my - 1;
          jb = nb + my;
        }
      }

      if ( ! ispz ) {
        if (ka < 0)   ka = 0;
        if (kb >= mz) kb = mz - 1;
      }
      else {
        if (ka-1 < (mz-1) - pzd) {
          /* atom influence wraps around low end of cell */
          int na = ((int) floorf(zg + pzd)) - mrk;
          if (na < 0) na = 0;
          ka = na - mz;
        }
        if (kb+1 > pzd) {
          /* atom influence wraps around high end of cell */
          int nb = ((int) floorf(zg - pzd)) + mrk + 1;
          if (nb >= mz) nb = mz - 1;
          kb = nb + mz;
        }
      }

      /* loop over surrounding map points, add contribution into epotmap */
      for (k = ka;  k <= kb;  k++) {
        rz = k*dz - z;
        kw = k;
        if (k < 0) {
          rz -= plz;
          kw += mz;
        }
        else if (k >= mz) {
          rz += plz;
          kw -= mz;
        }
        koff = kw*my;
        rz2 = rz*rz;

#ifdef MSMPOT_CHECK_CIRCLE_CPU
        /* clipping to the circle makes it slower */
        if (rz2 >= a2) continue;
#endif

        for (j = ja;  j <= jb;  j++) {
          ry = j*dy - y;
          jw = j;
          if (j < 0) {
            ry -= ply;
            jw += my;
          }
          else if (j >= my) {
            ry += ply;
            jw -= my;
          }
          jkoff = (koff + jw)*(long)mx;
          ryrz2 = ry*ry + rz2;

#ifdef MSMPOT_CHECK_CYLINDER_CPU
          /* clipping to the cylinder is faster */
          if (ryrz2 >= a2) continue;
#endif

#if 0
#if defined(__INTEL_COMPILER)
          for (i = ia;  i <= ib;  i++, pem++, rx += dx) {
            r2 = rx*rx + ryrz2;
            s = r2 * inv_a2;
            gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            *pem += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED! */
          }
#else
          for (i = ia;  i <= ib;  i++, pem++, rx += dx) {
            r2 = rx*rx + ryrz2;
            if (r2 >= a2) continue;
            s = r2 * inv_a2;
            gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            *pem += e;
          }
#endif
#else
          for (i = ia;  i <= ib;  i++) {
            rx = i*dx - x;
            iw = i;
            if (i < 0) {
              rx -= plx;
              iw += mx;
            }
            else if (i >= mx) {
              rx += plx;
              iw -= mx;
            }
            index = jkoff + iw;
            r2 = rx*rx + ryrz2;
            if (r2 >= a2) continue;
            s = r2 * inv_a2;
            SPOLY(&gs, s, split);  /* macro expands into switch */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            epotmap[index] += e;
          }
#endif

        }
      } /* end loop over surrounding map points */

    } /* end loop over atoms in grid cell */
  } /* end loop over grid cells */

  return MSMPOT_SUCCESS;
} /* linklist_evaluation() */
