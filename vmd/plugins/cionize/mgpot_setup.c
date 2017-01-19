/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_setup.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mgpot_defn.h"


int mgpot_setup(Mgpot *mg, float h, float a,
    long int nx, long int ny, long int nz,
    int scalexp, int interp, int split,
    const float *atoms, float *grideners,
    long int numplane, long int numcol, long int numpt, long int numatoms,
    float gridspacing, const unsigned char *excludepos,
    int numprocs, int emethod) {

  /*
   * must increase MAX_NU if we add higher order interpolants
   *
   * here, nu gives size of edge around mgpot grid for an interpolant
   * (whereas nu in thesis refers to stencil size)
   */
#define MAX_NU 2
#define STENCIL (2*MAX_NU + 2)

  long n;
  long gindex, ncell;
  float x, y, z, xmax, ymax, zmax;
  int maxlevels;
  int nu, omega, omega3;
  int ia, ib, ja, jb, ka, kb;
  long lastnelems, nelems;
  int level;
  float factor;
  int dim;
  float *gd;
  int i, j, k;
  int index;
  float s, t, gs, gt;
  int ni, nj, nk;
#ifdef MGPOT_FACTOR_INTERP
  int sdelta;
  float h_ionize;
#else
  int dim3;
  float (*phi_stencil)[STENCIL];
  float **phi;
  float h_ionize;
  floatLattice *w;
  float *wdata;
  int ii, jj, kk;
#endif

  /* check parameters */
  ASSERT(h > 0);
  ASSERT(a > h);
  ASSERT(nx > 1);
  ASSERT(ny > 1);
  ASSERT(nz > 1);
  ASSERT(scalexp >= 0);
  ASSERT(INTERP_NONE < interp && interp < INTERP_MAX);
  ASSERT(SPLIT_NONE < split && split < SPLIT_MAX);

  memset(mg, 0, sizeof(Mgpot));

  mg->atoms = atoms;
  mg->grideners = grideners;
  mg->numplane = numplane;
  mg->numcol = numcol;
  mg->numpt = numpt;
  mg->numatoms = numatoms;
  mg->gridspacing = gridspacing;
  mg->excludepos = excludepos;

  if (numprocs > 1) {
    mg->grideners_longrng
      = (float *) calloc(numplane*numcol*numpt, sizeof(float));
    if (NULL==mg->grideners_longrng) {
      return ERROR("calloc() failed\n");
    }
    mg->separate_longrng = 1;
  }
  else {
    mg->grideners_longrng = grideners;
  }

  /* mg->use_cuda = (emethod & (MLATCUTMASK | MBINMASK)); */
  mg->use_cuda = MGPOTUSECUDA(emethod);
#if ! defined(CUDA)
  if (mg->use_cuda) {
    return ERROR("CUDA support must be enabled\n");
  }
#endif

#if defined(MGPOT_GEOMHASH)
  /*
   * setup grid cells for geometric hashing of atoms
   */
  mg->inv_cellen = 1.f / MGPOT_CELLEN;
  /*
   * find max extent
   * note that system has already been shifted so that min extent is zero
   */
  xmax = ymax = zmax = 0;
  for (n = 0;  n < numatoms;  n++) {
    x = atoms[ INDEX_X(n) ];
    y = atoms[ INDEX_Y(n) ];
    z = atoms[ INDEX_Z(n) ];
    xmax = (xmax < x ? x : xmax);
    ymax = (ymax < y ? y : ymax);
    zmax = (zmax < z ? z : zmax);
  }

  /* number of cells in each dimension */
  mg->nxcell = (long int) floorf(xmax * mg->inv_cellen) + 1;
  mg->nycell = (long int) floorf(ymax * mg->inv_cellen) + 1;
  mg->nzcell = (long int) floorf(zmax * mg->inv_cellen) + 1;
  ncell = mg->nxcell * mg->nycell * mg->nzcell;

  /* allocate for cursor link list implementation */
  mg->first = (long int *) malloc(ncell * sizeof(long int));
  if (NULL==mg->first) {
    return ERROR("malloc() failed\n");
  }
  for (gindex = 0;  gindex < ncell;  gindex++) {
    mg->first[gindex] = -1;
  }
  mg->next = (long int *) malloc(numatoms * sizeof(long int));
  if (NULL==mg->next) {
    return ERROR("malloc() failed\n");
  }
  for (n = 0;  n < numatoms;  n++) {
    mg->next[n] = -1;
  }
#if defined(PERFPROF)
  mg->gnum = (long int *) calloc(ncell, sizeof(long int));
  if (NULL==mg->gnum) {
    return ERROR("calloc() failed\n");
  }
#endif
#endif /* MGPOT_GEOMHASH */

  /*
   * store scalar data members
   */
  mg->h = h;
  mg->h_1 = 1/h;
  mg->a = a;
  mg->a_1 = 1/a;
  mg->interp = interp;
  mg->split = split;
  mg->scalexp = scalexp;
  DEBUG( printf("interp=%d  split=%d\n", interp, split); );
  DEBUG( printf("scalexp=%d\n", scalexp); );

  /*
   * determine nlevels and allocate grid hierarchy
   *
   * number of grid levels shouldn't exceed:  maxlevels = lg(n)+1,
   * where n is max number of grid points among each dimension
   *
   * use this to allocate arrays qgrid of charge and egrid of potential
   */
  n = nx;
  if (n < ny) n = ny;
  if (n < nz) n = nz;
  for (maxlevels = 1;  n > 0;  n >>= 1)  maxlevels++;
  DEBUG( printf("maxlevels=%d\n", maxlevels); );

  mg->qgrid = (floatLattice **) calloc(maxlevels, sizeof(floatLattice *));
  mg->egrid = (floatLattice **) calloc(maxlevels, sizeof(floatLattice *));
  if (NULL==mg->qgrid || NULL==mg->egrid) {
    return ERROR("can\'t calloc() length %d arrays of floatLattice*\n",
        maxlevels);
  }

  /* set grid hierarchy recurrence parameters */
  switch (interp) {
    case CUBIC:
      nu = 1;
      omega = 6;
      break;
    case QUINTIC1:
      nu = 2;
      omega = 10;
      break;
    default:
      return ERROR("unknown interpolant\n");
  }
  omega3 = omega * omega * omega;
  DEBUG( printf("omega=%d  omega3=%d\n", omega, omega3); );
  /* set endpoints for finest level grid */
  ia = ja = ka = -nu;
  ib = nx + nu;
  jb = ny + nu;
  kb = nz + nu;
  if (mg->use_cuda) {
    /* when using cuda for lattice cutoff, make top level small as possible */
    lastnelems = omega3;
  }
  else {
    /* determine:  lastnelems = sqrt(nelems for finest level grid) */
    lastnelems = (long int) sqrtf( (ib-ia+1) * (jb-ja+1) * (kb-ka+1) );
  }
  DEBUG( printf("lastnelems=%d\n", lastnelems); );
  level = 0;
  do {
    DEBUG( printf("level=%d\n", level); )
    ASSERT(level < maxlevels);
    mg->qgrid[level] = new_floatLattice(ia, ib, ja, jb, ka, kb);
    mg->egrid[level] = new_floatLattice(ia, ib, ja, jb, ka, kb);
    if (NULL==mg->qgrid[level] || NULL==mg->egrid[level]) {
      return ERROR("new_floatLattice() failed for ia=%d, ib=%d, "
          "ja=%d, jb=%d, ka=%d, kb=%d\n", ia, ib, ja, jb, ka, kb);
    }
    nelems = (ib-ia+1) * (jb-ja+1) * (kb-ka+1);
    DEBUG( printf("nelems=%d\n", nelems); );
    DEBUG( printf("ia=%d  ib=%d  ja=%d  jb=%d  ka=%d  kb=%d\n",
          ia, ib, ja, jb, ka, kb); );
    ASSERT( (mg->qgrid[level])->nelems == nelems);
    ia = -((-ia+1)/2) - nu;
    ja = -((-ja+1)/2) - nu;
    ka = -((-ka+1)/2) - nu;
    ib = (ib+1)/2 + nu;
    jb = (jb+1)/2 + nu;
    kb = (kb+1)/2 + nu;
    level++;
  } while (nelems > lastnelems && nelems > omega3);
  mg->nlevels = level;

  /*
   * allocate storage for and calculate gdsum:  g(r) direct sum weights
   */
  mg->gdsum = (floatLattice **) calloc(mg->nlevels, sizeof(floatLattice *));
  if (NULL==mg->gdsum) {
    return ERROR("can\'t calloc() length %d array of floatLattice *\n",
        mg->nlevels);
  }

  /* radius for direct sum weights */
  n = (long int) ceilf(2*a/h) - 1;
  factor = 1;
  for (level = 0;  level < mg->nlevels-1;  level++) {
    mg->gdsum[level] = new_floatLattice(-n, n, -n, n, -n, n);
    if (NULL==mg->gdsum[level]) {
      return ERROR("new_floatLattice() failed for cube ia=ja=ka=%d, "
          "ib=jb=kb=%d\n", -n, n);
    }
    gd = (mg->gdsum[level])->data(mg->gdsum[level]);
    dim = 2*n + 1;
    for (k = -n;  k <= n;  k++) {
      for (j = -n;  j <= n;  j++) {
        for (i = -n;  i <= n;  i++) {
          index = (k*dim + j)*dim + i;
          ASSERT((mg->gdsum[level])->index(mg->gdsum[level], i, j, k) == index);
          s = (i*i + j*j + k*k) * h*h / (a*a);
          t = 0.25f * s;
          if (t >= 1) {
            gd[index] = 0;
          }
          else if (s >= 1) {
            gs = 1/sqrtf(s);
            mgpot_split(&gt, t, split);
            gd[index] = factor * (gs - 0.5f * gt)/a;
          }
          else {
            mgpot_split(&gs, s, split);
            mgpot_split(&gt, t, split);
            gd[index] = factor * (gs - 0.5f * gt)/a;
          }
        }
      }
    } /* end loop over gdsum weights for this grid level */
    factor *= 0.5f;
  } /* end loop over grid levels */

  /* compute coefficients for the last level - expect much bigger radius */
  ni = (mg->qgrid[level])->ni - 1;
  nj = (mg->qgrid[level])->nj - 1;
  nk = (mg->qgrid[level])->nk - 1;
  mg->gdsum[level] = new_floatLattice(-ni, ni, -nj, nj, -nk, nk);
  if (NULL==mg->gdsum[level]) {
    return ERROR("new_floatLattice() failed for ia=%d, ib=%d, ja=%d, jb=%d, "
        "ka=%d, kb=%d\n", -ni, ni, -nj, nj, -nk, nk);
  }
  gd = (mg->gdsum[level])->data( mg->gdsum[level] );
  for (k = -nk;  k <= nk;  k++) {
    for (j = -nj;  j <= nj;  j++) {
      for (i = -ni;  i <= ni;  i++) {
        index = (k*(2*nj+1) + j)*(2*ni+1) + i;
        ASSERT((mg->gdsum[level])->index(mg->gdsum[level], i, j, k) == index);
        s = (i*i + j*j + k*k) * h*h / (a*a);
        if (s >= 1) {
          gs = 1/sqrtf(s);
        }
        else {
          mgpot_split(&gs, s, split);
        }
        gd[index] = factor * gs/a;
      }
    }
  } /* end loop over gdsum weights for last grid level */

#ifdef MGPOT_FACTOR_INTERP
  dim = (1 << scalexp);
  sdelta = 2*nu*dim - 1;  /* radius of stencil on potential lattice */
  mg->eyzd = (float *) calloc(numcol*numplane, sizeof(float));
  mg->ezd = (float *) calloc(numplane, sizeof(float));
  mg->phibuffer = (float *) calloc(2*sdelta + 1, sizeof(float));
  mg->phi = mg->phibuffer + sdelta;  /* index -sdelta..sdelta */
  mg->nu = nu;
  mg->sdelta = sdelta;

  h_ionize = 1.f / dim;  /* grideners[] spacing normalized to h_mgpot=1 */
  switch (interp) {
    case CUBIC:
      for (i = -sdelta;  i <= sdelta;  i++) {
        t = i * h_ionize;
        if (t <= -1) {
          mg->phi[i] = 0.5f * (1 + t) * (2 + t) * (2 + t);
        }
        else if (t <= 0) {
          mg->phi[i] = (1 + t) * (1 - t - 1.5f * t * t);
        }
        else if (t <= 1) {
          mg->phi[i] = (1 - t) * (1 + t - 1.5f * t * t);
        }
        else if (t <= 2) {
          mg->phi[i] = 0.5f * (1 - t) * (2 - t) * (2 - t);
        }
        else {
          mg->phi[i] = 0;
        }
      }
      break;
    case QUINTIC1:
      for (i = -sdelta;  i <= sdelta;  i++) {
        t = i * h_ionize;
        if (t <= -2) {
          mg->phi[i] = (1.f/24) * (1+t) * (2+t) * (3+t) * (3+t) * (4+t);
        }
        else if (t <= -1) {
          mg->phi[i] = (1+t)*(2+t)*(3+t) * ((1.f/6) - t*(0.375f + (5.f/24)*t));
        }
        else if (t <= 0) {
          mg->phi[i] = (1-t*t) * (2+t) * (0.5f - t * (0.25f + (5.f/12)*t));
        }
        else if (t <= 1) {
          mg->phi[i] = (1-t*t) * (2-t) * (0.5f + t * (0.25f - (5.f/12)*t));
        }
        else if (t <= 2) {
          mg->phi[i] = (1-t)*(2-t)*(3-t) * ((1.f/6) + t*(0.375f - (5.f/24)*t));
        }
        else if (t <= 3) {
          mg->phi[i] = (1.f/24) * (1-t) * (2-t) * (3-t) * (3-t) * (4-t);
        }
        else {
          mg->phi[i] = 0;
        }
      }
      break;
    default:
      return ERROR("unknown interpolant\n");
  }
#else
  /*
   * allocate storage for and calculate potinterp lattices:
   * interpolation weights for computing grideners potentials
   * (a lattice of lattices)
   */

  /* length of potinterp array of lattices in one dimension */
  dim = (1 << scalexp);

  /* total length of potinterp array of lattices */
  dim3 = dim * dim * dim;

  mg->potinterp = (floatLattice **) calloc(dim3, sizeof(floatLattice *));
  if (NULL==mg->potinterp) {
    return ERROR("can\'t calloc() length %d array of floatLattice *\n", dim3);
  }

  /*
   * compute basis function contributions along one dimension
   * (requires 2*nu+2 space for each)
   */
  ASSERT(nu <= MAX_NU);
  phi_stencil = (float (*)[STENCIL]) calloc(dim, sizeof(float[STENCIL]));
  phi = (float **) calloc(dim, sizeof(float *));

  /* want to access phi[-nu..nu+1] */
  for (i = 0;  i < dim;  i++) {
    phi[i] = &(phi_stencil[i][nu]);
  }

  h_ionize = 1.f / dim;  /* grideners[] spacing normalized to h_mgpot=1 */

  switch (interp) {
    case CUBIC:
      phi[0][0] = 1;  /* for phi[0], grideners[] sits on mgpot grid point */
      for (i = 1;  i < dim;  i++) {
        t = i*h_ionize + 1;  /* nu=1 */
        phi[i][-1] = 0.5f * (1 - t) * (2 - t) * (2 - t);
        t--;
        phi[i][0] = (1 - t) * (1 + t - 1.5f * t * t);
        t--;
        phi[i][1] = (1 + t) * (1 - t - 1.5f * t * t);
        t--;
        phi[i][2] = 0.5f * (1 + t) * (2 + t) * (2 + t);
      }
      break;
    case QUINTIC1:
      phi[0][0] = 1;  /* for phi[0], grideners[] sits on mgpot grid point */
      for (i = 1;  i < dim;  i++) {
        t = i*h_ionize + 2;  /* nu=2 */
        phi[i][-2] = (1.f/24) * (1-t) * (2-t) * (3-t) * (3-t) * (4-t);
        t--;
        phi[i][-1] = (1-t)*(2-t)*(3-t) * ((1.f/6) + t * (0.375f - (5.f/24)*t));
        t--;
        phi[i][0] = (1-t*t) * (2-t) * (0.5f + t * (0.25f - (5.f/12)*t));
        t--;
        phi[i][1] = (1-t*t) * (2+t) * (0.5f - t * (0.25f + (5.f/12)*t));
        t--;
        phi[i][2] = (1+t)*(2+t)*(3+t) * ((1.f/6) - t * (0.375f + (5.f/24)*t));
        t--;
        phi[i][3] = (1.f/24) * (1+t) * (2+t) * (3+t) * (3+t) * (4+t);
      }
      break;
    default:
      return ERROR("unknown interpolant\n");
  }

  /* allocate and calculate each potinterp lattice */
  for (k = 0;  k < dim;  k++) {
    ka = (k==0 ? 0 : -nu);
    kb = (k==0 ? 0 : nu+1);
    for (j = 0;  j < dim;  j++) {
      ja = (j==0 ? 0 : -nu);
      jb = (j==0 ? 0 : nu+1);
      for (i = 0;  i < dim;  i++) {
        ia = (i==0 ? 0 : -nu);
        ib = (i==0 ? 0 : nu+1);
        index = (k*dim + j)*dim + i;
        mg->potinterp[index] = new_floatLattice(ia, ib, ja, jb, ka, kb);
        if (NULL==mg->potinterp[index]) {
          return ERROR("new_floatLattice() failed for ia=%d, ib=%d, "
              "ja=%d, jb=%d, ka=%d, kb=%d\n", ia, ib, ja, jb, ka, kb);
        }
        w = mg->potinterp[index];
        wdata = w->data(w);
        ni = w->ni;
        nj = w->nj;
        nk = w->nk;
        for (kk = ka;  kk <= kb;  kk++) {
          for (jj = ja;  jj <= jb;  jj++) {
            for (ii = ia;  ii <= ib;  ii++) {
              index = (kk*nj + jj)*ni + ii;
              ASSERT(w->index(w, ii, jj, kk) == index);
              wdata[index] = phi[i][ii] * phi[j][jj] * phi[k][kk];
            }
          }
        } /* end loop over this lattice */
      }
    }
  } /* end loop over all lattices */

  free(phi);
  free(phi_stencil);
#endif /* MGPOT_FACTOR_INTERP */

  return 0;
}


int mgpot_cleanup(Mgpot *mg) {
#ifdef MGPOT_FACTOR_INTERP
  int i;
#else
  int i, dim, dim3;
#endif

  ASSERT(mg != NULL);

#if defined(MGPOT_GEOMHASH)
  free(mg->first);
  free(mg->next);
#if defined(PERFPROF)
  free(mg->gnum);
#endif
#endif

  if (mg->separate_longrng) {
    free(mg->grideners_longrng);
  }
  for (i = 0;  i < mg->nlevels;  i++) {
    delete_floatLattice(mg->qgrid[i]);
    delete_floatLattice(mg->egrid[i]);
    delete_floatLattice(mg->gdsum[i]);
  }
  free(mg->qgrid);
  free(mg->egrid);
  free(mg->gdsum);
#ifdef MGPOT_FACTOR_INTERP
  free(mg->phibuffer);
  free(mg->ezd);
  free(mg->eyzd);
#else
  dim = (1 << mg->scalexp);
  dim3 = dim * dim * dim;
  for (i = 0;  i < dim3;  i++) {
    delete_floatLattice(mg->potinterp[i]);
  }
  free(mg->potinterp);
#endif
  return 0;
}
