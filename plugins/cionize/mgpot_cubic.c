/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 *
 * mgpot_cubic.c - smooth cubic "numerical Hermite" interpolant
 */

#include <math.h>

#undef MGPOT_TIMER
#define MGPOT_TIMER

#include "mgpot_defn.h"
#if defined(CUDA)
#include "mgpot_cuda.h"
#endif

#ifdef MGPOT_TIMER
#include "util.h"    /* timer code taken from Tachyon */
#define TIMING(x)  do { x } while (0)
#else
#define TIMING(x)
#endif

static int anterpolation(Mgpot *mg, const float *atoms, long int natoms);
static int interpolation(Mgpot *mg, float *grideners,
    long int numplane, long int numcol, long int numpt,
    const unsigned char *excludepos);
static int restriction(Mgpot *mg, int level);
static int prolongation(Mgpot *mg, int level);
static int latticecutoff(Mgpot *mg, int level);


int mgpot_longrng_cubic(Mgpot *mg) {
  const float *atoms = mg->atoms;
  float *grideners = mg->grideners_longrng;
  const long int numplane = mg->numplane;
  const long int numcol = mg->numcol;
  const long int numpt = mg->numpt;
  const long int natoms = mg->numatoms;
  const unsigned char *excludepos = mg->excludepos;

  int level;

#ifdef MGPOT_TIMER
  rt_timerhandle timer = rt_timer_create();
  float totaltime=0, lasttime, elapsedtime;
  float bench_latcut_cpu = 0;
  float bench_latcut_gpu = 0;
#endif

  if (mg->use_cuda & MLATCUTMASK) {
#if ! defined(CUDA)
    return ERROR("CUDA is not enabled\n");
#else

#undef DEBUGGING
#ifdef DEBUGGING
    floatLattice *q0 = mg->qgrid[0];
    floatLattice *e0 = mg->egrid[0];
    floatLattice *e1 = mg->egrid[1];
    floatLattice *e2 = mg->egrid[2];
    floatLattice *qalt[10];  // debug
    floatLattice *ealt[10];  // debug
    floatLattice **eorig;
#endif

    TIMING( rt_timer_start(timer); );

    /* finest level charge grid must be zeroed */
    (mg->qgrid[0])->zero(mg->qgrid[0]);

    /* anterpolation */
    TIMING( lasttime = rt_timer_timenow(timer); );
    if (anterpolation(mg, atoms, natoms)) {
      return ERROR("anterpolation() failed\n");
    }
    TIMING(
      elapsedtime = rt_timer_timenow(timer) - lasttime;
      printf(  "  time for anterpolation:             %.5f\n", elapsedtime);
      totaltime += elapsedtime;
    );

    /*
     * do all restriction operations first, then use cuda to perform
     * lattice cutoff simultaneously on all intermediate levels
     */

    for (level = 0;  level < mg->nlevels - 1;  level++) {

      /* restriction */
      TIMING( lasttime = rt_timer_timenow(timer); );
      if (restriction(mg, level)) {
        return ERROR("restriction() failed\n");
      }
      TIMING(
        elapsedtime = rt_timer_timenow(timer) - lasttime;
        printf("  time for level %d->%d restriction:    %.5f\n",
          level, level+1, elapsedtime);
        totaltime += elapsedtime;
      );
    }

#ifdef DEBUGGING
    // debug

#if 0
    /* clear all charges except for (4,4,4) on level 0 */
    for (level = 0;  level < mg->nlevels;  level++) {
      floatLattice *q = mg->qgrid[level];
      long memsz = q->ni * q->nj * q->nk * sizeof(float);
      memset(q->buffer, 0, memsz);
      if (0==level) {
        q->buffer[(4*q->nj + 4)*q->ni + 4] = 100;
      }
    }
#endif

    for (level = 0;  level < mg->nlevels;  level++) {
      long memsz;
      floatLattice *qg = mg->qgrid[level];
      floatLattice *eg = mg->egrid[level];
      qalt[level] = new_floatLattice(qg->ia, qg->ib, qg->ja, qg->jb,
          qg->ka, qg->kb);
      ealt[level] = new_floatLattice(eg->ia, eg->ib, eg->ja, eg->jb,
          eg->ka, eg->kb);
      memsz = qg->ni * qg->nj * qg->nk * sizeof(float);
      memcpy(qalt[level]->buffer, qg->buffer, memsz);
    }
    printf("qgrid(7,7,7):  q=%g  qalt=%g\n",
        q0->buffer[ (7*q0->nj + 7)*q0->ni + 7 ],
        qalt[0]->buffer[ (7*q0->nj + 7)*q0->ni + 7 ]);

    level = mg->nlevels-1;
    // end debug
#endif

    /* cuda lattice cutoff for intermediate levels */
    TIMING( lasttime = rt_timer_timenow(timer); );
    if (mgpot_cuda_condense_qgrids(mg)) {
      return ERROR("mgpot_cuda_condense_qgrids() failed\n");
    }
    switch (mg->use_cuda & MLATCUTMASK) {
      case MLATCUT01:
        printf("Calling latcut01 CUDA kernel\n");
        if (mgpot_cuda_latcut01(mg)) {
          return ERROR("mgpot_cuda_latcut01() failed\n");
        }
        break;
      case MLATCUT02:
        printf("Calling latcut02 CUDA kernel\n");
        if (mgpot_cuda_latcut02(mg)) {
          return ERROR("mgpot_cuda_latcut02() failed\n");
        }
        break;
      case MLATCUT03:
        printf("Calling latcut03 CUDA kernel\n");
        if (mgpot_cuda_latcut03(mg)) {
          return ERROR("mgpot_cuda_latcut03() failed\n");
        }
        break;
      case MLATCUT04:
        printf("Calling latcut04 CUDA kernel\n");
        if (mgpot_cuda_latcut04(mg)) {
          return ERROR("mgpot_cuda_latcut04() failed\n");
        }
        break;
      default:
        return ERROR("unrecognized CUDA lattice cutoff method\n");
    }
    if (mgpot_cuda_expand_egrids(mg)) {
      return ERROR("mgpot_cuda_expand_egrids() failed\n");
    }
    TIMING(
      elapsedtime = rt_timer_timenow(timer) - lasttime;
      printf(  "  time for levels %d..%d latcutoff:     %.5f\n",
        0, mg->lk_nlevels-1, elapsedtime);
      bench_latcut_gpu = elapsedtime;
      printf(  "BENCH_latcut_gpu:  %.5f\n", bench_latcut_gpu);
      totaltime += elapsedtime;
    );

#ifdef DEBUGGING
    // debug
    /*
    printf("egrid(7,7,7):  e=%g\n",
        e0->buffer[ (7*e0->nj + 7)*e0->ni + 7 ]);
    printf("egri1(7,7,7):  e=%g\n",
        e1->buffer[ (7*e1->nj + 7)*e1->ni + 7 ]);
    printf("egri2(7,7,7):  e=%g\n",
        e2->buffer[ (7*e2->nj + 7)*e2->ni + 7 ]);
        */

    /* swap out mg->egrid with ealt */
    eorig = mg->egrid;
    mg->egrid = (floatLattice **) calloc(mg->nlevels, sizeof(floatLattice *));
    memcpy(mg->egrid, ealt, mg->nlevels * sizeof(floatLattice *));

    for (level = 0;  level < mg->nlevels - 1;  level++) {

      /* lattice cutoff */
      TIMING( lasttime = rt_timer_timenow(timer); );
      if (latticecutoff(mg, level)) {
        return ERROR("latticecutoff() failed\n");
      }
      TIMING(
        elapsedtime = rt_timer_timenow(timer) - lasttime;
        printf("  time for level %d lattice cutoff:    %.5f\n",
          level, elapsedtime);
        totaltime += elapsedtime;
      );

    }

    /*
    printf("egrid(7,7,7):  e=%g  ealt=%g\n",
        e0->buffer[ (7*e0->nj + 7)*e0->ni + 7 ],
        ealt[0]->buffer[ (7*e0->nj + 7)*e0->ni + 7 ]);
    printf("egri1(7,7,7):  e=%g  ealt=%g\n",
        e1->buffer[ (7*e1->nj + 7)*e1->ni + 7 ],
        ealt[1]->buffer[ (7*e1->nj + 7)*e1->ni + 7 ]);
    printf("egri2(7,7,7):  e=%g  ealt=%g\n",
        e2->buffer[ (7*e2->nj + 7)*e2->ni + 7 ],
        ealt[2]->buffer[ (7*e2->nj + 7)*e2->ni + 7 ]);
        */

    /*
    printf("exact[0]=%.15e  eorig[0]=%.15e\n",
        ealt[5]->buffer[0], eorig[5]->buffer[0]);
    printf("exact[4]=%.15e  eorig[4]=%.15e\n",
        ealt[5]->buffer[4], eorig[5]->buffer[4]);
        */

    for (level = 0;  level < 6;  level++) {
      int ni = eorig[level]->ni;
      int nj = eorig[level]->nj;
      int nk = eorig[level]->nk;
      int i, j, k;
      printf("Dimensions level %d:  ni=%d  nj=%d  nk=%d\n",
          level, ni, nj, nk);
      for (k = 0;  k < nk;  k++) {
        for (j = 0;  j < nj;  j++) {
          for (i = 0;  i < ni;  i++) {
            float ep = eorig[level]->buffer[(k*nj + j)*ni +i];
            float e = ealt[level]->buffer[(k*nj + j)*ni +i];
            float err = 100*fabsf(ep-e)/fabsf(e);
            if (e != 0 && err > 1) {
              printf("level=%d  i=%d  j=%d  k=%d  e_exact=%g  e_gpu=%g  "
                  "err=%g\n", level, i, j, k, e, ep, err);
            }
          }
        }
      }
    }


    level = mg->nlevels-1;
    // end debug
#endif

    /* lattice cutoff - top level */
    TIMING( lasttime = rt_timer_timenow(timer); );
    if (latticecutoff(mg, level)) {
      return ERROR("latticecutoff() failed\n");
    }
    TIMING(
      elapsedtime = rt_timer_timenow(timer) - lasttime;
      printf(  "  time for level %d (top) latcutoff:   %.5f\n",
        level, elapsedtime);
      bench_latcut_cpu = elapsedtime;
      printf(  "BENCH_latcut_cpu:  %.5f\n", bench_latcut_cpu);
      printf(  "BENCH_latcut_total:  %.5f\n",
        bench_latcut_gpu + bench_latcut_cpu);
      totaltime += elapsedtime;
    );

    for (level--;  level >= 0;  level--) {

      /* prolongation */
      TIMING( lasttime = rt_timer_timenow(timer); );
      if (prolongation(mg, level)) {
        return ERROR("prolongation() failed\n");
      }
      TIMING(
        elapsedtime = rt_timer_timenow(timer) - lasttime;
        printf("  time for level %d->%d prolongation:   %.5f\n",
          level+1, level, elapsedtime);
        totaltime += elapsedtime;
      );
    }

    /* interpolation */
    TIMING( lasttime = rt_timer_timenow(timer); );
    if (interpolation(mg, grideners, numplane, numcol, numpt, excludepos)) {
      return ERROR("interpolation() failed\n");
    }
    TIMING(
      elapsedtime = rt_timer_timenow(timer) - lasttime;
      printf(  "  time for interpolation:             %.5f\n", elapsedtime);
      totaltime += elapsedtime;
    );

    TIMING(
      printf(  "total time for cubic interpolation:   %.5f\n", totaltime);
      printf(  "BENCH_long_range:  %.5f\n", totaltime);
    );
#endif
  }
  else {

    TIMING( rt_timer_start(timer); );

    /* finest level charge grid must be zeroed */
    (mg->qgrid[0])->zero(mg->qgrid[0]);

    /* computation is "inverse V-cycle" */

    /* anterpolation */
    TIMING( lasttime = rt_timer_timenow(timer); );
    if (anterpolation(mg, atoms, natoms)) {
      return ERROR("anterpolation() failed\n");
    }
    TIMING(
      elapsedtime = rt_timer_timenow(timer) - lasttime;
      printf(  "  time for anterpolation:             %.5f\n", elapsedtime);
      totaltime += elapsedtime;
    );

    for (level = 0;  level < mg->nlevels - 1;  level++) {

      /* lattice cutoff */
      TIMING( lasttime = rt_timer_timenow(timer); );
      if (latticecutoff(mg, level)) {
        return ERROR("latticecutoff() failed\n");
      }
      TIMING(
        elapsedtime = rt_timer_timenow(timer) - lasttime;
        printf("  time for level %d lattice cutoff:    %.5f\n",
          level, elapsedtime);
        totaltime += elapsedtime;
        bench_latcut_cpu += elapsedtime;
      );

      /* restriction */
      TIMING( lasttime = rt_timer_timenow(timer); );
      if (restriction(mg, level)) {
        return ERROR("restriction() failed\n");
      }
      TIMING(
        elapsedtime = rt_timer_timenow(timer) - lasttime;
        printf("  time for level %d->%d restriction:    %.5f\n",
          level, level+1, elapsedtime);
        totaltime += elapsedtime;
      );
    }

    /* lattice cutoff - top level */
    TIMING( lasttime = rt_timer_timenow(timer); );
    if (latticecutoff(mg, level)) {
      return ERROR("latticecutoff() failed\n");
    }
    TIMING(
      elapsedtime = rt_timer_timenow(timer) - lasttime;
      printf(  "  time for level %d (top) latcutoff:   %.5f\n",
        level, elapsedtime);
      totaltime += elapsedtime;
      bench_latcut_cpu += elapsedtime;
      printf(  "BENCH_latcut_cpu:  %.5f\n", bench_latcut_cpu);
      printf(  "BENCH_latcut_total:  %.5f\n", bench_latcut_cpu);
    );

    for (level--;  level >= 0;  level--) {

      /* prolongation */
      TIMING( lasttime = rt_timer_timenow(timer); );
      if (prolongation(mg, level)) {
        return ERROR("prolongation() failed\n");
      }
      TIMING(
        elapsedtime = rt_timer_timenow(timer) - lasttime;
        printf("  time for level %d->%d prolongation:   %.5f\n",
          level+1, level, elapsedtime);
        totaltime += elapsedtime;
      );
    }

    /* interpolation */
    TIMING( lasttime = rt_timer_timenow(timer); );
    if (interpolation(mg, grideners, numplane, numcol, numpt, excludepos)) {
      return ERROR("interpolation() failed\n");
    }
    TIMING(
      elapsedtime = rt_timer_timenow(timer) - lasttime;
      printf(  "  time for interpolation:             %.5f\n", elapsedtime);
      totaltime += elapsedtime;
    );

    TIMING(
      printf(  "total time for cubic interpolation:   %.5f\n", totaltime);
      printf(  "BENCH_long_range:  %.5f\n", totaltime);
    );
  }

  TIMING( rt_timer_destroy(timer); );

  return 0;
}


int anterpolation(Mgpot *mg, const float *atoms, long int natoms)
{
  float xphi[4], yphi[4], zphi[4];  /* phi grid func along x, y, z */
  float dx_h, dy_h, dz_h;           /* distance from origin */
  float t;                          /* normalized distance for phi */
  float ck, cjk;
  const float h_1 = mg->h_1;
  float q;

  floatLattice *qgrid = mg->qgrid[0];
  float *qh = qgrid->data(qgrid);
  const long int ni = qgrid->ni;
  const long int nj = qgrid->nj;

  long int n, i, j, k, ilo, jlo, klo, index;
  long int koff, jkoff;

  for (n = 0;  n < natoms;  n++) {

    /* atomic charge */
    q = atoms[ INDEX_Q(n) ];

    /* distance between atom and origin measured in grid points */
    dx_h = atoms[ INDEX_X(n) ] * h_1;
    dy_h = atoms[ INDEX_Y(n) ] * h_1;
    dz_h = atoms[ INDEX_Z(n) ] * h_1;

    /* find smallest numbered grid point in stencil */
    ilo = ((long int) dx_h) - 1;
    jlo = ((long int) dy_h) - 1;
    klo = ((long int) dz_h) - 1;

    /* find t for x dimension and compute xphi */
    t = dx_h - (float) ilo;
    xphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    xphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    xphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    xphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* find t for y dimension and compute yphi */
    t = dy_h - (float) jlo;
    yphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    yphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    yphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    yphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* find t for z dimension and compute zphi */
    t = dz_h - (float) klo;
    zphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    zphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    zphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    zphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* determine charge on 64=4*4*4 grid point stencil of qh */
    for (k = 0;  k < 4;  k++) {
      koff = (k + klo) * nj;
      ck = zphi[k] * q;
      for (j = 0;  j < 4;  j++) {
        jkoff = (koff + (j + jlo)) * ni;
        cjk = yphi[j] * ck;
        for (i = 0;  i < 4;  i++) {
          index = jkoff + (i + ilo);
          ASSERT(qgrid->index(qgrid,i+ilo,j+jlo,k+klo)==index);
          qh[index] += xphi[i] * cjk;
        }
      }
    }

  } /* end loop over atoms */
  return 0;
} /* anterpolation */


int interpolation(Mgpot *mg, float *grideners,
    long int numplane, long int numcol, long int numpt,
    const unsigned char *excludepos)
{
#ifdef MGPOT_FACTOR_INTERP
  const int scalexp = mg->scalexp;
  const int sdelta = mg->sdelta;

  const float *phi = mg->phi;
  float *ezd = mg->ezd;
  float *eyzd = mg->eyzd;

  floatLattice *egrid = mg->egrid[0];
  const float *eh = egrid->data(egrid);
  const long int ia = egrid->ia;
  const long int ib = egrid->ib;
  const long int ja = egrid->ja;
  const long int jb = egrid->jb;
  const long int ka = egrid->ka;
  const long int kb = egrid->kb;
  const long int nrow_eh = egrid->ni;
  const long int nstride_eh = nrow_eh * egrid->nj;

  const long int size_eyzd = numplane * numcol * sizeof(float);
  const long int size_ezd = numplane * sizeof(float);

  long int i, j, k;
  long int ih, jh, kh;
  long int im, jm, km;
  long int lower, upper;
  long int index_eh, index_plane_eh;
  long int offset, offset_k, index_jk, index;

  printf("doing factored interpolation\n");
  for (ih = ia;  ih <= ib;  ih++) {
    memset(eyzd, 0, size_eyzd);

    for (jh = ja;  jh <= jb;  jh++) {
      memset(ezd, 0, size_ezd);
      index_plane_eh = jh * nrow_eh + ih;

      for (kh = ka;  kh <= kb;  kh++) {
        index_eh = kh * nstride_eh + index_plane_eh;
        km = (kh << scalexp);
        lower = km - sdelta;
        if (lower < 0) lower = 0;
        upper = km + sdelta;
        if (upper >= numplane) upper = numplane-1;
        for (k = lower;  k <= upper;  k++) {
          ezd[k] += phi[k-km] * eh[index_eh];
        }
      }

      for (k = 0;  k < numplane;  k++) {
        offset = k * numcol;
        jm = (jh << scalexp);
        lower = jm - sdelta;
        if (lower < 0) lower = 0;
        upper = jm + sdelta;
        if (upper >= numcol) upper = numcol-1;
        for (j = lower;  j <= upper;  j++) {
          eyzd[offset + j] += phi[j-jm] * ezd[k];
        }
      }
    }

    for (k = 0;  k < numplane;  k++) {
      offset_k = k * numcol;

      for (j = 0;  j < numcol;  j++) {
        index_jk = offset_k + j;
        offset = index_jk * numpt;

        im = (ih << scalexp);
        lower = im - sdelta;
        if (lower < 0) lower = 0;
        upper = im + sdelta;
        if (upper >= numpt) upper = numpt-1;
        for (i = lower;  i <= upper;  i++) {
          grideners[offset + i] += phi[i-im] * eyzd[index_jk];
        }
      }
    }

  }

  for (k = 0;  k < numplane;  k++) {
    offset_k = k * numcol;
    for (j = 0;  j < numcol;  j++) {
      offset = (offset_k + j) * numpt;
      for (i = 0;  i < numpt;  i++) {
        index = offset + i;
        grideners[index] = (excludepos[index] ? 0 : grideners[index]);
      }
    }
  }

#else
  float e;

  const int scalexp = mg->scalexp;
  const int dim = (1 << scalexp);
  const long int mask = (long int)(dim-1);

  floatLattice *egrid = mg->egrid[0];
  const float *eh = egrid->data(egrid);
  const long int eni = egrid->ni;
  const long int enj = egrid->nj;

  long int index;
  long int i, j, k;
  long int im, jm, km;
  long int ip, jp, kp;

  floatLattice *w;
  const float *wt;
  long int wni, wnj;
  long int ia, ib, ja, jb, ka, kb;

  long int ii, jj, kk;
  long int woff_k, woff_jk, eoff_k, eoff_jk;

  /* loop over grideners */
  for (k = 0;  k < numplane;  k++) {
    for (j = 0;  j < numcol;  j++) {
      for (i = 0;  i < numpt;  i++) {

        /* index into grideners and excludepos arrays */
        index = (k*numcol + j)*numpt + i;
        if (excludepos[index]) continue;

	/* find closest mgpot point less than or equal to */
	im = (i >> scalexp);
	jm = (j >> scalexp);
	km = (k >> scalexp);

	/* find corresponding potinterp lattice */
	ip = (int)(i & mask);
	jp = (int)(j & mask);
	kp = (int)(k & mask);

	w = mg->potinterp[(kp*dim + jp)*dim + ip];
	wt = w->data(w);
	wni = w->ni;
	wnj = w->nj;
	ia = w->ia;
	ib = w->ib;
	ja = w->ja;
	jb = w->jb;
	ka = w->ka;
	kb = w->kb;

	/* loop over wt, summing weighted eh contributions to e */
	e = 0;
	for (kk = ka;  kk <= kb;  kk++) {
	  woff_k = kk*wnj;
	  eoff_k = (km + kk)*enj;
	  for (jj = ja;  jj <= jb;  jj++) {
	    woff_jk = (woff_k + jj)*wni;
	    eoff_jk = (eoff_k + (jm+jj))*eni;
	    for (ii = ia;  ii <= ib;  ii++) {
	      ASSERT(w->index(w, ii, jj, kk) == woff_jk + ii);
	      ASSERT(egrid->index(egrid, im+ii, jm+jj, km+kk)
		  == eoff_jk + (im+ii));
	      e += wt[woff_jk + ii] * eh[eoff_jk + (im+ii)];
	    }
	  }
	}
	grideners[index] += e;
	/* end loop over wt */

      }
    }
  } /* end loop over grideners */

#endif /* MGPOT_FACTOR_INTERP */

  return 0;
} /* interpolation */



/*
 * constants for grid transfer operations
 * cubic "numerical Hermite" interpolation
 */

/* length of stencil */
enum { NSTENCIL = 5 };

/* phi interpolating function along one dimension of grid stencil */
static const float Phi[NSTENCIL] = { -0.0625f, 0.5625f, 1, 0.5625f, -0.0625f };

/* stencil offsets from a central grid point on a finer grid level */
/* (these offsets are where phi weights above have been evaluated) */
static const int Offset[NSTENCIL] = { -3, -1, 0, 1, 3 };


int restriction(Mgpot *mg, int level)
{
  float cjk, q2h_sum;

  /* lattices of charge, finer grid and coarser grid */
  floatLattice *qhgrid = mg->qgrid[level];
  const float *qh = qhgrid->data(qhgrid);
  floatLattice *q2hgrid = mg->qgrid[level+1];
  float *q2h = q2hgrid->data(q2hgrid);

  /* finer grid index ranges and dimensions */
  const long int ia1 = qhgrid->ia;  /* lowest x-index */
  const long int ib1 = qhgrid->ib;  /* highest x-index */
  const long int ja1 = qhgrid->ja;  /* lowest y-index */
  const long int jb1 = qhgrid->jb;  /* highest y-index */
  const long int ka1 = qhgrid->ka;  /* lowest z-index */
  const long int kb1 = qhgrid->kb;  /* highest z-index */
  const long int ni1 = qhgrid->ni;  /* length along x-dim */
  const long int nj1 = qhgrid->nj;  /* length along y-dim */

  /* coarser grid index ranges and dimensions */
  const long int ia2 = q2hgrid->ia;  /* lowest x-index */
  const long int ib2 = q2hgrid->ib;  /* highest x-index */
  const long int ja2 = q2hgrid->ja;  /* lowest y-index */
  const long int jb2 = q2hgrid->jb;  /* highest y-index */
  const long int ka2 = q2hgrid->ka;  /* lowest z-index */
  const long int kb2 = q2hgrid->kb;  /* highest z-index */
  const long int ni2 = q2hgrid->ni;  /* length along x-dim */
  const long int nj2 = q2hgrid->nj;  /* length along y-dim */

  /* other variables */
  long int i1, j1, k1, index1, jk1off, k1off;
  long int i2, j2, k2, index2, jk2off, k2off;
  long int i, j, k;

  /* loop over coarser grid points */
  for (k2 = ka2;  k2 <= kb2;  k2++) {
    k2off = k2 * nj2;    /* coarser grid index offset for k-coord */
    k1 = k2 * 2;         /* k-coord of same-space point on finer grid */
    for (j2 = ja2;  j2 <= jb2;  j2++) {
      jk2off = (k2off + j2) * ni2;    /* add offset for j-coord coarser */
      j1 = j2 * 2;                    /* j-coord same-space finer grid */
      for (i2 = ia2;  i2 <= ib2;  i2++) {
        index2 = jk2off + i2;         /* index in coarser grid */
        i1 = i2 * 2;                  /* i-coord same-space finer grid */

        /* sum weighted charge contribution from finer grid stencil */
        q2h_sum = 0.0;
        for (k = 0;  k < NSTENCIL;  k++) {
          /* early loop termination if outside lattice */
          if (k1 + Offset[k] < ka1) continue;
          else if (k1 + Offset[k] > kb1) break;
          k1off = (k1 + Offset[k]) * nj1;  /* offset k-coord finer grid */
          for (j = 0;  j < NSTENCIL;  j++) {
            /* early loop termination if outside lattice */
            if (j1 + Offset[j] < ja1) continue;
            else if (j1 + Offset[j] > jb1) break;
            jk1off = (k1off + (j1 + Offset[j])) * ni1;  /* add offset j */
            cjk = Phi[j] * Phi[k];              /* mult weights in each dim */
            for (i = 0;  i < NSTENCIL;  i++) {
              /* early loop termination if outside lattice */
              if (i1 + Offset[i] < ia1) continue;
              else if (i1 + Offset[i] > ib1) break;
              index1 = jk1off + (i1 + Offset[i]);    /* index in finer grid */
	      ASSERT(qhgrid->index(qhgrid,
		    i1 + Offset[i], j1 + Offset[j], k1 + Offset[k]) == index1);
              q2h_sum += Phi[i] * cjk * qh[index1];  /* sum weighted charge */
            }
          }
        } /* end loop over finer grid stencil */

	ASSERT(q2hgrid->index(q2hgrid, i2, j2, k2) == index2);
        q2h[index2] = q2h_sum;  /* store charge to coarser grid */

      }
    }
  } /* end loop over each coarser grid points */
  return 0;
}


int prolongation(Mgpot *mg, int level)
{
  float ck, cjk;

  /* lattices of charge, finer grid and coarser grid */
  floatLattice *ehgrid = mg->egrid[level];
  float *eh = ehgrid->data(ehgrid);
  floatLattice *e2hgrid = mg->egrid[level+1];
  const float *e2h = e2hgrid->data(e2hgrid);

  /* finer grid index ranges and dimensions */
  const long int ia1 = ehgrid->ia;  /* lowest x-index */
  const long int ib1 = ehgrid->ib;  /* highest x-index */
  const long int ja1 = ehgrid->ja;  /* lowest y-index */
  const long int jb1 = ehgrid->jb;  /* highest y-index */
  const long int ka1 = ehgrid->ka;  /* lowest z-index */
  const long int kb1 = ehgrid->kb;  /* highest z-index */
  const long int ni1 = ehgrid->ni;  /* length along x-dim */
  const long int nj1 = ehgrid->nj;  /* length along y-dim */

  /* coarser grid index ranges and dimensions */
  const long int ia2 = e2hgrid->ia;  /* lowest x-index */
  const long int ib2 = e2hgrid->ib;  /* highest x-index */
  const long int ja2 = e2hgrid->ja;  /* lowest y-index */
  const long int jb2 = e2hgrid->jb;  /* highest y-index */
  const long int ka2 = e2hgrid->ka;  /* lowest z-index */
  const long int kb2 = e2hgrid->kb;  /* highest z-index */
  const long int ni2 = e2hgrid->ni;  /* length along x-dim */
  const long int nj2 = e2hgrid->nj;  /* length along y-dim */

  /* other variables */
  long int i1, j1, k1, index1, jk1off, k1off;
  long int i2, j2, k2, index2, jk2off, k2off;
  long int i, j, k;

  /* loop over coarser grid points */
  for (k2 = ka2;  k2 <= kb2;  k2++) {
    k2off = k2 * nj2;    /* coarser grid index offset for k-coord */
    k1 = k2 * 2;         /* k-coord of same-space point on finer grid */
    for (j2 = ja2;  j2 <= jb2;  j2++) {
      jk2off = (k2off + j2) * ni2;    /* add offset for j-coord coarser */
      j1 = j2 * 2;                    /* j-coord same-space finer grid */
      for (i2 = ia2;  i2 <= ib2;  i2++) {
        index2 = jk2off + i2;         /* index in coarser grid */
        i1 = i2 * 2;                  /* i-coord same-space finer grid */

        /* sum weighted charge contribution from finer grid stencil */
	ASSERT(e2hgrid->index(e2hgrid, i2, j2, k2) == index2);
        for (k = 0;  k < NSTENCIL;  k++) {
          /* early loop termination if outside lattice */
          if (k1 + Offset[k] < ka1) continue;
          else if (k1 + Offset[k] > kb1) break;
          k1off = (k1 + Offset[k]) * nj1;  /* offset k-coord finer grid */
	  ck = Phi[k] * e2h[index2];
          for (j = 0;  j < NSTENCIL;  j++) {
            /* early loop termination if outside lattice */
            if (j1 + Offset[j] < ja1) continue;
            else if (j1 + Offset[j] > jb1) break;
            jk1off = (k1off + (j1 + Offset[j])) * ni1;  /* add offset j */
            cjk = Phi[j] * ck;              /* mult weights in each dim */
            for (i = 0;  i < NSTENCIL;  i++) {
              /* early loop termination if outside lattice */
              if (i1 + Offset[i] < ia1) continue;
              else if (i1 + Offset[i] > ib1) break;
              index1 = jk1off + (i1 + Offset[i]);    /* index in finer grid */
	      ASSERT(ehgrid->index(ehgrid,
		    i1 + Offset[i], j1 + Offset[j], k1 + Offset[k]) == index1);
	      eh[index1] += Phi[i] * cjk;            /* sum weighted charge */
            }
          }
        } /* end loop over finer grid stencil */

      }
    }
  } /* end loop over each coarser grid points */
  return 0;
}


int latticecutoff(Mgpot *mg, int level)
{
  float eh_sum;

  /* lattices of charge and potential */
  floatLattice *qgrid = mg->qgrid[level];
  const float *qh = qgrid->data(qgrid);
  floatLattice *egrid = mg->egrid[level];
  float *eh = egrid->data(egrid);
  const long int ia = qgrid->ia;  /* lowest x-index */
  const long int ib = qgrid->ib;  /* highest x-index */
  const long int ja = qgrid->ja;  /* lowest y-index */
  const long int jb = qgrid->jb;  /* highest y-index */
  const long int ka = qgrid->ka;  /* lowest z-index */
  const long int kb = qgrid->kb;  /* highest z-index */
  const long int ni = qgrid->ni;  /* length along x-dim */
  const long int nj = qgrid->nj;  /* length along y-dim */

  /* lattice of weights for pairwise grid point interactions within cutoff */
  floatLattice *gdsum = mg->gdsum[level];
  const float *gd = gdsum->data(gdsum);
  const long int gia = gdsum->ia;  /* lowest x-index */
  const long int gib = gdsum->ib;  /* highest x-index */
  const long int gja = gdsum->ja;  /* lowest y-index */
  const long int gjb = gdsum->jb;  /* highest y-index */
  const long int gka = gdsum->ka;  /* lowest z-index */
  const long int gkb = gdsum->kb;  /* highest z-index */
  const long int gni = gdsum->ni;  /* length along x-dim */
  const long int gnj = gdsum->nj;  /* length along y-dim */
  /*
   * although gdsum lower levels are all spherical (cubes),
   * the highest gdsum level covers entire rectangular domain,
   * so ranges for all three dimensions are necessary
   */

  long int i, j, k;
  long int gia_clip, gib_clip;
  long int gja_clip, gjb_clip;
  long int gka_clip, gkb_clip;
  long int koff, jkoff, index;
  long int id, jd, kd;
  long int knoff, jknoff, nindex;
  long int kgoff, jkgoff, ngindex;

  /* loop over all grid points */
  for (k = ka;  k <= kb;  k++) {

    /* clip gdsum ranges to keep offset for k index within grid */
    gka_clip = (k + gka < ka ? ka - k : gka);
    gkb_clip = (k + gkb > kb ? kb - k : gkb);

    koff = k * nj;  /* find eh flat index */

    for (j = ja;  j <= jb;  j++) {

      /* clip gdsum ranges to keep offset for j index within grid */
      gja_clip = (j + gja < ja ? ja - j : gja);
      gjb_clip = (j + gjb > jb ? jb - j : gjb);

      jkoff = (koff + j) * ni;  /* find eh flat index */

      for (i = ia;  i <= ib;  i++) {

        /* clip gdsum ranges to keep offset for i index within grid */
        gia_clip = (i + gia < ia ? ia - i : gia);
        gib_clip = (i + gib > ib ? ib - i : gib);

        index = jkoff + i;  /* eh flat index */

        /* sum over "sphere" of weighted charge */
        eh_sum = 0.0;
        for (kd = gka_clip;  kd <= gkb_clip;  kd++) {
          knoff = (k + kd) * nj;  /* find qh flat index */
          kgoff = kd * gnj;       /* find gd flat index */

          for (jd = gja_clip;  jd <= gjb_clip;  jd++) {
            jknoff = (knoff + (j + jd)) * ni;  /* find qh flat index */
            jkgoff = (kgoff + jd) * gni;       /* find gd flat index */

            for (id = gia_clip;  id <= gib_clip;  id++) {
              nindex = jknoff + (i + id);  /* qh flat index */
              ngindex = jkgoff + id;       /* gd flat index */

	      ASSERT(qgrid->index(qgrid, i+id, j+jd, k+kd) == nindex);
	      ASSERT(gdsum->index(gdsum, id, jd, kd) == ngindex);
              eh_sum += qh[nindex] * gd[ngindex];  /* sum weighted charge */
            }
          }
        } /* end loop over "sphere" of charge */

	ASSERT(egrid->index(egrid, i, j, k) == index);
        eh[index] = eh_sum;  /* store potential */
      }
    }
  } /* end loop over all grid points */
  return 0;
}
