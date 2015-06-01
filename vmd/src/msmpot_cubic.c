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
 *      $RCSfile: msmpot_cubic.c,v $
 *      $Author: dhardy $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $      $Date: 2010/06/08 15:57:07 $
 *
 ***************************************************************************/

/*
 * smooth cubic "numerical Hermite" interpolation
 */

#include "msmpot_internal.h"

static int anterpolation(Msmpot *msm);
static int interpolation_factored(Msmpot *msm);
static int interpolation(Msmpot *msm);
static int restriction(Msmpot *msm, int level);
static int prolongation(Msmpot *msm, int level);
static int latticecutoff(Msmpot *msm, int level);

#undef OK
#define OK MSMPOT_SUCCESS

/*
 * The factored grid transfers are faster O(p M) versus O(p^3 M).
 * The implementation here supports periodic boundaries.
 *
 * (The non-factored implementation does not support periodic boundaries.)
 */
#define USE_FACTORED_GRID_TRANSFERS


/*
#define MSMPOT_CHECKMAPINDEX
#undef MSMPOT_CHECKMAPINDEX

#define MAPINDEX 0

#define MSMPOT_CHECKSUM
#undef MSMPOT_CHECKSUM
*/


int Msmpot_compute_longrng_cubic(Msmpot *msm) {
  int do_cpu_latcut = 1;  /* flag set to calculate lattice cutoff on CPU */
  int err = 0;
  int level;
  int nlevels = msm->nlevels;
#ifdef MSMPOT_REPORT
  char msg[120];
#endif

  REPORT("Using cubic interpolation for long-range part.");

  REPORT("Doing anterpolation.");
  err = anterpolation(msm);
  if (err) return ERROR(err);

  for (level = 0;  level < nlevels - 1;  level++) {
#ifdef MSMPOT_REPORT
    sprintf(msg, "Doing restriction of grid level %d.", level);
    REPORT(msg);
#endif
    err = restriction(msm, level);
    if (err) return ERROR(err);
  }

#ifdef MSMPOT_CUDA
  if (msm->use_cuda_latcut && nlevels > 1) {
#ifdef MSMPOT_REPORT
    sprintf(msg, "Computing lattice cutoff part with CUDA for grid %s %d.",
        (nlevels > 2 ? "levels 0 -" : "level "), nlevels-2);
    REPORT(msg);
#endif
    do_cpu_latcut = 0;
    if ((err = Msmpot_cuda_condense_qgrids(msm->msmcuda)) != OK ||
        (err = Msmpot_cuda_compute_latcut(msm->msmcuda)) != OK ||
        (err = Msmpot_cuda_expand_egrids(msm->msmcuda)) != OK) {
      if (msm->cuda_optional) {
        REPORT("Falling back on CPU for lattice cutoff part.");
        do_cpu_latcut = 1;  /* fall back on CPU latcut */
      }
      else return ERROR(err);
    }
  }
#endif

  if (do_cpu_latcut) {
    for (level = 0;  level < nlevels - 1;  level++) {
#ifdef MSMPOT_REPORT
      sprintf(msg, "Doing cutoff calculation on grid level %d.", level);
      REPORT(msg);
#endif
      err = latticecutoff(msm, level);
      if (err) return ERROR(err);
    }
  }

#ifdef MSMPOT_REPORT
  sprintf(msg, "Doing cutoff calculation on grid level %d.", level);
  REPORT(msg);
#endif
  err = latticecutoff(msm, level);  /* top level */
  if (err) return ERROR(err);

  for (level--;  level >= 0;  level--) {
#ifdef MSMPOT_REPORT
    sprintf(msg, "Doing prolongation to grid level %d.", level);
    REPORT(msg);
#endif
    err = prolongation(msm, level);
    if (err) return ERROR(err);
  }

#ifdef MSMPOT_VERBOSE
#ifdef MSMPOT_CHECKMAPINDEX
  printf("epotmap[%d]=%g\n", MAPINDEX, msm->epotmap[MAPINDEX]);
#endif
#endif

  if (msm->px == msm->lx && msm->py == msm->ly && msm->pz == msm->lz) {
    REPORT("Doing factored interpolation.");
    err = interpolation_factored(msm);
  }
  else {
    REPORT("Doing non-factored interpolation.");
    err = interpolation(msm);  /* slower */
  }
  if (err) return ERROR(err);

#ifdef MSMPOT_VERBOSE
#ifdef MSMPOT_CHECKMAPINDEX
  printf("epotmap[%d]=%g\n", MAPINDEX, msm->epotmap[MAPINDEX]);
#endif
#endif
  return OK;
}


int anterpolation(Msmpot *msm)
{
  const float *atom = msm->atom;
  const int natoms = msm->natoms;

  float xphi[4], yphi[4], zphi[4];  /* phi grid func along x, y, z */
  float rx_hx, ry_hy, rz_hz;        /* distance from origin */
  float t;                          /* normalized distance for phi */
  float ck, cjk;
  const float hx_1 = 1/msm->hx;
  const float hy_1 = 1/msm->hy;
  const float hz_1 = 1/msm->hz;
#if 1
  const float xm0 = msm->px0;
  const float ym0 = msm->py0;
  const float zm0 = msm->pz0;
#else
  const float xm0 = msm->lx0;
  const float ym0 = msm->ly0;
  const float zm0 = msm->lz0;
#endif
  float q;

  floatGrid *qhgrid = &(msm->qh[0]);
  float *qh = qhgrid->data;
  const int ni = qhgrid->ni;
  const int nj = qhgrid->nj;
  const int nk = qhgrid->nk;
  const int ia = qhgrid->i0;
  const int ib = ia + ni - 1;
  const int ja = qhgrid->j0;
  const int jb = ja + nj - 1;
  const int ka = qhgrid->k0;
  const int kb = ka + nk - 1;

  const int ispany = IS_SET_ANY(msm->isperiodic);
  int iswrap;

  int n, i, j, k, ilo, jlo, klo, koff;
  long jkoff, index;

  GRID_ZERO(qhgrid);

  for (n = 0;  n < natoms;  n++) {

    /* atomic charge */
    q = atom[4*n + 3];
    if (0==q) continue;

    /* distance between atom and origin measured in grid points */
    rx_hx = (atom[4*n    ] - xm0) * hx_1;
    ry_hy = (atom[4*n + 1] - ym0) * hy_1;
    rz_hz = (atom[4*n + 2] - zm0) * hz_1;

    /* find smallest numbered grid point in stencil */
    ilo = (int) floorf(rx_hx) - 1;
    jlo = (int) floorf(ry_hy) - 1;
    klo = (int) floorf(rz_hz) - 1;

    /* find t for x dimension and compute xphi */
    t = rx_hx - (float) ilo;
    xphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    xphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    xphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    xphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* find t for y dimension and compute yphi */
    t = ry_hy - (float) jlo;
    yphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    yphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    yphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    yphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* find t for z dimension and compute zphi */
    t = rz_hz - (float) klo;
    zphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    zphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    zphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    zphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* short-circuit tests for non-periodic boundaries */
    iswrap = ispany &&
      ( ilo < ia || (ilo+3) > ib ||
        jlo < ja || (jlo+3) > jb ||
        klo < ka || (klo+3) > kb);

    if ( ! iswrap ) {
      /* don't have to worry about wrapping */
      ASSERT(ia <= ilo && ilo + 3 <= ib);
      ASSERT(ja <= jlo && jlo + 3 <= jb);
      ASSERT(ka <= klo && klo + 3 <= kb);

      /* determine charge on 64=4*4*4 grid point stencil of qh */
      for (k = 0;  k < 4;  k++) {
        koff = (k + klo) * nj;
        ck = zphi[k] * q;
        for (j = 0;  j < 4;  j++) {
          jkoff = (koff + (j + jlo)) * (long)ni;
          cjk = yphi[j] * ck;
          for (i = 0;  i < 4;  i++) {
            index = jkoff + (i + ilo);
            GRID_INDEX_CHECK(qhgrid, i+ilo, j+jlo, k+klo);
            ASSERT(GRID_INDEX(qhgrid, i+ilo, j+jlo, k+klo) == index);
            qh[index] += xphi[i] * cjk;
          }
        }
      }
    } /* if */
    else {
      int ip, jp, kp;

      /* adjust ilo, jlo, klo so they are within lattice indexing */
      if      (ilo < ia) do { ilo += ni; } while (ilo < ia);
      else if (ilo > ib) do { ilo -= ni; } while (ilo > ib);
      if      (jlo < ja) do { jlo += nj; } while (jlo < ja);
      else if (jlo > jb) do { jlo -= nj; } while (jlo > jb);
      if      (klo < ka) do { klo += nk; } while (klo < ka);
      else if (klo > kb) do { klo -= nk; } while (klo > kb);

      /* determine charge on 64=4*4*4 grid point stencil of qh */
      for (k = 0, kp = klo;  k < 4;  k++, kp++) {
        if (kp > kb) kp = ka;  /* wrap stencil around grid */
        koff = kp * nj;
        ck = zphi[k] * q;
        for (j = 0, jp = jlo;  j < 4;  j++, jp++) {
          if (jp > jb) jp = ja;  /* wrap stencil around grid */
          jkoff = (koff + jp) * (long)ni;
          cjk = yphi[j] * ck;
          for (i = 0, ip = ilo;  i < 4;  i++, ip++) {
            if (ip > ib) ip = ia;  /* wrap stencil around grid */
            index = jkoff + ip;
            GRID_INDEX_CHECK(qhgrid, ip, jp, kp);
            ASSERT(GRID_INDEX(qhgrid, ip, jp, kp) == index);
            qh[index] += xphi[i] * cjk;
          }
        }
      }
    } /* else */

  } /* end loop over atoms */
#ifdef MSMPOT_DEBUG
  ck = 0;
  for (k = ka;  k <= kb;  k++) {
    for (j = ja;  j <= jb;  j++) {
      for (i = ia;  i <= ib;  i++) {
        index = (k*nj + j)*(long)ni + i;
        ck += qh[index];
      }
    }
  }
  printf("#  level = 0,  grid sum = %e\n", ck);
#endif
  return OK;
} /* anterpolation */


int interpolation_factored(Msmpot *msm) {
  float *epotmap = msm->epotmap;

  float *ezd = msm->ezd;
  float *eyzd = msm->eyzd;

  const floatGrid *ehgrid = &(msm->eh[0]);
  const float *eh = ehgrid->data;
  const int ia = ehgrid->i0;
  const int ib = ia + ehgrid->ni - 1;
  const int ja = ehgrid->j0;
  const int jb = ja + ehgrid->nj - 1;
  const int ka = ehgrid->k0;
  const int kb = ka + ehgrid->nk - 1;
  const int nrow_eh = ehgrid->ni;
  const int nstride_eh = nrow_eh * ehgrid->nj;

  const int mx = msm->mx;
  const int my = msm->my;
  const int mz = msm->mz;

  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);

  const size_t size_ezd = mz * sizeof(float);
  const size_t size_eyzd = my * mz * sizeof(float);

  const int ih_phi_cycle = msm->cycle_x;
  const int jh_phi_cycle = msm->cycle_y;
  const int kh_phi_cycle = msm->cycle_z;
  int ih_phi, jh_phi, kh_phi;

  const int rmap_x = msm->rmap_x;
  const int rmap_y = msm->rmap_y;
  const int rmap_z = msm->rmap_z;

  const int diam_x = 2*rmap_x + 1;
  const int diam_y = 2*rmap_y + 1;
  const int diam_z = 2*rmap_z + 1;

  const float *base_phi_x = msm->phi_x;
  const float *base_phi_y = msm->phi_y;
  const float *base_phi_z = msm->phi_z;
  const float *phi = NULL;

  const float hx_dx = msm->hx_dx;
  const float hy_dy = msm->hy_dy;
  const float hz_dz = msm->hz_dz;

  int ih, jh, kh;
  int im, jm, km;
  int i, j, k;
  int index_plane_eh, index_eh;
  int index_jk, offset_k;
  long offset;

  ih_phi = ia;
  while (ih_phi < 0) ih_phi += ih_phi_cycle;
  jh_phi = ja;
  while (jh_phi < 0) jh_phi += jh_phi_cycle;
  kh_phi = ka;
  while (kh_phi < 0) kh_phi += kh_phi_cycle;

  for (ih = ia;  ih <= ib;  ih++, ih_phi++) {
    if (ih_phi == ih_phi_cycle) ih_phi = 0;
    memset(eyzd, 0, size_eyzd);

    for (jh = ja;  jh <= jb;  jh++, jh_phi++) {
      if (jh_phi == jh_phi_cycle) jh_phi = 0;
      memset(ezd, 0, size_ezd);
      index_plane_eh = jh * nrow_eh + ih;

      for (kh = ka;  kh <= kb;  kh++, kh_phi++) {
        if (kh_phi == kh_phi_cycle) kh_phi = 0;
        index_eh = kh * nstride_eh + index_plane_eh;
        km = (int) floorf(kh * hz_dz);
        if ( ! ispz ) {  /* nonperiodic */
          int lower = km - rmap_z;
          int upper = km + rmap_z;
          if (lower < 0)   lower = 0;
          if (upper >= mz) upper = mz-1;  /* clip upper and lower */
          phi = base_phi_z + diam_z * kh_phi + rmap_z;
          for (k = lower;  k <= upper;  k++) {
            ezd[k] += phi[k-km] * eh[index_eh];
          }
        }
        else {  /* periodic */
          int kp = km - rmap_z;
          if (kp < 0) do { kp += mz; } while (kp < 0);  /* start kp inside */
          phi = base_phi_z + diam_z * kh_phi;
          for (k = 0;  k < diam_z;  k++, kp++) {
            if (kp == mz) kp -= mz;  /* wrap kp */
            ezd[kp] += phi[k] * eh[index_eh];
          }
        }
      }

      for (k = 0;  k < mz;  k++) {
        offset = k * my;
        jm = (int) floorf(jh * hy_dy);
        if ( ! ispy ) {  /* nonperiodic */
          int lower = jm - rmap_y;
          int upper = jm + rmap_y;
          if (lower < 0)   lower = 0;
          if (upper >= my) upper = my-1;  /* clip upper and lower */
          phi = base_phi_y + diam_y * jh_phi + rmap_y;
          for (j = lower;  j <= upper;  j++) {
            eyzd[offset + j] += phi[j-jm] * ezd[k];
          }
        }
        else {  /* periodic */
          int jp = jm - rmap_z;
          if (jp < 0) do { jp += my; } while (jp < 0);  /* start jp inside */
          phi = base_phi_y + diam_y * jh_phi;
          for (j = 0;  j < diam_y;  j++, jp++) {
            if (jp == my) jp -= my;  /* wrap jp */
            eyzd[offset + jp] += phi[j] * ezd[k];
          }
        }
      }
    }

    for (k = 0;  k < mz;  k++) {
      offset_k = k * my;

      for (j = 0;  j < my;  j++) {
        index_jk = offset_k + j;
        offset = index_jk * (long)mx;
        im = (int) floorf(ih * hx_dx);
        if ( ! ispx ) {  /* nonperiodic */
          int lower = im - rmap_x;
          int upper = im + rmap_x;
          if (lower < 0)   lower = 0;
          if (upper >= mx) upper = mx-1;  /* clip upper and lower */
          phi = base_phi_x + diam_x * ih_phi + rmap_x;
          for (i = lower;  i <= upper;  i++) {
            epotmap[offset + i] += phi[i-im] * eyzd[index_jk];
          }
        }
        else {  /* periodic */
          int ip = im - rmap_z;
          if (ip < 0) do { ip += mx; } while (ip < 0);  /* start ip inside */
          phi = base_phi_x + diam_x * ih_phi;
          for (i = 0;  i < diam_x;  i++, ip++) {
            if (ip == mx) ip -= mx;  /* wrap ip */
            epotmap[offset + ip] += phi[i] * eyzd[index_jk];
          }
        }
      }
    }

  }
  return OK;
} /* interpolation_factored() */


int interpolation(Msmpot *msm)
{
  float *epotmap = msm->epotmap;

  float xphi[4], yphi[4], zphi[4];  /* phi grid func along x, y, z */
  float rx_hx, ry_hy, rz_hz;        /* distance from origin */
  float t;                          /* normalized distance for phi */
  float ck, cjk;
  const float hx_1 = 1/msm->hx;
  const float hy_1 = 1/msm->hy;
  const float hz_1 = 1/msm->hz;
  const float px0 = msm->px0;
  const float py0 = msm->py0;
  const float pz0 = msm->pz0;

  const int mx = msm->mx;
  const int my = msm->my;
  const int mz = msm->mz;
  const float dx = msm->dx;
  const float dy = msm->dy;
  const float dz = msm->dz;
  const float lx0 = msm->lx0;
  const float ly0 = msm->ly0;
  const float lz0 = msm->lz0;

  const floatGrid *ehgrid = &(msm->eh[0]);
  const float *eh = ehgrid->data;
  const int ni = ehgrid->ni;
  const int nj = ehgrid->nj;
  const int nk = ehgrid->nk;
  const int ia = ehgrid->i0;
  const int ib = ia + ni - 1;
  const int ja = ehgrid->j0;
  const int jb = ja + nj - 1;
  const int ka = ehgrid->k0;
  const int kb = ka + nk - 1;

  const int ispany = IS_SET_ANY(msm->isperiodic);
  int iswrap;

  float x, y, z, esum;

  int i, j, k, ii, jj, kk, ilo, jlo, klo;
  int koff, kmoff;
  long index, mindex, jkoff, jkmoff;

  for (kk = 0;  kk < mz;  kk++) {
    kmoff = kk * my;
    z = kk*dz + lz0;

    for (jj = 0;  jj < my;  jj++) {
      jkmoff = (kmoff + jj) * (long)mx;
      y = jj*dy + ly0;

      for (ii = 0;  ii < mx;  ii++) {
        mindex = jkmoff + ii;
        x = ii*dx + lx0;

        /* distance between atom and origin measured in grid points */
        rx_hx = (x - px0) * hx_1;
        ry_hy = (y - py0) * hy_1;
        rz_hz = (z - pz0) * hz_1;

        /* find smallest numbered grid point in stencil */
        ilo = (int) floorf(rx_hx) - 1;
        jlo = (int) floorf(ry_hy) - 1;
        klo = (int) floorf(rz_hz) - 1;

        /* find t for x dimension and compute xphi */
        t = rx_hx - (float) ilo;
        xphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
        t--;
        xphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
        t--;
        xphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
        t--;
        xphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

        /* find t for y dimension and compute yphi */
        t = ry_hy - (float) jlo;
        yphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
        t--;
        yphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
        t--;
        yphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
        t--;
        yphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

        /* find t for z dimension and compute zphi */
        t = rz_hz - (float) klo;
        zphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
        t--;
        zphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
        t--;
        zphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
        t--;
        zphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

        /* short-circuit tests for non-periodic boundaries */
        iswrap = ispany &&
          ( ilo < ia || (ilo+3) > ib ||
            jlo < ja || (jlo+3) > jb ||
            klo < ka || (klo+3) > kb);

        if ( ! iswrap ) {
          /* don't have to worry about wrapping */
          ASSERT(ia <= ilo && ilo + 3 <= ib);
          ASSERT(ja <= jlo && jlo + 3 <= jb);
          ASSERT(ka <= klo && klo + 3 <= kb);

          /* determine 64=4*4*4 eh grid stencil contribution to potential */
          esum = 0;
          for (k = 0;  k < 4;  k++) {
            koff = (k + klo) * nj;
            ck = zphi[k];
            for (j = 0;  j < 4;  j++) {
              jkoff = (koff + (j + jlo)) * (long)ni;
              cjk = yphi[j] * ck;
              for (i = 0;  i < 4;  i++) {
                index = jkoff + (i + ilo);
                GRID_INDEX_CHECK(ehgrid, i+ilo, j+jlo, k+klo);
                ASSERT(GRID_INDEX(ehgrid, i+ilo, j+jlo, k+klo) == index);
                esum += eh[index] * xphi[i] * cjk;
              }
            }
          }
        } /* if */
        else {
          int ip, jp, kp;

          /* adjust ilo, jlo, klo so they are within lattice indexing */
          if      (ilo < ia) do { ilo += ni; } while (ilo < ia);
          else if (ilo > ib) do { ilo -= ni; } while (ilo > ib);
          if      (jlo < ja) do { jlo += nj; } while (jlo < ja);
          else if (jlo > jb) do { jlo -= nj; } while (jlo > jb);
          if      (klo < ka) do { klo += nk; } while (klo < ka);
          else if (klo > kb) do { klo -= nk; } while (klo > kb);

          /* determine charge on 64=4*4*4 grid point stencil of qh */
          esum = 0;
          for (k = 0, kp = klo;  k < 4;  k++, kp++) {
            if (kp > kb) kp = ka;  /* wrap stencil around grid */
            koff = kp * nj;
            ck = zphi[k];
            for (j = 0, jp = jlo;  j < 4;  j++, jp++) {
              if (jp > jb) jp = ja;  /* wrap stencil around grid */
              jkoff = (koff + jp) * (long)ni;
              cjk = yphi[j] * ck;
              for (i = 0, ip = ilo;  i < 4;  i++, ip++) {
                if (ip > ib) ip = ia;  /* wrap stencil around grid */
                index = jkoff + ip;
                GRID_INDEX_CHECK(ehgrid, ip, jp, kp);
                ASSERT(GRID_INDEX(ehgrid, ip, jp, kp) == index);
                esum += eh[index] * xphi[i] * cjk;
              }
            }
          }
        } /* else */

#ifdef MSMPOT_CHECKMAPINDEX
        if (MAPINDEX==mindex) {
          printf("shortrng=%g  longrng=%g  epotmap[%ld]=%g\n",
              epotmap[mindex], esum, mindex, epotmap[mindex]+esum);
        }
#endif

        epotmap[mindex] += esum;
      }
    }
  } /* end map loop */

  return OK;
} /* interpolation() */


#if !defined(USE_FACTORED_GRID_TRANSFERS)

/* constants for grid transfer operations
 * cubic "numerical Hermite" interpolation */

/* length of stencil */
enum { NSTENCIL = 5 };

/* phi interpolating function along one dimension of grid stencil */
static const float Phi[NSTENCIL] = { -0.0625f, 0.5625f, 1, 0.5625f, -0.0625f };

/* stencil offsets from a central grid point on a finer grid level */
/* (these offsets are where phi weights above have been evaluated) */
static const int Offset[NSTENCIL] = { -3, -1, 0, 1, 3 };


int restriction(Msmpot *msm, int level)
{
  float cjk, q2h_sum;

  /* lattices of charge, finer grid and coarser grid */
  const floatGrid *qhgrid = &(msm->qh[level]);
  const float *qh = qhgrid->data;
  floatGrid *q2hgrid = &(msm->qh[level+1]);
  float *q2h = q2hgrid->data;

  /* finer grid index ranges and dimensions */
  const int ia1 = qhgrid->i0;             /* lowest x-index */
  const int ib1 = ia1 + qhgrid->ni - 1;   /* highest x-index */
  const int ja1 = qhgrid->j0;             /* lowest y-index */
  const int jb1 = ja1 + qhgrid->nj - 1;   /* highest y-index */
  const int ka1 = qhgrid->k0;             /* lowest z-index */
  const int kb1 = ka1 + qhgrid->nk - 1;   /* highest z-index */
  const int ni1 = qhgrid->ni;             /* length along x-dim */
  const int nj1 = qhgrid->nj;             /* length along y-dim */

  /* coarser grid index ranges and dimensions */
  const int ia2 = q2hgrid->i0;            /* lowest x-index */
  const int ib2 = ia2 + q2hgrid->ni - 1;  /* highest x-index */
  const int ja2 = q2hgrid->j0;            /* lowest y-index */
  const int jb2 = ja2 + q2hgrid->nj - 1;  /* highest y-index */
  const int ka2 = q2hgrid->k0;            /* lowest z-index */
  const int kb2 = ka2 + q2hgrid->nk - 1;  /* highest z-index */
  const int ni2 = q2hgrid->ni;            /* length along x-dim */
  const int nj2 = q2hgrid->nj;            /* length along y-dim */

  /* other variables */
  int i1, j1, k1, k1off;
  int i2, j2, k2, k2off;
  int i, j, k;
  long index1, jk1off;
  long index2, jk2off;

  if (msm->isperiodic) {
    return ERRMSG(MSMPOT_ERROR_SUPPORT,
        "non-factored grid transfer does not support periodic boundaries");
  }

  /* loop over coarser grid points */
  for (k2 = ka2;  k2 <= kb2;  k2++) {
    k2off = k2 * nj2;    /* coarser grid index offset for k-coord */
    k1 = k2 * 2;         /* k-coord of same-space point on finer grid */
    for (j2 = ja2;  j2 <= jb2;  j2++) {
      jk2off = (k2off + j2) * (long)ni2;  /* add offset for j-coord coarser */
      j1 = j2 * 2;                    /* j-coord same-space finer grid */
      for (i2 = ia2;  i2 <= ib2;  i2++) {
        index2 = jk2off + i2;         /* index in coarser grid */
        i1 = i2 * 2;                  /* i-coord same-space finer grid */

        /* sum weighted charge contribution from finer grid stencil */
        q2h_sum = 0;
        for (k = 0;  k < NSTENCIL;  k++) {
          /* early loop termination if outside lattice */
          if (k1 + Offset[k] < ka1) continue;
          else if (k1 + Offset[k] > kb1) break;
          k1off = (k1 + Offset[k]) * nj1;  /* offset k-coord finer grid */
          for (j = 0;  j < NSTENCIL;  j++) {
            /* early loop termination if outside lattice */
            if (j1 + Offset[j] < ja1) continue;
            else if (j1 + Offset[j] > jb1) break;
            jk1off = (k1off + (j1 + Offset[j])) * (long)ni1; /* add offset j */
            cjk = Phi[j] * Phi[k];              /* mult weights in each dim */
            for (i = 0;  i < NSTENCIL;  i++) {
              /* early loop termination if outside lattice */
              if (i1 + Offset[i] < ia1) continue;
              else if (i1 + Offset[i] > ib1) break;
              index1 = jk1off + (i1 + Offset[i]);    /* index in finer grid */
              GRID_INDEX_CHECK(qhgrid,
                  i1+Offset[i], j1+Offset[j], k1+Offset[k]);
              ASSERT(GRID_INDEX(qhgrid,
                    i1+Offset[i], j1+Offset[j], k1+Offset[k]) == index1);
              q2h_sum += Phi[i] * cjk * qh[index1];  /* sum weighted charge */
            }
          }
        } /* end loop over finer grid stencil */

        GRID_INDEX_CHECK(q2hgrid, i2, j2, k2);
	ASSERT(GRID_INDEX(q2hgrid, i2, j2, k2) == index2);
        q2h[index2] = q2h_sum;  /* store charge to coarser grid */

      }
    }
  } /* end loop over each coarser grid points */
  return OK;
}


int prolongation(Msmpot *msm, int level)
{
  float ck, cjk;

  /* lattices of potential, finer grid and coarser grid */
  floatGrid *ehgrid = &(msm->eh[level]);
  float *eh = ehgrid->data;
  const floatGrid *e2hgrid = &(msm->eh[level+1]);
  const float *e2h = e2hgrid->data;

  /* finer grid index ranges and dimensions */
  const int ia1 = ehgrid->i0;             /* lowest x-index */
  const int ib1 = ia1 + ehgrid->ni - 1;   /* highest x-index */
  const int ja1 = ehgrid->j0;             /* lowest y-index */
  const int jb1 = ja1 + ehgrid->nj - 1;   /* highest y-index */
  const int ka1 = ehgrid->k0;             /* lowest z-index */
  const int kb1 = ka1 + ehgrid->nk - 1;   /* highest z-index */
  const int ni1 = ehgrid->ni;             /* length along x-dim */
  const int nj1 = ehgrid->nj;             /* length along y-dim */

  /* coarser grid index ranges and dimensions */
  const int ia2 = e2hgrid->i0;            /* lowest x-index */
  const int ib2 = ia2 + e2hgrid->ni - 1;  /* highest x-index */
  const int ja2 = e2hgrid->j0;            /* lowest y-index */
  const int jb2 = ja2 + e2hgrid->nj - 1;  /* highest y-index */
  const int ka2 = e2hgrid->k0;            /* lowest z-index */
  const int kb2 = ka2 + e2hgrid->nk - 1;  /* highest z-index */
  const int ni2 = e2hgrid->ni;            /* length along x-dim */
  const int nj2 = e2hgrid->nj;            /* length along y-dim */

  /* other variables */
  int i1, j1, k1, k1off;
  int i2, j2, k2, k2off;
  int i, j, k;
  long index1, jk1off;
  long index2, jk2off;

  if (msm->isperiodic) {
    return ERRMSG(MSMPOT_ERROR_SUPPORT,
        "non-factored grid transfer does not support periodic boundaries");
  }

  /* loop over coarser grid points */
  for (k2 = ka2;  k2 <= kb2;  k2++) {
    k2off = k2 * nj2;    /* coarser grid index offset for k-coord */
    k1 = k2 * 2;         /* k-coord of same-space point on finer grid */
    for (j2 = ja2;  j2 <= jb2;  j2++) {
      jk2off = (k2off + j2) * (long)ni2;  /* add offset for j-coord coarser */
      j1 = j2 * 2;                    /* j-coord same-space finer grid */
      for (i2 = ia2;  i2 <= ib2;  i2++) {
        index2 = jk2off + i2;         /* index in coarser grid */
        i1 = i2 * 2;                  /* i-coord same-space finer grid */

        /* sum weighted charge contribution from finer grid stencil */
        GRID_INDEX_CHECK(e2hgrid, i2, j2, k2);
        ASSERT(GRID_INDEX(e2hgrid, i2, j2, k2) == index2);
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
            jk1off = (k1off + (j1 + Offset[j])) * (long)ni1; /* add offset j */
            cjk = Phi[j] * ck;              /* mult weights in each dim */
            for (i = 0;  i < NSTENCIL;  i++) {
              /* early loop termination if outside lattice */
              if (i1 + Offset[i] < ia1) continue;
              else if (i1 + Offset[i] > ib1) break;
              index1 = jk1off + (i1 + Offset[i]);    /* index in finer grid */
              GRID_INDEX_CHECK(ehgrid,
                  i1+Offset[i], j1+Offset[j], k1+Offset[k]);
              ASSERT(GRID_INDEX(ehgrid,
                    i1+Offset[i], j1+Offset[j], k1+Offset[k]) == index1);
	      eh[index1] += Phi[i] * cjk;            /* sum weighted charge */
            }
          }
        } /* end loop over finer grid stencil */

      }
    }
  } /* end loop over each coarser grid points */
  return OK;
}

#else

/* constants for grid transfer operations
 * cubic "numerical Hermite" interpolation */

enum {
  R_STENCIL    = 3,                /* radius of stencil */
  DIAM_STENCIL = 2*R_STENCIL + 1,  /* diameter of stencil */
};

static const float PHI_FACTORED[DIAM_STENCIL] = {
  -0.0625f, 0, 0.5625f, 1, 0.5625f, 0, -0.0625f
};


int restriction(Msmpot *msm, int level) {
  /* lattices of potential, finer grid and coarser grid */
  const floatGrid *qhgrid = &(msm->qh[level]);
  const float *qh = qhgrid->data;
  floatGrid *q2hgrid = &(msm->qh[level+1]);
  float *q2h = q2hgrid->data;

  /* finer grid index ranges and dimensions */
  const int ia1 = qhgrid->i0;             /* lowest x-index */
  const int ib1 = ia1 + qhgrid->ni - 1;   /* highest x-index */
  const int ja1 = qhgrid->j0;             /* lowest y-index */
  const int jb1 = ja1 + qhgrid->nj - 1;   /* highest y-index */
  const int ka1 = qhgrid->k0;             /* lowest z-index */
  const int kb1 = ka1 + qhgrid->nk - 1;   /* highest z-index */
  const int ni1 = qhgrid->ni;             /* length along x-dim */
  const int nj1 = qhgrid->nj;             /* length along y-dim */
  const int nk1 = qhgrid->nk;             /* length along z-dim */

  /* coarser grid index ranges and dimensions */
  const int ia2 = q2hgrid->i0;            /* lowest x-index */
  const int ib2 = ia2 + q2hgrid->ni - 1;  /* highest x-index */
  const int ja2 = q2hgrid->j0;            /* lowest y-index */
  const int jb2 = ja2 + q2hgrid->nj - 1;  /* highest y-index */
  const int ka2 = q2hgrid->k0;            /* lowest z-index */
  const int kb2 = ka2 + q2hgrid->nk - 1;  /* highest z-index */
  const int nrow_q2 = q2hgrid->ni;
  const int nstride_q2 = nrow_q2 * q2hgrid->nj;

  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);

  /* set buffer using indexing offset, so that indexing matches qh grid */
  float *qzd = msm->lzd + (-ka1);
  float *qyzd = msm->lyzd + (-ka1*nj1 + -ja1);
  float qsum;

  const float *phi = NULL;

  int i2, j2, k2;
  int im, jm, km;
  int i, j, k;
  int index_plane_q2, index_q2;
  int index_jk, offset_k;
  long offset;

  for (i2 = ia2;  i2 <= ib2;  i2++) {

    for (k = ka1;  k <= kb1;  k++) {
      offset_k = k * nj1;

      for (j = ja1;  j <= jb1;  j++) {
        index_jk = offset_k + j;
        offset = index_jk * (long)ni1;
        im = (i2 << 1);  /* = 2*i2 */
        qsum = 0;
        if ( ! ispx ) {  /* nonperiodic */
          int lower = im - R_STENCIL;
          int upper = im + R_STENCIL;
          if (lower < ia1) lower = ia1;
          if (upper > ib1) upper = ib1;  /* clip edges */
          phi = PHI_FACTORED + R_STENCIL;  /* center of stencil */
          for (i = lower;  i <= upper;  i++) {
            qsum += phi[i-im] * qh[offset + i];
          }
        }
        else {  /* periodic */
          int ip = im - R_STENCIL;  /* index at left end of stencil */
          if (ip < ia1) do { ip += ni1; } while (ip < ia1);  /* start inside */
          phi = PHI_FACTORED;  /* left end of stencil */
          for (i = 0;  i < DIAM_STENCIL;  i++, ip++) {
            if (ip > ib1) ip = ia1;  /* wrap around edge of lattice */
            qsum += phi[i] * qh[offset + ip];
          }
        }
        qyzd[index_jk] = qsum;
      } /* for j */

    } /* for k */

    for (j2 = ja2;  j2 <= jb2;  j2++) {
      index_plane_q2 = j2 * nrow_q2 + i2;

      for (k = ka1;  k <= kb1;  k++) {
        offset = k * nj1;
        jm = (j2 << 1);  /* = 2*j2 */
        qsum = 0;
        if ( ! ispy ) {  /* nonperiodic */
          int lower = jm - R_STENCIL;
          int upper = jm + R_STENCIL;
          if (lower < ja1) lower = ja1;
          if (upper > jb1) upper = jb1;  /* clip edges */
          phi = PHI_FACTORED + R_STENCIL;  /* center of stencil */
          for (j = lower;  j <= upper;  j++) {
            qsum += phi[j-jm] * qyzd[offset + j];
          }
        }
        else {  /* periodic */
          int jp = jm - R_STENCIL;  /* index at left end of stencil */
          if (jp < ja1) do { jp += nj1; } while (jp < ja1);  /* start inside */
          phi = PHI_FACTORED;  /* left end of stencil */
          for (j = 0;  j < DIAM_STENCIL;  j++, jp++) {
            if (jp > jb1) jp = ja1;  /* wrap around edge of lattice */
            qsum += phi[j] * qyzd[offset + jp];
          }
        }
        qzd[k] = qsum;
      } /* for k */

      for (k2 = ka2;  k2 <= kb2;  k2++) {
        index_q2 = k2 * nstride_q2 + index_plane_q2;
        km = (k2 << 1);  /* = 2*k2 */
        qsum = 0;
        if ( ! ispz ) {  /* nonperiodic */
          int lower = km - R_STENCIL;
          int upper = km + R_STENCIL;
          if (lower < ka1) lower = ka1;
          if (upper > kb1) upper = kb1;  /* clip edges */
          phi = PHI_FACTORED + R_STENCIL;  /* center of stencil */
          for (k = lower;  k <= upper;  k++) {
            qsum += phi[k-km] * qzd[k];
          }
        }
        else {  /* periodic */
          int kp = km - R_STENCIL;  /* index at left end of stencil */
          if (kp < ka1) do { kp += nk1; } while (kp < ka1);  /* start inside */
          phi = PHI_FACTORED;  /* left end of stencil */
          for (k = 0;  k < DIAM_STENCIL;  k++, kp++) {
            if (kp > kb1) kp = ka1;  /* wrap around edge of lattice */
            qsum += phi[k] * qzd[kp];
          }
        }
        q2h[index_q2] = qsum;
      } /* for k2 */

    } /* for j2 */

  } /* for i2 */
#ifdef MSMPOT_DEBUG
  qsum = 0;
  for (k = ka2;  k <= kb2;  k++) {
    for (j = ja2;  j <= jb2;  j++) {
      for (i = ia2;  i <= ib2;  i++) {
        index_q2 = k*nstride_q2 + j*nrow_q2 + i;
        qsum += q2h[index_q2];
      }
    }
  }
  printf("#  level = %d,  grid sum = %e\n", level+1, qsum);
#endif
  return OK;
} /* restriction, factored */


int prolongation(Msmpot *msm, int level) {
  /* lattices of potential, finer grid and coarser grid */
  floatGrid *ehgrid = &(msm->eh[level]);
  float *eh = ehgrid->data;
  const floatGrid *e2hgrid = &(msm->eh[level+1]);
  const float *e2h = e2hgrid->data;

  /* finer grid index ranges and dimensions */
  const int ia1 = ehgrid->i0;             /* lowest x-index */
  const int ib1 = ia1 + ehgrid->ni - 1;   /* highest x-index */
  const int ja1 = ehgrid->j0;             /* lowest y-index */
  const int jb1 = ja1 + ehgrid->nj - 1;   /* highest y-index */
  const int ka1 = ehgrid->k0;             /* lowest z-index */
  const int kb1 = ka1 + ehgrid->nk - 1;   /* highest z-index */
  const int ni1 = ehgrid->ni;             /* length along x-dim */
  const int nj1 = ehgrid->nj;             /* length along y-dim */
  const int nk1 = ehgrid->nk;             /* length along z-dim */

  /* coarser grid index ranges and dimensions */
  const int ia2 = e2hgrid->i0;            /* lowest x-index */
  const int ib2 = ia2 + e2hgrid->ni - 1;  /* highest x-index */
  const int ja2 = e2hgrid->j0;            /* lowest y-index */
  const int jb2 = ja2 + e2hgrid->nj - 1;  /* highest y-index */
  const int ka2 = e2hgrid->k0;            /* lowest z-index */
  const int kb2 = ka2 + e2hgrid->nk - 1;  /* highest z-index */
  const int nrow_e2 = e2hgrid->ni;
  const int nstride_e2 = nrow_e2 * e2hgrid->nj;

  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);

  /* set buffer using indexing offset, so that indexing matches eh grid */
  float *ezd = msm->lzd + (-ka1);
  float *eyzd = msm->lyzd + (-ka1*nj1 + -ja1);

  const size_t size_lzd = nk1 * sizeof(float);
  const size_t size_lyzd = nj1 * nk1 * sizeof(float);

  const float *phi = NULL;

  int i2, j2, k2;
  int im, jm, km;
  int i, j, k;
  int index_plane_e2, index_e2;
  int index_jk, offset_k;
  long offset;

  for (i2 = ia2;  i2 <= ib2;  i2++) {
    memset(msm->lyzd, 0, size_lyzd);

    for (j2 = ja2;  j2 <= jb2;  j2++) {
      memset(msm->lzd, 0, size_lzd);
      index_plane_e2 = j2 * nrow_e2 + i2;

      for (k2 = ka2;  k2 <= kb2;  k2++) {
        index_e2 = k2 * nstride_e2 + index_plane_e2;
        km = (k2 << 1);  /* = 2*k2 */
        if ( ! ispz ) {  /* nonperiodic */
          int lower = km - R_STENCIL;
          int upper = km + R_STENCIL;
          if (lower < ka1) lower = ka1;
          if (upper > kb1) upper = kb1;  /* clip edges */
          phi = PHI_FACTORED + R_STENCIL;  /* center of stencil */
          for (k = lower;  k <= upper;  k++) {
            ezd[k] += phi[k-km] * e2h[index_e2];
          }
        }
        else {  /* periodic */
          int kp = km - R_STENCIL;  /* index at left end of stencil */
          if (kp < ka1) do { kp += nk1; } while (kp < ka1);  /* start inside */
          phi = PHI_FACTORED;  /* left end of stencil */
          for (k = 0;  k < DIAM_STENCIL;  k++, kp++) {
            if (kp > kb1) kp = ka1;  /* wrap around edge of lattice */
            ezd[kp] += phi[k] * e2h[index_e2];
          }
        }
      } /* for k2 */

      for (k = ka1;  k <= kb1;  k++) {
        offset = k * nj1;
        jm = (j2 << 1);  /* = 2*j2 */
        if ( ! ispy ) {  /* nonperiodic */
          int lower = jm - R_STENCIL;
          int upper = jm + R_STENCIL;
          if (lower < ja1) lower = ja1;
          if (upper > jb1) upper = jb1;  /* clip edges */
          phi = PHI_FACTORED + R_STENCIL;  /* center of stencil */
          for (j = lower;  j <= upper;  j++) {
            eyzd[offset + j] += phi[j-jm] * ezd[k];
          }
        }
        else {  /* periodic */
          int jp = jm - R_STENCIL;  /* index at left end of stencil */
          if (jp < ja1) do { jp += nj1; } while (jp < ja1);  /* start inside */
          phi = PHI_FACTORED;  /* left end of stencil */
          for (j = 0;  j < DIAM_STENCIL;  j++, jp++) {
            if (jp > jb1) jp = ja1;  /* wrap around edge of lattice */
            eyzd[offset + jp] += phi[j] * ezd[k];
          }
        }
      } /* for k */

    } /* for j2 */

    for (k = ka1;  k <= kb1;  k++) {
      offset_k = k * nj1;

      for (j = ja1;  j <= jb1;  j++) {
        index_jk = offset_k + j;
        offset = index_jk * (long)ni1;
        im = (i2 << 1);  /* = 2*i2 */
        if ( ! ispx ) {  /* nonperiodic */
          int lower = im - R_STENCIL;
          int upper = im + R_STENCIL;
          if (lower < ia1) lower = ia1;
          if (upper > ib1) upper = ib1;  /* clip edges */
          phi = PHI_FACTORED + R_STENCIL;  /* center of stencil */
          for (i = lower;  i <= upper;  i++) {
            eh[offset + i] += phi[i-im] * eyzd[index_jk];
          }
        }
        else {  /* periodic */
          int ip = im - R_STENCIL;  /* index at left end of stencil */
          if (ip < ia1) do { ip += ni1; } while (ip < ia1);  /* start inside */
          phi = PHI_FACTORED;  /* left end of stencil */
          for (i = 0;  i < DIAM_STENCIL;  i++, ip++) {
            if (ip > ib1) ip = ia1;  /* wrap around edge of lattice */
            eh[offset + ip] += phi[i] * eyzd[index_jk];
          }
        }
      } /* for j */

    } /* for k */

  } /* for i2 */

  return OK;
} /* prolongation, factored */

#endif


int latticecutoff(Msmpot *msm, int level)
{
  float eh_sum;

  /* lattices of charge and potential */
  const floatGrid *qhgrid = &(msm->qh[level]);
  const float *qh = qhgrid->data;
  floatGrid *ehgrid = &(msm->eh[level]);
  float *eh = ehgrid->data;
  const int ia = qhgrid->i0;            /* lowest x-index */
  const int ib = ia + qhgrid->ni - 1;   /* highest x-index */
  const int ja = qhgrid->j0;            /* lowest y-index */
  const int jb = ja + qhgrid->nj - 1;   /* highest y-index */
  const int ka = qhgrid->k0;            /* lowest z-index */
  const int kb = ka + qhgrid->nk - 1;   /* highest z-index */
  const int ni = qhgrid->ni;            /* length along x-dim */
  const int nj = qhgrid->nj;            /* length along y-dim */
  const int nk = qhgrid->nk;            /* length along z-dim */

  /* lattice of weights for pairwise grid point interactions within cutoff */
  const floatGrid *gcgrid = &(msm->gc[level]);
  const float *gc = gcgrid->data;
  const int gia = gcgrid->i0;            /* lowest x-index */
  const int gib = gia + gcgrid->ni - 1;  /* highest x-index */
  const int gja = gcgrid->j0;            /* lowest y-index */
  const int gjb = gja + gcgrid->nj - 1;  /* highest y-index */
  const int gka = gcgrid->k0;            /* lowest z-index */
  const int gkb = gka + gcgrid->nk - 1;  /* highest z-index */
  const int gni = gcgrid->ni;            /* length along x-dim */
  const int gnj = gcgrid->nj;            /* length along y-dim */

  const int ispx = (IS_SET_X(msm->isperiodic) != 0);
  const int ispy = (IS_SET_Y(msm->isperiodic) != 0);
  const int ispz = (IS_SET_Z(msm->isperiodic) != 0);

  const int ispnone = !(ispx || ispy || ispz);

  int i, j, k;
  int gia_clip, gib_clip;
  int gja_clip, gjb_clip;
  int gka_clip, gkb_clip;
  int koff;
  long jkoff, index;
  int id, jd, kd;
  int knoff;
  long jknoff, nindex;
  int kgoff, jkgoff, ngindex;

  if ( ispnone ) {  /* nonperiodic boundaries */

    /* loop over all grid points */
    for (k = ka;  k <= kb;  k++) {

      /* clip gc ranges to keep offset for k index within grid */
      gka_clip = (k + gka < ka ? ka - k : gka);
      gkb_clip = (k + gkb > kb ? kb - k : gkb);

      koff = k * nj;  /* find eh flat index */

      for (j = ja;  j <= jb;  j++) {

        /* clip gc ranges to keep offset for j index within grid */
        gja_clip = (j + gja < ja ? ja - j : gja);
        gjb_clip = (j + gjb > jb ? jb - j : gjb);

        jkoff = (koff + j) * (long)ni;  /* find eh flat index */

        for (i = ia;  i <= ib;  i++) {

          /* clip gc ranges to keep offset for i index within grid */
          gia_clip = (i + gia < ia ? ia - i : gia);
          gib_clip = (i + gib > ib ? ib - i : gib);

          index = jkoff + i;  /* eh flat index */

          /* sum over "sphere" of weighted charge */
          eh_sum = 0;
          for (kd = gka_clip;  kd <= gkb_clip;  kd++) {
            knoff = (k + kd) * nj;  /* find qh flat index */
            kgoff = kd * gnj;       /* find gc flat index */

            for (jd = gja_clip;  jd <= gjb_clip;  jd++) {
              jknoff = (knoff + (j + jd)) * (long)ni;  /* find qh flat index */
              jkgoff = (kgoff + jd) * gni;       /* find gc flat index */

              for (id = gia_clip;  id <= gib_clip;  id++) {
                nindex = jknoff + (i + id);  /* qh flat index */
                ngindex = jkgoff + id;       /* gc flat index */

                GRID_INDEX_CHECK(qhgrid, i+id, j+jd, k+kd);
                ASSERT(GRID_INDEX(qhgrid, i+id, j+jd, k+kd) == nindex);

                GRID_INDEX_CHECK(gcgrid, id, jd, kd);
                ASSERT(GRID_INDEX(gcgrid, id, jd, kd) == ngindex);

                eh_sum += qh[nindex] * gc[ngindex];  /* sum weighted charge */
              }
            }
          } /* end loop over "sphere" of charge */

          GRID_INDEX_CHECK(ehgrid, i, j, k);
          ASSERT(GRID_INDEX(ehgrid, i, j, k) == index);
          eh[index] = eh_sum;  /* store potential */
        }
      }
    } /* end loop over all grid points */

  } /* if nonperiodic boundaries */
  else {
    /* some boundary is periodic */
    int ilo, jlo, klo;
    int ip, jp, kp;

    /* loop over all grid points */
    for (k = ka;  k <= kb;  k++) {
      klo = k + gka;
      if ( ! ispz ) {  /* nonperiodic z */
        /* clip gc ranges to keep offset for k index within grid */
        gka_clip = (k + gka < ka ? ka - k : gka);
        gkb_clip = (k + gkb > kb ? kb - k : gkb);
        if (klo < ka) klo = ka;  /* keep lowest qh index within grid */
      }
      else {  /* periodic z */
        gka_clip = gka;
        gkb_clip = gkb;
        if (klo < ka) do { klo += nk; } while (klo < ka);
      }
      ASSERT(klo <= kb);

      koff = k * nj;  /* find eh flat index */

      for (j = ja;  j <= jb;  j++) {
        jlo = j + gja;
        if ( ! ispy ) {  /* nonperiodic y */
          /* clip gc ranges to keep offset for j index within grid */
          gja_clip = (j + gja < ja ? ja - j : gja);
          gjb_clip = (j + gjb > jb ? jb - j : gjb);
          if (jlo < ja) jlo = ja;  /* keep lowest qh index within grid */
        }
        else {  /* periodic y */
          gja_clip = gja;
          gjb_clip = gjb;
          if (jlo < ja) do { jlo += nj; } while (jlo < ja);
        }
        ASSERT(jlo <= jb);

        jkoff = (koff + j) * (long)ni;  /* find eh flat index */

        for (i = ia;  i <= ib;  i++) {
          ilo = i + gia;
          if ( ! ispx ) {  /* nonperiodic x */
            /* clip gc ranges to keep offset for i index within grid */
            gia_clip = (i + gia < ia ? ia - i : gia);
            gib_clip = (i + gib > ib ? ib - i : gib);
            if (ilo < ia) ilo = ia;  /* keep lowest qh index within grid */
          }
          else {  /* periodic x */
            gia_clip = gia;
            gib_clip = gib;
            if (ilo < ia) do { ilo += ni; } while (ilo < ia);
          }
          ASSERT(ilo <= ib);

          index = jkoff + i;  /* eh flat index */

          /* sum over "sphere" of weighted charge */
          eh_sum = 0;
          for (kd = gka_clip, kp = klo;  kd <= gkb_clip;  kd++, kp++) {
            /* clipping makes conditional always fail for nonperiodic */
            if (kp > kb) kp = ka;  /* wrap z direction */
            knoff = kp * nj;       /* find qh flat index */
            kgoff = kd * gnj;      /* find gc flat index */

            for (jd = gja_clip, jp = jlo;  jd <= gjb_clip;  jd++, jp++) {
              /* clipping makes conditional always fail for nonperiodic */
              if (jp > jb) jp = ja;              /* wrap y direction */
              jknoff = (knoff + jp) * (long)ni;  /* find qh flat index */
              jkgoff = (kgoff + jd) * gni;       /* find gc flat index */

              for (id = gia_clip, ip = ilo;  id <= gib_clip;  id++, ip++) {
                /* clipping makes conditional always fail for nonperiodic */
                if (ip > ib) ip = ia;   /* wrap x direction */
                nindex = jknoff +  ip;  /* qh flat index */
                ngindex = jkgoff + id;  /* gc flat index */

                GRID_INDEX_CHECK(qhgrid, ip, jp, kp);
                ASSERT(GRID_INDEX(qhgrid, ip, jp, kp) == nindex);

                GRID_INDEX_CHECK(gcgrid, id, jd, kd);
                ASSERT(GRID_INDEX(gcgrid, id, jd, kd) == ngindex);

                eh_sum += qh[nindex] * gc[ngindex];  /* sum weighted charge */

              }
            }
          } /* end loop over "sphere" of charge */

          GRID_INDEX_CHECK(ehgrid, i, j, k);
          ASSERT(GRID_INDEX(ehgrid, i, j, k) == index);
          eh[index] = eh_sum;  /* store potential */
        }
      }
    } /* end loop over all grid points */

  } /* else some boundary is periodic */

  return OK;
}
