#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "force/fnbcut.h"
#include "moltypes/vecops.h"
#include "moltypes/const.h"


int Fnbcut_init(Fnbcut *f, const Exclude *ex) {
  int s;  /* error status */

  /* check function arguments */
  if (NULL==ex) return ERROR(ERR_EXPECT);
  ASSERT(ex->topo != NULL);
  ASSERT(ex->topo->fprm != NULL);

  /* init data storage */
  memset(f, 0, sizeof(Fnbcut));
  f->fprm = ex->topo->fprm;
  f->topo = ex->topo;
  f->exclude = ex;

  /* init for gridcell hashing */
  if ((s=Array_init(&(f->cell), sizeof(FnbcutCell))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->next), sizeof(int32))) != OK) return ERROR(s);

  /* init for interaction energies */
  if ((s=Array_init(&(f->idmap), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->alltrue), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->isatomID1), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->isatomID2), sizeof(char))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->atomID12), sizeof(int32))) != OK) return ERROR(s);
  if ((s=Array_init(&(f->fsubtr), sizeof(dvec))) != OK) return ERROR(s);

  return OK;
}


void Fnbcut_done(Fnbcut *f) {
  Array_done(&(f->cell));
  Array_done(&(f->next));
  Array_done(&(f->idmap));
  Array_done(&(f->alltrue));
  Array_done(&(f->isatomID1));
  Array_done(&(f->isatomID2));
  Array_done(&(f->atomID12));
  Array_done(&(f->fsubtr));
}


int Fnbcut_find_rminmax(const Fnbcut *f, dvec *rmin, dvec *rmax,
    const dvec *pos, int32 n) {
  dvec min, max;
  int32 i;
  if (n < 1) return ERROR(ERR_VALUE);
  if (NULL==rmin || NULL==rmax) return ERROR(ERR_EXPECT);
  min = max = pos[0];
  for (i = 1;  i < n;  i++) {
    if      (min.x > pos[i].x)  min.x = pos[i].x;
    else if (max.x < pos[i].x)  max.x = pos[i].x;
    if      (min.y > pos[i].y)  min.y = pos[i].y;
    else if (max.y < pos[i].y)  max.y = pos[i].y;
    if      (min.z > pos[i].z)  min.z = pos[i].z;
    else if (max.z < pos[i].z)  max.z = pos[i].z;
  }
  *rmin = min;
  *rmax = max;
  return OK;
}


int Fnbcut_setup(Fnbcut *f, const FnbcutPrm *prm,
    const Domain *d, const dvec *rmin, const dvec *rmax) {
  const NonbPrm *nonbprm;
  int32 natoms;
  dreal cutoff2;
  dvec u, v, w;               /* the basis vectors */
  dreal ulen, vlen, wlen;     /* lengths of basis vector */
  dvec ucv, vcw, wcu;         /* for cross products */
  dvec pu, pv, pw;            /* for orthogonal projections */
  dreal pulen, pvlen, pwlen;  /* lengths of othogonal projections */
  dreal sc;
  dvec smin = { 0, 0, 0 };
  dvec smax = { 1, 1, 1 };    /* recip space bounds for periodic case */
  const boolean is_rminmax_null = (NULL==rmin && NULL==rmax);
  const boolean is_rminmax_defn = (NULL!=rmin && NULL!=rmax);
  boolean is_allperiodic;
  int es;  /* error status */

  /* check function arguments */
  if (NULL==prm) return ERROR(ERR_EXPECT);
  if (NULL==d) return ERROR(ERR_EXPECT);

  /* check sanity of FnbcutPrm */
  if (prm->hgroupCutoff < 0) return ERROR(ERR_VALUE);
  if (prm->pairlistDist < 0) return ERROR(ERR_VALUE);
  if (prm->cellDepth.x < 0 ||
      prm->cellDepth.y < 0 ||
      prm->cellDepth.z < 0) return ERROR(ERR_VALUE);
  if (0==prm->nbpairType ||
      (prm->nbpairType & ~FNBCUT_MASK) != 0) {
    return ERROR(ERR_VALUE);
  }
  if ((prm->nbpairType & FNBCUT_ELEC_MASK) >= FNBCUT_ELEC_END) {
    return ERROR(ERR_RANGE);
  }
  f->nbprm = *prm;

  /* cell depth must fit within max */
  if (0==f->nbprm.cellDepth.x || f->nbprm.cellDepth.x > FNBCUT_CELL_MAXDEPTH) {
    f->nbprm.cellDepth.x = FNBCUT_CELL_MAXDEPTH;
  }
  if (0==f->nbprm.cellDepth.y || f->nbprm.cellDepth.y > FNBCUT_CELL_MAXDEPTH) {
    f->nbprm.cellDepth.y = FNBCUT_CELL_MAXDEPTH;
  }
  if (0==f->nbprm.cellDepth.z || f->nbprm.cellDepth.z > FNBCUT_CELL_MAXDEPTH) {
    f->nbprm.cellDepth.z = FNBCUT_CELL_MAXDEPTH;
  }
  VEC(f->nbprm.cellDepth);

  /* init force field parameters */
  nonbprm = ForcePrm_nonbprm(f->fprm);
  if (nonbprm->cutoff <= 0 || nonbprm->dielectric <= 0) {
    return ERROR(ERR_VALUE);
  }
  f->cutoff = nonbprm->cutoff;
  cutoff2 = f->cutoff * f->cutoff;
  f->extcutoff = f->cutoff + prm->hgroupCutoff + prm->pairlistDist;
  f->mincellsz.x = f->extcutoff / f->nbprm.cellDepth.x;
  f->mincellsz.y = f->extcutoff / f->nbprm.cellDepth.y;
  f->mincellsz.z = f->extcutoff / f->nbprm.cellDepth.z;
  VEC(f->mincellsz);
  f->elec_const = COULOMB / nonbprm->dielectric;
  FLT(f->elec_const);
  f->switchdist2 = nonbprm->switchdist * nonbprm->switchdist;
  f->inv_cutoff2 = 1. / cutoff2;
  if (cutoff2 > f->switchdist2) {
    f->inv_denom_switch = 1. / ((cutoff2 - f->switchdist2) *
        (cutoff2 - f->switchdist2) * (cutoff2 - f->switchdist2));
  }
  else {
    f->inv_denom_switch = 0;
  }
  f->ewald_coef = 0;  /* need to set this from pme library? */
  f->ewald_grad_coef = (2. / sqrt(M_PI)) * f->ewald_coef;
  f->scaling14 = nonbprm->scaling14;
  f->exclpolicy = nonbprm->exclude;
  if ((f->nbprm.nbpairType & FNBCUT_ELEC_MASK) == FNBCUT_ELEC) {
    f->nbprm.nbpairType &= ~FNBCUT_ELEC_MASK;  /* reset option */
    if (nonbprm->switching) {
      f->nbprm.nbpairType |= FNBCUT_ELEC_CUTOFF;
    }
    else {
      f->nbprm.nbpairType |= FNBCUT_ELEC_INFTY;
    }
  }
  if ((f->nbprm.nbpairType & FNBCUT_VDW_MASK) == FNBCUT_VDW) {
    f->nbprm.nbpairType &= ~FNBCUT_VDW_MASK;  /* reset option */
    if (nonbprm->switching) {
      f->nbprm.nbpairType |= FNBCUT_VDW_CUTOFF;
    }
    else {
      f->nbprm.nbpairType |= FNBCUT_VDW_INFTY;
    }
  }
  if ((f->nbprm.nbpairType & FNBCUT_BUCK_MASK) == FNBCUT_BUCK) {
    f->nbprm.nbpairType &= ~FNBCUT_BUCK_MASK;  /* reset option */
    if (nonbprm->switching) {
      f->nbprm.nbpairType |= FNBCUT_BUCK_CUTOFF;
    }
    else {
      f->nbprm.nbpairType |= FNBCUT_BUCK_INFTY;
    }
  }
  HEX(f->nbprm.nbpairType);

  is_allperiodic = d->periodic_x && d->periodic_y && d->periodic_z;
  if (!is_rminmax_null && !is_rminmax_defn) return ERROR(ERR_EXPECT);
  if (!is_allperiodic && is_rminmax_null) return ERROR(ERR_EXPECT);

  f->domain = d;
  u = d->basis[0];
  v = d->basis[1];
  w = d->basis[2];
  VEC(u);
  VEC(v);
  VEC(w);

  if (!is_allperiodic) {
    /* find in recip space the extent of real space bounding box */
    dvec r[8], s[8];
    int32 i, j, k, index;
    for (k = 0;  k < 2;  k++) {
      for (j = 0;  j < 2;  j++) {
        for (i = 0;  i < 2;  i++) {
          index = (k*2 + j)*2 + i;
          r[index].x = (0==i ? rmin->x : rmax->x);
          r[index].y = (0==j ? rmin->y : rmax->y);
          r[index].z = (0==k ? rmin->z : rmax->z);
        }
      }
    }
    for (i = 0;  i < 8;  i++) {
      ivec n;
      Domain_normalize_vec(d, &s[i], &n, &r[i]);
      VECADD(s[i], s[i], n);  /* add in the image offset */
    }
    smin = smax = s[0];
    for (i = 1;  i < 8;  i++) {
      if      (smin.x > s[i].x)  smin.x = s[i].x;
      else if (smax.x < s[i].x)  smax.x = s[i].x;
      if      (smin.y > s[i].y)  smin.y = s[i].y;
      else if (smax.y < s[i].y)  smax.y = s[i].y;
      if      (smin.z > s[i].z)  smin.z = s[i].z;
      else if (smax.z < s[i].z)  smax.z = s[i].z;
    }
    /* handle degenerate cases */
    if (smax.x - smin.x < 1) {
      dreal dx = 0.5 * (1. - (smax.x - smin.x));
      smax.x += dx;
      smin.x -= dx;
    }
    if (smax.y - smin.y < 1) {
      dreal dy = 0.5 * (1. - (smax.y - smin.y));
      smax.y += dy;
      smin.y -= dy;
    }
    if (smax.z - smin.z < 1) {
      dreal dz = 0.5 * (1. - (smax.z - smin.z));
      smax.z += dz;
      smin.z -= dz;
    }
    VEC(smin);
    VEC(smax);
  } /* if (!is_allperiodic) */

  /* find orthogonal projection of u onto v cross w */
  VECCROSS(vcw, v, w);
  sc = VECDOT(u, vcw) / VECLEN2(vcw);
  VECMUL(pu, sc, vcw);
  pulen = sqrt(VECLEN2(pu));
  ulen = sqrt(VECLEN2(u));

  /* find orthogonal projection of v onto w cross u */
  VECCROSS(wcu, w, u);
  sc = VECDOT(v, wcu) / VECLEN2(wcu);
  VECMUL(pv, sc, wcu);
  pvlen = sqrt(VECLEN2(pv));
  vlen = sqrt(VECLEN2(v));

  /* find orthogonal projection of w onto u cross v */
  VECCROSS(ucv, u, v);
  sc = VECDOT(w, ucv) / VECLEN2(ucv);
  VECMUL(pw, sc, ucv);
  pwlen = sqrt(VECLEN2(pw));
  wlen = sqrt(VECLEN2(w));

  /* find grid cell hashing parameters in u direction */
  if (d->periodic_x) {
    if (pulen < f->extcutoff) return ERROR(ERR_EXPECT);
    f->celldim.x = (int32) floor(pulen / f->mincellsz.x);
    ASSERT(f->celldim.x > 0);
    f->hashfactor.x = f->celldim.x; 
    f->hashorigin.x = 0;
    f->cellsz.x = ulen / f->celldim.x;
  }
  else {
    f->cellsz.x = f->mincellsz.x * (ulen / pulen);
    ASSERT(f->cellsz.x >= f->mincellsz.x);
    f->hashfactor.x = ulen / f->cellsz.x;
    f->hashorigin.x = smin.x;
    f->celldim.x = (int32) ceil((smax.x - smin.x) * f->hashfactor.x);
  }

  /* find grid cell hashing parameters in v direction */
  if (d->periodic_y) {
    if (pvlen < f->extcutoff) return ERROR(ERR_EXPECT);
    f->celldim.y = (int32) floor(pvlen / f->mincellsz.y);
    ASSERT(f->celldim.y > 0);
    f->hashfactor.y = f->celldim.y; 
    f->hashorigin.y = 0;
    f->cellsz.y = vlen / f->celldim.y;
  }
  else {
    f->cellsz.y = f->mincellsz.y * (vlen / pvlen);
    ASSERT(f->cellsz.y >= f->mincellsz.y);
    f->hashfactor.y = vlen / f->cellsz.y;
    f->hashorigin.y = smin.y;
    f->celldim.y = (int32) ceil((smax.y - smin.y) * f->hashfactor.y);
  }

  /* find grid cell hashing parameters in w direction */
  if (d->periodic_z) {
    if (pwlen < f->extcutoff) return ERROR(ERR_EXPECT);
    f->celldim.z = (int32) floor(pwlen / f->mincellsz.z);
    ASSERT(f->celldim.z > 0);
    f->hashfactor.z = f->celldim.z; 
    f->hashorigin.z = 0;
    f->cellsz.z = wlen / f->celldim.z;
  }
  else {
    f->cellsz.z = f->mincellsz.z * (wlen / pwlen);
    ASSERT(f->cellsz.z >= f->mincellsz.z);
    f->hashfactor.z = wlen / f->cellsz.z;
    f->hashorigin.z = smin.z;
    f->celldim.z = (int32) ceil((smax.z - smin.z) * f->hashfactor.z);
  }
  VEC(f->cellsz);
  VEC(f->celldim);
  VEC(f->hashfactor);
  VEC(f->hashorigin);
  ASSERT(Fnbcut_check_gridcell(f)==OK);

  /* setup for cursor linked list */
  natoms = Topology_atom_array_length(f->topo);
  if ((es=Array_resize(&(f->next), natoms)) != OK) return ERROR(es);

  /* setup grid cells */
  f->ncells = f->celldim.x * f->celldim.y * f->celldim.z;
  ASSERT(f->ncells > 0);
  if ((es=Array_resize(&(f->cell), f->ncells)) != OK) return ERROR(es);
  if ((es=Array_erase(&(f->cell))) != OK) return ERROR(es);
  if ((es=Fnbcut_setup_cellnbrs(f)) != OK) return ERROR(es);

  /* setup for interaction energies */
  if ((es=Array_resize(&(f->idmap), natoms)) != OK) return ERROR(es);
  else {
    int32 *idmap = Array_data(&(f->idmap));
    int32 i;
    for (i = 0;  i < natoms;  i++) {
      idmap[i] = i;
    }
  }
  if ((es=Array_resize(&(f->alltrue), natoms)) != OK) return ERROR(es);
  else {
    char *a = Array_data(&(f->alltrue));
    int32 i;
    for (i = 0;  i < natoms;  i++) {
      a[i] = (char)TRUE;
    }
  }
  /* don't allocate the other arrays unless they are needed */

  return OK;
}


/* flat indexing of 3x3x3 cube using {-1,0,1}^3 indices */
#define IMAGE_INDEX(i,j,k)  ((((k)*3 + (j))*3 + (i)) + 13)

int Fnbcut_setup_cellnbrs(Fnbcut *f) {
  FnbcutCell *cell = Array_data(&(f->cell));
  const int32 depth_x = f->nbprm.cellDepth.x;
  const int32 depth_y = f->nbprm.cellDepth.y;
  const int32 depth_z = f->nbprm.cellDepth.z;
  const int32 dim_x = f->celldim.x;
  const int32 dim_y = f->celldim.y;
  const int32 dim_z = f->celldim.z;
  const boolean is_periodic_x = f->domain->periodic_x;
  const boolean is_periodic_y = f->domain->periodic_y;
  const boolean is_periodic_z = f->domain->periodic_z;
  int32 i, j, k, n, ii, jj, kk, nn, in, jn, kn;
  int32 image_x, image_y, image_z, imageIndex, nbrcnt;
  dvec w, a0, a1, a2;

  ASSERT(f->domain != NULL);

  /* setup image table */
  a0 = f->domain->basis[0];
  a1 = f->domain->basis[1];
  a2 = f->domain->basis[2];
  for (k = -1;  k <= 1;  k++) {
    for (j = -1;  j <= 1;  j++) {
      for (i = -1;  i <= 1;  i++) {
        w.x = -i*a0.x + -j*a1.x + -k*a2.x;
        w.y = -i*a0.y + -j*a1.y + -k*a2.y;
        w.z = -i*a0.z + -j*a1.z + -k*a2.z;
        f->imageTable[ IMAGE_INDEX(i,j,k) ] = w;
      }
    }
  }

  ASSERT(cell != NULL);
  ASSERT(Array_length(&(f->cell)) == dim_x * dim_y * dim_z);

  /* loop over all cells */
  for (k = 0;  k < dim_z;  k++) {
    for (j = 0;  j < dim_y;  j++) {
      for (i = 0;  i < dim_x;  i++) {

        /* index of this cell */
        n = (k * dim_y + j) * dim_x + i;
        ASSERT(n >= 0 && n < f->ncells);

        /* loop over half-shell of neighbor cells */
        nbrcnt = 0;
        for (kn = 0;  kn <= depth_z;  kn++) {
          kk = k + kn;
          image_z = 0;
          if (kk >= dim_z) {
            if (is_periodic_z) {
              kk -= dim_z;
              image_z = -1;
            }
            else continue;
          }

          for (jn = (0==kn ? 0 : -depth_y);  jn <= depth_y;  jn++) {
            jj = j + jn;
            image_y = 0;
            if (jj >= dim_y) {
              if (is_periodic_y) {
                jj -= dim_y;
                image_y = -1;
              }
              else continue;
            }
            else if (jj < 0) {
              if (is_periodic_y) {
                jj += dim_y;
                image_y = 1;
              }
              else continue;
            }

            for (in = (0==kn && 0==jn ? 0 : -depth_x);  in <= depth_x;  in++) {
              ii = i + in;
              image_x = 0;
              if (ii >= dim_x) {
                if (is_periodic_x) {
                  ii -= dim_x;
                  image_x = -1;
                }
                else continue;
              }
              else if (ii < 0) {
                if (is_periodic_x) {
                  ii += dim_x;
                  image_x = 1;
                }
                else continue;
              }

              /* determine index into image table for this neighbor cell */
              imageIndex = IMAGE_INDEX(image_x, image_y, image_z);

              /* index of neighbor cell */
              nn = (kk * dim_y + jj) * dim_x + ii;
              ASSERT(nn >= 0 && nn < f->ncells);

              /* store neighbor index and image index */
              cell[n].nbr[nbrcnt] = nn;
              cell[n].image[nbrcnt] = imageIndex;
              nbrcnt++;
            }
          }
        } /* end loop over half-shell of neighbor cells */
        cell[n].nbrcnt = nbrcnt;
      }
    }
  } /* end loop over all cells */
  return OK;
}


int Fnbcut_check_gridcell(const Fnbcut *f) {
  dvec u = f->domain->basis[0];
  dvec v = f->domain->basis[1];
  dvec w = f->domain->basis[2];
  dreal invlen;
  const dvec cellsz = f->cellsz;
  const dreal cutoff2 = f->extcutoff * f->extcutoff;
  const ivec cellDepth = f->nbprm.cellDepth;
  int32 i, j, k;
  int32 im, jm, km;
  int32 in, jn, kn;
  int32 cnt;

  /* need unit vectors */
  invlen = 1./sqrt(VECLEN2(u));
  VECMUL(u, invlen, u);
  invlen = 1./sqrt(VECLEN2(v));
  VECMUL(v, invlen, v);
  invlen = 1./sqrt(VECLEN2(w));
  VECMUL(w, invlen, w);

  /* loop over outer blocks */
  for (k = -cellDepth.z;  k <= cellDepth.z;  k++) {
    for (j = -cellDepth.y;  j <= cellDepth.y;  j++) {
      for (i = -cellDepth.x;  i <= cellDepth.x;  i++) {
        /* check only outer blocks */
        if (i != -cellDepth.x && i != cellDepth.x &&
            j != -cellDepth.y && j != cellDepth.y &&
            k != -cellDepth.z && k != cellDepth.z) continue;

        /* for all 8 corners of center block */
        for (km = 0;  km < 2;  km++) {
          for (jm = 0;  jm < 2;  jm++) {
            for (im = 0;  im < 2;  im++) {
              dvec rm;
              VECZERO(rm);
              VECMADD(rm, im*cellsz.x, u, rm);
              VECMADD(rm, jm*cellsz.y, v, rm);
              VECMADD(rm, km*cellsz.z, w, rm);

              /* check distances of all 8 corners of edge block */
              cnt = 0;
              for (kn = 0;  kn < 2;  kn++) {
                for (jn = 0;  jn < 2;  jn++) {
                  for (in = 0;  in < 2;  in++) {
                    dvec rn, delta;
                    VECZERO(rn);
                    VECMADD(rn, (i+in)*cellsz.x, u, rn);
                    VECMADD(rn, (j+jn)*cellsz.y, v, rn);
                    VECMADD(rn, (k+kn)*cellsz.z, w, rn);
                    VECSUB(delta, rn, rm);
                    if (VECLEN2(delta) >= cutoff2) cnt++;
                  }
                }
              }
              /* at least four of points checked must be outside cutoff */
              if (cnt < 4) return FAIL;
            }
          }
        } /* for km, jm, im */

      }
    }
  } /* for k, j, i */

  return OK;
}
