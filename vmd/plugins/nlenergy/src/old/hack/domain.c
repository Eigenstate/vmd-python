/* domain.c */

#include <string.h>
#include <math.h>
#include "moltypes/domain.h"
#include "moltypes/vecops.h"


int Transform_init(Transform *t, dvec col[3]) {
  /* test singularity of transformation */
  const dreal tol = 1e-4;
  dreal vol;
  dvec u;
  VECCROSS(u, col[0], col[1]);
  vol = VECDOT(u, col[2]);   /* compute volume */
  if (fabs(vol) < tol) {  /* fails if near-singular */
    memset(t, 0, sizeof(Transform));
    return FAIL;
  }
  /* transpose */
  t->row[0].x = col[0].x, t->row[0].y = col[1].x, t->row[0].z = col[2].x;
  t->row[1].x = col[0].y, t->row[1].y = col[1].y, t->row[1].z = col[2].y;
  t->row[2].x = col[0].z, t->row[2].y = col[1].z, t->row[2].z = col[2].z;
  return OK;
}


void Transform_done(Transform *t) {
  /* nothing to do! */
}


void Transform_apply(const Transform *t, dvec *tr, const dvec *r) {
  const dvec t0 = t->row[0];
  const dvec t1 = t->row[1];
  const dvec t2 = t->row[2];
  tr->x = VECDOT(t0, *r);
  tr->y = VECDOT(t1, *r);
  tr->z = VECDOT(t2, *r);
}


static void calc_recip(Domain *d) {
  const dvec a0 = d->basis[0];
  const dvec a1 = d->basis[1];
  const dvec a2 = d->basis[2];
  dvec b0, b1, b2;
  dvec v;
  dreal s;

  VECCROSS(v, a1, a2);
  s = 1.0 / VECDOT(a0, v);
  VECMUL(b0, s, v);

  VECCROSS(v, a2, a0);
  s = 1.0 / VECDOT(a1, v);
  VECMUL(b1, s, v);

  VECCROSS(v, a0, a1);
  s = 1.0 / VECDOT(a2, v);
  VECMUL(b2, s, v);

  d->recip[0] = b0;
  d->recip[1] = b1;
  d->recip[2] = b2;
}


int Domain_init(Domain *d) {
  memset(d, 0, sizeof(Domain));
  return OK;
}


void Domain_done(Domain *d) {
  /* nothing to do! */
}


/* intended to be called once after init() */
int Domain_setup(Domain *d, const dvec *center,
    const dvec *basis_v1, const dvec *basis_v2, const dvec *basis_v3) {
  const dvec e_x = { 1.0, 0.0, 0.0 };
  const dvec e_z = { 0.0, 0.0, 1.0 };
  dvec a0, a1, a2;
  dreal s;
  boolean is_periodic_x, is_periodic_y, is_periodic_z;

  d->center = *center;
  a0 = *basis_v1;
  a1 = *basis_v2;
  a2 = *basis_v3;
  is_periodic_x = (VECLEN2(a0)!=0.0 ? TRUE : FALSE);
  is_periodic_y = (VECLEN2(a1)!=0.0 ? TRUE : FALSE);
  is_periodic_z = (VECLEN2(a2)!=0.0 ? TRUE : FALSE);
  if ( ! is_periodic_x ) {
    a0 = e_x;
  }
  if ( ! is_periodic_y ) {
    /*
     * need to choose "y" direction for nonperiodic domain
     * choose (0,0,1) cross a0, unless these are too close to lying
     * on the same line, in which case choose (1,0,0) cross a0
     *
     * note that for a0==(1,0,0) we get a1==(0,1,0)
     */
    dvec na0;
    s = 1.0 / sqrt(VECLEN2(a0));
    VECMUL(na0, s, a0);     /* normalize a0 */
    if (fabs(VECDOT(e_z, na0)) < 0.9) {
      VECCROSS(a1, e_z, a0);
    }
    else {
      VECCROSS(a1, e_x, a0);
    }
    s = 1.0 / sqrt(VECLEN2(a1));
    VECMUL(a1, s, a1);      /* normalize a1 */
  }
  if ( ! is_periodic_z ) {
    VECCROSS(a2, a0, a1);
    s = 1.0 / sqrt(VECLEN2(a2));
    VECMUL(a2, s, a2);      /* normalize a2 */
  }
  d->basis[0] = a0;
  d->basis[1] = a1;
  d->basis[2] = a2;
  d->periodic_x = is_periodic_x;
  d->periodic_y = is_periodic_y;
  d->periodic_z = is_periodic_z;

  if (Domain_volume(d) < 0.0) {
    d->basis[2].x = -a2.x;
    d->basis[2].y = -a2.y;
    d->basis[2].z = -a2.z;
  }
  calc_recip(d);
  return OK;
}


/* calculate volume of domain */
dreal Domain_volume(const Domain *d) {
  const dvec a0 = d->basis[0];
  const dvec a1 = d->basis[1];
  const dvec a2 = d->basis[2];
  if (d->periodic_x && d->periodic_y && d->periodic_z) {
    dvec v;
    VECCROSS(v, a0, a1);
    return VECDOT(v, a2);
  }
  return 0.0;
}


/*
 * rescale domain by applying linear transformation,
 * transformation can be applied in spite of periodicity
 */
int Domain_rescale(Domain *d, const Transform *t) {
  const dvec a0 = d->basis[0];
  const dvec a1 = d->basis[1];
  const dvec a2 = d->basis[2];
  const dvec t0 = t->row[0];
  const dvec t1 = t->row[1];
  const dvec t2 = t->row[2];
  dvec na0, na1, na2;
  na0.x = VECDOT(t0, a0);
  na0.y = VECDOT(t1, a0);
  na0.z = VECDOT(t2, a0);
  na1.x = VECDOT(t0, a1);
  na1.y = VECDOT(t1, a1);
  na1.z = VECDOT(t2, a1);
  na2.x = VECDOT(t0, a2);
  na2.y = VECDOT(t1, a2);
  na2.z = VECDOT(t2, a2);
  d->basis[0] = na0;
  d->basis[1] = na1;
  d->basis[2] = na2;
  calc_recip(d);
  return OK;
}


/*
 * calculate shortest vector r_ij from r_i to r_j,
 * i.e., shortest (r_j-r_i) over all periodic images
 */
void Domain_shortest_vec(const Domain *d,
    dvec *r_ij, const dvec *r_j, const dvec *r_i) {
  const dvec a0 = d->basis[0];
  const dvec a1 = d->basis[1];
  const dvec a2 = d->basis[2];
  const dvec b0 = d->recip[0];
  const dvec b1 = d->recip[1];
  const dvec b2 = d->recip[2];
  dvec u, v, m;
  VECSUB(u, *r_j, *r_i);
  v.x = VECDOT(b0, u) + 0.5;
  v.y = VECDOT(b1, u) + 0.5;
  v.z = VECDOT(b2, u) + 0.5;
  m.x = (d->periodic_x ? floor(v.x) : 0.0);
  m.y = (d->periodic_y ? floor(v.y) : 0.0);
  m.z = (d->periodic_z ? floor(v.z) : 0.0);
  u.x -= (m.x * a0.x + m.y * a1.x + m.z * a2.x);
  u.y -= (m.x * a0.y + m.y * a1.y + m.z * a2.y);
  u.z -= (m.x * a0.z + m.y * a1.z + m.z * a2.z);
  *r_ij = u;
}


/*
 * calculate normalization of vector r to reciprocal space:
 *   s = A^(-1)(r - c) + 1/2
 * where n counts the number of basis vector displacements in each dimension;
 * for wrapped periodic system, s is in [0,1)^3, up to roundoff error
 */
void Domain_normalize_vec(const Domain *d, dvec *s, ivec *n, const dvec *r) {
  const dvec b0 = d->recip[0];
  const dvec b1 = d->recip[1];
  const dvec b2 = d->recip[2];
  dvec u, v, m;
  VECSUB(u, *r, d->center);
  v.x = VECDOT(b0, u) + 0.5;
  v.y = VECDOT(b1, u) + 0.5;
  v.z = VECDOT(b2, u) + 0.5;
  m.x = (d->periodic_x ? floor(v.x) : 0.0);
  m.y = (d->periodic_y ? floor(v.y) : 0.0);
  m.z = (d->periodic_z ? floor(v.z) : 0.0);
  /*
   * NOTE: setting s by  VECSUB(*s, v, m);
   * to guarantee s in [0,1)^3 does not always work for wrapped
   * systems due to roundoff error, e.g., you have to either
   * readjust (re-wrap) the positions or you have to undo the
   * subtraction so that s corresponds to the original wrapping
   */
  *s = v;
  n->x = (int32) m.x;
  n->y = (int32) m.y;
  n->z = (int32) m.z;
}


/*
 * calculate displacement vector w bringing vector r into real space domain:
 *   w = -An, where n = floor(A^{-1}(r-c) + 1/2)
 * where n counts the number of basis vector displacements in each dimension,
 * so you can regain r from r+w by (r+w)+An
 *
 * on return, r+w is inside of domain (it "wraps" r)
 */
void Domain_wrap_vec(const Domain *d, dvec *w, ivec *n, const dvec *r) {
  const dvec a0 = d->basis[0];
  const dvec a1 = d->basis[1];
  const dvec a2 = d->basis[2];
  const dvec b0 = d->recip[0];
  const dvec b1 = d->recip[1];
  const dvec b2 = d->recip[2];
  dvec u, m, v;
  VECSUB(u, *r, d->center);
  v.x = VECDOT(b0, u) + 0.5;
  v.y = VECDOT(b1, u) + 0.5;
  v.z = VECDOT(b2, u) + 0.5;
  m.x = (d->periodic_x ? floor(v.x) : 0.0);
  m.y = (d->periodic_y ? floor(v.y) : 0.0);
  m.z = (d->periodic_z ? floor(v.z) : 0.0);
  w->x = -(m.x * a0.x + m.y * a1.x + m.z * a2.x);
  w->y = -(m.x * a0.y + m.y * a1.y + m.z * a2.y);
  w->z = -(m.x * a0.z + m.y * a1.z + m.z * a2.z);
  n->x = (int32) m.x;
  n->y = (int32) m.y;
  n->z = (int32) m.z;
}


/*
 * calculate displacement vector w bringing vector r closest to center c
 * of real space domain:
 *   w = -An, where n in Z^3 such that | r+w - c | is minimized
 * where n counts the number of basis vector displacements in each dimension,
 * so you can regain r from r+w by (r+w)+An
 *
 * on return, r+w is the closest vector to c, over all possible w
 *
 * with the right choice of basis vectors, this can be used to obtain
 * other periodic domains beyond parallelepipeds
 */
void Domain_wrap_vec_center(const Domain *d, dvec *w, ivec *n, const dvec *r) {
  Domain_wrap_vec_location(d, w, n, r, &(d->center));
#if 0
  const dvec a0 = d->basis[0];
  const dvec a1 = d->basis[1];
  const dvec a2 = d->basis[2];
  const dvec b0 = d->recip[0];
  const dvec b1 = d->recip[1];
  const dvec b2 = d->recip[2];
  const int p0 = d->periodic_x;
  const int p1 = d->periodic_y;
  const int p2 = d->periodic_z;
  int32 i, j, k;
  int32 min_ni, min_nj, min_nk;
  dvec u, m, v, w0;
  dvec delta, du, min_delta;
  dreal ulen2, dulen2;
  VECSUB(u, *r, d->center);
  v.x = VECDOT(b0, u) + 0.5;
  v.y = VECDOT(b1, u) + 0.5;
  v.z = VECDOT(b2, u) + 0.5;
  m.x = (p0 ? floor(v.x) : 0.0);
  m.y = (p1 ? floor(v.y) : 0.0);
  m.z = (p2 ? floor(v.z) : 0.0);
  w0.x = -(m.x * a0.x + m.y * a1.x + m.z * a2.x);
  w0.y = -(m.x * a0.y + m.y * a1.y + m.z * a2.y);
  w0.z = -(m.x * a0.z + m.y * a1.z + m.z * a2.z);
  VECADD(u, u, w0);
  ulen2 = VECLEN2(u);
  VECZERO(min_delta);
  min_ni = 0;
  min_nj = 0;
  min_nk = 0;
  for (k = -p2;  k <= p2;  k++) {
    for (j = -p1;  j <= p1;  j++) {
      for (i = -p0;  i <= p0;  i++) {
        delta.x = -(i * a0.x + j * a1.x + k * a2.x);
        delta.y = -(i * a0.y + j * a1.y + k * a2.y);
        delta.z = -(i * a0.z + j * a1.z + k * a2.z);
        VECADD(du, delta, u);
        dulen2 = VECLEN2(du);
        if (dulen2 < ulen2) {
          ulen2 = dulen2;
          min_delta = delta;
          min_ni = i;
          min_nj = j;
          min_nk = k;
        }
      }
    }
  }
  VECADD(*w, w0, min_delta);
  n->x = ((int32) m.x) + min_ni;
  n->y = ((int32) m.y) + min_nj;
  n->z = ((int32) m.z) + min_nk;
#endif
}


/*
 * calculate displacement vector w bringing vector r closest to location p
 * of real space domain:
 *   w = -An, where n in Z^3 such that | r+w - p | is minimized
 * where n counts the number of basis vector displacements in each dimension,
 * so you can regain r from r+w by (r+w)+An
 *
 * on return, r+w is the closest vector to p, over all possible w
 *
 * this generalizes Domain_wrap_vec_center()
 */
void Domain_wrap_vec_location(const Domain *d, dvec *w, ivec *n,
    const dvec *r, const dvec *p) {
  const dvec a0 = d->basis[0];
  const dvec a1 = d->basis[1];
  const dvec a2 = d->basis[2];
  const dvec b0 = d->recip[0];
  const dvec b1 = d->recip[1];
  const dvec b2 = d->recip[2];
  const int p0 = d->periodic_x;
  const int p1 = d->periodic_y;
  const int p2 = d->periodic_z;
  int32 i, j, k;
  int32 min_ni, min_nj, min_nk;
  dvec u, m, v, w0;
  dvec delta, du, min_delta;
  dreal ulen2, dulen2;
  VECSUB(u, *r, *p);
  v.x = VECDOT(b0, u) + 0.5;
  v.y = VECDOT(b1, u) + 0.5;
  v.z = VECDOT(b2, u) + 0.5;
  m.x = (p0 ? floor(v.x) : 0.0);
  m.y = (p1 ? floor(v.y) : 0.0);
  m.z = (p2 ? floor(v.z) : 0.0);
  w0.x = -(m.x * a0.x + m.y * a1.x + m.z * a2.x);
  w0.y = -(m.x * a0.y + m.y * a1.y + m.z * a2.y);
  w0.z = -(m.x * a0.z + m.y * a1.z + m.z * a2.z);
  VECADD(u, u, w0);
  ulen2 = VECLEN2(u);
  VECZERO(min_delta);
  min_ni = 0;
  min_nj = 0;
  min_nk = 0;
  for (k = -p2;  k <= p2;  k++) {
    for (j = -p1;  j <= p1;  j++) {
      for (i = -p0;  i <= p0;  i++) {
        delta.x = -(i * a0.x + j * a1.x + k * a2.x);
        delta.y = -(i * a0.y + j * a1.y + k * a2.y);
        delta.z = -(i * a0.z + j * a1.z + k * a2.z);
        VECADD(du, delta, u);
        dulen2 = VECLEN2(du);
        if (dulen2 < ulen2) {
          ulen2 = dulen2;
          min_delta = delta;
          min_ni = i;
          min_nj = j;
          min_nk = k;
        }
      }
    }
  }
  VECADD(*w, w0, min_delta);
  n->x = ((int32) m.x) + min_ni;
  n->y = ((int32) m.y) + min_nj;
  n->z = ((int32) m.z) + min_nk;
}
