/* forceprm_nonb.c */

#include <math.h>
#include "moltypes/forceprm.h"


/*
 * manage nonbonded parameters
 */

int ForcePrm_set_nonbprm(ForcePrm *f, const NonbPrm *p) {
  /* check correctness of NonbPrm */
  if (p->switchdist < 0 || p->cutoff < p->switchdist) {
    return ERROR(ERR_VALUE);
  }
  if (p->dielectric <= 0) {
    return ERROR(ERR_VALUE);
  }
  if (p->exclude < EXCL_NONE || p->exclude > EXCL_SCALED14) {
    return ERROR(ERR_VALUE);
  }
  if (p->charge_model < CHARGE_FIXED || p->charge_model > CHARGE_FLUCQ) {
    return ERROR(ERR_VALUE);
  }
  if (p->water_model < WATER_TIP3 || p->water_model > WATER_TIP4) {
    return ERROR(ERR_VALUE);
  }
  f->nonbprm = *p;  /* store it */
  return OK;
}


const NonbPrm *ForcePrm_nonbprm(const ForcePrm *f) {
  return &(f->nonbprm);
}


/*
 * manage table of van der Waals interaction parameters
 */

int ForcePrm_setup_vdwtable(ForcePrm *f) {
  const Array *atomprm = &(f->atomprm);
  const int32 atomprmlen = Array_length(atomprm);
  const Array *vdwpairprm = &(f->vdwpairprm);
  const int32 vdwpairprmlen = Array_length(vdwpairprm);
  Array *vdwtable = &(f->vdwtable);
  int s;  /* error status */
  int32 i, j, k;

  f->vdwtablelen = atomprmlen;
  INT(atomprmlen);
  if ((s=Array_resize(vdwtable, atomprmlen * atomprmlen)) != OK) {
    return ERROR(s);
  }

  for (j = 0;  j < atomprmlen;  j++) {
    for (i = 0;  i < atomprmlen;  i++) {
      VdwTableElem *p_ij = Array_elem(vdwtable, j * atomprmlen + i);
      VdwTableElem *p_ji = Array_elem(vdwtable, i * atomprmlen + j);
      const AtomPrm *prm_i = Array_elem_const(atomprm, i);
      const AtomPrm *prm_j = Array_elem_const(atomprm, j);

      if (NULL == p_ij || NULL == p_ji) return ERROR(ERR_EXPECT);
      if (NULL == prm_i || NULL == prm_j) return ERROR(ERR_EXPECT);

      if (i > 0 && j > 0) {
        dreal neg_emin = sqrt(prm_i->emin * prm_j->emin);
        dreal rmin = 0.5 * (prm_i->rmin + prm_j->rmin);
        dreal neg_emin14 = sqrt(prm_i->emin14 * prm_j->emin14);
        dreal rmin14 = 0.5 * (prm_i->rmin14 + prm_j->rmin14);

        dreal rmin6 = rmin * rmin * rmin;
        dreal rmin6_14 = rmin14 * rmin14 * rmin14;
        rmin6 = rmin6 * rmin6;
        rmin6_14 = rmin6_14 * rmin6_14;

        p_ij->a   = p_ji->a   = neg_emin * rmin6 * rmin6;
        p_ij->b   = p_ji->b   = 2.0 * neg_emin * rmin6;
        p_ij->a14 = p_ji->a14 = neg_emin14 * rmin6_14 * rmin6_14;
        p_ij->b14 = p_ji->b14 = 2.0 * neg_emin14 * rmin6_14;
      }
      else {  /* one of these AtomPrms is the "zero"-value */
        p_ij->a   = p_ji->a   = 0;
        p_ij->b   = p_ji->b   = 0;
        p_ij->a14 = p_ji->a14 = 0;
        p_ij->b14 = p_ji->b14 = 0;
      }

    } /* end j loop */
  } /* end i loop */

  for (k = 0;  k < vdwpairprmlen;  k++) {
    const VdwpairPrm *prm = ForcePrm_vdwpairprm(f, k);

    if (NULL == prm ||
        (i = ForcePrm_getid_atomprm(f, prm->atomType[0])) < 0 ||
        (j = ForcePrm_getid_atomprm(f, prm->atomType[1])) < 0) {
      continue;  /* this VdwpairPrm has been removed */
    }
    else {
      VdwTableElem *p_ij = Array_elem(vdwtable, j * atomprmlen + i);
      VdwTableElem *p_ji = Array_elem(vdwtable, i * atomprmlen + j);

      if (NULL == p_ij || NULL == p_ji) return ERROR(ERR_EXPECT);
      else {
        dreal neg_emin = -prm->emin;
        dreal rmin = prm->rmin;
        dreal neg_emin14 = -prm->emin14;
        dreal rmin14 = prm->rmin14;

        dreal rmin6 = rmin * rmin * rmin;
        dreal rmin6_14 = rmin14 * rmin14 * rmin14;
        rmin6 = rmin6 * rmin6;
        rmin6_14 = rmin6_14 * rmin6_14;

        p_ij->a   = p_ji->a   = neg_emin * rmin6 * rmin6;
        p_ij->b   = p_ji->b   = 2.0 * neg_emin * rmin6;
        p_ij->a14 = p_ji->a14 = neg_emin14 * rmin6_14 * rmin6_14;
        p_ij->b14 = p_ji->b14 = 2.0 * neg_emin14 * rmin6_14;
      }
    }

  } /* end k loop */

  return OK;
}


const VdwTableElem *ForcePrm_vdwtable(const ForcePrm *f,
    int32 apid0, int32 apid1) {
  const VdwTableElem *pbase = Array_data_const(&(f->vdwtable));
  const int32 tablelen = f->vdwtablelen;
  apid0++;
  apid1++;
  if (apid0 < 0 || apid0 >= tablelen || apid1 < 0 || apid1 >= tablelen) {
    return pbase;  /* leading element is "zero"-valued */
  }
  return pbase + (apid1*tablelen) + apid0;
}


const VdwTableElem *ForcePrm_vdwtable_array(const ForcePrm *f) {
  const VdwTableElem *pbase = Array_data_const(&(f->vdwtable));
  return pbase + f->vdwtablelen + 1;
}


int32 ForcePrm_vdwtable_length(const ForcePrm *f) {
  return f->vdwtablelen;
}


/*
 * manage table of Buckingham interaction parameters
 */

int ForcePrm_setup_bucktable(ForcePrm *f) {
  const Array *atomprm = &(f->atomprm);
  const int32 atomprmlen = Array_length(atomprm);
  const Array *buckpairprm = &(f->buckpairprm);
  const int32 buckpairprmlen = Array_length(buckpairprm);
  Array *bucktable = &(f->bucktable);
  int s;  /* error status */
  int32 i, j, k;

  f->bucktablelen = atomprmlen;
  if ((s=Array_resize(bucktable, atomprmlen * atomprmlen)) != OK) {
    return ERROR(s);
  }
  if ((s=Array_erase(bucktable)) != OK) return ERROR(s);

  for (k = 0;  k < buckpairprmlen;  k++) {
    const BuckpairPrm *prm = ForcePrm_buckpairprm(f, k);

    if (NULL == prm ||
        (i = ForcePrm_getid_atomprm(f, prm->atomType[0])) < 0 ||
        (j = ForcePrm_getid_atomprm(f, prm->atomType[1])) < 0) {
      continue;  /* this BuckpairPrm has been removed */
    }
    else {
      BuckTableElem *p_ij = Array_elem(bucktable, j * atomprmlen + i);
      BuckTableElem *p_ji = Array_elem(bucktable, i * atomprmlen + j);

      if (NULL == p_ij || NULL == p_ji) return ERROR(ERR_EXPECT);
      else {
        dreal as, bs, rs, urs, rmax, urmax;
        if ((s=ForcePrm_calc_safebuck(f, &as, &bs, &rs, &urs, &rmax, &urmax,
                prm->a, prm->b, prm->c)) != OK) {
          return ERROR(s);
        }
        p_ij->a   = p_ji->a   = prm->a;
        p_ij->b   = p_ji->b   = prm->b;
        p_ij->c   = p_ji->c   = prm->c;
        p_ij->as  = p_ji->as  = as;
        p_ij->bs  = p_ji->bs  = bs;
        p_ij->rs2 = p_ji->rs2 = rs * rs;
      }
    }

  } /* end k loop */

  return OK;
}


const BuckTableElem *ForcePrm_bucktable(const ForcePrm *f,
    int32 apid0, int32 apid1) {
  const BuckTableElem *pbase = Array_data_const(&(f->bucktable));
  const int32 tablelen = f->bucktablelen;
  if (apid0 < 0 || apid0 >= tablelen || apid1 < 0 || apid1 >= tablelen) {
    return pbase;  /* leading element is "zero"-valued */
  }
  return pbase + (apid1*tablelen) + apid0;
}


const BuckTableElem *ForcePrm_bucktable_array(const ForcePrm *f) {
  const BuckTableElem *pbase = Array_data_const(&(f->bucktable));
  return pbase + f->bucktablelen + 1;
}


int32 ForcePrm_bucktable_length(const ForcePrm *f) {
  return f->bucktablelen;
}


/*
 * Determine parameters for smooth extension of Buckingham potential.
 *
 * Attach to  U(r) = a exp(-r*b) - c/r^6  at the change in curvature
 * point along the wall,  U''(r_0) = 0  such that  U(r_0) > 0, a piece
 * of the form  A/r^6 + B  (A>0, B>0)  such that the join is C^1.
 * This avoids the non-physical well and can be safely used for dynamics.
 * It will be especially beneficial for minimization.
 */


/*
 * Buckingham potential and first two derivatives computed
 * as function of r and constant params a, b, c.
 */
static dreal buck(dreal r, dreal a, dreal b, dreal c);
static dreal d_buck(dreal r, dreal a, dreal b, dreal c);
static dreal dd_buck(dreal r, dreal a, dreal b, dreal c);

/*
 * Customized bisection method for Buckingham.
 */
static int bisection(
    dreal *root,
    dreal (*f)(dreal x, dreal a, dreal b, dreal c),
    dreal x0,
    dreal x1,
    dreal a,
    dreal b,
    dreal c,
    dreal tol);


#define R_SMALL  (1./128)
#define MAX_ITER 100
#define TOL_LO   1e-6
#define TOL_HI   1e-12

/*
 * To be called externally to determine extra parameters to Buckingham
 * potential,  a exp(-r*b) - c/r^6,  for piecewise-defined smooth extension
 * having form  A/r^6 + B.
 *
 *   A  -- return parameter A
 *   B  -- return parameter B
 *   Rswitch -- return switch distance within which extension is to be active
 *   uRswitch -- return energy at switch distance (kcal/mol)
 *   Rtop -- return distance for top of potential barrier
 *   uRtop -- return energy at top of potential barrier
 *   a  -- constant for Buckingham
 *   b  -- constant for Buckingham
 *   c  -- constant for Buckingham
 *
 * Returns 0 on success or FORCE_FAIL if something goes wrong.
 */
int ForcePrm_calc_safebuck(ForcePrm *f,
    dreal *A,
    dreal *B,
    dreal *Rswitch,
    dreal *uRswitch,
    dreal *Rtop,
    dreal *uRtop,
    dreal a,
    dreal b,
    dreal c
    )
{
  dreal r0;
  dreal ur0;
  dreal dur0;
  dreal r1;
  dreal ur1;
  dreal fac;
  dreal root0;
  dreal root1;
  dreal droot0;
  dreal ddroot0;
  dreal (*U)(dreal, dreal, dreal, dreal) = buck;
  dreal (*dU)(dreal, dreal, dreal, dreal) = d_buck;
  dreal (*ddU)(dreal, dreal, dreal, dreal) = dd_buck;
  int cnt;

  if (a < 0 || b < 0 || c < 0) {
    /* method won't work, set everything to zero and return ERROR */
    *A = 0;
    *B = 0;
    *Rswitch = 0;
    *uRswitch = 0;
    *Rtop = 0;
    *uRtop = 0;
    return ERROR(ERR_VALUE);
  }
  else if (0==a || 0==b || 0==c) {
    /* method won't work, set everything to zero and return OK */
    *A = 0;
    *B = 0;
    *Rswitch = 0;
    *uRswitch = 0;
    *Rtop = 0;
    *uRtop = 0;
    return OK;
  }

  r0 = R_SMALL;
  ur0 = U(r0, a, b, c);
  fac = (ur0 < 0 ? 1.5 : 0.75);
  r1 = r0;
  ur1 = ur0;

  /* bracket root: find sign change in U */
  cnt = 0;
  while (ur0 * ur1 >= 0) {
    if (cnt == MAX_ITER) {
      return ERROR(ERR_EXPECT);  /* unable to bracket root */
    }
    r1 *= fac;
    ur1 = U(r1, a, b, c);
    cnt++;
  }
  /* this should be smallest root of U (no need for high tolerance) */
  if (bisection(&root0, U, r0, r1, a, b, c, TOL_LO)) {
    return ERROR(ERR_EXPECT);  /* bisection method failed to find root */
  }

  r0 = (r1 > root0 ? r1 : r0);
  ur0 = U(r0, a, b, c);
  r1 = r0;
  ur1 = ur0;
  /* (sanity check) */
  if (ur0 < 0) {
    return ERROR(ERR_EXPECT);  /* sign error */
  }
  fac = 1.5;

  /* bracket root: find sign change in U */
  cnt = 0;
  while (ur0 * ur1 >= 0) {
    if (cnt == MAX_ITER) {
      return ERROR(ERR_EXPECT);  /* unable to bracket root */
    }
    r1 *= fac;
    ur1 = U(r1, a, b, c);
    cnt++;
  }
  /* this should be largest root of U (no need for high tolerance) */
  if (bisection(&root1, U, r0, r1, a, b, c, TOL_LO)) {
    return ERROR(ERR_EXPECT);  /* bisection method failed to find root */
  }

  /* (sanity check) */
  if (root0 >= root1) {
    return ERROR(ERR_EXPECT);  /* failed to find expected roots */
  }

  /* this should be smallest root of U'' (want high tolerance) */
  if (bisection(&ddroot0, ddU, root0, root1, a, b, c, TOL_HI)) {
    return ERROR(ERR_EXPECT);  /* bisection method failed to find root */
  }
  r0 = ddroot0;
  ur0 = U(r0, a, b, c);
  dur0 = dU(r0, a, b, c);
  *A = (-1./6) * dur0 * ((r0*r0*r0)*(r0*r0*r0)*r0);
  *B = ur0 - *A / ((r0*r0*r0)*(r0*r0*r0));
  *Rswitch = r0;
  *uRswitch = ur0;

  /* find barrier height - for diagnostic purposes */
  if (bisection(&droot0, dU, root0, root1, a, b, c, TOL_HI)) {
    return ERROR(ERR_EXPECT);  /* bisection method failed to find root */
  }
  *Rtop = droot0;
  *uRtop = U(droot0, a, b, c);

#if 0
  {
    dreal droot0;
    /* find barrier height */
    if (bisection(&droot0, dU, root0, root1, a, b, c, TOL_HI)) {
      printf("# ERROR (bucksafe.c, line %d): bisection failed to find root\n",
          __LINE__);
      return FORCE_FAIL;
    }
    printf("# Buckingham: barrier r=%.12g A, height U(r)=%.12g kcal/mol\n",
        droot0, U(droot0, a, b, c));
    printf("# extending with join at r=%.12g A, height U(r)=%.12g kcal/mol\n",
        ddroot0, U(ddroot0, a, b, c));
  }
#endif

  return 0;
}


dreal buck(dreal r, dreal a, dreal b, dreal c)
{
  dreal r6 = r*r*r * r*r*r;
  return (a * exp(-r*b) - c/r6);
}


dreal d_buck(dreal r, dreal a, dreal b, dreal c)
{
  dreal r7 = (r*r*r) * (r*r*r) * r;
  return (-a*b * exp(-r*b) + 6.*c/r7);
}


dreal dd_buck(dreal r, dreal a, dreal b, dreal c)
{
  dreal r8 = ((r*r)*(r*r)) * ((r*r)*(r*r));
  return (a*b*b * exp(-r*b) - 42.*c/r8);
}


int bisection(
    dreal *root,
    dreal (*f)(dreal x, dreal a, dreal b, dreal c),
    dreal x0,
    dreal x1,
    dreal a,
    dreal b,
    dreal c,
    dreal tol)
{
  dreal fx0;
  dreal fx1;
  dreal m = 0.5 * (x0+x1);  /* needs to be initialized */
  dreal fm;
  int sfx0;
  int sfx1;
  int sfm;
  int cnt;

  /* swap values if in wrong order */
  if (x1 < x0) {
    dreal tmp = x1;
    x1 = x0;
    x0 = tmp;
#if 0
#ifdef DEBUG_WATCH
    NL_printf("bisection (DIAGNOSTIC) swapped x0 and x1\n");
    NL_printf("x0=%.12g  x1=%.12g\n", x0, x1);
#endif
#endif
  }

  /* make sure these endpoints give f(x) of opposite signs */
  fx0 = f(x0, a, b, c);
  fx1 = f(x1, a, b, c);
  sfx0 = (fx0 > 0 ? 1 : -1);
  sfx1 = (fx1 > 0 ? 1 : -1);
  if (sfx0 == sfx1) {
#if 0
#ifdef DEBUG_WATCH
    printf("bisection (FAILURE) function endpoints must have opposite sign\n");
    printf("f(x0)=%.12g  f(x1)=%.12g\n", fx0, fx1);
#endif
#endif
    return ERROR(ERR_EXPECT);  /* expected opposite signs */
  }

  cnt = 0;
  while ((x1-x0) > tol) {
    if (cnt == MAX_ITER) {
#if 0
#ifdef DEBUG_WATCH
      printf("bisection (FAILURE) exceeded max iteration count %d\n", cnt);
      printf("remaining interval:  x0=%.12g  x1=%.12g\n", x0, x1);
#endif
#endif
      return ERROR(ERR_EXPECT);  /* exceeded max iteration count */
    }
    m = 0.5 * (x0+x1);  /* midpoint */
    fm = f(m, a, b, c);
    sfm = (fm > 0 ? 1 : -1);
    if (sfm == sfx0) {
      x0 = m;
    }
    else {
      x1 = m;
    }
    cnt++;
  }
#if 0
#ifdef DEBUG_WATCH
  printf("bisection (SUCCESS) found root %.12g\n", m);
#endif
#endif
  *root = m;
  return 0;
}
