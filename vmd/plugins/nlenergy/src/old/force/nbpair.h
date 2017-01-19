/*
 * Copyright (C) 2007 by David J. Hardy.  All rights reserved.
 */

/**@file    force/nbpair.h
 * @brief   Nonbonded pair interaction functions.
 * @author  David J. Hardy
 * @date    Aug. 2007
 */

#ifndef FORCE_NBPAIR_H
#define FORCE_NBPAIR_H

#include <math.h>
#include "nlbase/nlbase.h"

#ifdef __cplusplus
extern "C" {
#endif

  int Nbpair_elec_infty(dreal *u, dreal *du_r, dreal r2, dreal c);
  int Nbpair_elec_cutoff(dreal *u, dreal *du_r, dreal r2, dreal c,
      dreal inv_cutoff2);
  int Nbpair_elec_ewald(dreal *u, dreal *du_r, dreal r2, dreal c,
      dreal ewald_coef, dreal ewald_grad_coef);

  int Nbpair_vdw_infty(dreal *u, dreal *du_r, dreal r2, dreal a, dreal b);
  int Nbpair_vdw_cutoff(dreal *u, dreal *du_r, dreal r2, dreal a, dreal b,
      dreal roff2, dreal ron2, dreal inv_denom_switch);

  int Nbpair_buck_infty(dreal *u, dreal *du_r, dreal r2,
      dreal a, dreal b, dreal c, dreal as, dreal bs, dreal rs2);
  int Nbpair_buck_cutoff(dreal *u, dreal *du_r, dreal r2,
      dreal a, dreal b, dreal c, dreal as, dreal bs, dreal rs2,
      dreal roff2, dreal ron2, dreal inv_denom_switch);


  /*
   * evaluate electrostatic potential U(r) and force scaling (1/r)*(dU/dr)
   *
   * u returns potential
   * du_r returns force scaling
   * r2 is square of pairwise distance r
   * c is constant
   */
#define EVAL_NBPAIR_ELEC_INFTY(u, du_r, r2, c) \
  do { \
    const dreal _r2 = (r2); \
    dreal *_u = (u); \
    dreal inv_r;   /* 1/r */ \
    dreal inv_r2;  /* 1/r^2 */ \
    ASSERT(_r2 > 0.0); \
    inv_r2 = 1.0 / _r2; \
    inv_r = sqrt(inv_r2); \
    *_u = (c) * inv_r; \
    *(du_r) = -(*_u) * inv_r2; \
  } while (0)

  /*
   * evaluate electrostatic potential U(r) and force scaling (1/r)*(dU/dr)
   *
   * u returns smooth potential (shifted to 0 at cutoff)
   * du_r returns force scaling
   * r2 is square of distance
   * c is constant
   * inv_rc2 is inverse square of elec cutoff
   *
   * ASSUME ATOMS ARE WITHIN CUTOFF (i.e. r2 < cutoff2)
   */
#define EVAL_NBPAIR_ELEC_CUTOFF(u, du_r, r2, c, inv_rc2) \
  do { \
    const dreal _r2 = (r2); \
    const dreal _inv_rc2 = (inv_rc2); \
    dreal inv_r;   /* 1/r */ \
    dreal inv_r2;  /* 1/r^2 */ \
    dreal w;       /* full elec potential */ \
    dreal dw_r;    /* dw/r */ \
    dreal s;       /* smoothing function */ \
    dreal ds_r;    /* ds/r */ \
    ASSERT(_r2 > 0.0); \
    inv_r2 = 1.0 / _r2; \
    ASSERT(inv_r2 >= _inv_rc2); \
    inv_r = sqrt(inv_r2); \
    w = (c) * inv_r; \
    dw_r = -w * inv_r2; \
    s = (1.0 - _r2 * _inv_rc2) * (1.0 - _r2 * _inv_rc2); \
    ds_r = -4.0 * _inv_rc2 * (1.0 - _r2 * _inv_rc2); \
    *(u) = w * s; \
    *(du_r) = w * ds_r + dw_r * s; \
  } while (0)

  /*
   * evaluate electrostatic potential U(r) and force scaling (1/r)*(dU/dr)
   *
   * u returns Ewald direct space potential
   * du_r returns force scaling
   * r2 is square of distance
   * c is electrostatic constant
   * ewald_coef is coefficient for direct Ewald energy
   * grad_coef is coefficient for gradient of direct Ewald energy
   *
   * ASSUME ATOMS ARE WITHIN CUTOFF (i.e. r2 < cutoff2)
   */
#define EVAL_NBPAIR_ELEC_EWALD(u, du_r, r2, c, ewald_coef, grad_coef) \
  do { \
    const dreal _r2 = (r2); \
    dreal *_u = (u); \
    dreal r; \
    dreal inv_r;   /* 1/r */ \
    dreal inv_r2;  /* 1/r^2 */ \
    dreal a, b; \
    const dreal _c = (c); \
    ASSERT(_r2 > 0.0); \
    r = sqrt(_r2); \
    inv_r = 1.0 / r; \
    inv_r2 = inv_r * inv_r; \
    a = r * (ewald_coef); \
    b = erfc(a); \
    *_u = _c * b * inv_r; \
    *(du_r) = -(_c * (grad_coef) * exp(-a*a) + *_u) * inv_r2; \
  } while (0)


  /*
   * evaluate van der Waals potential U(r) and force scaling (1/r)*(dU/dr)
   *
   * u returns potential
   * du_r returns force scaling
   * r2 is square of pairwise distance r
   * a, b are constants
   */
#define EVAL_NBPAIR_VDW_INFTY(u, du_r, r2, a, b) \
  do { \
    const dreal _r2 = (r2); \
    dreal inv_r2;   /* 1/r^2 */ \
    dreal inv_r6;   /* 1/r^6 */ \
    dreal inv_r12;  /* 1/r^12 */ \
    dreal a_r12;    /* a/r^12 */ \
    dreal b_r6;     /* b/r^6 */ \
    ASSERT(_r2 > 0.0); \
    inv_r2 = 1.0 / _r2; \
    inv_r6 = inv_r2 * inv_r2 * inv_r2; \
    inv_r12 = inv_r6 * inv_r6; \
    a_r12 = (a) * inv_r12; \
    b_r6 = (b) * inv_r6; \
    *(u) = a_r12 - b_r6; \
    *(du_r) = (-12.0 * a_r12 + 6.0 * b_r6) * inv_r2; \
  } while (0)

  /*
   * evaluate van der Waals potential U(r) and force scaling (1/r)*(dU/dr)
   *
   * u returns smooth potential (switched from switchdist to 0 at cutoff)
   * du_r returns force scaling
   * r2 is square of pairwise distance
   * a, b are constants
   * roff2 is square of cutoff distance (cutoff2)
   * ron2 is square of switching distance (switchdist2)
   * denom is inverse of denominator of switching function
   *
   * ASSUME ATOMS ARE WITHIN CUTOFF (i.e. r2 < cutoff2)
   */
#define EVAL_NBPAIR_VDW_CUTOFF(u, du_r, r2, a, b, roff2, ron2, denom) \
  do { \
    const dreal _r2 = r2; \
    const dreal _ron2 = ron2; \
    const dreal _roff2 = roff2; \
    const dreal _denom = denom; \
    dreal inv_r2;   /* 1/r^2 */ \
    dreal inv_r6;   /* 1/r^6 */ \
    dreal inv_r12;  /* 1/r^12 */ \
    dreal a_r12;    /* a/r^12 */ \
    dreal b_r6;     /* b/r^6 */ \
    dreal w;        /* full vdw potential */ \
    dreal dw_r;     /* dw/r */ \
    dreal s;        /* switching function */ \
    dreal ds_r;     /* ds/r */ \
    ASSERT(_ron2 < _roff2); \
    ASSERT(_r2 <= _roff2); \
    ASSERT(_r2 > 0.0); \
    inv_r2 = 1.0 / _r2; \
    inv_r6 = inv_r2 * inv_r2 * inv_r2; \
    inv_r12 = inv_r6 * inv_r6; \
    a_r12 = (a) * inv_r12; \
    b_r6 = (b) * inv_r6; \
    w = a_r12 - b_r6; \
    dw_r = (-12.0 * a_r12 + 6.0 * b_r6) * inv_r2; \
    if (_r2 > _ron2) { \
      s = (_roff2 - _r2) * (_roff2 - _r2) \
        * (_roff2 + 2.0 * _r2 - 3.0 * _ron2) * _denom; \
      ds_r = 12.0 * (_roff2 - _r2) * (_ron2 - _r2) * _denom; \
      *(u) = w * s; \
      *(du_r) = w * ds_r + dw_r * s; \
    } \
    else { \
      *(u) = w; \
      *(du_r) = dw_r; \
    } \
  } while (0)


  /*
   * evaluate Buckingham potential U(r) and force scaling (1/r)*(dU/dr)
   * with switched inner part for energy minimization
   *
   * u returns potential
   * du_r returns force scaling
   * r2 is square of pairwise distance
   * a, b, c are constant parameters of Buckingham potential
   * an, bn are constant parameters for inner function
   * rn2 is square inner switching distance
   */
#define EVAL_NBPAIR_BUCK_INFTY(u, du_r, r2, a, b, c, an, bn, rn2) \
  do { \
    const dreal _r2 = (r2); \
    dreal inv_r2;     /* 1/r^2 */ \
    dreal inv_r6;     /* 1/r^6 */ \
    dreal r; \
    dreal nbr;        /* -b*r */ \
    dreal c_r6;       /* c/r^6 */ \
    dreal a_exp_nbr;  /* a*exp(-b*r) */ \
    dreal an_r6;      /* an/r^6 */ \
    ASSERT(_r2 > 0.0); \
    if (_r2 > (rn2)) { \
      r = sqrt(_r2); \
      nbr = -r * (b); \
      a_exp_nbr = (a) * exp(nbr); \
      inv_r2 = 1.0 / _r2; \
      inv_r6 = inv_r2 * inv_r2 * inv_r2; \
      c_r6 = (c) * inv_r6; \
      *(u) = a_exp_nbr - c_r6; \
      *(du_r) = inv_r2 * (6.0 * c_r6 + nbr * a_exp_nbr); \
    } \
    else { \
      inv_r2 = 1.0 / _r2; \
      inv_r6 = inv_r2 * inv_r2 * inv_r2; \
      an_r6 = (an) * inv_r6; \
      *(u) = an_r6 + (bn); \
      *(du_r) = -6.0 * an_r6 * inv_r2; \
    } \
  } while (0)

  /*
   * evaluate Buckingham potential U(r) and force scaling (1/r)*(dU/dr)
   * with switched inner part for energy minimization
   *
   * u returns smooth potential (switched from switchdist to 0 at cutoff)
   * du_r returns force scaling
   * r2 is square of pairwise distance
   * a, b, c are constant parameters
   * an, bn are constant parameters for inner function
   * rn2 is square inner switching distance
   * roff2 is square of cutoff distance (cutoff2)
   * ron2 is square of switching distance (switchdist2)
   * denom is inverse of denominator of switching function
   *
   * ASSUME ATOMS ARE WITHIN CUTOFF (i.e. r2 < cutoff2)
   */
#define EVAL_NBPAIR_BUCK_CUTOFF(u, du_r, r2, a, b, c, an, bn, rn2, \
    roff2, ron2, denom) \
  do { \
    const dreal _r2 = (r2); \
    const dreal _ron2 = (ron2); \
    const dreal _roff2 = (roff2); \
    const dreal _denom = (denom); \
    dreal inv_r2;     /* 1/r^2 */ \
    dreal inv_r6;     /* 1/r^6 */ \
    dreal r; \
    dreal nbr;        /* -b*r */ \
    dreal c_r6;       /* c/r^6 */ \
    dreal a_exp_nbr;  /* a*exp(-b*r) */ \
    dreal an_r6;      /* an/r^6 */ \
    dreal w;          /* full Buckingham potential */ \
    dreal dw_r;       /* dw/r */ \
    dreal s;          /* switching function */ \
    dreal ds_r;       /* ds/r */ \
    ASSERT(_ron2 < _roff2); \
    ASSERT(_r2 <= _roff2); \
    ASSERT(_r2 > 0.0); \
    if (_r2 > (rn2)) { \
      r = sqrt(_r2); \
      nbr = -r * (b); \
      a_exp_nbr = (a) * exp(nbr); \
      inv_r2 = 1.0 / _r2; \
      inv_r6 = inv_r2 * inv_r2 * inv_r2; \
      c_r6 = (c) * inv_r6; \
      w = a_exp_nbr - c_r6; \
      dw_r = inv_r2 * (6.0 * c_r6 + nbr * a_exp_nbr); \
      if (_r2 > _ron2) { \
        s = (_roff2 - _r2) * (_roff2 - _r2) \
          * (_roff2 + 2.0 * _r2 - 3.0 * _ron2) * _denom; \
        ds_r = 12.0 * (_roff2 - _r2) * (_ron2 - _r2) * _denom; \
        *(u) = w * s; \
        *(du_r) = w * ds_r + dw_r * s; \
      } \
      else { \
        *(u) = w; \
        *(du_r) = dw_r; \
      } \
    } \
    else { \
      inv_r2 = 1.0 / _r2; \
      inv_r6 = inv_r2 * inv_r2 * inv_r2; \
      an_r6 = (an) * inv_r6; \
      *(u) = an_r6 + (bn); \
      *(du_r) = -6.0 * an_r6 * inv_r2; \
    } \
  } while (0)


#ifdef FAST_NBPAIR

#define NBPAIR_ELEC_INFTY  EVAL_NBPAIR_ELEC_INFTY
#define NBPAIR_ELEC_CUTOFF EVAL_NBPAIR_ELEC_CUTOFF
#define NBPAIR_ELEC_EWALD  EVAL_NBPAIR_ELEC_EWALD

#define NBPAIR_VDW_INFTY   EVAL_NBPAIR_VDW_INFTY
#define NBPAIR_VDW_CUTOFF  EVAL_NBPAIR_VDW_CUTOFF

#define NBPAIR_BUCK_INFTY  EVAL_NBPAIR_BUCK_INFTY
#define NBPAIR_BUCK_CUTOFF EVAL_NBPAIR_BUCK_CUTOFF

#else

#define NBPAIR_ELEC_INFTY  Nbpair_elec_infty
#define NBPAIR_ELEC_CUTOFF Nbpair_elec_cutoff
#define NBPAIR_ELEC_EWALD  Nbpair_elec_ewald

#define NBPAIR_VDW_INFTY   Nbpair_vdw_infty
#define NBPAIR_VDW_CUTOFF  Nbpair_vdw_cutoff

#define NBPAIR_BUCK_INFTY  Nbpair_buck_infty
#define NBPAIR_BUCK_CUTOFF Nbpair_buck_cutoff

#endif


#ifdef __cplusplus
}
#endif

#endif /* FORCE_NBPAIR_H */
