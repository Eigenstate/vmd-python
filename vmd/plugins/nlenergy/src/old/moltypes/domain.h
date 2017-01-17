/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    moltypes/domain.h
 * @brief   Domain and transformations for (semi-)periodic parallelepiped.
 * @author  David J. Hardy
 * @date    Mar. 2008
 */

#ifndef MOLTYPES_DOMAIN_H
#define MOLTYPES_DOMAIN_H

#include "nlbase/nlbase.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**@brief Linear transformation in 3D space.
   *
   * Must be nonsingular.
   */
  typedef struct Transform_t {
    dvec row[3];     /**< row vectors */
  } Transform;

  int Transform_init(Transform *, dvec col[3]);
  void Transform_done(Transform *);

  /* apply transformation to vector r resulting in vector tr */
  void Transform_apply(const Transform *, dvec *tr, const dvec *r);


  /**@brief Simulation domain for atomic system.
   * 
   * Modeled as a (semi-)periodic parallelepiped.
   * Initialize non-periodic dimensions using the 0-vector.
   */
  typedef struct Domain_t {
    dvec center;         /**< center of domain */
    dvec basis[3];       /**< basis vectors defining domain */
    dvec recip[3];       /**< row vector transformation to reciprocal space */
    boolean periodic_x;  /**< is periodic in x-direction? */
    boolean periodic_y;  /**< is periodic in y-direction? */
    boolean periodic_z;  /**< is periodic in z-direction? */
  } Domain;

  int Domain_init(Domain *);
  void Domain_done(Domain *);

  /* intended to be called once after init() */
  int Domain_setup(Domain *, const dvec *center,
      const dvec *basis_v1, const dvec *basis_v2, const dvec *basis_v3);

  /* calculate volume of domain */
  dreal Domain_volume(const Domain *);

  /*
   * rescale domain by applying linear transformation,
   * transformation can be applied in spite of periodicity
   */
  int Domain_rescale(Domain *, const Transform *t);

  /*
   * calculate shortest vector r_ij from r_i to r_j,
   * i.e., shortest (r_j-r_i) over all periodic images
   */
  void Domain_shortest_vec(const Domain *,
      dvec *r_ij, const dvec *r_j, const dvec *r_i);

  /*
   * calculate normalization of vector r to reciprocal space:
   *   s = A^(-1)(r - c) + 1/2
   * where n counts the number of basis vector displacements in each dimension;
   * for wrapped periodic system, s is in [0,1)^3, up to roundoff error
   */
  void Domain_normalize_vec(const Domain *, dvec *s, ivec *n, const dvec *r);

  /*
   * calculate displacement vector w bringing vector r into real space domain:
   *   w = -An, where n = floor(A^{-1}(r-c) + 1/2)
   * where n counts the number of basis vector displacements in each dimension,
   * so you can regain r from r+w by (r+w)+An
   *
   * on return, r+w is inside of domain (it "wraps" r)
   */
  void Domain_wrap_vec(const Domain *, dvec *w, ivec *n, const dvec *r);

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
  void Domain_wrap_vec_center(const Domain *, dvec *w, ivec *n, const dvec *r);


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
      const dvec *r, const dvec *p);


#ifdef __cplusplus
}
#endif

#endif /* MOLTYPES_DOMAIN_H */
