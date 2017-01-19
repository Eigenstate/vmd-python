/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

/**@file    force/fbonded.h
 * @brief   Evaluate bonded interactions.
 * @author  David J. Hardy
 * @date    Apr. 2008
 */

#ifndef FORCE_FBONDED_H
#define FORCE_FBONDED_H

#include "moltypes/moltypes.h"

#ifdef __cplusplus
extern "C" {
#endif


  typedef struct Fbonded_t {
    const ForcePrm *fprm;
    const Topology *topo;
    const Domain *domain;
  } Fbonded;

  int Fbonded_init(Fbonded *, const Topology *);
  void Fbonded_done(Fbonded *);

  int Fbonded_setup(Fbonded *, const Domain *);

  /**@brief Evaluate all bonded forces. */
  int Fbonded_eval(Fbonded *, const dvec *pos, dvec *f, Energy *en);

  /**@brief Evaluate the bond-length forces. */
  int Fbonded_eval_bond(Fbonded *, const dvec *pos, dvec *f, Energy *en);

  /**@brief Evaluate one bond-length force contribution. */
  int Fbonded_eval_bond_term(Fbonded *,
      const BondPrm *bondprm,
      const dvec *pos_i,
      const dvec *pos_j,
      dvec *f_bond_i,
      dvec *f_bond_j,
      dreal *u_bond_ij,
      dreal virial_bond_ij[NELEMS_VIRIAL]);

  /**@brief Evaluate the angle forces. */
  int Fbonded_eval_angle(Fbonded *, const dvec *pos, dvec *f, Energy *en);

  /**@brief Evaluate one angle force contribution. */
  int Fbonded_eval_angle_term(Fbonded *,
      const AnglePrm *angleprm,
      const dvec *pos_i,
      const dvec *pos_j,
      const dvec *pos_k,
      dvec *f_angle_i,
      dvec *f_angle_j,
      dvec *f_angle_k,
      dreal *u_angle_ijk,
      dreal virial_angle_ijk[NELEMS_VIRIAL]);

  /**@brief Evaluate the dihedral forces. */
  int Fbonded_eval_dihed(Fbonded *, const dvec *pos, dvec *f, Energy *en);

  /**@brief Evaluate one dihedral force contribution. */
  int Fbonded_eval_dihed_term(Fbonded *,
      const DihedPrm *dihedprm,
      const dvec *pos_i,
      const dvec *pos_j,
      const dvec *pos_k,
      const dvec *pos_l,
      dvec *f_dihed_i,
      dvec *f_dihed_j,
      dvec *f_dihed_k,
      dvec *f_dihed_l,
      dreal *u_dihed_ijkl,
      dreal virial_dihed_ijkl[NELEMS_VIRIAL]);

  /**@brief Evaluate the improper forces. */
  int Fbonded_eval_impr(Fbonded *, const dvec *pos, dvec *f, Energy *en);

  /**@brief Evaluate one improper force contribution. */
  int Fbonded_eval_impr_term(Fbonded *,
      const ImprPrm *imprprm,
      const dvec *pos_i,
      const dvec *pos_j,
      const dvec *pos_k,
      const dvec *pos_l,
      dvec *f_impr_i,
      dvec *f_impr_j,
      dvec *f_impr_k,
      dvec *f_impr_l,
      dreal *u_impr_ijkl,
      dreal virial_impr_ijkl[NELEMS_VIRIAL]);


#ifdef __cplusplus
}
#endif

#endif /* FORCE_FBONDED_H */
