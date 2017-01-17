/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <math.h>
#include "moltypes/vecops.h"
#include "force/fbonded.h"


int Fbonded_eval_bond(Fbonded *p, const dvec *pos, dvec *f, Energy *e) {
  const BondPrm *bondprm = ForcePrm_bondprm_array(p->fprm);
  const Bond *bond = Topology_bond_array(p->topo);
  int32 nbonds = Topology_bond_array_length(p->topo);
  int32 n, i, j, pid;
  int s;

  for (n = 0;  n < nbonds;  n++) {
    pid = bond[n].bondPrmID;
    if (FAIL == pid) continue;
    i = bond[n].atomID[0];
    j = bond[n].atomID[1];
    if ((s=Fbonded_eval_bond_term(p, &bondprm[pid],
            &pos[i], &pos[j], &f[i], &f[j],
            &(e->pe_bond), e->f_virial)) != OK) return ERROR(s);
  }
  e->pe += e->pe_bond;
  return OK;
}


int Fbonded_eval_bond_term(Fbonded *p,
    const BondPrm *prm,
    const dvec *pos_i,
    const dvec *pos_j,
    dvec *f_i,
    dvec *f_j,
    dreal *u,
    dreal virial[NELEMS_VIRIAL]) {
  dvec r_ij, f_ij;
  dreal r_ij_len, dis, coef, energy;

  Domain_shortest_vec(p->domain, &r_ij, pos_j, pos_i);
  if (prm->r0 != 0) {
    r_ij_len = sqrt(VECLEN2(r_ij));
    dis = r_ij_len - prm->r0;
    coef = -2 * prm->k * dis / r_ij_len;
    energy = prm->k * dis * dis;
  }
  else {
    coef = -2 * prm->k;
    energy = prm->k * VECLEN2(r_ij);
  }
  *u += energy;
  VECMUL(f_ij, coef, r_ij);
  VECSUB(*f_i, *f_i, f_ij);
  VECADD(*f_j, *f_j, f_ij);
  virial[VIRIAL_XX] += f_ij.x * r_ij.x;
  virial[VIRIAL_XY] += f_ij.x * r_ij.y;
  virial[VIRIAL_XZ] += f_ij.x * r_ij.z;
  virial[VIRIAL_YY] += f_ij.y * r_ij.y;
  virial[VIRIAL_YZ] += f_ij.y * r_ij.z;
  virial[VIRIAL_ZZ] += f_ij.z * r_ij.z;
  return OK;
}
