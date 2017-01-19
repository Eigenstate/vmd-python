/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <math.h>
#include "moltypes/vecops.h"
#include "force/fbonded.h"


int Fbonded_eval_angle(Fbonded *p, const dvec *pos, dvec *f, Energy *e) {
  const AnglePrm *angleprm = ForcePrm_angleprm_array(p->fprm);
  const Angle *angle = Topology_angle_array(p->topo);
  int32 nangles = Topology_angle_array_length(p->topo);
  int32 n, i, j, k, pid;
  int s;

  for (n = 0;  n < nangles;  n++) {
    pid = angle[n].anglePrmID;
    if (FAIL == pid) continue;
    i = angle[n].atomID[0];
    j = angle[n].atomID[1];
    k = angle[n].atomID[2];
    if ((s=Fbonded_eval_angle_term(p, &angleprm[pid],
            &pos[i], &pos[j], &pos[k], &f[i], &f[j], &f[k],
            &(e->pe_angle), e->f_virial)) != OK) return ERROR(s);
  }
  e->pe += e->pe_angle;
  return OK;
}


int Fbonded_eval_angle_term(Fbonded *p,
      const AnglePrm *prm,
      const dvec *pos_i,
      const dvec *pos_j,
      const dvec *pos_k,
      dvec *f_i,
      dvec *f_j,
      dvec *f_k,
      dreal *u,
      dreal virial[NELEMS_VIRIAL]) {
  dvec r21, r23, f1, f2, f3, r13, f13;
  dreal inv_r21len, inv_r23len, cos_theta, sin_theta, delta_theta, coef;
  dreal energy, r13len, dist;

  Domain_shortest_vec(p->domain, &r21, pos_i, pos_j);
  Domain_shortest_vec(p->domain, &r23, pos_k, pos_j);
  inv_r21len = 1 / sqrt(VECLEN2(r21));
  inv_r23len = 1 / sqrt(VECLEN2(r23));
  cos_theta = VECDOT(r21, r23) * inv_r21len * inv_r23len;
  /* cos(theta) should be in [-1,1],
   * however, we need to correct in case of roundoff error */
  if (cos_theta > 1.0)        cos_theta = 1.0;
  else if (cos_theta < -1.0)  cos_theta = -1.0;
  sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  delta_theta = acos(cos_theta) - prm->theta0;
  coef = -2.0 * prm->k_theta * delta_theta / sin_theta;
  energy = prm->k_theta * delta_theta * delta_theta;

  f1.x = coef * (cos_theta * r21.x * inv_r21len - r23.x * inv_r23len)
         * inv_r21len;
  f1.y = coef * (cos_theta * r21.y * inv_r21len - r23.y * inv_r23len)
         * inv_r21len;
  f1.z = coef * (cos_theta * r21.z * inv_r21len - r23.z * inv_r23len)
         * inv_r21len;
  f3.x = coef * (cos_theta * r23.x * inv_r23len - r21.x * inv_r21len)
         * inv_r23len;
  f3.y = coef * (cos_theta * r23.y * inv_r23len - r21.y * inv_r21len)
         * inv_r23len;
  f3.z = coef * (cos_theta * r23.z * inv_r23len - r21.z * inv_r21len)
         * inv_r23len;
  VECADD(f2, f1, f3);

  /* Urey-Bradley term effects only atoms 1 and 3 */
  if (prm->k_ub != 0.0) {
    r13.x = r23.x - r21.x;
    r13.y = r23.y - r21.y;
    r13.z = r23.z - r21.z;
    r13len = sqrt(r13.x * r13.x + r13.y * r13.y + r13.z * r13.z);
    dist = r13len - prm->r_ub;
    coef = -2.0 * prm->k_ub * dist / r13len;
    energy += prm->k_ub * dist * dist;
    f13.x = coef * r13.x;
    f13.y = coef * r13.y;
    f13.z = coef * r13.z;
    f1.x -= f13.x;
    f1.y -= f13.y;
    f1.z -= f13.z;
    f3.x += f13.x;
    f3.y += f13.y;
    f3.z += f13.z;
  }
  *u += energy;
  VECADD(*f_i, *f_i, f1);
  VECSUB(*f_j, *f_j, f2);
  VECADD(*f_k, *f_k, f3);
  virial[VIRIAL_XX] += (f1.x * r21.x + f3.x * r23.x);
  virial[VIRIAL_XY] += (f1.x * r21.y + f3.x * r23.y);
  virial[VIRIAL_XZ] += (f1.x * r21.z + f3.x * r23.z);
  virial[VIRIAL_YY] += (f1.y * r21.y + f3.y * r23.y);
  virial[VIRIAL_YZ] += (f1.y * r21.z + f3.y * r23.z);
  virial[VIRIAL_ZZ] += (f1.z * r21.z + f3.z * r23.z);
  return OK;
}
