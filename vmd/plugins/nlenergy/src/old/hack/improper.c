/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#include <math.h>
#include "moltypes/vecops.h"
#include "force/fbonded.h"


int Fbonded_eval_impr(Fbonded *p, const dvec *pos, dvec *f, Energy *e) {
  const ImprPrm *imprprm = ForcePrm_imprprm_array(p->fprm);
  const Impr *impr = Topology_impr_array(p->topo);
  int32 nimprs = Topology_impr_array_length(p->topo);
  int32 n, i, j, k, l, pid;
  int s;

  for (n = 0;  n < nimprs;  n++) {
    pid = impr[n].imprPrmID;
    if (FAIL == pid) continue;
    i = impr[n].atomID[0];
    j = impr[n].atomID[1];
    k = impr[n].atomID[2];
    l = impr[n].atomID[3];
    if ((s=Fbonded_eval_impr_term(p, &imprprm[pid],
            &pos[i], &pos[j], &pos[k], &pos[l], &f[i], &f[j], &f[k], &f[l],
            &(e->pe_impr), e->f_virial)) != OK) return ERROR(s);
  }
  e->pe += e->pe_impr;
  return OK;
}


int Fbonded_eval_impr_term(Fbonded *p,
    const ImprPrm *prm,
    const dvec *pos_i,
    const dvec *pos_j,
    const dvec *pos_k,
    const dvec *pos_l,
    dvec *f_i,
    dvec *f_j,
    dvec *f_k,
    dvec *f_l,
    dreal *u,
    dreal virial[NELEMS_VIRIAL]) {
  dvec r12, r23, r34;
  dvec A, B, C;
  dreal rA, rB, rC;
  dreal cos_phi, sin_phi, phi;
  dreal K, K1;
  dvec f1, f2, f3;

  dreal k = prm->k_impr;
  dreal delta = prm->psi0;
  dreal diff;

  Domain_shortest_vec(p->domain, &r12, pos_i, pos_j);
  Domain_shortest_vec(p->domain, &r23, pos_j, pos_k);
  Domain_shortest_vec(p->domain, &r34, pos_k, pos_l);

  VECCROSS(A, r12, r23);
  VECCROSS(B, r23, r34);
  VECCROSS(C, r23, A);

  rA = 1 / sqrt(VECLEN2(A));
  rB = 1 / sqrt(VECLEN2(B));
  rC = 1 / sqrt(VECLEN2(C));

  VECMUL(B, rB, B);  /* normalize B */
  cos_phi = VECDOT(A, B) * rA;
  sin_phi = VECDOT(C, B) * rC;

  phi = -atan2(sin_phi, cos_phi);

  diff = phi - delta;
  if      (diff < -M_PI)  diff += 2.0 * M_PI;
  else if (diff >  M_PI)  diff -= 2.0 * M_PI;
  K = k * diff * diff;
  K1 = 2.0 * k * diff;

  if (fabs(sin_phi) > 0.1) {
    dvec dcosdA, dcosdB;
    dvec tv1, tv2;

    /* use sine version to avoid 1/cos terms */

    VECMUL(A, rA, A);  /* normalize A */
    VECMSUB(dcosdA, cos_phi, A, B);
    VECMUL(dcosdA, rA, dcosdA);
    VECMSUB(dcosdB, cos_phi, B, A);
    VECMUL(dcosdB, rB, dcosdB);

    K1 /= sin_phi;
    VECCROSS(f1, r23, dcosdA);
    VECMUL(f1, K1, f1);
    VECCROSS(f3, dcosdB, r23);
    VECMUL(f3, K1, f3);

    VECCROSS(tv1, dcosdA, r12);
    VECCROSS(tv2, r34, dcosdB);
    VECADD(f2, tv1, tv2);
    VECMUL(f2, K1, f2);
  }
  else {
    dvec dsindB, dsindC;

    /* phi is too close to 0 or pi, use cos version to avoid 1/sin */

    VECMUL(C, rC, C);  /* normalize C */
    VECMSUB(dsindC, sin_phi, C, B);
    VECMUL(dsindC, rC, dsindC);
    VECMSUB(dsindB, sin_phi, B, C);
    VECMUL(dsindB, rB, dsindB);

    K1 /= -cos_phi;
    f1.x = K1 * ((r23.y * r23.y + r23.z * r23.z) * dsindC.x
                 - r23.x * r23.y * dsindC.y
                 - r23.x * r23.z * dsindC.z);
    f1.y = K1 * ((r23.z * r23.z + r23.x * r23.x) * dsindC.y
                 - r23.y * r23.z * dsindC.z
                 - r23.y * r23.x * dsindC.x);
    f1.z = K1 * ((r23.x * r23.x + r23.y * r23.y) * dsindC.z
                 - r23.z * r23.x * dsindC.x
                 - r23.z * r23.y * dsindC.y);

    VECCROSS(f3, dsindB, r23);
    VECMUL(f3, K1, f3);

    f2.x = K1 * (-(r23.y * r12.y + r23.z * r12.z) * dsindC.x
                 + (2.0 * r23.x * r12.y - r12.x * r23.y) * dsindC.y
                 + (2.0 * r23.x * r12.z - r12.x * r23.z) * dsindC.z
                 + dsindB.z * r34.y - dsindB.y * r34.z);
    f2.y = K1 * (-(r23.z * r12.z + r23.x * r12.x) * dsindC.y
                 + (2.0 * r23.y * r12.z - r12.y * r23.z) * dsindC.z
                 + (2.0 * r23.y * r12.x - r12.y * r23.x) * dsindC.x
                 + dsindB.x * r34.z - dsindB.z * r34.x);
    f2.z = K1 * (-(r23.x * r12.x + r23.y * r12.y) * dsindC.z
                 + (2.0 * r23.z * r12.x - r12.z * r23.x) * dsindC.x
                 + (2.0 * r23.z * r12.y - r12.z * r23.y) * dsindC.y
                 + dsindB.y * r34.x - dsindB.x * r34.y);
  }
  *u += K;
  VECADD(*f_i, *f_i, f1);
  f_j->x += f2.x - f1.x;
  f_j->y += f2.y - f1.y;
  f_j->z += f2.z - f1.z;
  f_k->x += f3.x - f2.x;
  f_k->y += f3.y - f2.y;
  f_k->z += f3.z - f2.z;
  VECSUB(*f_l, *f_l, f3);
  virial[VIRIAL_XX] += (f1.x * r12.x + f2.x * r23.x + f3.x * r34.x);
  virial[VIRIAL_XY] += (f1.x * r12.y + f2.x * r23.y + f3.x * r34.y);
  virial[VIRIAL_XZ] += (f1.x * r12.z + f2.x * r23.z + f3.x * r34.z);
  virial[VIRIAL_YY] += (f1.y * r12.y + f2.y * r23.y + f3.y * r34.y);
  virial[VIRIAL_YZ] += (f1.y * r12.z + f2.y * r23.z + f3.y * r34.z);
  virial[VIRIAL_ZZ] += (f1.z * r12.z + f2.z * r23.z + f3.z * r34.z);
  return OK;
}
