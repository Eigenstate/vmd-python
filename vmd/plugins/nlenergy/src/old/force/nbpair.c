#include "force/nbpair.h"


int Nbpair_elec_infty(dreal *u, dreal *du_r, dreal r2, dreal c)
{
  EVAL_NBPAIR_ELEC_INFTY(u, du_r, r2, c);
  return OK;
}

int Nbpair_elec_cutoff(dreal *u, dreal *du_r, dreal r2, dreal c,
    dreal inv_cutoff2)
{
  EVAL_NBPAIR_ELEC_CUTOFF(u, du_r, r2, c, inv_cutoff2);
  return OK;
}

int Nbpair_elec_ewald(dreal *u, dreal *du_r, dreal r2, dreal c,
    dreal ewald_coef, dreal ewald_grad_coef)
{
  EVAL_NBPAIR_ELEC_EWALD(u, du_r, r2, c, ewald_coef, ewald_grad_coef);
  return OK;
}


int Nbpair_vdw_infty(dreal *u, dreal *du_r, dreal r2, dreal a, dreal b)
{
  EVAL_NBPAIR_VDW_INFTY(u, du_r, r2, a, b);
  return OK;
}

int Nbpair_vdw_cutoff(dreal *u, dreal *du_r, dreal r2, dreal a, dreal b,
    dreal roff2, dreal ron2, dreal inv_denom_switch)
{
  EVAL_NBPAIR_VDW_CUTOFF(u, du_r, r2, a, b, roff2, ron2, inv_denom_switch);
  return OK;
}


int Nbpair_buck_infty(dreal *u, dreal *du_r, dreal r2,
    dreal a, dreal b, dreal c, dreal as, dreal bs, dreal rs2)
{
  EVAL_NBPAIR_BUCK_INFTY(u, du_r, r2, a, b, c, as, bs, rs2);
  return OK;
}

int Nbpair_buck_cutoff(dreal *u, dreal *du_r, dreal r2,
    dreal a, dreal b, dreal c, dreal as, dreal bs, dreal rs2,
    dreal roff2, dreal ron2, dreal inv_denom_switch)
{
  EVAL_NBPAIR_BUCK_CUTOFF(u, du_r, r2, a, b, c, as, bs, rs2,
      roff2, ron2, inv_denom_switch);
  return OK;
}
