#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/nlenergy.h"


int NLEnergy_eval_force(NLEnergy *p) {
  Fnbcut *fnbcut = &(p->fnbcut);
  Fbonded *fbon = &(p->fbon);
  Energy *e = &(p->ener);
  Exclude *exclude = &(p->exclude);
  Coord *coord = &(p->coord);
  dvec *f = Coord_force(coord);
  const dvec *pos = Coord_pos_const(coord);
  ForcePrm *fprm = &(p->fprm);
  const Topology *topo = &(p->topo);
  const int32 natoms = Topology_atom_array_length(topo);
  const int32 nbonds = Topology_bond_array_length(topo);
  const int32 nangles = Topology_angle_array_length(topo);
  const int32 ndiheds = Topology_dihed_array_length(topo);
  const int32 nimprs = Topology_impr_array_length(topo);
  const char *bondsel = Array_data_const(&(p->bondsel));
  const char *anglesel = Array_data_const(&(p->anglesel));
  const char *dihedsel = Array_data_const(&(p->dihedsel));
  const char *imprsel = Array_data_const(&(p->imprsel));
  const int32 *idnonb = Array_data_const(&(p->idnonb));
  const int32 *idnonb_b = Array_data_const(&(p->idnonb_b));
  const int32 *idnbvdw = Array_data_const(&(p->idnbvdw));
  const int32 *idnbvdw_b = Array_data_const(&(p->idnbvdw_b));
  const int32 idnonb_len = Array_length(&(p->idnonb));
  const int32 idnonb_b_len = Array_length(&(p->idnonb_b));
  const int32 idnbvdw_len = Array_length(&(p->idnbvdw));
  const int32 idnbvdw_b_len = Array_length(&(p->idnbvdw_b));
  int32 nbpairType = 0;
  int32 id;
  int s;

  TEXT("eval force");
  memset(e, 0, sizeof(Energy));
  memset(f, 0, natoms * sizeof(dvec));

  /* evaluate bonds */
  for (id = 0;  id < nbonds;  id++) {
    if (bondsel[id]) {
      const BondPrm *bondprm = ForcePrm_bondprm_array(fprm);
      const Bond *bond = Topology_bond(topo, id);
      int32 pid;
      int32 i, j;
      if (NULL==bond || (pid=bond->bondPrmID)==FAIL) continue;
      i = bond->atomID[0];
      j = bond->atomID[1];
      if ((s=Fbonded_eval_bond_term(fbon, &bondprm[pid],
              &pos[i], &pos[j], &f[i], &f[j],
              &(e->pe_bond), e->f_virial)) != OK) return ERROR(s);
    }
  }

  /* evaluate angles */
  for (id = 0;  id < nangles;  id++) {
    if (anglesel[id]) {
      const AnglePrm *angleprm = ForcePrm_angleprm_array(fprm);
      const Angle *angle = Topology_angle(topo, id);
      int32 pid;
      int32 i, j, k;
      if (NULL==angle || (pid=angle->anglePrmID)==FAIL) continue;
      i = angle->atomID[0];
      j = angle->atomID[1];
      k = angle->atomID[2];
      if ((s=Fbonded_eval_angle_term(fbon, &angleprm[pid],
              &pos[i], &pos[j], &pos[k], &f[i], &f[j], &f[k],
              &(e->pe_angle), e->f_virial)) != OK) return ERROR(s);
    }
  }

  /* evaluate dihedrals */
  for (id = 0;  id < ndiheds;  id++) {
    if (dihedsel[id]) {
      const DihedPrm *dihedprm = ForcePrm_dihedprm_array(fprm);
      const Dihed *dihed = Topology_dihed(topo, id);
      int32 pid;
      int32 i, j, k, l;
      if (NULL==dihed || (pid=dihed->dihedPrmID)==FAIL) continue;
      i = dihed->atomID[0];
      j = dihed->atomID[1];
      k = dihed->atomID[2];
      l = dihed->atomID[3];
      if ((s=Fbonded_eval_dihed_term(fbon, &dihedprm[pid],
              &pos[i], &pos[j], &pos[k], &pos[l],
              &f[i], &f[j], &f[k], &f[l],
              &(e->pe_dihed), e->f_virial)) != OK) return ERROR(s);
    }
  }

  /* evaluate impropers */
  for (id = 0;  id < nimprs;  id++) {
    if (imprsel[id]) {
      const ImprPrm *imprprm = ForcePrm_imprprm_array(fprm);
      const Impr *impr = Topology_impr(topo, id);
      int32 pid;
      int32 i, j, k, l;
      if (NULL==impr || (pid=impr->imprPrmID)==FAIL) continue;
      i = impr->atomID[0];
      j = impr->atomID[1];
      k = impr->atomID[2];
      l = impr->atomID[3];
      if ((s=Fbonded_eval_impr_term(fbon, &imprprm[pid],
              &pos[i], &pos[j], &pos[k], &pos[l],
              &f[i], &f[j], &f[k], &f[l],
              &(e->pe_impr), e->f_virial)) != OK) return ERROR(s);
    }
  }

  /* evaluate cutoff nonbonded */
  nbpairType = p->fnbcut_all | p->fnbcut_subset | p->fnbcut_disjoint;
  HEX(nbpairType);
  if (nbpairType) {
    dvec rmin, rmax;
    FnbcutPrm fnbcutprm;

    /* preliminary setup */
    memset(&fnbcutprm, 0, sizeof(FnbcutPrm));
    if (nbpairType & FNBCUT_VDW) {
      if ((s=ForcePrm_setup_vdwtable(fprm)) != OK) return ERROR(s);
    }
    if ((s=Exclude_setup(exclude)) != OK) return ERROR(s);
    if ((s=Fnbcut_find_rminmax(fnbcut, &rmin, &rmax, pos, natoms)) != OK) {
      return ERROR(s);
    }

    INT(p->nb_overlap);
    HEX(p->fnbcut_all);
    HEX(p->fnbcut_subset);
    HEX(p->fnbcut_disjoint);
    if (p->nb_overlap || (nbpairType & FNBCUT_ELEC)) {
      fnbcutprm.nbpairType = (p->nb_overlap ? nbpairType : FNBCUT_ELEC);
      if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
              &rmin, &rmax)) != OK) {
        return ERROR(s);
      }
      if (p->fnbcut_all & FNBCUT_ELEC) {
        if ((s=Fnbcut_eval(fnbcut, pos, f, e)) != OK) return ERROR(s);
      }
      else if (p->fnbcut_subset & FNBCUT_ELEC) {
        if ((s=Fnbcut_eval_subset(fnbcut, pos, f, e,
                idnonb, idnonb_len)) != OK) {
          return ERROR(s);
        }
      }
      else {
        if ((s=Fnbcut_eval_disjoint_subsets(fnbcut, pos, f, e,
                idnonb, idnonb_len, idnonb_b, idnonb_b_len)) != OK) {
          return ERROR(s);
        }
      }
    } /* if overlapped or elec */

    if ( ! p->nb_overlap && (nbpairType & FNBCUT_VDW)) {
      fnbcutprm.nbpairType = FNBCUT_VDW;
      if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
              &rmin, &rmax)) != OK) {
        return ERROR(s);
      }
      if (p->fnbcut_all & FNBCUT_VDW) {
        if ((s=Fnbcut_eval(fnbcut, pos, f, e)) != OK) return ERROR(s);
      }
      else if (p->fnbcut_subset & FNBCUT_VDW) {
        if ((s=Fnbcut_eval_subset(fnbcut, pos, f, e,
                idnbvdw, idnbvdw_len)) != OK) {
          return ERROR(s);
        }
      }
      else {
        if ((s=Fnbcut_eval_disjoint_subsets(fnbcut, pos, f, e,
                idnbvdw, idnbvdw_len, idnbvdw_b, idnbvdw_b_len)) != OK) {
          return ERROR(s);
        }
      }
    } /* if not overlapped */

  } /* if nonbonded */

  /* accumulate the potential */
  e->pe = e->pe_bond + e->pe_angle + e->pe_dihed + e->pe_impr +
    e->pe_elec + e->pe_vdw;

#if 0
    if (p->fnbcut_disjoint != 0) {
      if (p->nb_overlap) {
        fnbcutprm.nbpairType = nbpairType;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval_disjoint_subsets(fnbcut, pos, f, e,
                idnonb, idnonb_len, idnonb_b, idnonb_b_len)) != OK) {
          return ERROR(s);
        }
      }
      else if (p->fnbcut_disjoint & FNBCUT_ELEC) {
        fnbcutprm.nbpairType = FNBCUT_ELEC;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval_disjoint_subsets(fnbcut, pos, f, e,
                idnonb, idnonb_len, idnonb_b, idnonb_b_len)) != OK) {
          return ERROR(s);
        }
      }
      else if (p->fnbcut_disjoint & FNBCUT_VDW) {
        fnbcutprm.nbpairType = FNBCUT_VDW;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval_disjoint_subsets(fnbcut, pos, f, e,
                idnbvdw, idnbvdw_len, idnbvdw_b, idnbvdw_b_len)) != OK) {
          return ERROR(s);
        }
      }
    } /* if disjoint */

    if (p->fnbcut_subset != 0) {
      if (p->nb_overlap) {
        fnbcutprm.nbpairType = nbpairType;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval_subset(fnbcut, pos, f, e,
                idnonb, idnonb_len)) != OK) {
          return ERROR(s);
        }
      }
      else if (p->fnbcut_subset & FNBCUT_ELEC) {
        fnbcutprm.nbpairType = FNBCUT_ELEC;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval_subset(fnbcut, pos, f, e,
                idnonb, idnonb_len)) != OK) {
          return ERROR(s);
        }
      }
      else if (p->fnbcut_subset & FNBCUT_VDW) {
        fnbcutprm.nbpairType = FNBCUT_VDW;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval_subset(fnbcut, pos, f, e,
                idnbvdw, idnbvdw_len)) != OK) {
          return ERROR(s);
        }
      }
    } /* if subset */

    if (p->fnbcut_all != 0) {
      if (p->nb_overlap) {
        fnbcutprm.nbpairType = nbpairType;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval(fnbcut, pos, f, e)) != OK) return ERROR(s);
      }
      else if (p->fnbcut_subset & FNBCUT_ELEC) {
        fnbcutprm.nbpairType = FNBCUT_ELEC;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval(fnbcut, pos, f, e)) != OK) return ERROR(s);
      }
      else if (p->fnbcut_subset & FNBCUT_VDW) {
        fnbcutprm.nbpairType = FNBCUT_VDW;
        if ((s=Fnbcut_setup(fnbcut, &fnbcutprm, Coord_domain(coord),
                &rmin, &rmax)) != OK) {
          return ERROR(s);
        }
        if ((s=Fnbcut_eval(fnbcut, pos, f, e)) != OK) return ERROR(s);
      }
    } /* if all */
#endif

  return OK;
}
