#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/tclwrap.h"


int NLEnergy_read_xplor(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("xplor");
  if (objc != 1) return ERROR(ERR_EXPECT);
  else {
    const char *fname = Tcl_GetString(objv[0]);
    int s;
    if ((s=ForcePrm_read_xplor(&(p->fprm), fname)) != OK) return ERROR(s);
    if ((s=Topology_setprm_atom_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_bond_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_angle_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_dihed_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_impr_array(&(p->topo))) < FAIL) return ERROR(s);
  }
  return OK;
}


int NLEnergy_read_charmm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("charmm");
  if (objc != 1) return ERROR(ERR_EXPECT);
  else {
    const char *fname = Tcl_GetString(objv[0]);
    int s;
    if ((s=ForcePrm_read_charmm(&(p->fprm), fname)) != OK) return ERROR(s);
    if ((s=Topology_setprm_atom_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_bond_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_angle_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_dihed_array(&(p->topo))) < FAIL) return ERROR(s);
    if ((s=Topology_setprm_impr_array(&(p->topo))) < FAIL) return ERROR(s);
  }
  return OK;
}


static int tclean(Topology *topo, int retval) {
  Topology_done(topo);
  NL_free(topo);
  return retval;
}

int NLEnergy_read_psf(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("psf");
  if (objc != 1) return ERROR(ERR_EXPECT);
  else {
    const char *fname = Tcl_GetString(objv[0]);
    Topology *t = NL_calloc(1, sizeof(Topology));
    const int32 *atomid = Array_data_const(&(p->atomid));
    const Atom *atom = NULL;
    const Bond *bond = NULL;
    const Angle *angle = NULL;
    const Dihed *dihed = NULL;
    const Impr *impr = NULL;
    const Excl *excl = NULL;
    int32 natoms, nbonds, nangles, ndiheds, nimprs, nexcls;
    int32 i;
    int s;

    if (NULL==t) return ERROR(ERR_MEMALLOC);
    if ((s=Topology_init(t, &(p->fprm))) != OK) {
      return tclean(t,ERROR(s));
    }
    if ((s=Topology_read_psf(t, fname)) != OK) {
      return tclean(t,ERROR(s));
    }
    if (Topology_atom_array_length(t) <= p->lastid) {
      return tclean(t,ERROR(ERR_EXPECT));
    }

    natoms = Topology_atom_array_length(t);
    atom = Topology_atom_array(t);
    for (i = 0;  i < natoms;  i++) {
      int32 a0=FAIL, n=i;
      if (n < p->firstid || n > p->lastid) continue;
      if ((a0 = atomid[n - p->firstid])==FAIL) continue;
      if ((s=Topology_update_atom(&(p->topo),&atom[i],a0)) != OK) {
        return tclean(t,ERROR(s));
      }
    }

    nbonds = Topology_bond_array_length(t);
    bond = Topology_bond_array(t);
    for (i = 0;  i < nbonds;  i++) {
      Bond a;
      int32 a0=FAIL, a1=FAIL, n;
      n = bond[i].atomID[0];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a0 = atomid[n - p->firstid])==FAIL) continue;
      n = bond[i].atomID[1];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a1 = atomid[n - p->firstid])==FAIL) continue;
      a.atomID[0] = a0;
      a.atomID[1] = a1;
      if ((n=Topology_add_bond(&(p->topo),&a)) < FAIL) {
        return tclean(t,ERROR(n));
      }
    }

    nangles = Topology_angle_array_length(t);
    angle = Topology_angle_array(t);
    for (i = 0;  i < nangles;  i++) {
      Angle a;
      int32 a0=FAIL, a1=FAIL, a2=FAIL, n;
      n = angle[i].atomID[0];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a0 = atomid[n - p->firstid])==FAIL) continue;
      n = angle[i].atomID[1];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a1 = atomid[n - p->firstid])==FAIL) continue;
      n = angle[i].atomID[2];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a2 = atomid[n - p->firstid])==FAIL) continue;
      a.atomID[0] = a0;
      a.atomID[1] = a1;
      a.atomID[2] = a2;
      if ((n=Topology_add_angle(&(p->topo),&a)) < FAIL) {
        return tclean(t,ERROR(n));
      }
    }

    ndiheds = Topology_dihed_array_length(t);
    dihed = Topology_dihed_array(t);
    for (i = 0;  i < ndiheds;  i++) {
      Dihed a;
      int32 a0=FAIL, a1=FAIL, a2=FAIL, a3=FAIL, n;
      n = dihed[i].atomID[0];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a0 = atomid[n - p->firstid])==FAIL) continue;
      n = dihed[i].atomID[1];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a1 = atomid[n - p->firstid])==FAIL) continue;
      n = dihed[i].atomID[2];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a2 = atomid[n - p->firstid])==FAIL) continue;
      n = dihed[i].atomID[3];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a3 = atomid[n - p->firstid])==FAIL) continue;
      a.atomID[0] = a0;
      a.atomID[1] = a1;
      a.atomID[2] = a2;
      a.atomID[3] = a3;
      if ((n=Topology_add_dihed(&(p->topo),&a)) < FAIL) {
        return tclean(t,ERROR(n));
      }
    }

    nimprs = Topology_impr_array_length(t);
    impr = Topology_impr_array(t);
    for (i = 0;  i < nimprs;  i++) {
      Impr a;
      int32 a0=FAIL, a1=FAIL, a2=FAIL, a3=FAIL, n;
      n = impr[i].atomID[0];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a0 = atomid[n - p->firstid])==FAIL) continue;
      n = impr[i].atomID[1];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a1 = atomid[n - p->firstid])==FAIL) continue;
      n = impr[i].atomID[2];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a2 = atomid[n - p->firstid])==FAIL) continue;
      n = impr[i].atomID[3];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a3 = atomid[n - p->firstid])==FAIL) continue;
      a.atomID[0] = a0;
      a.atomID[1] = a1;
      a.atomID[2] = a2;
      a.atomID[3] = a3;
      if ((n=Topology_add_impr(&(p->topo),&a)) < FAIL) {
        return tclean(t,ERROR(n));
      }
    }

    nexcls = Topology_excl_array_length(t);
    excl = Topology_excl_array(t);
    for (i = 0;  i < nexcls;  i++) {
      Excl a;
      int32 a0=FAIL, a1=FAIL, n;
      n = excl[i].atomID[0];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a0 = atomid[n - p->firstid])==FAIL) continue;
      n = excl[i].atomID[1];
      if (n < p->firstid || n > p->lastid) continue;
      if ((a1 = atomid[n - p->firstid])==FAIL) continue;
      a.atomID[0] = a0;
      a.atomID[1] = a1;
      if ((n=Topology_add_excl(&(p->topo),&a)) < FAIL) {
        return tclean(t,ERROR(n));
      }
    }

    Topology_done(t);
    NL_free(t);

    if ((s=Exclude_setup(&(p->exclude))) != OK) return ERROR(s);
  }
  return OK;
}


int NLEnergy_read_pdb(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("pdb");
  if (objc != 1) return ERROR(ERR_EXPECT);
  else {
    PdbAtom pdb;
    const int32 natoms = Topology_atom_array_length(&(p->topo));
    const char *fname = Tcl_GetString(objv[0]);
    int s;
    if ((s=PdbAtom_init(&pdb)) != OK) return ERROR(s);
    if ((s=PdbAtom_setup(&pdb, natoms)) != OK) {
      PdbAtom_done(&pdb);
      return ERROR(s);
    }
    if ((s=PdbAtom_read(&pdb, PDBATOM_POS, fname)) != OK) {
      PdbAtom_done(&pdb);
      return ERROR(s);
    }
    ASSERT(PdbAtom_numatoms(&pdb)==natoms);
    if ((s=Coord_set_pos(&(p->coord), PdbAtom_coord_const(&pdb),
            PdbAtom_numatoms(&pdb), UPDATE_ALL)) != OK) {
      PdbAtom_done(&pdb);
      return ERROR(s);
    }
    PdbAtom_done(&pdb);
  }
  return OK;
}


int NLEnergy_read_namdbin(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("namdbin");
  return OK;
}
