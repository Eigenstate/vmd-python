#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/tclwrap.h"


int NLEnergy_remove_bond(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j;
  int s;

  TEXT("bond");
  if (objc != 2) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0) return ERROR(ERR_EXPECT);
  if ((id=Topology_getid_bond(topo, i, j)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=Topology_remove_bond(topo, id)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_angle(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j, k;
  int s;

  TEXT("angle");
  if (objc != 3) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0 ||
      (k=atomid_from_obj(p,interp,objv[2])) < 0) return ERROR(ERR_EXPECT);
  if ((id=Topology_getid_angle(topo, i, j, k)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=Topology_remove_angle(topo, id)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_dihed(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j, k, l;
  int s;

  TEXT("dihed");
  if (objc != 4) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0 ||
      (k=atomid_from_obj(p,interp,objv[2])) < 0 ||
      (l=atomid_from_obj(p,interp,objv[3])) < 0) return ERROR(ERR_EXPECT);
  if ((id=Topology_getid_dihed(topo, i, j, k, l)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=Topology_remove_dihed(topo, id)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_impr(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j, k, l;
  int s;

  TEXT("impr");
  if (objc != 4) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0 ||
      (k=atomid_from_obj(p,interp,objv[2])) < 0 ||
      (l=atomid_from_obj(p,interp,objv[3])) < 0) return ERROR(ERR_EXPECT);
  if ((id=Topology_getid_impr(topo, i, j, k, l)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=Topology_remove_impr(topo, id)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_atomprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const char *t0;
  int32 id;
  int s;

  TEXT("atomprm");
  if (objc != 1) return ERROR(ERR_EXPECT);
  t0 = Tcl_GetString(objv[0]);
  if ((id=ForcePrm_getid_atomprm(fprm, t0)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=ForcePrm_remove_atomprm(fprm, id)) != OK) return ERROR(s);
  if ((s=Topology_setprm_atom_array(&(p->topo)))!=OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_bondprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const char *t0, *t1;
  int32 id;
  int s;

  TEXT("bondprm");
  if (objc != 2) return ERROR(ERR_EXPECT);
  t0 = Tcl_GetString(objv[0]);
  t1 = Tcl_GetString(objv[1]);
  if ((id=ForcePrm_getid_bondprm(fprm, t0, t1)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=ForcePrm_remove_bondprm(fprm, id)) != OK) return ERROR(s);
  if ((s=Topology_setprm_bond_array(&(p->topo)))!=OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_angleprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const char *t0, *t1, *t2;
  int32 id;
  int s;

  TEXT("angleprm");
  if (objc != 3) return ERROR(ERR_EXPECT);
  t0 = Tcl_GetString(objv[0]);
  t1 = Tcl_GetString(objv[1]);
  t2 = Tcl_GetString(objv[2]);
  if ((id=ForcePrm_getid_angleprm(fprm, t0, t1, t2)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=ForcePrm_remove_angleprm(fprm, id)) != OK) return ERROR(s);
  if ((s=Topology_setprm_angle_array(&(p->topo)))!=OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_dihedprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const char *t0, *t1, *t2, *t3;
  int32 id;
  int s;

  TEXT("dihedprm");
  if (objc != 4) return ERROR(ERR_EXPECT);
  t0 = Tcl_GetString(objv[0]);
  t1 = Tcl_GetString(objv[1]);
  t2 = Tcl_GetString(objv[2]);
  t3 = Tcl_GetString(objv[3]);
  if ((id=ForcePrm_getid_dihedprm(fprm, t0, t1, t2, t3)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=ForcePrm_remove_dihedprm(fprm, id)) != OK) return ERROR(s);
  if ((s=Topology_setprm_dihed_array(&(p->topo)))!=OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_imprprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const char *t0, *t1, *t2, *t3;
  int32 id;
  int s;

  TEXT("imprprm");
  if (objc != 4) return ERROR(ERR_EXPECT);
  t0 = Tcl_GetString(objv[0]);
  t1 = Tcl_GetString(objv[1]);
  t2 = Tcl_GetString(objv[2]);
  t3 = Tcl_GetString(objv[3]);
  if ((id=ForcePrm_getid_imprprm(fprm, t0, t1, t2, t3)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=ForcePrm_remove_imprprm(fprm, id)) != OK) return ERROR(s);
  if ((s=Topology_setprm_impr_array(&(p->topo)))!=OK) return ERROR(s);
  return OK;
}


int NLEnergy_remove_vdwpairprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const char *t0, *t1;
  int32 id;
  int s;

  TEXT("vdwpairprm");
  if (objc != 4) return ERROR(ERR_EXPECT);
  t0 = Tcl_GetString(objv[0]);
  t1 = Tcl_GetString(objv[1]);
  if ((id=ForcePrm_getid_vdwpairprm(fprm, t0, t1)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((s=ForcePrm_remove_vdwpairprm(fprm, id)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_missing_atomprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const Atom *atom = Topology_atom_array(&(p->topo));
  const int32 natoms = Topology_atom_array_length(&(p->topo));
  int32 i;
  Tcl_Obj *alist = NULL;
  int s;

  TEXT("atomprm");
  if (objc != 0) return ERROR(ERR_EXPECT);
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  for (i = 0;  i < natoms;  i++) {
    int32 n = atom[i].atomPrmID;
    if (FAIL==n && (s=list_append_atomid(p, interp, alist, i)) != OK) {
      return ERROR(s);
    }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_missing_bondprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const Bond *bond = Topology_bond_array(&(p->topo));
  const int32 nbonds = Topology_bond_array_length(&(p->topo));
  int32 i;
  Tcl_Obj *alist = NULL;
  int s;

  TEXT("bondprm");
  if (objc != 0) return ERROR(ERR_EXPECT);
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  for (i = 0;  i < nbonds;  i++) {
    int32 n = bond[i].bondPrmID;
    if (FAIL==n) {
      Tcl_Obj *a = NULL;
      int32 a0 = bond[i].atomID[0];
      int32 a1 = bond[i].atomID[1];
      if ((s=new_list(interp, &a)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a0)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a1)) != OK) return ERROR(s);
      if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
    }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_missing_angleprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const Angle *angle = Topology_angle_array(&(p->topo));
  const int32 nangles = Topology_angle_array_length(&(p->topo));
  int32 i;
  Tcl_Obj *alist = NULL;
  int s;

  TEXT("angleprm");
  if (objc != 0) return ERROR(ERR_EXPECT);
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  for (i = 0;  i < nangles;  i++) {
    int32 n = angle[i].anglePrmID;
    if (FAIL==n) {
      Tcl_Obj *a = NULL;
      int32 a0 = angle[i].atomID[0];
      int32 a1 = angle[i].atomID[1];
      int32 a2 = angle[i].atomID[2];
      if ((s=new_list(interp, &a)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a0)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a1)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a2)) != OK) return ERROR(s);
      if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
    }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_missing_dihedprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const Dihed *dihed = Topology_dihed_array(&(p->topo));
  const int32 ndiheds = Topology_dihed_array_length(&(p->topo));
  int32 i;
  Tcl_Obj *alist = NULL;
  int s;

  TEXT("dihedprm");
  if (objc != 0) return ERROR(ERR_EXPECT);
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  for (i = 0;  i < ndiheds;  i++) {
    int32 n = dihed[i].dihedPrmID;
    if (FAIL==n) {
      Tcl_Obj *a = NULL;
      int32 a0 = dihed[i].atomID[0];
      int32 a1 = dihed[i].atomID[1];
      int32 a2 = dihed[i].atomID[2];
      int32 a3 = dihed[i].atomID[3];
      if ((s=new_list(interp, &a)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a0)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a1)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a2)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a3)) != OK) return ERROR(s);
      if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
    }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_missing_imprprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const Impr *impr = Topology_impr_array(&(p->topo));
  const int32 nimprs = Topology_impr_array_length(&(p->topo));
  int32 i;
  Tcl_Obj *alist = NULL;
  int s;

  TEXT("imprprm");
  if (objc != 0) return ERROR(ERR_EXPECT);
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  for (i = 0;  i < nimprs;  i++) {
    int32 n = impr[i].imprPrmID;
    if (FAIL==n) {
      Tcl_Obj *a = NULL;
      int32 a0 = impr[i].atomID[0];
      int32 a1 = impr[i].atomID[1];
      int32 a2 = impr[i].atomID[2];
      int32 a3 = impr[i].atomID[3];
      if ((s=new_list(interp, &a)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a0)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a1)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a2)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, a3)) != OK) return ERROR(s);
      if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
    }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}
