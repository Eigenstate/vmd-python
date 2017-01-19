#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/tclwrap.h"


int NLEnergy_add_bond(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j;
  Bond a;

  TEXT("bond");
  if (objc != 2) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0) return ERROR(ERR_EXPECT);
  a.atomID[0] = i;
  a.atomID[1] = j;
  a.bondPrmID = -1;
  if ((id=Topology_add_bond(topo, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  return OK;
}


int NLEnergy_add_angle(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j, k;
  Angle a;

  TEXT("angle");
  if (objc != 3) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0 ||
      (k=atomid_from_obj(p,interp,objv[2])) < 0) return ERROR(ERR_EXPECT);
  a.atomID[0] = i;
  a.atomID[1] = j;
  a.atomID[2] = k;
  a.anglePrmID = -1;
  if ((id=Topology_add_angle(topo, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  return OK;
}


int NLEnergy_add_dihed(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j, k, l;
  Dihed a;

  TEXT("dihed");
  if (objc != 4) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0 ||
      (k=atomid_from_obj(p,interp,objv[2])) < 0 ||
      (l=atomid_from_obj(p,interp,objv[3])) < 0) return ERROR(ERR_EXPECT);
  a.atomID[0] = i;
  a.atomID[1] = j;
  a.atomID[2] = k;
  a.atomID[3] = l;
  a.dihedPrmID = -1;
  if ((id=Topology_add_dihed(topo, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  return OK;
}


int NLEnergy_add_impr(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Topology *topo = &(p->topo);
  int32 id;
  int i, j, k, l;
  Impr a;

  TEXT("impr");
  if (objc != 4) return ERROR(ERR_EXPECT);
  if ((i=atomid_from_obj(p,interp,objv[0])) < 0 ||
      (j=atomid_from_obj(p,interp,objv[1])) < 0 ||
      (k=atomid_from_obj(p,interp,objv[2])) < 0 ||
      (l=atomid_from_obj(p,interp,objv[3])) < 0) return ERROR(ERR_EXPECT);
  a.atomID[0] = i;
  a.atomID[1] = j;
  a.atomID[2] = k;
  a.atomID[3] = l;
  a.imprPrmID = -1;
  if ((id=Topology_add_impr(topo, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  return OK;
}


int NLEnergy_add_atomprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  AtomPrm a;
  const char *t = NULL;
  int n;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int32 id;

  TEXT("atomprm");
  if (objc != 2) return ERROR(ERR_EXPECT);
  t = Tcl_GetStringFromObj(objv[0], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[0], t);
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[1], &aobjc, &aobjv)
      || aobjc != 4) return ERROR(ERR_EXPECT);
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d)
      || d > 0) return ERROR(ERR_EXPECT);
  a.emin = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.rmin = d;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[2], &d)
      || d > 0) return ERROR(ERR_EXPECT);
  a.emin14 = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[3], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.rmin14 = d;
  if ((id=ForcePrm_add_atomprm(fprm, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((n=Topology_setprm_atom_array(&(p->topo))) < FAIL) return ERROR(n);
  return OK;
}


int NLEnergy_add_bondprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  BondPrm a;
  const char *t = NULL;
  int n;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int32 id;

  TEXT("bondprm");
  if (objc != 3) return ERROR(ERR_EXPECT);
  t = Tcl_GetStringFromObj(objv[0], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[0], t);
  t = Tcl_GetStringFromObj(objv[1], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[1], t);
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[2], &aobjc, &aobjv)
      || aobjc != 2) return ERROR(ERR_EXPECT);
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.k = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.r0 = d;
  if ((id=ForcePrm_add_bondprm(fprm, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((n=Topology_setprm_bond_array(&(p->topo))) < FAIL) return ERROR(n);
  return OK;
}


int NLEnergy_add_angleprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  AnglePrm a;
  const char *t = NULL;
  int n;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int32 id;

  TEXT("angleprm");
  if (objc != 4) return ERROR(ERR_EXPECT);
  t = Tcl_GetStringFromObj(objv[0], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[0], t);
  t = Tcl_GetStringFromObj(objv[1], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[1], t);
  t = Tcl_GetStringFromObj(objv[2], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[2], t);
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[3], &aobjc, &aobjv)
      || aobjc != 4) return ERROR(ERR_EXPECT);
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.k_theta = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)) {
    return ERROR(ERR_EXPECT);
  }
  a.theta0 = d * RADIANS;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[2], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.k_ub = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[3], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.r_ub = d;
  if ((id=ForcePrm_add_angleprm(fprm, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((n=Topology_setprm_angle_array(&(p->topo))) < FAIL) return ERROR(n);
  return OK;
}


static int dpclean(DihedPrm *p, int retval) {
  DihedPrm_done(p);
  return retval;
}

int NLEnergy_add_dihedprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  DihedPrm dp;
  const char *t = NULL;
  int n;
  Tcl_Obj **aobjv;
  int aobjc;
  int s;
  int32 id;

  TEXT("dihedprm");
  if (objc != 5) return ERROR(ERR_EXPECT);
  if ((s=DihedPrm_init(&dp)) != OK) return ERROR(s);
  t = Tcl_GetStringFromObj(objv[0], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) {
    return dpclean(&dp,ERROR(ERR_EXPECT));
  }
  strcpy(dp.atomType[0], t);
  t = Tcl_GetStringFromObj(objv[1], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) {
    return dpclean(&dp,ERROR(ERR_EXPECT));
  }
  strcpy(dp.atomType[1], t);
  t = Tcl_GetStringFromObj(objv[2], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) {
    return dpclean(&dp,ERROR(ERR_EXPECT));
  }
  strcpy(dp.atomType[2], t);
  t = Tcl_GetStringFromObj(objv[3], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) {
    return dpclean(&dp,ERROR(ERR_EXPECT));
  }
  strcpy(dp.atomType[3], t);
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[4], &aobjc, &aobjv)
      || aobjc < 2) return dpclean(&dp,ERROR(ERR_EXPECT));
  if (TCL_ERROR==Tcl_GetIntFromObj(interp, aobjv[0], &n)
      || n != aobjc-1) return dpclean(&dp,ERROR(ERR_EXPECT));
  if ((s=DihedPrm_setmaxnum_term(&dp, n)) != OK) {
    return dpclean(&dp,ERROR(s));
  }
  aobjv++, aobjc--;
  for (n = 0;  n < aobjc;  n++) {
    Tcl_Obj **tobjv;
    int tobjc;
    DihedTerm dterm;
    double d;
    int m;
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, aobjv[n],
	  &tobjc, &tobjv) || tobjc != 3) {
      return dpclean(&dp,ERROR(ERR_EXPECT));
    }
    if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, tobjv[0], &d) || d < 0) {
      return dpclean(&dp,ERROR(ERR_EXPECT));
    }
    dterm.k_dihed = d * ENERGY_INTERNAL;
    if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, tobjv[1], &d)) {
      return dpclean(&dp,ERROR(ERR_EXPECT));
    }
    dterm.phi0 = d * RADIANS;
    if (TCL_ERROR==Tcl_GetIntFromObj(interp, tobjv[2], &m) || m <= 0) {
      return dpclean(&dp,ERROR(ERR_EXPECT));
    }
    dterm.n = m;
    if ((s=DihedPrm_add_term(&dp, &dterm)) != OK) {
      return dpclean(&dp,ERROR(s));
    }
  }
  if ((id=ForcePrm_add_dihedprm(fprm, &dp)) < OK) {
    return dpclean(&dp,(id < FAIL ? ERROR(id) : FAIL));
  }
  if ((n=Topology_setprm_dihed_array(&(p->topo))) < FAIL) {
    return dpclean(&dp,ERROR(n));
  }
  return dpclean(&dp,OK);
}


int NLEnergy_add_imprprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  ImprPrm a;
  const char *t = NULL;
  int n;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int32 id;

  TEXT("imprprm");
  if (objc != 5) return ERROR(ERR_EXPECT);
  t = Tcl_GetStringFromObj(objv[0], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[0], t);
  t = Tcl_GetStringFromObj(objv[1], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[1], t);
  t = Tcl_GetStringFromObj(objv[2], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[2], t);
  t = Tcl_GetStringFromObj(objv[3], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[3], t);
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[4], &aobjc, &aobjv)
      || aobjc != 2) return ERROR(ERR_EXPECT);
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.k_impr = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)) {
    return ERROR(ERR_EXPECT);
  }
  a.psi0 = d * RADIANS;
  if ((id=ForcePrm_add_imprprm(fprm, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  if ((n=Topology_setprm_impr_array(&(p->topo))) < FAIL) return ERROR(n);
  return OK;
}


int NLEnergy_add_vdwpairprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  VdwpairPrm a;
  const char *t = NULL;
  int n;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int32 id;

  TEXT("vdwpairprm");
  if (objc != 3) return ERROR(ERR_EXPECT);
  t = Tcl_GetStringFromObj(objv[0], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[0], t);
  t = Tcl_GetStringFromObj(objv[1], &n);
  if (n >= sizeof(AtomType) || 0==t[0]) return ERROR(ERR_EXPECT);
  strcpy(a.atomType[1], t);
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[2], &aobjc, &aobjv)
      || aobjc != 4) return ERROR(ERR_EXPECT);
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d)
      || d > 0) return ERROR(ERR_EXPECT);
  a.emin = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.rmin = d;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[2], &d)
      || d > 0) return ERROR(ERR_EXPECT);
  a.emin14 = d * ENERGY_INTERNAL;
  if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[3], &d)
      || d < 0) return ERROR(ERR_EXPECT);
  a.rmin14 = d;
  if ((id=ForcePrm_add_vdwpairprm(fprm, &a)) < OK) {
    return (id < FAIL ? ERROR(id) : FAIL);
  }
  return OK;
}
