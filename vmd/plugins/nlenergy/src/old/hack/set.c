#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/tclwrap.h"


int NLEnergy_set_coord(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const int32 natoms = Coord_numatoms(&(p->coord));
  dvec *pos = Coord_pos(&(p->coord));
  dvec v;
  Tcl_Obj **cobjv = NULL, **vobjv = NULL;
  int cobjc = 0, vobjc = 0;
  int i = -1;  /* uninitialized */

  TEXT("coord");
  if (1==objc) {
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[0],
          &cobjc, &cobjv)) return ERROR(ERR_EXPECT);
    else if (cobjc != natoms) return ERROR(ERR_EXPECT);
    for (i = 0;  i < natoms;  i++) {
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, cobjv[i],
            &vobjc, &vobjv)) return ERROR(ERR_EXPECT);
      else if (vobjc != 3) return ERROR(ERR_EXPECT);
      else if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, vobjv[0],
            &v.x)) return ERROR(ERR_EXPECT);
      else if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, vobjv[1],
            &v.y)) return ERROR(ERR_EXPECT);
      else if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, vobjv[2],
            &v.z)) return ERROR(ERR_EXPECT);
      pos[i] = v;
    }
  }
  else if (2==objc) {
    if ((i=atomid_from_obj(p,interp,objv[0])) < 0) return ERROR(ERR_EXPECT);
    else if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[1],
          &vobjc, &vobjv)) return ERROR(ERR_EXPECT);
    else if (vobjc != 3) return ERROR(ERR_EXPECT);
    else if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, vobjv[0],
          &v.x)) return ERROR(ERR_EXPECT);
    else if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, vobjv[1],
          &v.y)) return ERROR(ERR_EXPECT);
    else if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, vobjv[2],
          &v.z)) return ERROR(ERR_EXPECT);
    pos[i] = v;
  }
  else return ERROR(ERR_EXPECT);
  return OK;
}


int NLEnergy_set_atom(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    MASS,
    CHARGE,
    NAME,
    TYPE,
    RESIDUE,
    RESNAME
  };
  Topology *topo = &(p->topo);
  const Atom *atom = Topology_atom_array(topo);
  Atom a;
  double d;
  const char *str;
  int n;
  int i = -1;
  Tcl_Obj **aobjv;
  int aobjc;
  int action = ALL;
  int s;

  TEXT("atom");
  if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
  else if ((i=atomid_from_obj(p,interp,objv[0])) < 0) return ERROR(ERR_EXPECT);
  else if (3==objc) {
    if (strcmp(Tcl_GetString(objv[1]),"mass")==0) action=MASS;
    else if (strcmp(Tcl_GetString(objv[1]),"charge")==0) action=CHARGE;
    else if (strcmp(Tcl_GetString(objv[1]),"name")==0) action=NAME;
    else if (strcmp(Tcl_GetString(objv[1]),"type")==0) action=TYPE;
    else if (strcmp(Tcl_GetString(objv[1]),"residue")==0) action=RESIDUE;
    else if (strcmp(Tcl_GetString(objv[1]),"resname")==0) action=RESNAME;
    else return ERROR(ERR_EXPECT);
  }
  a = atom[i];  /* preserve previous values */
  switch (action) {
    case MASS:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[2], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.m = d;
      break;
    case CHARGE:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[2], &d)) {
        return ERROR(ERR_EXPECT);
      }
      a.q = d;
      break;
    case NAME:
      str = Tcl_GetStringFromObj(objv[2], &n);
      if (n >= sizeof(AtomName)) return ERROR(ERR_EXPECT);
      strcpy(a.atomName, str);
      break;
    case TYPE:
      str = Tcl_GetStringFromObj(objv[2], &n);
      if (n >= sizeof(AtomType)) return ERROR(ERR_EXPECT);
      strcpy(a.atomType, str);
      break;
    case RESIDUE:
      if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[2], &n) || n < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.residue = n;
      break;
    case RESNAME:
      str = Tcl_GetStringFromObj(objv[2], &n);
      if (n >= sizeof(ResName)) return ERROR(ERR_EXPECT);
      strcpy(a.resName, str);
      break;
    default:  /* ALL */
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[1], &aobjc, &aobjv)
          || aobjc != 6) return ERROR(ERR_EXPECT);
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.m = d;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)) {
        return ERROR(ERR_EXPECT);
      }
      a.q = d;
      str = Tcl_GetStringFromObj(aobjv[2], &n);
      if (n >= sizeof(AtomName)) return ERROR(ERR_EXPECT);
      strcpy(a.atomName, str);
      str = Tcl_GetStringFromObj(aobjv[3], &n);
      if (n >= sizeof(AtomType)) return ERROR(ERR_EXPECT);
      strcpy(a.atomType, str);
      if (TCL_ERROR==Tcl_GetIntFromObj(interp, aobjv[4], &n) || n < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.residue = n;
      str = Tcl_GetStringFromObj(aobjv[5], &n);
      if (n >= sizeof(ResName)) return ERROR(ERR_EXPECT);
      strcpy(a.resName, str);
  }
  if ((s=Topology_update_atom(topo, &a, i)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_set_atomprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    EMIN,
    RMIN,
    EMIN14,
    RMIN14
  };
  ForcePrm *fprm = &(p->fprm);
  const AtomPrm *atomprm = ForcePrm_atomprm_array(fprm);
  AtomPrm a;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int action = ALL;
  int32 id;
  int32 s;

  TEXT("atomprm");
  if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
  else {
    const char *typ = Tcl_GetString(objv[0]);
    if (3==objc) {
      if (strcmp(Tcl_GetString(objv[1]),"emin")==0) action=EMIN;
      else if (strcmp(Tcl_GetString(objv[1]),"rmin")==0) action=RMIN;
      else if (strcmp(Tcl_GetString(objv[1]),"emin14")==0) action=EMIN14;
      else if (strcmp(Tcl_GetString(objv[1]),"rmin14")==0) action=RMIN14;
      else return ERROR(ERR_EXPECT);
    }
    if ((id=ForcePrm_getid_atomprm(fprm, typ)) < OK) {
      return (id < FAIL ? ERROR(id) : FAIL);
    }
  }
  a = atomprm[id];  /* preserve previous values */
  switch (action) {
    case EMIN:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[2], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin = d * ENERGY_INTERNAL;
      break;
    case RMIN:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[2], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin = d;
      break;
    case EMIN14:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[2], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin14 = d * ENERGY_INTERNAL;
      break;
    case RMIN14:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[2], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin14 = d;
      break;
    default:  /* ALL */
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[1], &aobjc, &aobjv)
          || aobjc != 4) return ERROR(ERR_EXPECT);
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin = d;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[2], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin14 = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[3], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin14 = d;
  }
  if ((s=ForcePrm_update_atomprm(fprm, &a)) != id) {
    return (s < FAIL ? ERROR(s) : FAIL);
  }
  return OK;
}


int NLEnergy_set_bondprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    K,
    R0
  };
  ForcePrm *fprm = &(p->fprm);
  const BondPrm *bondprm = ForcePrm_bondprm_array(fprm);
  BondPrm a;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int action = ALL;
  int32 id;
  int32 s;

  TEXT("bondprm");
  if (objc < 3 || objc > 4) return ERROR(ERR_EXPECT);
  else {
    const char *typ0 = Tcl_GetString(objv[0]);
    const char *typ1 = Tcl_GetString(objv[1]);
    if (4==objc) {
      if (strcmp(Tcl_GetString(objv[2]),"k")==0) action=K;
      else if (strcmp(Tcl_GetString(objv[2]),"r0")==0) action=R0;
      else return ERROR(ERR_EXPECT);
    }
    if ((id=ForcePrm_getid_bondprm(fprm, typ0, typ1)) < OK) {
      return (id < FAIL ? ERROR(id) : FAIL);
    }
  }
  a = bondprm[id];  /* preserve previous values */
  switch (action) {
    case K:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[3], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k = d * ENERGY_INTERNAL;
      break;
    case R0:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[3], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.r0 = d;
      break;
    default:  /* ALL */
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[2], &aobjc, &aobjv)
          || aobjc != 2) return ERROR(ERR_EXPECT);
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.r0 = d;
  }
  if ((s=ForcePrm_update_bondprm(fprm, &a)) != id) {
    return (s < FAIL ? ERROR(s) : FAIL);
  }
  return OK;
}


int NLEnergy_set_angleprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    KTHETA,
    THETA0,
    KUB,
    RUB
  };
  ForcePrm *fprm = &(p->fprm);
  const AnglePrm *angleprm = ForcePrm_angleprm_array(fprm);
  AnglePrm a;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int action = ALL;
  int32 id;
  int32 s;

  TEXT("angleprm");
  if (objc < 4 || objc > 5) return ERROR(ERR_EXPECT);
  else {
    const char *typ0 = Tcl_GetString(objv[0]);
    const char *typ1 = Tcl_GetString(objv[1]);
    const char *typ2 = Tcl_GetString(objv[2]);
    if (5==objc) {
      if (strcmp(Tcl_GetString(objv[3]),"ktheta")==0) action=KTHETA;
      else if (strcmp(Tcl_GetString(objv[3]),"theta0")==0) action=THETA0;
      else if (strcmp(Tcl_GetString(objv[3]),"kub")==0) action=KUB;
      else if (strcmp(Tcl_GetString(objv[3]),"rub")==0) action=RUB;
      else return ERROR(ERR_EXPECT);
    }
    if ((id=ForcePrm_getid_angleprm(fprm, typ0, typ1, typ2)) < OK) {
      return (id < FAIL ? ERROR(id) : FAIL);
    }
  }
  a = angleprm[id];  /* preserve previous values */
  switch (action) {
    case KTHETA:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[4], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k_theta = d * ENERGY_INTERNAL;
      break;
    case THETA0:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[4], &d)) {
        return ERROR(ERR_EXPECT);
      }
      a.theta0 = d * RADIANS;
      break;
    case KUB:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[4], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k_ub = d * ENERGY_INTERNAL;
      break;
    case RUB:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[4], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.r_ub = d;
      break;
    default:  /* ALL */
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[3], &aobjc, &aobjv)
          || aobjc != 4) return ERROR(ERR_EXPECT);
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k_theta = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)) {
        return ERROR(ERR_EXPECT);
      }
      a.theta0 = d * RADIANS;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[2], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k_ub = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[3], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.r_ub = d;
  }
  if ((s=ForcePrm_update_angleprm(fprm, &a)) != id) {
    return (s < FAIL ? ERROR(s) : FAIL);
  }
  return OK;
}


static int dpclean(DihedPrm *d, int retval) {
  DihedPrm_done(d);
  return retval;
}

int NLEnergy_set_dihedprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    TERM,
    KDIHED,
    PHI0,
    N
  };
  ForcePrm *fprm = &(p->fprm);
  const DihedPrm *dihedprm = ForcePrm_dihedprm_array(fprm);
  DihedPrm dp;
  int action = ALL;
  int termid = -1;
  int ndterms = -1;
  DihedTerm *dterm = NULL;
  int32 id;
  int32 s;
  double d;
  Tcl_Obj **dobjv, **tobjv;
  int dobjc, tobjc;
  int n;

  TEXT("dihedprm");
  if ((s=DihedPrm_init(&dp)) != OK) return ERROR(s);
  if (objc < 5 || objc > 8) return dpclean(&dp,ERROR(ERR_EXPECT));
  else {
    const char *typ0 = Tcl_GetString(objv[0]);
    const char *typ1 = Tcl_GetString(objv[1]);
    const char *typ2 = Tcl_GetString(objv[2]);
    const char *typ3 = Tcl_GetString(objv[3]);
    if (6==objc) return dpclean(&dp,ERROR(ERR_EXPECT));
    else if (objc >= 7) {
      if (strcmp(Tcl_GetString(objv[4]),"term")!=0) {
        return dpclean(&dp,ERROR(ERR_EXPECT));
      }
      if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[5], &termid)
          || termid < 0) return dpclean(&dp,ERROR(ERR_EXPECT));
      if (7==objc) action=TERM;
      else if (strcmp(Tcl_GetString(objv[6]),"kdihed")==0) action=KDIHED;
      else if (strcmp(Tcl_GetString(objv[6]),"phi0")==0) action=PHI0;
      else if (strcmp(Tcl_GetString(objv[6]),"n")==0) action=N;
      else return dpclean(&dp,ERROR(ERR_EXPECT));
    }
    if ((id=ForcePrm_getid_dihedprm(fprm, typ0, typ1, typ2, typ3)) < OK) {
      return dpclean(&dp,(id < FAIL ? ERROR(id) : FAIL));
    }
    if ((s=DihedPrm_copy(&dp, &dihedprm[id])) != OK) {
      return dpclean(&dp,ERROR(s));
    }
    ndterms = DihedPrm_term_array_length(&dp);
    if (termid >= ndterms) return dpclean(&dp,ERROR(ERR_EXPECT));
    dterm = Array_data(&dp.dihedterm);
    if (NULL==dterm) return dpclean(&dp,ERROR(ERR_EXPECT));
  }
  switch (action) {
    case TERM:
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[6], &tobjc, &tobjv)
          || tobjc != 3) return dpclean(&dp,ERROR(ERR_EXPECT));
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, tobjv[0], &d) || d < 0) {
        return dpclean(&dp,ERROR(ERR_EXPECT));
      }
      dterm[termid].k_dihed = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, tobjv[1], &d)) {
        return dpclean(&dp,ERROR(ERR_EXPECT));
      }
      dterm[termid].phi0 = d * RADIANS;
      if (TCL_ERROR==Tcl_GetIntFromObj(interp, tobjv[2], &n) || n < 0) {
        return dpclean(&dp,ERROR(ERR_EXPECT));
      }
      dterm[termid].n = n;
      break;
    case KDIHED:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[7], &d) || d < 0) {
        return dpclean(&dp,ERROR(ERR_EXPECT));
      }
      dterm[termid].k_dihed = d * ENERGY_INTERNAL;
      break;
    case PHI0:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[7], &d)) {
        return dpclean(&dp,ERROR(ERR_EXPECT));
      }
      dterm[termid].phi0 = d * RADIANS;
      break;
    case N:
      if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[7], &n) || n < 0) {
        return dpclean(&dp,ERROR(ERR_EXPECT));
      }
      dterm[termid].n = n;
      break;
    default:
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[4], &dobjc, &dobjv)
          || dobjc <= 0) return dpclean(&dp,ERROR(ERR_EXPECT));
      if (dobjc != ndterms) {
        ndterms = dobjc;
        if ((s=Array_resize(&dp.dihedterm,ndterms)) != OK) {
          return dpclean(&dp,ERROR(s));
        }
        dterm = Array_data(&dp.dihedterm);
      }
      for (termid = 0;  termid < ndterms;  termid++) {
        if (TCL_ERROR==Tcl_ListObjGetElements(interp, dobjv[termid],
              &tobjc, &tobjv) || tobjc != 3) {
          return dpclean(&dp,ERROR(ERR_EXPECT));
        }
        if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, tobjv[0], &d) || d < 0) {
          return dpclean(&dp,ERROR(ERR_EXPECT));
        }
        dterm[termid].k_dihed = d * ENERGY_INTERNAL;
        if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, tobjv[1], &d)) {
          return dpclean(&dp,ERROR(ERR_EXPECT));
        }
        dterm[termid].phi0 = d * RADIANS;
        if (TCL_ERROR==Tcl_GetIntFromObj(interp, tobjv[2], &n) || n < 0) {
          return dpclean(&dp,ERROR(ERR_EXPECT));
        }
        dterm[termid].n = n;
      }
  }
  if ((s=ForcePrm_update_dihedprm(fprm, &dp)) != id) {
    return dpclean(&dp,(s < FAIL ? ERROR(s) : FAIL));
  }
  return dpclean(&dp,OK);
}


int NLEnergy_set_imprprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    KIMPR,
    PSI0
  };
  ForcePrm *fprm = &(p->fprm);
  const ImprPrm *imprprm = ForcePrm_imprprm_array(fprm);
  ImprPrm a;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int action = ALL;
  int32 id;
  int32 s;

  TEXT("imprprm");
  if (objc < 5 || objc > 6) return ERROR(ERR_EXPECT);
  else {
    const char *typ0 = Tcl_GetString(objv[0]);
    const char *typ1 = Tcl_GetString(objv[1]);
    const char *typ2 = Tcl_GetString(objv[2]);
    const char *typ3 = Tcl_GetString(objv[3]);
    if (6==objc) {
      if (strcmp(Tcl_GetString(objv[4]),"kimpr")==0) action=KIMPR;
      else if (strcmp(Tcl_GetString(objv[4]),"psi0")==0) action=PSI0;
      else return ERROR(ERR_EXPECT);
    }
    if ((id=ForcePrm_getid_imprprm(fprm, typ0, typ1, typ2, typ3)) < OK) {
      return (id < FAIL ? ERROR(id) : FAIL);
    }
  }
  a = imprprm[id];  /* preserve previous values */
  switch (action) {
    case KIMPR:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[5], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k_impr = d * ENERGY_INTERNAL;
      break;
    case PSI0:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[5], &d)) {
        return ERROR(ERR_EXPECT);
      }
      a.psi0 = d * RADIANS;
      break;
    default:  /* ALL */
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[4], &aobjc, &aobjv)
          || aobjc != 2) return ERROR(ERR_EXPECT);
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.k_impr = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)) {
        return ERROR(ERR_EXPECT);
      }
      a.psi0 = d * RADIANS;
  }
  if ((s=ForcePrm_update_imprprm(fprm, &a)) != id) {
    return (s < FAIL ? ERROR(s) : FAIL);
  }
  return OK;
}


int NLEnergy_set_vdwpairprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    EMIN,
    RMIN,
    EMIN14,
    RMIN14
  };
  ForcePrm *fprm = &(p->fprm);
  const VdwpairPrm *vdwpairprm = ForcePrm_vdwpairprm_array(fprm);
  VdwpairPrm a;
  double d;
  Tcl_Obj **aobjv;
  int aobjc;
  int action = ALL;
  int32 id;
  int32 s;

  TEXT("vdwpairprm");
  if (objc < 3 || objc > 4) return ERROR(ERR_EXPECT);
  else {
    const char *typ0 = Tcl_GetString(objv[0]);
    const char *typ1 = Tcl_GetString(objv[1]);
    if (4==objc) {
      if (strcmp(Tcl_GetString(objv[2]),"emin")==0) action=EMIN;
      else if (strcmp(Tcl_GetString(objv[2]),"rmin")==0) action=RMIN;
      else if (strcmp(Tcl_GetString(objv[2]),"emin14")==0) action=EMIN14;
      else if (strcmp(Tcl_GetString(objv[2]),"rmin14")==0) action=RMIN14;
      else return ERROR(ERR_EXPECT);
    }
    if ((id=ForcePrm_getid_vdwpairprm(fprm, typ0, typ1)) < OK) {
      return (id < FAIL ? ERROR(id) : FAIL);
    }
  }
  a = vdwpairprm[id];  /* preserve previous values */
  switch (action) {
    case EMIN:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[3], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin = d * ENERGY_INTERNAL;
      break;
    case RMIN:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[3], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin = d;
      break;
    case EMIN14:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[3], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin14 = d * ENERGY_INTERNAL;
      break;
    case RMIN14:
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[3], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin14 = d;
      break;
    default:  /* ALL */
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[2], &aobjc, &aobjv)
          || aobjc != 4) return ERROR(ERR_EXPECT);
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin = d;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[2], &d) || d > 0) {
        return ERROR(ERR_EXPECT);
      }
      a.emin14 = d * ENERGY_INTERNAL;
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[3], &d) || d < 0) {
        return ERROR(ERR_EXPECT);
      }
      a.rmin14 = d;
  }
  if ((s=ForcePrm_update_vdwpairprm(fprm, &a)) != id) {
    return (s < FAIL ? ERROR(s) : FAIL);
  }
  return OK;
}


int NLEnergy_set_nonbprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const NonbPrm *p_nonbprm = ForcePrm_nonbprm(fprm);
  const char *t;

  TEXT("nonbprm");
  if (objc != 2) return ERROR(ERR_EXPECT);
  t = Tcl_GetString(objv[0]);
  if (strcmp(t,"fulldirect")==0) {
    const char *str = Tcl_GetString(objv[1]);
    if (strcmp(str,"on")==0)       p->fulldirect = TRUE;
    else if (strcmp(str,"off")==0) p->fulldirect = FALSE;
    else return ERROR(ERR_EXPECT);
  }
  else if (strcmp(t,"fulldirectvdw")==0) {
    const char *str = Tcl_GetString(objv[1]);
    if (strcmp(str,"on")==0)       p->fulldirectvdw = TRUE;
    else if (strcmp(str,"off")==0) p->fulldirectvdw = FALSE;
    else return ERROR(ERR_EXPECT);
  }
  else {
    NonbPrm nonbprm = *p_nonbprm;
    double r;
    int s;
    if (strcmp(t,"cutoff")==0) {
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[1], &r)
  	|| r < nonbprm.switchdist) {
        return ERROR(ERR_EXPECT);
      }
      nonbprm.cutoff = r;
    }
    else if (strcmp(t,"switching")==0) {
      const char *str = Tcl_GetString(objv[1]);
      if (strcmp(str,"on")==0)       nonbprm.switching = TRUE;
      else if (strcmp(str,"off")==0) nonbprm.switching = FALSE;
      else return ERROR(ERR_EXPECT);
    }
    else if (strcmp(t,"switchdist")==0) {
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[1], &r)
  	|| r < 0 || r > nonbprm.cutoff) {
        return ERROR(ERR_EXPECT);
      }
      nonbprm.switchdist = r;
    }
    else if (strcmp(t,"exclude")==0) {
      const char *str = Tcl_GetString(objv[1]);
      if (strcmp(str,"none")==0)           nonbprm.exclude = EXCL_NONE;
      else if (strcmp(str,"1-2")==0)       nonbprm.exclude = EXCL_12;
      else if (strcmp(str,"1-3")==0)       nonbprm.exclude = EXCL_13;
      else if (strcmp(str,"1-4")==0)       nonbprm.exclude = EXCL_14;
      else if (strcmp(str,"scaled1-4")==0) nonbprm.exclude = EXCL_SCALED14;
      else return ERROR(ERR_EXPECT);
    }
    else if (strcmp(t,"dielectric")==0) {
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[1], &r) || r <= 0) {
        return ERROR(ERR_EXPECT);
      }
      nonbprm.dielectric = r;
    }
    else if (strcmp(t,"1-4scaling")==0) {
      if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, objv[1], &r) || r <= 0) {
        return ERROR(ERR_EXPECT);
      }
      nonbprm.scaling14 = r;
    }
    else return ERROR(ERR_EXPECT);
    if ((s=ForcePrm_set_nonbprm(fprm, &nonbprm)) != OK) return ERROR(s);
  }
  return OK;
}


int NLEnergy_set_cellbasis(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  Coord *coord = &(p->coord);
  const Domain *domain = Coord_domain(coord);
  const dvec *pcen = &(domain->center);
  dvec a[3];
  int s, i;

  TEXT("cellbasis");
  if (objc != 3) return ERROR(ERR_EXPECT);
  for (i = 0;  i < 3;  i++) {
    Tcl_Obj **aobjv;
    int aobjc;
    double d;
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, objv[i], &aobjc, &aobjv)
        || aobjc != 3) return ERROR(ERR_EXPECT);
    if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[0], &d)) {
      return ERROR(ERR_EXPECT);
    }
    a[i].x = d;
    if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[1], &d)) {
      return ERROR(ERR_EXPECT);
    }
    a[i].y = d;
    if (TCL_ERROR==Tcl_GetDoubleFromObj(interp, aobjv[2], &d)) {
      return ERROR(ERR_EXPECT);
    }
    a[i].z = d;
  }
  if ((s=Coord_setup_basis(coord, pcen, &a[0], &a[1], &a[2])) != OK) {
    return ERROR(s);
  }
  if ((s=Coord_update_pos(coord, UPDATE_ALL)) != OK) return ERROR(s);
  return OK;
}
