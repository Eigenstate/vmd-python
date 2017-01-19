#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/tclwrap.h"


int NLEnergy_parse_get(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("get");
  if (objc >= 1) {
    const char *t = Tcl_GetString(objv[0]);
    if (strcmp(t,"coord")==0) {
      return NLEnergy_get_coord(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"atom")==0) {
      return NLEnergy_get_atom(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"bond")==0) {
      return NLEnergy_get_bond(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angle")==0) {
      return NLEnergy_get_angle(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihed")==0) {
      return NLEnergy_get_dihed(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"impr")==0) {
      return NLEnergy_get_impr(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"atomprm")==0) {
      return NLEnergy_get_atomprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"bondprm")==0) {
      return NLEnergy_get_bondprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angleprm")==0) {
      return NLEnergy_get_angleprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihedprm")==0) {
      return NLEnergy_get_dihedprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"imprprm")==0) {
      return NLEnergy_get_imprprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"vdwpairprm")==0) {
      return NLEnergy_get_vdwpairprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"nonbprm")==0) {
      return NLEnergy_get_nonbprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"cellbasis")==0) {
      return NLEnergy_get_cellbasis(p, interp, objc-1, objv+1);
    }
  }
  return ERROR(ERR_EXPECT);
}


int NLEnergy_parse_set(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("set");
  if (objc >= 1) {
    const char *t = Tcl_GetString(objv[0]);
    if (strcmp(t,"coord")==0) {
      return NLEnergy_set_coord(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"atom")==0) {
      return NLEnergy_set_atom(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"atomprm")==0) {
      return NLEnergy_set_atomprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"bondprm")==0) {
      return NLEnergy_set_bondprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angleprm")==0) {
      return NLEnergy_set_angleprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihedprm")==0) {
      return NLEnergy_set_dihedprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"imprprm")==0) {
      return NLEnergy_set_imprprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"vdwpairprm")==0) {
      return NLEnergy_set_vdwpairprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"nonbprm")==0) {
      return NLEnergy_set_nonbprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"cellbasis")==0) {
      return NLEnergy_set_cellbasis(p, interp, objc-1, objv+1);
    }
  }
  return ERROR(ERR_EXPECT);
}


int NLEnergy_parse_add(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("add");
  if (objc >= 1) {
    const char *t = Tcl_GetString(objv[0]);
    if (strcmp(t,"bond")==0) {
      return NLEnergy_add_bond(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angle")==0) {
      return NLEnergy_add_angle(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihed")==0) {
      return NLEnergy_add_dihed(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"impr")==0) {
      return NLEnergy_add_impr(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"atomprm")==0) {
      return NLEnergy_add_atomprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"bondprm")==0) {
      return NLEnergy_add_bondprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angleprm")==0) {
      return NLEnergy_add_angleprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihedprm")==0) {
      return NLEnergy_add_dihedprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"imprprm")==0) {
      return NLEnergy_add_imprprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"vdwpairprm")==0) {
      return NLEnergy_add_vdwpairprm(p, interp, objc-1, objv+1);
    }
  }
  return ERROR(ERR_EXPECT);
}


int NLEnergy_parse_remove(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("remove");
  if (objc >= 1) {
    const char *t = Tcl_GetString(objv[0]);
    if (strcmp(t,"bond")==0) {
      return NLEnergy_remove_bond(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angle")==0) {
      return NLEnergy_remove_angle(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihed")==0) {
      return NLEnergy_remove_dihed(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"impr")==0) {
      return NLEnergy_remove_impr(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"atomprm")==0) {
      return NLEnergy_remove_atomprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"bondprm")==0) {
      return NLEnergy_remove_bondprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angleprm")==0) {
      return NLEnergy_remove_angleprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihedprm")==0) {
      return NLEnergy_remove_dihedprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"imprprm")==0) {
      return NLEnergy_remove_imprprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"vdwpairprm")==0) {
      return NLEnergy_remove_vdwpairprm(p, interp, objc-1, objv+1);
    }
  }
  return ERROR(ERR_EXPECT);
}


int NLEnergy_parse_missing(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("missing");
  if (objc >= 1) {
    const char *t = Tcl_GetString(objv[0]);
    if (strcmp(t,"atomprm")==0) {
      return NLEnergy_missing_atomprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"bondprm")==0) {
      return NLEnergy_missing_bondprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angleprm")==0) {
      return NLEnergy_missing_angleprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihedprm")==0) {
      return NLEnergy_missing_dihedprm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"imprprm")==0) {
      return NLEnergy_missing_imprprm(p, interp, objc-1, objv+1);
    }
  }
  return ERROR(ERR_EXPECT);
}


int NLEnergy_parse_read(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("read");
  if (objc >= 1) {
    const char *t = Tcl_GetString(objv[0]);
    if (strcmp(t,"xplor")==0) {
      return NLEnergy_read_xplor(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"charmm")==0) {
      return NLEnergy_read_charmm(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"psf")==0) {
      return NLEnergy_read_psf(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"pdb")==0) {
      return NLEnergy_read_pdb(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"namdbin")==0) {
      return NLEnergy_read_namdbin(p, interp, objc-1, objv+1);
    }
  }
  return ERROR(ERR_EXPECT);
}


int NLEnergy_parse_write(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  TEXT("write");
  return OK;
}


static int32 parse_bond(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj);
static int parse_bondlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert);
static int32 parse_angle(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj);
static int parse_anglelist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert);
static int32 parse_dihed(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj);
static int parse_dihedlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert);
static int32 parse_impr(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj);
static int parse_imprlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert);
static int32 parse_atom(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj);
static int parse_atomlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert, int mark);
static int select_from_atomlist(NLEnergy *p);
static int atomlist_contrib(NLEnergy *p);

int NLEnergy_parse_eval(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[], int evalType) {
  enum { KEYWD, ATOM, BOND, ANGLE, DIHED, IMPR, ELEC, VDW, NONB };
  const Topology *topo = &(p->topo);
  const int32 natoms = Topology_atom_array_length(topo);
  const int32 nbonds = Topology_bond_array_length(topo);
  const int32 nangles = Topology_angle_array_length(topo);
  const int32 ndiheds = Topology_dihed_array_length(topo);
  const int32 nimprs = Topology_impr_array_length(topo);
  int32 ninvs;
  char *atomsel, *nonbsel, *bondsel, *anglesel, *dihedsel, *imprsel, *invsel;
  int32 i;
  int state = KEYWD, s;
  int setnum = 0, mark = FALSE;
  boolean invert = FALSE;

  TEXT("eval");
  if (Array_length(&(p->atomsel)) != natoms
      && (s=Array_resize(&(p->atomsel), natoms)) != OK) return ERROR(s);
  atomsel = Array_data(&(p->atomsel));
  if (Array_length(&(p->nonbsel)) != natoms
      && (s=Array_resize(&(p->nonbsel), natoms)) != OK) return ERROR(s);
  nonbsel = Array_data(&(p->nonbsel));
  if (Array_length(&(p->bondsel)) != nbonds
      && (s=Array_resize(&(p->bondsel), nbonds)) != OK) return ERROR(s);
  bondsel = Array_data(&(p->bondsel));
  if (Array_length(&(p->anglesel)) != nangles
      && (s=Array_resize(&(p->anglesel), nangles)) != OK) return ERROR(s);
  anglesel = Array_data(&(p->anglesel));
  if (Array_length(&(p->dihedsel)) != ndiheds
      && (s=Array_resize(&(p->dihedsel), ndiheds)) != OK) return ERROR(s);
  dihedsel = Array_data(&(p->dihedsel));
  if (Array_length(&(p->imprsel)) != nimprs
      && (s=Array_resize(&(p->imprsel), nimprs)) != OK) return ERROR(s);
  imprsel = Array_data(&(p->imprsel));

  /* find max length for inverse selection array */
  ninvs = natoms;
  if (ninvs < nbonds)  ninvs = nbonds;
  if (ninvs < nangles) ninvs = nangles;
  if (ninvs < ndiheds) ninvs = ndiheds;
  if (ninvs < nimprs)  ninvs = nimprs;
  if (Array_length(&(p->invsel)) != ninvs
      && (s=Array_resize(&(p->invsel), ninvs)) != OK) return ERROR(s);
  invsel = Array_data(&(p->invsel));

  if (0 == objc) {
    memset(atomsel, 0, natoms);
    memset(nonbsel, ASEL_NONB, natoms);
    memset(bondsel, TRUE, nbonds);
    memset(anglesel, TRUE, nangles);
    memset(dihedsel, TRUE, ndiheds);
    memset(imprsel, TRUE, nimprs);
  }
  else {
    const char *t = NULL;
    state = KEYWD;
    memset(atomsel, 0, natoms);
    memset(nonbsel, 0, natoms);
    memset(bondsel, 0, nbonds);
    memset(anglesel, 0, nangles);
    memset(dihedsel, 0, ndiheds);
    memset(imprsel, 0, nimprs);
    i = 0;
    INT(objc);
    while (i <= objc) {
      INT(i);
      switch (state) {
        case KEYWD:
          if (i == objc) { i++; break; }
          t = Tcl_GetString(objv[i]);
          setnum = 0;
          invert = FALSE;
          if ('-'==t[0]) { invert = TRUE;  t++; }
          else if ('+'==t[0])  { t++; }
          if (strcmp(t,"atom")==0)       state = ATOM;
          else if (strcmp(t,"bond")==0)  state = BOND;
          else if (strcmp(t,"angle")==0) state = ANGLE;
          else if (strcmp(t,"dihed")==0) state = DIHED;
          else if (strcmp(t,"impr")==0)  state = IMPR;
          else if (strcmp(t,"elec")==0)  state = ELEC;
          else if (strcmp(t,"vdw")==0)   state = VDW;
          else if (strcmp(t,"nonb")==0)  state = NONB;
          else return ERROR(ERR_EXPECT);
          i++;
          break;
        case BOND:
          s = FAIL;
          if (i<objc && (s=parse_bondlist(p,interp,objv[i],invert))==OK) i++;
          //else if (s < FAIL) return ERROR(s);
          else if ( ! invert ) memset(bondsel, TRUE, nbonds);
          state = KEYWD;
          break;
        case ANGLE:
          s = FAIL;
          if (i<objc && (s=parse_anglelist(p,interp,objv[i],invert))==OK) i++;
          //else if (s < FAIL) return ERROR(s);
          else if ( ! invert ) memset(anglesel, TRUE, nangles);
          state = KEYWD;
          break;
        case DIHED:
          s = FAIL;
          if (i<objc && (s=parse_dihedlist(p,interp,objv[i],invert))==OK) i++;
          //else if (s < FAIL) return ERROR(s);
          else if ( ! invert ) memset(dihedsel, TRUE, ndiheds);
          state = KEYWD;
          break;
        case IMPR:
          s = FAIL;
          if (i<objc && (s=parse_imprlist(p,interp,objv[i],invert))==OK) i++;
          //else if (s < FAIL) return ERROR(s);
          else if ( ! invert ) memset(imprsel, TRUE, nimprs);
          state = KEYWD;
          break;
        default:  /* ATOM, ELEC, VDW, or NONB */
          if (i==objc && setnum > 0) { state = KEYWD; continue; }
          if (ATOM==state) mark = (0==setnum ? -ASEL_NONB : -ASEL_NONB_B);
          else if (ELEC==state) mark = (0==setnum ? ASEL_ELEC : ASEL_ELEC_B);
          else if (VDW==state)  mark = (0==setnum ? ASEL_VDW  : ASEL_VDW_B);
          else                  mark = (0==setnum ? ASEL_NONB : ASEL_NONB_B);
          INT(ASEL_NONB==mark);
          INT(ASEL_NONB_B==mark);
          s = FAIL;
          if (i<objc && (s=parse_atomlist(p,interp,objv[i],invert,mark))==OK) {
            i++;
            setnum++;
            if (invert || 2==setnum) state = KEYWD;
          }
#if 0
          else if (s < FAIL) {
            if (setnum > 0) continue;
            else return ERROR(s);
          }
#endif
          else if (0==setnum &&  !invert) {
            if (mark > 0) {
              memset(nonbsel, mark, natoms);
            }
            else {
              memset(atomsel, -mark, natoms);
            }
            state = KEYWD;
          }
          else state = KEYWD;
      } /* switch */
    } /* while */
  } /* else */

  if ((s=select_from_atomlist(p)) != OK) return ERROR(s);

  /* evaluation */
  if (EVAL_ENERGY==evalType || EVAL_FORCE==evalType) {
    if ((s=NLEnergy_eval_force(p)) != OK) return ERROR(s);
  }
  else {
    /* minimize not yet supported */
    return ERROR(ERR_EXPECT);
  }

  /* output */
  if (EVAL_ENERGY==evalType) {
    Tcl_Obj *a = NULL;
    if ((s=NLEnergy_new_obj_dreal(interp, &a,
            p->ener.pe * ENERGY_EXTERNAL)) != OK) {
      return ERROR(s);
    }
    if ((s=NLEnergy_set_obj_result(interp, a)) != OK) return ERROR(s);
  }
  else if (EVAL_FORCE==evalType || EVAL_MINIMIZE==evalType) {
    const dvec *f = Coord_force_const(&(p->coord));
    Tcl_Obj *r = NULL;  /* return list of lists */
    Tcl_Obj *a = NULL;  /* list of atom index */
    Tcl_Obj *b = NULL;  /* list of force or potentials (MINIMIZE) */
    if ((s=atomlist_contrib(p)) != OK) return ERROR(s);
    if ((s=new_list(interp, &r)) != OK) return ERROR(s);
    if ((s=new_list(interp, &a)) != OK) return ERROR(s);
    if ((s=new_list(interp, &b)) != OK) return ERROR(s);
    for (i = 0;  i < natoms;  i++) {
      if (atomsel[i]) {
        if ((s=list_append_atomid(p,interp,a,i)) != OK) return ERROR(s);
        if (EVAL_FORCE==evalType) {
          dvec fs;
          VECMUL(fs, ENERGY_EXTERNAL, f[i]);
          if ((s=list_append_dvec(interp,b,&fs)) != OK) return ERROR(s);
        }
      }
    }
    if (EVAL_MINIMIZE==evalType) {
      return ERROR(ERR_EXPECT);
    }
    if ((s=list_append_obj(interp, r, a)) != OK) return ERROR(s);
    if ((s=list_append_obj(interp, r, b)) != OK) return ERROR(s);
    if ((s=set_obj_result(interp, r)) != OK) return ERROR(s);
  }
  else {
    /* nothing else is supported */
    return ERROR(ERR_EXPECT);
  }

#if 0
  if (objc >= 1) {
    const char *t = Tcl_GetString(objv[0]);
    if (strcmp(t,"bond")==0) {
      return NLEnergy_energy_bond(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"angle")==0) {
      return NLEnergy_energy_angle(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"dihed")==0) {
      return NLEnergy_energy_dihed(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"impr")==0) {
      return NLEnergy_energy_impr(p, interp, objc-1, objv+1);
    }
    if (strcmp(t,"elec")==0) {
      return NLEnergy_energy_nonbonded(p, FNBCUT_ELEC, interp, objc-1, objv+1);
    }
    if (strcmp(t,"vdw")==0) {
      return NLEnergy_energy_nonbonded(p, FNBCUT_VDW, interp, objc-1, objv+1);
    }
    if (strcmp(t,"nonbonded")==0) {
      return NLEnergy_energy_nonbonded(p, FNBCUT_ELEC | FNBCUT_VDW,
	  interp, objc-1, objv+1);
    }
  }
  return ERROR(ERR_EXPECT);
#endif

  return OK;
}


int32 parse_bond(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj) {
  int32 a0, a1;
  Tcl_Obj **objv;
  int objc = 0;
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)
      || objc != 2
      || (a0=atomid_from_obj(p,interp,objv[0])) < 0
      || (a1=atomid_from_obj(p,interp,objv[1])) < 0) {
    return FAIL;
  }
  return Topology_getid_bond(&(p->topo), a0, a1);
}


int parse_bondlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert) {
  char *bondsel = Array_data(&(p->bondsel));
  char *invsel = Array_data(&(p->invsel));
  char *sel = (invert ? invsel : bondsel);
  const int32 nbonds = Topology_bond_array_length(&(p->topo));
  int32 id;
  Tcl_Obj **objv;
  int objc, n;

  if (invert) {
    memset(invsel, 0, nbonds);
  }
  if ((id=parse_bond(p,interp,obj)) >= 0) {  /* could be a singleton */
    sel[id] = TRUE;
  }
  else {  /* could be a list of bonds */
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
      return FAIL;
    }
    for (n = 0;  n < objc;  n++) {
      if ((id=parse_bond(p,interp,objv[n])) < 0) {
        return FAIL;
      }
      sel[id] = TRUE;
    }
  }
  if (invert) {
    for (id = 0;  id < nbonds;  id++) {
      if (FALSE==invsel[id]) bondsel[id] = TRUE;
    }
  }
  return OK;
}


int32 parse_angle(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj) {
  int32 a0, a1, a2;
  Tcl_Obj **objv;
  int objc;
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)
      || objc != 3
      || (a0=atomid_from_obj(p,interp,objv[0])) < 0
      || (a1=atomid_from_obj(p,interp,objv[1])) < 0
      || (a2=atomid_from_obj(p,interp,objv[2])) < 0) {
    return FAIL;
  }
  return Topology_getid_angle(&(p->topo), a0, a1, a2);
}


int parse_anglelist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert) {
  char *anglesel = Array_data(&(p->anglesel));
  char *invsel = Array_data(&(p->invsel));
  char *sel = (invert ? invsel : anglesel);
  const int32 nangles = Topology_angle_array_length(&(p->topo));
  int32 id;
  Tcl_Obj **objv;
  int objc, n;

  if (invert) {
    memset(invsel, 0, nangles);
  }
  if ((id=parse_angle(p,interp,obj)) >= 0) {  /* could be a singleton */
    sel[id] = TRUE;
  }
  else {  /* its a list of angles */
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
      return FAIL;
    }
    for (n = 0;  n < objc;  n++) {
      if ((id=parse_angle(p,interp,objv[n])) < 0) {
        return FAIL;
      }
      sel[id] = TRUE;
    }
  }
  if (invert) {
    for (id = 0;  id < nangles;  id++) {
      if (FALSE==invsel[id]) anglesel[id] = TRUE;
    }
  }
  return OK;
}


int32 parse_dihed(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj) {
  int32 a0, a1, a2, a3;
  Tcl_Obj **objv;
  int objc;
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)
      || objc != 4
      || (a0=atomid_from_obj(p,interp,objv[0])) < 0
      || (a1=atomid_from_obj(p,interp,objv[1])) < 0
      || (a2=atomid_from_obj(p,interp,objv[2])) < 0
      || (a3=atomid_from_obj(p,interp,objv[3])) < 0) {
    return FAIL;
  }
  return Topology_getid_dihed(&(p->topo), a0, a1, a2, a3);
}


int parse_dihedlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert) {
  char *dihedsel = Array_data(&(p->dihedsel));
  char *invsel = Array_data(&(p->invsel));
  char *sel = (invert ? invsel : dihedsel);
  const int32 ndiheds = Topology_dihed_array_length(&(p->topo));
  int32 id;
  Tcl_Obj **objv;
  int objc, n;

  if (invert) {
    memset(invsel, 0, ndiheds);
  }
  if ((id=parse_dihed(p,interp,obj)) >= 0) {  /* could be a singleton */
    sel[id] = TRUE;
  }
  else {  /* its a list of diheds */
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
      return FAIL;
    }
    for (n = 0;  n < objc;  n++) {
      if ((id=parse_dihed(p,interp,objv[n])) < 0) {
        return FAIL;
      }
      sel[id] = TRUE;
    }
  }
  if (invert) {
    for (id = 0;  id < ndiheds;  id++) {
      if (FALSE==invsel[id]) dihedsel[id] = TRUE;
    }
  }
  return OK;
}


int32 parse_impr(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj) {
  int32 a0, a1, a2, a3;
  Tcl_Obj **objv;
  int objc;
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)
      || objc != 4
      || (a0=atomid_from_obj(p,interp,objv[0])) < 0
      || (a1=atomid_from_obj(p,interp,objv[1])) < 0
      || (a2=atomid_from_obj(p,interp,objv[2])) < 0
      || (a3=atomid_from_obj(p,interp,objv[3])) < 0) {
    return FAIL;
  }
  return Topology_getid_impr(&(p->topo), a0, a1, a2, a3);
}


int parse_imprlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert) {
  char *imprsel = Array_data(&(p->imprsel));
  char *invsel = Array_data(&(p->invsel));
  char *sel = (invert ? invsel : imprsel);
  const int32 nimprs = Topology_impr_array_length(&(p->topo));
  int32 id;
  Tcl_Obj **objv;
  int objc, n;

  if (invert) {
    memset(invsel, 0, nimprs);
  }
  if ((id=parse_impr(p,interp,obj)) >= 0) {  /* could be a singleton */
    sel[id] = TRUE;
  }
  else {  /* its a list of imprs */
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
      return FAIL;
    }
    for (n = 0;  n < objc;  n++) {
      if ((id=parse_impr(p,interp,objv[n])) < 0) {
        return FAIL;
      }
      sel[id] = TRUE;
    }
  }
  if (invert) {
    for (id = 0;  id < nimprs;  id++) {
      if (FALSE==invsel[id]) imprsel[id] = TRUE;
    }
  }
  return OK;
}


int32 parse_atom(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *const obj) {
  int32 a0;
  if ((a0=atomid_from_obj(p,interp,obj)) < 0) {
    return FAIL;
  }
  return a0;
}


int parse_atomlist(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *const obj, boolean invert, int mark) {
  char *atomsel = Array_data(&(p->atomsel));
  char *nonbsel = Array_data(&(p->nonbsel));
  char *asel = (mark > 0 ? nonbsel : atomsel);
  char *invsel = Array_data(&(p->invsel));
  char *sel = (invert ? invsel : asel);
  const int32 natoms = Topology_atom_array_length(&(p->topo));
  int32 id;
  Tcl_Obj **objv;
  int objc, n;
  int m = (mark > 0 ? mark : -mark);

  if (invert) {
    memset(invsel, 0, natoms);
  }
  if ((id=parse_atom(p,interp,obj)) >= 0) {  /* could be a singleton */
    sel[id] |= (char)m;
  }
  else {  /* its a list of atoms */
    if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &objc, &objv)) {
      return FAIL;
    }
    for (n = 0;  n < objc;  n++) {
      if ((id=parse_atom(p,interp,objv[n])) < 0) {
        return FAIL;
      }
      sel[id] |= (char)m;
    }
  }
  if (invert) {
    for (id = 0;  id < natoms;  id++) {
      if (FALSE==invsel[id]) asel[id] |= (char)m;
    }
  }
  return OK;
}


int select_from_atomlist(NLEnergy *p) {
  char *atomsel = Array_data(&(p->atomsel));
  char *nonbsel = Array_data(&(p->nonbsel));
  char *bondsel = Array_data(&(p->bondsel));
  char *anglesel = Array_data(&(p->anglesel));
  char *dihedsel = Array_data(&(p->dihedsel));
  char *imprsel = Array_data(&(p->imprsel));
  const Topology *topo = &(p->topo);
  const Bond *bond = Topology_bond_array(topo);
  const Angle *angle = Topology_angle_array(topo);
  const Dihed *dihed = Topology_dihed_array(topo);
  const Impr *impr = Topology_impr_array(topo);
  const int32 natoms = Topology_atom_array_length(topo);
  int32 i, id;
  Idseq seq;
  int32 numelec, numelec_b, numvdw, numvdw_b;
  boolean overlap;
  int s;

  for (i = 0;  i < natoms;  i++) {
    nonbsel[i] |= atomsel[i];
    /* select bonds */
    if ((s=Idseq_init(&seq, Topology_atom_bondlist(topo, i))) != OK) {
      return ERROR(s);
    }
    while ((id=Idseq_getid(&seq)) >= 0) {
      if (atomsel[ bond[id].atomID[0] ]
          && atomsel[ bond[id].atomID[1] ]) {
        bondsel[id] = TRUE;
      }
    }
    Idseq_done(&seq);
    /* select angles */
    if ((s=Idseq_init(&seq, Topology_atom_anglelist(topo, i))) != OK) {
      return ERROR(s);
    }
    while ((id=Idseq_getid(&seq)) >= 0) {
      if (atomsel[ angle[id].atomID[0] ]
          && atomsel[ angle[id].atomID[1] ]
          && atomsel[ angle[id].atomID[2] ]) {
        anglesel[id] = TRUE;
      }
    }
    Idseq_done(&seq);
    /* select dihedrals */
    if ((s=Idseq_init(&seq, Topology_atom_dihedlist(topo, i))) != OK) {
      return ERROR(s);
    }
    while ((id=Idseq_getid(&seq)) >= 0) {
      if (atomsel[ dihed[id].atomID[0] ]
          && atomsel[ dihed[id].atomID[1] ]
          && atomsel[ dihed[id].atomID[2] ]
          && atomsel[ dihed[id].atomID[3] ]) {
        dihedsel[id] = TRUE;
      }
    }
    Idseq_done(&seq);
    /* select impropers */
    if ((s=Idseq_init(&seq, Topology_atom_imprlist(topo, i))) != OK) {
      return ERROR(s);
    }
    while ((id=Idseq_getid(&seq)) >= 0) {
      if (atomsel[ impr[id].atomID[0] ]
          && atomsel[ impr[id].atomID[1] ]
          && atomsel[ impr[id].atomID[2] ]
          && atomsel[ impr[id].atomID[3] ]) {
        imprsel[id] = TRUE;
      }
    }
    Idseq_done(&seq);
  }

  /* check validity of nonbonded selection */
  numelec = 0;
  numelec_b = 0;
  numvdw = 0;
  numvdw_b = 0;
  overlap = TRUE;
  //INT(natoms);
  for (i = 0;  i < natoms;  i++) {
    if ((nonbsel[i] & (ASEL_ELEC | ASEL_ELEC_B))
        == (ASEL_ELEC | ASEL_ELEC_B) ||
        (nonbsel[i] & (ASEL_VDW | ASEL_VDW_B))
        == (ASEL_VDW | ASEL_VDW_B)) {
      return ERROR(ERR_EXPECT);
    }
    if (overlap && nonbsel[i] != 0 &&
        nonbsel[i] != ASEL_NONB && nonbsel[i] != ASEL_NONB_B) {
      overlap = FALSE;
    }
    //INT(i);
    //HEX(nonbsel[i]);
    if (nonbsel[i] & ASEL_ELEC)   numelec++;
    if (nonbsel[i] & ASEL_VDW)    numvdw++;
    if (nonbsel[i] & ASEL_ELEC_B) numelec_b++;
    if (nonbsel[i] & ASEL_VDW_B)  numvdw_b++;
  }
  if ((0==numelec && numelec_b > 0) || (0==numvdw && numvdw_b > 0)) {
    INT(numelec);
    INT(numelec_b);
    INT(numvdw);
    INT(numvdw_b);
    return ERROR(ERR_EXPECT);
  }
  INT(numvdw);

  p->nb_overlap = overlap;
  //INT(p->nb_overlap);
  p->fnbcut_all = (natoms==numelec ? FNBCUT_ELEC : 0)
    | (natoms==numvdw ? FNBCUT_VDW : 0);
  //HEX(p->fnbcut_all);
  p->fnbcut_subset =
    (0 < numelec && numelec < natoms && 0==numelec_b ? FNBCUT_ELEC : 0)
    | (0 < numvdw && numvdw < natoms && 0==numvdw_b ? FNBCUT_VDW : 0);
  //HEX(p->fnbcut_subset);
  p->fnbcut_disjoint = (numelec > 0 && numelec_b > 0 ? FNBCUT_ELEC : 0)
    | (numvdw > 0 && numvdw_b > 0 ? FNBCUT_VDW : 0);
  //HEX(p->fnbcut_disjoint);

  /* start by resetting all index array lengths */
  if ((s=Array_resize(&(p->idnonb), 0)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->idnonb_b), 0)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->idnbvdw), 0)) != OK) return ERROR(s);
  if ((s=Array_resize(&(p->idnbvdw_b), 0)) != OK) return ERROR(s);

  if ((p->fnbcut_all & FNBCUT_ELEC)==0) {
    if ((p->fnbcut_subset & FNBCUT_ELEC) ||
        (p->fnbcut_disjoint & FNBCUT_ELEC)) {
      if ((s=Array_resize(&(p->idnonb), numelec)) != OK) return ERROR(s);
      /* reset uselen for Array_append() */
      if ((s=Array_resize(&(p->idnonb), 0)) != OK) return ERROR(s);
      for (i = 0;  i < natoms;  i++) {
        if ((nonbsel[i] & ASEL_ELEC)
            && (s=Array_append(&(p->idnonb), &i)) != OK) {
          return ERROR(s);
        }
      }
    }
    if ((p->fnbcut_disjoint & FNBCUT_ELEC)) {
      if ((s=Array_resize(&(p->idnonb_b), numelec_b)) != OK) return ERROR(s);
      /* reset uselen for Array_append() */
      if ((s=Array_resize(&(p->idnonb_b), 0)) != OK) return ERROR(s);
      for (i = 0;  i < natoms;  i++) {
        if ((nonbsel[i] & ASEL_ELEC_B)
            && (s=Array_append(&(p->idnonb_b), &i)) != OK) {
          return ERROR(s);
        }
      }
    }
  }

  if (!overlap && (p->fnbcut_all & FNBCUT_VDW)==0) {
    if ((p->fnbcut_subset & FNBCUT_VDW) ||
        (p->fnbcut_disjoint & FNBCUT_VDW)) {
      if ((s=Array_resize(&(p->idnbvdw), numvdw)) != OK) return ERROR(s);
      /* reset uselen for Array_append() */
      if ((s=Array_resize(&(p->idnbvdw), 0)) != OK) return ERROR(s);
      for (i = 0;  i < natoms;  i++) {
        if ((nonbsel[i] & ASEL_VDW)
            && (s=Array_append(&(p->idnbvdw), &i)) != OK) {
          return ERROR(s);
        }
      }
    }
    if ((p->fnbcut_disjoint & FNBCUT_VDW)) {
      if ((s=Array_resize(&(p->idnbvdw_b), numvdw_b)) != OK) return ERROR(s);
      /* reset uselen for Array_append() */
      if ((s=Array_resize(&(p->idnbvdw_b), 0)) != OK) return ERROR(s);
      for (i = 0;  i < natoms;  i++) {
        if ((nonbsel[i] & ASEL_VDW_B)
            && (s=Array_append(&(p->idnbvdw_b), &i)) != OK) {
          return ERROR(s);
        }
      }
    }
  }

  return OK;
}


int atomlist_contrib(NLEnergy *p) {
  char *atomsel = Array_data(&(p->atomsel));
  const char *nonbsel = Array_data_const(&(p->nonbsel));
  const char *bondsel = Array_data_const(&(p->bondsel));
  const char *anglesel = Array_data_const(&(p->anglesel));
  const char *dihedsel = Array_data_const(&(p->dihedsel));
  const char *imprsel = Array_data_const(&(p->imprsel));
  const Topology *topo = &(p->topo);
  const Bond *bond = Topology_bond_array(topo);
  const Angle *angle = Topology_angle_array(topo);
  const Dihed *dihed = Topology_dihed_array(topo);
  const Impr *impr = Topology_impr_array(topo);
  const int32 natoms = Topology_atom_array_length(topo);
  const int32 nbonds = Topology_bond_array_length(topo);
  const int32 nangles = Topology_angle_array_length(topo);
  const int32 ndiheds = Topology_dihed_array_length(topo);
  const int32 nimprs = Topology_impr_array_length(topo);
  int32 i;

  memset(atomsel, 0, natoms);
  for (i = 0;  i < natoms;  i++) {
    if (nonbsel[i]) atomsel[i]=TRUE;
  }
  for (i = 0;  i < nbonds;  i++) {
    if (bondsel[i]) {
      atomsel[ bond[i].atomID[0] ] = TRUE;
      atomsel[ bond[i].atomID[1] ] = TRUE;
    }
  }
  for (i = 0;  i < nangles;  i++) {
    if (anglesel[i]) {
      atomsel[ angle[i].atomID[0] ] = TRUE;
      atomsel[ angle[i].atomID[1] ] = TRUE;
      atomsel[ angle[i].atomID[2] ] = TRUE;
    }
  }
  for (i = 0;  i < ndiheds;  i++) {
    if (dihedsel[i]) {
      atomsel[ dihed[i].atomID[0] ] = TRUE;
      atomsel[ dihed[i].atomID[1] ] = TRUE;
      atomsel[ dihed[i].atomID[2] ] = TRUE;
      atomsel[ dihed[i].atomID[3] ] = TRUE;
    }
  }
  for (i = 0;  i < nimprs;  i++) {
    if (imprsel[i]) {
      atomsel[ impr[i].atomID[0] ] = TRUE;
      atomsel[ impr[i].atomID[1] ] = TRUE;
      atomsel[ impr[i].atomID[2] ] = TRUE;
      atomsel[ impr[i].atomID[3] ] = TRUE;
    }
  }
  return OK;
}
