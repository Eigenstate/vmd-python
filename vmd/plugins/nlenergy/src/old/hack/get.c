#include <string.h>
#include <math.h>
#include "moltypes/const.h"
#include "moltypes/vecops.h"
#include "nlenergy/tclwrap.h"


int NLEnergy_get_coord(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const int32 natoms = Coord_numatoms(&(p->coord));
  const dvec *pos = Coord_pos_const(&(p->coord));
  int n = -1;  /* uninitialized */
  int s;

  TEXT("coord");
  if (objc > 1 || (1==objc && (n=atomid_from_obj(p,interp,objv[0])) < 0)) {
    return ERROR(ERR_EXPECT);
  }
  else if (n == -1) {
    Tcl_Obj *coord = NULL;  /* list of coordinates */
    int32 i;
    if ((s=new_list(interp, &coord)) != OK) return ERROR(s);
    for (i = 0;  i < natoms;  i++) {
      if ((s=list_append_dvec(interp, coord, pos+i)) != OK) return ERROR(s);
    }
    if ((s=set_obj_result(interp, coord)) != OK) return ERROR(s);
  }
  else {
    Tcl_Obj *a = NULL;      /* a coordinate */
    ASSERT(n >= 0 && n < natoms);
    if ((s=new_obj_dvec(interp, &a, pos+n)) != OK) return ERROR(s);
    if ((s=set_obj_result(interp, a)) != OK) return ERROR(s);
  }
  return OK;
}


int NLEnergy_get_atom(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const int32 natoms = Topology_atom_array_length(&(p->topo));
  const Atom *atom = Topology_atom_array(&(p->topo));
  int n = -1;
  int s;

  TEXT("atom");
  if (objc > 2) return ERROR(ERR_EXPECT);
  else if (objc >= 1) {
    if ((n=atomid_from_obj(p,interp,objv[0])) < 0) return ERROR(ERR_EXPECT);
    if (2==objc) {
      const char *field = Tcl_GetString(objv[1]);
      Tcl_Obj *obj = NULL;
      STR(field);
      if (strcmp(field,"index")==0 &&
          (s=new_obj_atomid(p, interp, &obj, n)) != OK) return ERROR(s);
      else if (strcmp(field,"mass")==0 &&
          (s=new_obj_dreal(interp, &obj, atom[n].m)) != OK) return ERROR(s);
      else if (strcmp(field,"charge")==0 &&
          (s=new_obj_dreal(interp, &obj, atom[n].q)) != OK) return ERROR(s);
      else if (strcmp(field,"name")==0 &&
          (s=new_obj_string(interp, &obj, atom[n].atomName)) != OK) {
        return ERROR(s);
      }
      else if (strcmp(field,"type")==0 &&
          (s=new_obj_string(interp, &obj, atom[n].atomType)) != OK) {
        return ERROR(s);
      }
      else if (strcmp(field,"residue")==0 &&
          (s=new_obj_int32(interp, &obj, atom[n].residue)) != OK) {
        return ERROR(s);
      }
      else if (strcmp(field,"resname")==0 &&
          (s=new_obj_string(interp, &obj, atom[n].resName)) != OK) {
        return ERROR(s);
      }
      else if (NULL==obj) return ERROR(ERR_EXPECT);
      else if ((s=set_obj_result(interp, obj)) != OK) return ERROR(s);
      return OK;
    }
  }
  if (-1 == n) {
    Tcl_Obj *alist = NULL;   /* list of atoms */
    Tcl_Obj *a = NULL;       /* one atom */
    int32 i;
    if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
    for (i = 0;  i < natoms;  i++) {
      if ((s=new_list(interp, &a)) != OK) return ERROR(s);
      if ((s=list_append_atomid(p, interp, a, i)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, a, atom[i].m)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, a, atom[i].q)) != OK) return ERROR(s);
      if ((s=list_append_string(interp, a, atom[i].atomName)) != OK) {
        return ERROR(s);
      }
      if ((s=list_append_string(interp, a, atom[i].atomType)) != OK) {
        return ERROR(s);
      }
      if ((s=list_append_int32(interp, a, atom[i].residue)) != OK) {
        return ERROR(s);
      }
      if ((s=list_append_string(interp, a, atom[i].resName)) != OK) {
        return ERROR(s);
      }
      if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
    }
    if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  }
  else {
    Tcl_Obj *a = NULL;  /* one atom */
    ASSERT(n >= 0 && n < natoms);
    if ((s=new_list(interp, &a)) != OK) return ERROR(s);
    if ((s=list_append_atomid(p, interp, a, n)) != OK) return ERROR(s);
    if ((s=list_append_dreal(interp, a, atom[n].m)) != OK) return ERROR(s);
    if ((s=list_append_dreal(interp, a, atom[n].q)) != OK) return ERROR(s);
    if ((s=list_append_string(interp, a, atom[n].atomName)) != OK) {
      return ERROR(s);
    }
    if ((s=list_append_string(interp, a, atom[n].atomType)) != OK) {
      return ERROR(s);
    }
    if ((s=list_append_int32(interp, a, atom[n].residue)) != OK) {
      return ERROR(s);
    }
    if ((s=list_append_string(interp, a, atom[n].resName)) != OK) {
      return ERROR(s);
    }
    if ((s=set_obj_result(interp, a)) != OK) return ERROR(s);
  }
  return OK;
}


int NLEnergy_get_bond(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,      /* return list of all bonds */
    ATOM,     /* return bonds involving specified atom */
    BOND,     /* return specified bond */
    RESIDUE,  /* return list of bonds within residue */
    RESNAME,  /* return list of bonds within some residue name */
    TYPE      /* return list of bonds of specified type */
  };
  Topology *topo = &(p->topo);
  int action = ALL;
  int i = -1, j = -1;
  int s;

  TEXT("bond");
  if (objc > 0) {
    if (strcmp(Tcl_GetString(objv[0]),"residue")==0) action=RESIDUE;
    else if (strcmp(Tcl_GetString(objv[0]),"resname")==0) action=RESNAME;
    else if (strcmp(Tcl_GetString(objv[0]),"type")==0) action=TYPE;
    else if (1==objc) action=ATOM;
    else if (2==objc) action=BOND;
    else return ERROR(ERR_EXPECT);
  }
  switch (action) {
    case ATOM:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0) {
        const Idlist *idlist = Topology_atom_bondlist(topo, i);
        Idseq seq;
        int32 id;
        Tcl_Obj *blist = NULL;  /* list of bonds */
        Tcl_Obj *b = NULL;      /* one bond */
        if (NULL==idlist) return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &blist)) != OK) return ERROR(s);
        if ((s=Idseq_init(&seq, idlist)) != OK) return ERROR(s);
        while (FAIL != (id = Idseq_getid(&seq))) {
          const Bond *tb = Topology_bond(topo, id);
          if (NULL==tb) return ERROR(ERR_EXPECT);
          if ((s=new_list(interp, &b)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, b, tb->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, b, tb->atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, blist, b)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, blist)) != OK) return ERROR(s);
        Idseq_done(&seq);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case BOND:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0 &&
          (j=atomid_from_obj(p,interp,objv[1])) >= 0) {
        int32 id = Topology_getid_bond(topo, i, j);
        Tcl_Obj *b = NULL;  /* one bond */
        if ((s=new_list(interp, &b)) != OK) return ERROR(s);
        if (id != FAIL) {
          const Bond *tb = Topology_bond(topo, id);
          if (NULL==tb) return ERROR(ERR_EXPECT);
          if ((s=list_append_atomid(p, interp, b, tb->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, b, tb->atomID[1])) != OK) {
            return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, b)) != OK) return ERROR(s);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case RESIDUE:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_bond_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Bond *bond = Topology_bond_array(topo);
        const int32 nbonds = Topology_bond_array_length(topo);
        int32 n;
        int resid;
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *blist = NULL;  /* list of bonds */
        Tcl_Obj *b = NULL;      /* one bond */
        if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[1], &resid)
            || resid < 0 /* || resid > atom[natoms-1].residue */) {
          return ERROR(ERR_EXPECT);
        }
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &blist)) != OK) return ERROR(s);
        for (n = 0;  n < nbonds;  n++) {
          const int32 a0 = bond[n].atomID[0];
          const int32 a1 = bond[n].atomID[1];
          int32 r0, r1;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          r0 = atom[a0].residue;
          r1 = atom[a1].residue;
          if ((isself && r0==resid && r1==resid)
              || (isall && (r0==resid || r1==resid))
              || (isjoin && ((r0==resid && r1!=resid)
                  || (r0!=resid && r1==resid)))) {
            if ((s=new_list(interp, &b)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,b,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,b,a1)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, blist, b)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, blist)) != OK) return ERROR(s);
      }
      break;
    case RESNAME:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_bond_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Bond *bond = Topology_bond_array(topo);
        const int32 nbonds = Topology_bond_array_length(topo);
        int32 n;
        const char *resname = Tcl_GetString(objv[1]);
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *blist = NULL;  /* list of bonds */
        Tcl_Obj *b = NULL;      /* one bond */
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &blist)) != OK) return ERROR(s);
        for (n = 0;  n < nbonds;  n++) {
          const int32 a0 = bond[n].atomID[0];
          const int32 a1 = bond[n].atomID[1];
          const char *r0, *r1;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          r0 = atom[a0].resName;
          r1 = atom[a1].resName;
          if ((isself && strcmp(resname,r0)==0 && strcmp(resname,r1)==0)
              || (isall && (strcmp(resname,r0)==0 || strcmp(resname,r1)==0))
              || (isjoin && ((strcmp(resname,r0)==0 && strcmp(resname,r1)!=0)
                  || (strcmp(resname,r0)!=0 && strcmp(resname,r1)==0)))) {
            if ((s=new_list(interp, &b)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,b,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,b,a1)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, blist, b)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, blist)) != OK) return ERROR(s);
      }
      break;
    case TYPE:
      /* O(N) search */
      if (objc != 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_bond_array(topo)) != OK) return ERROR(s);
      else {
        const char *typ0 = Tcl_GetString(objv[1]);
        const char *typ1 = Tcl_GetString(objv[2]);
        const Atom *atom = Topology_atom_array(topo);
        const Bond *bond = Topology_bond_array(topo);
        const int32 nbonds = Topology_bond_array_length(topo);
        int32 n;
        Tcl_Obj *blist = NULL;  /* list of bonds */
        Tcl_Obj *b = NULL;      /* one bond */
        if ((s=new_list(interp, &blist)) != OK) return ERROR(s);
        for (n = 0;  n < nbonds;  n++) {
          const int32 a0 = bond[n].atomID[0];
          const int32 a1 = bond[n].atomID[1];
          const char *at0, *at1;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          at0 = atom[a0].atomType;
          at1 = atom[a1].atomType;
          if ((strcmp(typ0,at0)==0 && strcmp(typ1,at1)==0)
              || (strcmp(typ0,at1)==0 && strcmp(typ1,at0)==0)) {
            if ((s=new_list(interp, &b)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,b,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,b,a1)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, blist, b)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, blist)) != OK) return ERROR(s);
      }
      break;
    default:
      /* return all bonds after compacting array */
      if ((s=Topology_compact_bond_array(topo)) != OK) return ERROR(s);
      else {
        const Bond *bond = Topology_bond_array(topo);
        const int32 nbonds = Topology_bond_array_length(topo);
        int32 n;
        Tcl_Obj *blist = NULL;  /* list of bonds */
        Tcl_Obj *b = NULL;      /* one bond */
        if ((s=new_list(interp, &blist)) != OK) return ERROR(s);
        for (n = 0;  n < nbonds;  n++) {
          if ((s=new_list(interp, &b)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, b, bond[n].atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, b, bond[n].atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, blist, b)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, blist)) != OK) return ERROR(s);
      }
  }
  return OK;
}


int NLEnergy_get_angle(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,      /* return list of all angles */
    ATOM,     /* return angles involving specified atom */
    ANGLE,    /* return specified angle */
    RESIDUE,  /* return list of angles within residue */
    RESNAME,  /* return list of angles within some residue name */
    TYPE      /* return list of angles of specified type */
  };
  Topology *topo = &(p->topo);
  int action = ALL;
  int i = -1, j = -1, k = -1;
  int s;

  TEXT("angle");
  if (objc > 0) {
    if (strcmp(Tcl_GetString(objv[0]),"residue")==0) action=RESIDUE;
    else if (strcmp(Tcl_GetString(objv[0]),"resname")==0) action=RESNAME;
    else if (strcmp(Tcl_GetString(objv[0]),"type")==0) action=TYPE;
    else if (1==objc) action=ATOM;
    else if (3==objc) action=ANGLE;
    else return ERROR(ERR_EXPECT);
  }
  switch (action) {
    case ATOM:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0) {
        const Idlist *idlist = Topology_atom_anglelist(topo, i);
        Idseq seq;
        int32 id;
        Tcl_Obj *alist = NULL;  /* list of angles */
        Tcl_Obj *a = NULL;      /* one angle */
        if (NULL==idlist) return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        if ((s=Idseq_init(&seq, idlist)) != OK) return ERROR(s);
        while (FAIL != (id = Idseq_getid(&seq))) {
          const Angle *ta = Topology_angle(topo, id);
          if (NULL==ta) return ERROR(ERR_EXPECT);
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, a, ta->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
        Idseq_done(&seq);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case ANGLE:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0 &&
          (j=atomid_from_obj(p,interp,objv[1])) >= 0 &&
          (k=atomid_from_obj(p,interp,objv[2])) >= 0) {
        int32 id = Topology_getid_angle(topo, i, j, k);
        Tcl_Obj *a = NULL;  /* one angle */
        if ((s=new_list(interp, &a)) != OK) return ERROR(s);
        if (id != FAIL) {
          const Angle *ta = Topology_angle(topo, id);
          if (NULL==ta) return ERROR(ERR_EXPECT);
          if ((s=list_append_atomid(p, interp, a, ta->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[2])) != OK) {
            return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, a)) != OK) return ERROR(s);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case RESIDUE:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_angle_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Angle *angle = Topology_angle_array(topo);
        const int32 nangles = Topology_angle_array_length(topo);
        int32 n;
        int resid;
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *alist = NULL;  /* list of angles */
        Tcl_Obj *a = NULL;      /* one angle */
        if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[1], &resid)
            || resid < 0 /* || resid > atom[natoms-1].residue */) {
          return ERROR(ERR_EXPECT);
        }
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nangles;  n++) {
          const int32 a0 = angle[n].atomID[0];
          const int32 a1 = angle[n].atomID[1];
          const int32 a2 = angle[n].atomID[2];
          int32 r0, r1, r2;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          r0 = atom[a0].residue;
          r1 = atom[a1].residue;
          r2 = atom[a2].residue;
          if ((isself && r0==resid && r1==resid && r2==resid)
              || (isall && (r0==resid || r1==resid || r2==resid))
              || (isjoin && ((r0==resid && (r1!=resid || r2!=resid))
                  || (r1==resid && (r0!=resid || r2!=resid))
                  || (r2==resid && (r0!=resid || r1!=resid))))) {
            if ((s=new_list(interp, &a)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a2)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
      break;
    case RESNAME:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_angle_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Angle *angle = Topology_angle_array(topo);
        const int32 nangles = Topology_angle_array_length(topo);
        int32 n;
        const char *resname = Tcl_GetString(objv[1]);
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *alist = NULL;  /* list of angles */
        Tcl_Obj *a = NULL;      /* one angle */
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nangles;  n++) {
          const int32 a0 = angle[n].atomID[0];
          const int32 a1 = angle[n].atomID[1];
          const int32 a2 = angle[n].atomID[2];
          const char *r0, *r1, *r2;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          r0 = atom[a0].resName;
          r1 = atom[a1].resName;
          r2 = atom[a2].resName;
          if ((isself && strcmp(resname,r0)==0 && strcmp(resname,r1)==0
                && strcmp(resname,r2)==0)
              || (isall && (strcmp(resname,r0)==0 || strcmp(resname,r1)==0
                  || strcmp(resname,r2)==0))
              || (isjoin && ((strcmp(resname,r0)==0 &&
                    (strcmp(resname,r1)!=0 || strcmp(resname,r2)!=0))
                  || (strcmp(resname,r1)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r2)!=0))
                  || (strcmp(resname,r2)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r1)!=0))))) {
            if ((s=new_list(interp, &a)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a2)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
      break;
    case TYPE:
      /* O(N) search */
      if (objc != 4) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_angle_array(topo)) != OK) return ERROR(s);
      else {
        const char *typ0 = Tcl_GetString(objv[1]);
        const char *typ1 = Tcl_GetString(objv[2]);
        const char *typ2 = Tcl_GetString(objv[3]);
        const Atom *atom = Topology_atom_array(topo);
        const Angle *angle = Topology_angle_array(topo);
        const int32 nangles = Topology_angle_array_length(topo);
        int32 n;
        Tcl_Obj *alist = NULL;  /* list of angles */
        Tcl_Obj *a = NULL;      /* one angle */
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nangles;  n++) {
          const int32 a0 = angle[n].atomID[0];
          const int32 a1 = angle[n].atomID[1];
          const int32 a2 = angle[n].atomID[2];
          const char *at0, *at1, *at2;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          at0 = atom[a0].atomType;
          at1 = atom[a1].atomType;
          at2 = atom[a2].atomType;
          if ((strcmp(typ0,at0)==0 && strcmp(typ1,at1)==0
                && strcmp(typ2,at2)==0)
              || (strcmp(typ0,at2)==0 && strcmp(typ1,at1)==0
                && strcmp(typ2,at0)==0)) {
            if ((s=new_list(interp, &a)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a2)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
      break;
    default:
      /* return all angles after compacting array */
      if ((s=Topology_compact_angle_array(topo)) != OK) return ERROR(s);
      else {
        const Angle *angle = Topology_angle_array(topo);
        const int32 nangles = Topology_angle_array_length(topo);
        int32 n;
        Tcl_Obj *alist = NULL;  /* list of angles */
        Tcl_Obj *a = NULL;      /* one angle */
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nangles;  n++) {
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, a, angle[n].atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, angle[n].atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, angle[n].atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
  }
  return OK;
}


int NLEnergy_get_dihed(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,      /* return list of all dihedrals */
    ATOM,     /* return dihedrals involving specified atom */
    PAIR,     /* return dihedrals centered on bonded pair of atoms */
    DIHED,    /* return specified dihedral */
    RESIDUE,  /* return list of dihedrals within residue */
    RESNAME,  /* return list of dihedrals within some residue name */
    TYPE      /* return list of dihedrals of specified type */
  };
  Topology *topo = &(p->topo);
  int action = ALL;
  int i = -1, j = -1, k = -1, l = -1;
  int s;

  TEXT("dihed");
  if (objc > 0) {
    if (strcmp(Tcl_GetString(objv[0]),"residue")==0) action=RESIDUE;
    else if (strcmp(Tcl_GetString(objv[0]),"resname")==0) action=RESNAME;
    else if (strcmp(Tcl_GetString(objv[0]),"type")==0) action=TYPE;
    else if (1==objc) action=ATOM;
    else if (2==objc) action=PAIR;
    else if (4==objc) action=DIHED;
    else return ERROR(ERR_EXPECT);
  }
  switch (action) {
    case ATOM:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0) {
        const Idlist *idlist = Topology_atom_dihedlist(topo, i);
        Idseq seq;
        int32 id;
        Tcl_Obj *dlist = NULL;  /* list of diheds */
        Tcl_Obj *d = NULL;      /* one dihed */
        if (NULL==idlist) return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &dlist)) != OK) return ERROR(s);
        if ((s=Idseq_init(&seq, idlist)) != OK) return ERROR(s);
        while (FAIL != (id = Idseq_getid(&seq))) {
          const Dihed *td = Topology_dihed(topo, id);
          if (NULL==td) return ERROR(ERR_EXPECT);
          if ((s=new_list(interp, &d)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, d, td->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, td->atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, td->atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, td->atomID[3])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, dlist, d)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, dlist)) != OK) return ERROR(s);
        Idseq_done(&seq);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case PAIR:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0 &&
          (j=atomid_from_obj(p,interp,objv[1])) >= 0) {
        const Idlist *idlist = Topology_atom_dihedlist(topo, i);
        Idseq seq;
        int32 id;
        Tcl_Obj *dlist = NULL;  /* list of diheds */
        Tcl_Obj *d = NULL;      /* one dihed */
        if (NULL==idlist) return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &dlist)) != OK) return ERROR(s);
        if ((s=Idseq_init(&seq, idlist)) != OK) return ERROR(s);
        while (FAIL != (id = Idseq_getid(&seq))) {
          const Dihed *td = Topology_dihed(topo, id);
          if (NULL==td) return ERROR(ERR_EXPECT);
          if ((td->atomID[1]==i && td->atomID[2]==j)
              || (td->atomID[2]==i && td->atomID[1]==j)) {
            if ((s=new_list(interp, &d)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p, interp, d, td->atomID[0])) != OK) {
              return ERROR(s);
            }
            if ((s=list_append_atomid(p, interp, d, td->atomID[1])) != OK) {
              return ERROR(s);
            }
            if ((s=list_append_atomid(p, interp, d, td->atomID[2])) != OK) {
              return ERROR(s);
            }
            if ((s=list_append_atomid(p, interp, d, td->atomID[3])) != OK) {
              return ERROR(s);
            }
            if ((s=list_append_obj(interp, dlist, d)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, dlist)) != OK) return ERROR(s);
        Idseq_done(&seq);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case DIHED:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0 &&
          (j=atomid_from_obj(p,interp,objv[1])) >= 0 &&
          (k=atomid_from_obj(p,interp,objv[2])) >= 0 &&
          (l=atomid_from_obj(p,interp,objv[3])) >= 0) {
        int32 id = Topology_getid_dihed(topo, i, j, k, l);
        Tcl_Obj *d = NULL;  /* one dihed */
        if ((s=new_list(interp, &d)) != OK) return ERROR(s);
        if (id != FAIL) {
          const Dihed *td = Topology_dihed(topo, id);
          if (NULL==td) return ERROR(ERR_EXPECT);
          if ((s=list_append_atomid(p, interp, d, td->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, td->atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, td->atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, td->atomID[3])) != OK) {
            return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, d)) != OK) return ERROR(s);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case RESIDUE:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_dihed_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Dihed *dihed = Topology_dihed_array(topo);
        const int32 ndiheds = Topology_dihed_array_length(topo);
        int32 n;
        int resid;
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *dlist = NULL;  /* list of diheds */
        Tcl_Obj *d = NULL;      /* one dihed */
        if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[1], &resid)
            || resid < 0 /* || resid > atom[natoms-1].residue */) {
          return ERROR(ERR_EXPECT);
        }
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &dlist)) != OK) return ERROR(s);
        for (n = 0;  n < ndiheds;  n++) {
          const int32 a0 = dihed[n].atomID[0];
          const int32 a1 = dihed[n].atomID[1];
          const int32 a2 = dihed[n].atomID[2];
          const int32 a3 = dihed[n].atomID[3];
          int32 r0, r1, r2, r3;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          ASSERT(a3 >= 0 && a3 < Topology_atom_array_length(topo));
          r0 = atom[a0].residue;
          r1 = atom[a1].residue;
          r2 = atom[a2].residue;
          r3 = atom[a3].residue;
          if ((isself && r0==resid && r1==resid && r2==resid && r3==resid)
              || (isall && (r0==resid || r1==resid || r2==resid || r3==resid))
              || (isjoin && ((r0==resid
                    && (r1!=resid || r2!=resid || r3!=resid))
                  || (r1==resid
                    && (r0!=resid || r2!=resid || r3!=resid))
                  || (r2==resid
                    && (r0!=resid || r1!=resid || r3!=resid))
                  || (r3==resid
                    && (r0!=resid || r1!=resid || r2!=resid))))) {
            if ((s=new_list(interp, &d)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a2)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a3)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, dlist, d)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, dlist)) != OK) return ERROR(s);
      }
      break;
    case RESNAME:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_dihed_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Dihed *dihed = Topology_dihed_array(topo);
        const int32 ndiheds = Topology_dihed_array_length(topo);
        int32 n;
        const char *resname = Tcl_GetString(objv[1]);
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *dlist = NULL;  /* list of diheds */
        Tcl_Obj *d = NULL;      /* one dihed */
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &dlist)) != OK) return ERROR(s);
        for (n = 0;  n < ndiheds;  n++) {
          const int32 a0 = dihed[n].atomID[0];
          const int32 a1 = dihed[n].atomID[1];
          const int32 a2 = dihed[n].atomID[2];
          const int32 a3 = dihed[n].atomID[3];
          const char *r0, *r1, *r2, *r3;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          ASSERT(a3 >= 0 && a3 < Topology_atom_array_length(topo));
          r0 = atom[a0].resName;
          r1 = atom[a1].resName;
          r2 = atom[a2].resName;
          r3 = atom[a3].resName;
          if ((isself && strcmp(resname,r0)==0 && strcmp(resname,r1)==0
                && strcmp(resname,r2)==0 && strcmp(resname,r3)==0)
              || (isall && (strcmp(resname,r0)==0 || strcmp(resname,r1)==0
                  || strcmp(resname,r2)==0 || strcmp(resname,r3)==0))
              || (isjoin && ((strcmp(resname,r0)==0 &&
                    (strcmp(resname,r1)!=0 || strcmp(resname,r2)!=0
                     || strcmp(resname,r3)!=0))
                  || (strcmp(resname,r1)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r2)!=0
                     || strcmp(resname,r3)!=0))
                  || (strcmp(resname,r2)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r1)!=0
                     || strcmp(resname,r3)!=0))
                  || (strcmp(resname,r3)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r1)!=0
                     || strcmp(resname,r2)!=0))))) {
            if ((s=new_list(interp, &d)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a2)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a3)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, dlist, d)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, dlist)) != OK) return ERROR(s);
      }
      break;
    case TYPE:
      /* O(N) search */
      if (objc != 5) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_dihed_array(topo)) != OK) return ERROR(s);
      else {
        const char *typ0 = Tcl_GetString(objv[1]);
        const char *typ1 = Tcl_GetString(objv[2]);
        const char *typ2 = Tcl_GetString(objv[3]);
        const char *typ3 = Tcl_GetString(objv[4]);
        const Atom *atom = Topology_atom_array(topo);
        const Dihed *dihed = Topology_dihed_array(topo);
        const int32 ndiheds = Topology_dihed_array_length(topo);
        int32 n;
        Tcl_Obj *dlist = NULL;  /* list of diheds */
        Tcl_Obj *d = NULL;      /* one dihed */
        if ((s=new_list(interp, &dlist)) != OK) return ERROR(s);
        for (n = 0;  n < ndiheds;  n++) {
          const int32 a0 = dihed[n].atomID[0];
          const int32 a1 = dihed[n].atomID[1];
          const int32 a2 = dihed[n].atomID[2];
          const int32 a3 = dihed[n].atomID[3];
          const char *at0, *at1, *at2, *at3;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          ASSERT(a3 >= 0 && a3 < Topology_atom_array_length(topo));
          at0 = atom[a0].atomType;
          at1 = atom[a1].atomType;
          at2 = atom[a2].atomType;
          at3 = atom[a3].atomType;
          if ((strcmp(typ0,at0)==0 && strcmp(typ1,at1)==0
                && strcmp(typ2,at2)==0 && strcmp(typ3,at3)==0)
              || (strcmp(typ0,at3)==0 && strcmp(typ1,at2)==0
                && strcmp(typ2,at1)==0 && strcmp(typ3,at0)==0)) {
            if ((s=new_list(interp, &d)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a2)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,d,a3)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, dlist, d)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, dlist)) != OK) return ERROR(s);
      }
      break;
    default:
      /* return all diheds after compacting array */
      if ((s=Topology_compact_dihed_array(topo)) != OK) return ERROR(s);
      else {
        const Dihed *dihed = Topology_dihed_array(topo);
        const int32 ndiheds = Topology_dihed_array_length(topo);
        int32 n;
        Tcl_Obj *dlist = NULL;  /* list of diheds */
        Tcl_Obj *d = NULL;      /* one dihed */
        if ((s=new_list(interp, &dlist)) != OK) return ERROR(s);
        for (n = 0;  n < ndiheds;  n++) {
          if ((s=new_list(interp, &d)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, d, dihed[n].atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, dihed[n].atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, dihed[n].atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, d, dihed[n].atomID[3])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, dlist, d)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, dlist)) != OK) return ERROR(s);
      }
  }
  return OK;
}


int NLEnergy_get_impr(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,      /* return list of all impropers */
    ATOM,     /* return impropers involving specified atom */
    IMPR,     /* return specified improper */
    RESIDUE,  /* return list of impropers within residue */
    RESNAME,  /* return list of impropers within some residue name */
    TYPE      /* return list of impropers of specified type */
  };
  Topology *topo = &(p->topo);
  int action = ALL;
  int i = -1, j = -1, k = -1, l = -1;
  int s;

  TEXT("impr");
  if (objc > 0) {
    if (strcmp(Tcl_GetString(objv[0]),"residue")==0) action=RESIDUE;
    else if (strcmp(Tcl_GetString(objv[0]),"resname")==0) action=RESNAME;
    else if (strcmp(Tcl_GetString(objv[0]),"type")==0) action=TYPE;
    else if (1==objc) action=ATOM;
    else if (4==objc) action=IMPR;
    else return ERROR(ERR_EXPECT);
  }
  switch (action) {
    case ATOM:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0) {
        const Idlist *idlist = Topology_atom_imprlist(topo, i);
        Idseq seq;
        int32 id;
        Tcl_Obj *alist = NULL;  /* list of imprs */
        Tcl_Obj *a = NULL;      /* one impr */
        if (NULL==idlist) return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        if ((s=Idseq_init(&seq, idlist)) != OK) return ERROR(s);
        while (FAIL != (id = Idseq_getid(&seq))) {
          const Impr *ta = Topology_impr(topo, id);
          if (NULL==ta) return ERROR(ERR_EXPECT);
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, a, ta->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[3])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
        Idseq_done(&seq);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case IMPR:
      if ((i=atomid_from_obj(p,interp,objv[0])) >= 0 &&
          (j=atomid_from_obj(p,interp,objv[1])) >= 0 &&
          (k=atomid_from_obj(p,interp,objv[2])) >= 0 &&
          (l=atomid_from_obj(p,interp,objv[3])) >= 0) {
        int32 id = Topology_getid_impr(topo, i, j, k, l);
        Tcl_Obj *a = NULL;  /* one impr */
        if ((s=new_list(interp, &a)) != OK) return ERROR(s);
        if (id != FAIL) {
          const Impr *ta = Topology_impr(topo, id);
          if (NULL==ta) return ERROR(ERR_EXPECT);
          if ((s=list_append_atomid(p, interp, a, ta->atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, ta->atomID[3])) != OK) {
            return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, a)) != OK) return ERROR(s);
      }
      else return ERROR(ERR_EXPECT);
      break;
    case RESIDUE:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_impr_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Impr *impr = Topology_impr_array(topo);
        const int32 nimprs = Topology_impr_array_length(topo);
        int32 n;
        int resid;
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *alist = NULL;  /* list of imprs */
        Tcl_Obj *a = NULL;      /* one impr */
        if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[1], &resid)
            || resid < 0 /* || resid > atom[natoms-1].residue */) {
          return ERROR(ERR_EXPECT);
        }
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nimprs;  n++) {
          const int32 a0 = impr[n].atomID[0];
          const int32 a1 = impr[n].atomID[1];
          const int32 a2 = impr[n].atomID[2];
          const int32 a3 = impr[n].atomID[3];
          int32 r0, r1, r2, r3;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          ASSERT(a3 >= 0 && a3 < Topology_atom_array_length(topo));
          r0 = atom[a0].residue;
          r1 = atom[a1].residue;
          r2 = atom[a2].residue;
          r3 = atom[a3].residue;
          if ((isself && r0==resid && r1==resid && r2==resid && r3==resid)
              || (isall && (r0==resid || r1==resid || r2==resid || r3==resid))
              || (isjoin && ((r0==resid
                    && (r1!=resid || r2!=resid || r3!=resid))
                  || (r1==resid
                    && (r0!=resid || r2!=resid || r3!=resid))
                  || (r2==resid
                    && (r0!=resid || r1!=resid || r3!=resid))
                  || (r3==resid
                    && (r0!=resid || r1!=resid || r2!=resid))))) {
            if ((s=new_list(interp, &a)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a2)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a3)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
      break;
    case RESNAME:
      /* O(N) search */
      if (objc < 2 || objc > 3) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_impr_array(topo)) != OK) return ERROR(s);
      else {
        /* default option is "-self" */
        const char *opt = (2==objc ? "-self" : Tcl_GetString(objv[2]));
        const Atom *atom = Topology_atom_array(topo);
        const Impr *impr = Topology_impr_array(topo);
        const int32 nimprs = Topology_impr_array_length(topo);
        int32 n;
        const char *resname = Tcl_GetString(objv[1]);
        boolean isself=FALSE, isall=FALSE, isjoin=FALSE;
        Tcl_Obj *alist = NULL;  /* list of imprs */
        Tcl_Obj *a = NULL;      /* one impr */
        if (strcmp(opt,"-self")==0) isself=TRUE;
        else if (strcmp(opt,"-all")==0) isall=TRUE;
        else if (strcmp(opt,"-join")==0) isjoin=TRUE;
        else return ERROR(ERR_EXPECT);
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nimprs;  n++) {
          const int32 a0 = impr[n].atomID[0];
          const int32 a1 = impr[n].atomID[1];
          const int32 a2 = impr[n].atomID[2];
          const int32 a3 = impr[n].atomID[3];
          const char *r0, *r1, *r2, *r3;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          ASSERT(a3 >= 0 && a3 < Topology_atom_array_length(topo));
          r0 = atom[a0].resName;
          r1 = atom[a1].resName;
          r2 = atom[a2].resName;
          r3 = atom[a3].resName;
          if ((isself && strcmp(resname,r0)==0 && strcmp(resname,r1)==0
                && strcmp(resname,r2)==0 && strcmp(resname,r3)==0)
              || (isall && (strcmp(resname,r0)==0 || strcmp(resname,r1)==0
                  || strcmp(resname,r2)==0 || strcmp(resname,r3)==0))
              || (isjoin && ((strcmp(resname,r0)==0 &&
                    (strcmp(resname,r1)!=0 || strcmp(resname,r2)!=0
                     || strcmp(resname,r3)!=0))
                  || (strcmp(resname,r1)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r2)!=0
                     || strcmp(resname,r3)!=0))
                  || (strcmp(resname,r2)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r1)!=0
                     || strcmp(resname,r3)!=0))
                  || (strcmp(resname,r3)==0 &&
                    (strcmp(resname,r0)!=0 || strcmp(resname,r1)!=0
                     || strcmp(resname,r2)!=0))))) {
            if ((s=new_list(interp, &a)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a2)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a3)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
      break;
    case TYPE:
      /* O(N) search */
      if (objc != 5) return ERROR(ERR_EXPECT);
      else if ((s=Topology_compact_impr_array(topo)) != OK) return ERROR(s);
      else {
        const char *typ0 = Tcl_GetString(objv[1]);
        const char *typ1 = Tcl_GetString(objv[2]);
        const char *typ2 = Tcl_GetString(objv[3]);
        const char *typ3 = Tcl_GetString(objv[4]);
        const Atom *atom = Topology_atom_array(topo);
        const Impr *impr = Topology_impr_array(topo);
        const int32 nimprs = Topology_impr_array_length(topo);
        int32 n;
        Tcl_Obj *alist = NULL;  /* list of imprs */
        Tcl_Obj *a = NULL;      /* one impr */
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nimprs;  n++) {
          const int32 a0 = impr[n].atomID[0];
          const int32 a1 = impr[n].atomID[1];
          const int32 a2 = impr[n].atomID[2];
          const int32 a3 = impr[n].atomID[3];
          const char *at0, *at1, *at2, *at3;
          ASSERT(a0 >= 0 && a0 < Topology_atom_array_length(topo));
          ASSERT(a1 >= 0 && a1 < Topology_atom_array_length(topo));
          ASSERT(a2 >= 0 && a2 < Topology_atom_array_length(topo));
          ASSERT(a3 >= 0 && a3 < Topology_atom_array_length(topo));
          at0 = atom[a0].atomType;
          at1 = atom[a1].atomType;
          at2 = atom[a2].atomType;
          at3 = atom[a3].atomType;
          if ((strcmp(typ0,at0)==0 && strcmp(typ1,at1)==0
                && strcmp(typ2,at2)==0 && strcmp(typ3,at3)==0)
              || (strcmp(typ0,at3)==0 && strcmp(typ1,at2)==0
                && strcmp(typ2,at1)==0 && strcmp(typ3,at0)==0)) {
            if ((s=new_list(interp, &a)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a0)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a1)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a2)) != OK) return ERROR(s);
            if ((s=list_append_atomid(p,interp,a,a3)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
          }
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
      break;
    default:
      /* return all imprs after compacting array */
      if ((s=Topology_compact_impr_array(topo)) != OK) return ERROR(s);
      else {
        const Impr *impr = Topology_impr_array(topo);
        const int32 nimprs = Topology_impr_array_length(topo);
        int32 n;
        Tcl_Obj *alist = NULL;  /* list of imprs */
        Tcl_Obj *a = NULL;      /* one impr */
        if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
        for (n = 0;  n < nimprs;  n++) {
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_atomid(p, interp, a, impr[n].atomID[0])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, impr[n].atomID[1])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, impr[n].atomID[2])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_atomid(p, interp, a, impr[n].atomID[3])) != OK) {
            return ERROR(s);
          }
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
        if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
      }
  }
  return OK;
}


int NLEnergy_get_atomprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    TYPE,
    EMIN,
    RMIN,
    EMIN14,
    RMIN14
  };
  ForcePrm *fprm = &(p->fprm);
  const AtomPrm *atomprm = ForcePrm_atomprm_array(fprm);
  int32 natomprms = ForcePrm_atomprm_array_length(fprm);
  Tcl_Obj *alist = NULL;
  int32 i = -1;
  int action = ALL;
  int s;

  TEXT("atomprm");
  if (objc > 0) {
    const char *typ;
    if (1==objc) action=TYPE;
    else if (2==objc) {
      if (strcmp(Tcl_GetString(objv[1]),"emin")==0) action=EMIN;
      else if (strcmp(Tcl_GetString(objv[1]),"rmin")==0) action=RMIN;
      else if (strcmp(Tcl_GetString(objv[1]),"emin14")==0) action=EMIN14;
      else if (strcmp(Tcl_GetString(objv[1]),"rmin14")==0) action=RMIN14;
      else return ERROR(ERR_EXPECT);
    }
    else return ERROR(ERR_EXPECT);
    typ = Tcl_GetString(objv[0]);  /* get atom type */
    if ((i=ForcePrm_getid_atomprm(fprm, typ)) < OK) {
      return (FAIL==i ? OK: ERROR(i));  /* failed match gives empty list */
    }
    /* otherwise return subset of ith atomprm */
  }
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  switch (action) {
    case TYPE:
      if ((s=list_append_string(interp, alist,
              atomprm[i].atomType[0])) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].emin * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].rmin)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].emin14 * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].rmin14)) != OK) return ERROR(s);
      break;
    case EMIN:
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].emin * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case RMIN:
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].rmin)) != OK) return ERROR(s);
      break;
    case EMIN14:
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].emin14 * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case RMIN14:
      if ((s=list_append_dreal(interp, alist,
              atomprm[i].rmin14)) != OK) return ERROR(s);
      break;
    default:  /* ALL */
      for (i = 0;  i < natomprms;  i++) {
        if (atomprm[i].atomType[0][0] != 0) {
          Tcl_Obj *a = NULL;
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  atomprm[i].atomType[0])) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  atomprm[i].emin * ENERGY_EXTERNAL)) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  atomprm[i].rmin)) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  atomprm[i].emin14 * ENERGY_EXTERNAL)) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  atomprm[i].rmin14)) != OK) return ERROR(s);
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
      }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_get_bondprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    TYPE,
    K,
    R0
  };
  ForcePrm *fprm = &(p->fprm);
  const BondPrm *bondprm = ForcePrm_bondprm_array(fprm);
  int32 nbondprms = ForcePrm_bondprm_array_length(fprm);
  Tcl_Obj *alist = NULL;
  int32 i = -1;
  int action = ALL;
  int s;

  TEXT("bondprm");
  if (objc > 0) {
    const char *typ0, *typ1;
    if (2==objc) action=TYPE;
    else if (3==objc) {
      if (strcmp(Tcl_GetString(objv[2]),"k")==0) action=K;
      else if (strcmp(Tcl_GetString(objv[2]),"r0")==0) action=R0;
      else return ERROR(ERR_EXPECT);
    }
    else return ERROR(ERR_EXPECT);
    typ0 = Tcl_GetString(objv[0]);  /* get atom type */
    typ1 = Tcl_GetString(objv[1]);  /* get atom type */
    if ((i=ForcePrm_getid_bondprm(fprm, typ0, typ1)) < OK) {
      return (FAIL==i ? OK: ERROR(i));  /* failed match gives empty list */
    }
    /* otherwise return subset of ith bondprm */
  }
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  switch (action) {
    case TYPE:
      if ((s=list_append_string(interp, alist,
              bondprm[i].atomType[0])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              bondprm[i].atomType[1])) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              bondprm[i].k * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              bondprm[i].r0)) != OK) return ERROR(s);
      break;
    case K:
      if ((s=list_append_dreal(interp, alist,
              bondprm[i].k * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case R0:
      if ((s=list_append_dreal(interp, alist,
              bondprm[i].r0)) != OK) return ERROR(s);
      break;
    default:  /* ALL */
      for (i = 0;  i < nbondprms;  i++) {
        if (bondprm[i].atomType[0][0] != 0) {
          Tcl_Obj *a = NULL;
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  bondprm[i].atomType[0])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  bondprm[i].atomType[1])) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  bondprm[i].k * ENERGY_EXTERNAL)) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  bondprm[i].r0)) != OK) return ERROR(s);
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
      }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_get_angleprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    TYPE,
    KTHETA,
    THETA0,
    KUB,
    RUB
  };
  ForcePrm *fprm = &(p->fprm);
  const AnglePrm *angleprm = ForcePrm_angleprm_array(fprm);
  int32 nangleprms = ForcePrm_angleprm_array_length(fprm);
  Tcl_Obj *alist = NULL;
  int32 i = -1;
  int action = ALL;
  int s;

  TEXT("angleprm");
  if (objc > 0) {
    const char *typ0, *typ1, *typ2;
    if (3==objc) action=TYPE;
    else if (4==objc) {
      if (strcmp(Tcl_GetString(objv[3]),"ktheta")==0) action=KTHETA;
      else if (strcmp(Tcl_GetString(objv[3]),"theta0")==0) action=THETA0;
      else if (strcmp(Tcl_GetString(objv[3]),"kub")==0) action=KUB;
      else if (strcmp(Tcl_GetString(objv[3]),"rub")==0) action=RUB;
      else return ERROR(ERR_EXPECT);
    }
    else return ERROR(ERR_EXPECT);
    typ0 = Tcl_GetString(objv[0]);  /* get atom type */
    typ1 = Tcl_GetString(objv[1]);  /* get atom type */
    typ2 = Tcl_GetString(objv[2]);  /* get atom type */
    if ((i=ForcePrm_getid_angleprm(fprm, typ0, typ1, typ2)) < OK) {
      return (FAIL==i ? OK: ERROR(i));  /* failed match gives empty list */
    }
    /* otherwise return subset of ith angleprm */
  }
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  switch (action) {
    case TYPE:
      if ((s=list_append_string(interp, alist,
              angleprm[i].atomType[0])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              angleprm[i].atomType[1])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              angleprm[i].atomType[2])) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].k_theta * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].theta0 * DEGREES)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].k_ub * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].r_ub)) != OK) return ERROR(s);
      break;
    case KTHETA:
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].k_theta * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case THETA0:
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].theta0 * DEGREES)) != OK) return ERROR(s);
      break;
    case KUB:
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].k_ub * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case RUB:
      if ((s=list_append_dreal(interp, alist,
              angleprm[i].r_ub)) != OK) return ERROR(s);
      break;
    default:  /* ALL */
      for (i = 0;  i < nangleprms;  i++) {
        if (angleprm[i].atomType[0][0] != 0) {
          Tcl_Obj *a = NULL;
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  angleprm[i].atomType[0])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  angleprm[i].atomType[1])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  angleprm[i].atomType[2])) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  angleprm[i].k_theta * ENERGY_EXTERNAL))!=OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  angleprm[i].theta0 * DEGREES)) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  angleprm[i].k_ub * ENERGY_EXTERNAL)) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  angleprm[i].r_ub)) != OK) return ERROR(s);
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
      }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_get_dihedprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    TYPE,
    TERM,
    ATERM,
    KDIHED,
    PHI0,
    N
  };
  ForcePrm *fprm = &(p->fprm);
  const DihedPrm *dihedprm = ForcePrm_dihedprm_array(fprm);
  int32 ndihedprms = ForcePrm_dihedprm_array_length(fprm);
  const DihedTerm *dterm = NULL;
  int32 ndterms = 0;
  Tcl_Obj *alist = NULL;
  int32 i = -1, j;
  int action = ALL;
  int termid = -1;
  int s;

  TEXT("dihedprm");
  if (objc > 0) {
    const char *typ0, *typ1, *typ2, *typ3;
    if (4==objc) action=TYPE;
    else if (objc > 4) {
      if (strcmp(Tcl_GetString(objv[4]),"term")!=0) return ERROR(ERR_EXPECT);
      if (5==objc) action=TERM;
      else if (objc > 5) {
        if (TCL_ERROR==Tcl_GetIntFromObj(interp, objv[5], &termid)
            || termid < 0) return ERROR(ERR_EXPECT);
	if (6==objc) action=ATERM;
	else if (7==objc) {
          if (strcmp(Tcl_GetString(objv[6]),"kdihed")==0) action=KDIHED;
          else if (strcmp(Tcl_GetString(objv[6]),"phi0")==0) action=PHI0;
          else if (strcmp(Tcl_GetString(objv[6]),"n")==0) action=N;
          else return ERROR(ERR_EXPECT);
	}
	else return ERROR(ERR_EXPECT);
      }
      else return ERROR(ERR_EXPECT);
    }
    typ0 = Tcl_GetString(objv[0]);  /* get atom type */
    typ1 = Tcl_GetString(objv[1]);  /* get atom type */
    typ2 = Tcl_GetString(objv[2]);  /* get atom type */
    typ3 = Tcl_GetString(objv[3]);  /* get atom type */
    if ((i=ForcePrm_matchid_dihedprm(fprm, typ0, typ1, typ2, typ3)) < OK) {
      return (FAIL==i ? OK: ERROR(i));  /* failed match gives empty list */
    }
    dterm = DihedPrm_term_array(&dihedprm[i]);
    ndterms = DihedPrm_term_array_length(&dihedprm[i]);
    if (termid >= ndterms) return OK;  /* return empty list */
    /* otherwise return subset of ith dihedprm */
  } /* if */
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  switch (action) {
    case TYPE:
      if ((s=list_append_string(interp, alist,
              dihedprm[i].atomType[0])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              dihedprm[i].atomType[1])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              dihedprm[i].atomType[2])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              dihedprm[i].atomType[3])) != OK) return ERROR(s);
      if ((s=list_append_int32(interp, alist, ndterms))!=OK) return ERROR(s);
      for (j = 0;  j < ndterms;  j++) {  /* all terms */
        Tcl_Obj *a = NULL;
        if ((s=new_list(interp, &a)) != OK) return ERROR(s);
        if ((s=list_append_dreal(interp, a,
                dterm[j].k_dihed * ENERGY_EXTERNAL)) != OK) return ERROR(s);
        if ((s=list_append_dreal(interp, a,
                dterm[j].phi0 * DEGREES)) != OK) return ERROR(s);
        if ((s=list_append_int32(interp, a,
                dterm[j].n)) != OK) return ERROR(s);
        if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
      }
      break;
    case TERM:
      if ((s=list_append_int32(interp, alist, ndterms))!=OK) return ERROR(s);
      break;
    case ATERM:
      if ((s=list_append_dreal(interp, alist,
              dterm[termid].k_dihed * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              dterm[termid].phi0 * DEGREES)) != OK) return ERROR(s);
      if ((s=list_append_int32(interp, alist,
              dterm[termid].n)) != OK) return ERROR(s);
      break;
    case KDIHED:
      if ((s=list_append_dreal(interp, alist,
              dterm[termid].k_dihed * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case PHI0:
      if ((s=list_append_dreal(interp, alist,
              dterm[termid].phi0 * DEGREES)) != OK) return ERROR(s);
      break;
    case N:
      if ((s=list_append_int32(interp, alist,
              dterm[termid].n)) != OK) return ERROR(s);
      break;
    default:  /* ALL */
      for (i = 0;  i < ndihedprms;  i++) {
        if (dihedprm[i].atomType[0][0] != 0) {
          Tcl_Obj *b = NULL;
          dterm = DihedPrm_term_array(&dihedprm[i]);
          ndterms = DihedPrm_term_array_length(&dihedprm[i]);
          if ((s=new_list(interp, &b)) != OK) return ERROR(s);
          if ((s=list_append_string(interp, b,
                  dihedprm[i].atomType[0])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, b,
                  dihedprm[i].atomType[1])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, b,
                  dihedprm[i].atomType[2])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, b,
                  dihedprm[i].atomType[3])) != OK) return ERROR(s);
          if ((s=list_append_int32(interp, b, ndterms))!=OK) return ERROR(s);
          for (j = 0;  j < ndterms;  j++) {  /* all terms */
            Tcl_Obj *a = NULL;
            if ((s=new_list(interp, &a)) != OK) return ERROR(s);
            if ((s=list_append_dreal(interp, a,
                    dterm[j].k_dihed * ENERGY_EXTERNAL))!=OK) return ERROR(s);
            if ((s=list_append_dreal(interp, a,
                    dterm[j].phi0 * DEGREES)) != OK) return ERROR(s);
            if ((s=list_append_int32(interp, a,
                    dterm[j].n)) != OK) return ERROR(s);
            if ((s=list_append_obj(interp, b, a)) != OK) return ERROR(s);
          }
          if ((s=list_append_obj(interp, alist, b)) != OK) return ERROR(s);
        } /* if */
      } /* for */
  } /* switch */
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_get_imprprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    TYPE,
    KIMPR,
    PSI0
  };
  ForcePrm *fprm = &(p->fprm);
  const ImprPrm *imprprm = ForcePrm_imprprm_array(fprm);
  int32 nimprprms = ForcePrm_imprprm_array_length(fprm);
  Tcl_Obj *alist = NULL;
  int32 i = -1;
  int action = ALL;
  int s;

  TEXT("imprprm");
  if (objc > 0) {
    const char *typ0, *typ1, *typ2, *typ3;
    if (4==objc) action=TYPE;
    else if (5==objc) {
      if (strcmp(Tcl_GetString(objv[4]),"kimpr")==0) action=KIMPR;
      else if (strcmp(Tcl_GetString(objv[4]),"psi0")==0) action=PSI0;
      else return ERROR(ERR_EXPECT);
    }
    else return ERROR(ERR_EXPECT);
    typ0 = Tcl_GetString(objv[0]);  /* get atom type */
    typ1 = Tcl_GetString(objv[1]);  /* get atom type */
    typ2 = Tcl_GetString(objv[2]);  /* get atom type */
    typ3 = Tcl_GetString(objv[3]);  /* get atom type */
    if ((i=ForcePrm_matchid_imprprm(fprm, typ0, typ1, typ2, typ3)) < OK) {
      return (FAIL==i ? OK: ERROR(i));  /* failed match gives empty list */
    }
    /* otherwise return subset of ith imprprm */
  }
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  switch (action) {
    case TYPE:
      if ((s=list_append_string(interp, alist,
              imprprm[i].atomType[0])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              imprprm[i].atomType[1])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              imprprm[i].atomType[2])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              imprprm[i].atomType[3])) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              imprprm[i].k_impr * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              imprprm[i].psi0 * DEGREES)) != OK) return ERROR(s);
      break;
    case KIMPR:
      if ((s=list_append_dreal(interp, alist,
              imprprm[i].k_impr * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case PSI0:
      if ((s=list_append_dreal(interp, alist,
              imprprm[i].psi0 * DEGREES)) != OK) return ERROR(s);
      break;
    default:  /* ALL */
      for (i = 0;  i < nimprprms;  i++) {
        if (imprprm[i].atomType[0][0] != 0) {
          Tcl_Obj *a = NULL;
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  imprprm[i].atomType[0])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  imprprm[i].atomType[1])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  imprprm[i].atomType[2])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  imprprm[i].atomType[3])) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  imprprm[i].k_impr * ENERGY_EXTERNAL))!=OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  imprprm[i].psi0 * DEGREES)) != OK) return ERROR(s);
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
      }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_get_vdwpairprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  enum {
    ALL,
    TYPE,
    EMIN,
    RMIN,
    EMIN14,
    RMIN14
  };
  ForcePrm *fprm = &(p->fprm);
  const VdwpairPrm *vdwpairprm = ForcePrm_vdwpairprm_array(fprm);
  int32 nvdwpairprms = ForcePrm_vdwpairprm_array_length(fprm);
  Tcl_Obj *alist = NULL;
  int32 i = -1;
  int action = ALL;
  int s;

  TEXT("vdwpairprm");
  if (objc > 0) {
    const char *typ0, *typ1;
    if (2==objc) action=TYPE;
    else if (3==objc) {
      if (strcmp(Tcl_GetString(objv[2]),"emin")==0) action=EMIN;
      else if (strcmp(Tcl_GetString(objv[2]),"rmin")==0) action=RMIN;
      else if (strcmp(Tcl_GetString(objv[2]),"emin14")==0) action=EMIN14;
      else if (strcmp(Tcl_GetString(objv[2]),"rmin14")==0) action=RMIN14;
      else return ERROR(ERR_EXPECT);
    }
    else return ERROR(ERR_EXPECT);
    typ0 = Tcl_GetString(objv[0]);  /* get atom type */
    typ1 = Tcl_GetString(objv[1]);  /* get atom type */
    if ((i=ForcePrm_getid_vdwpairprm(fprm, typ0, typ1)) < OK) {
      return (FAIL==i ? OK: ERROR(i));  /* failed match gives empty list */
    }
    /* otherwise return subset of ith vdwpairprm */
  }
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  switch (action) {
    case TYPE:
      if ((s=list_append_string(interp, alist,
              vdwpairprm[i].atomType[0])) != OK) return ERROR(s);
      if ((s=list_append_string(interp, alist,
              vdwpairprm[i].atomType[1])) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].emin * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].rmin)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].emin14 * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].rmin14)) != OK) return ERROR(s);
      break;
    case EMIN:
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].emin * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case RMIN:
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].rmin)) != OK) return ERROR(s);
      break;
    case EMIN14:
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].emin14 * ENERGY_EXTERNAL)) != OK) return ERROR(s);
      break;
    case RMIN14:
      if ((s=list_append_dreal(interp, alist,
              vdwpairprm[i].rmin14)) != OK) return ERROR(s);
      break;
    default:  /* ALL */
      for (i = 0;  i < nvdwpairprms;  i++) {
        if (vdwpairprm[i].atomType[0][0] != 0) {
          Tcl_Obj *a = NULL;
          if ((s=new_list(interp, &a)) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  vdwpairprm[i].atomType[0])) != OK) return ERROR(s);
          if ((s=list_append_string(interp, a,
                  vdwpairprm[i].atomType[1])) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  vdwpairprm[i].emin*ENERGY_EXTERNAL))!=OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  vdwpairprm[i].rmin)) != OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  vdwpairprm[i].emin14*ENERGY_EXTERNAL))!=OK) return ERROR(s);
          if ((s=list_append_dreal(interp, a,
                  vdwpairprm[i].rmin14)) != OK) return ERROR(s);
          if ((s=list_append_obj(interp, alist, a)) != OK) return ERROR(s);
        }
      }
  }
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_get_nonbprm(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  ForcePrm *fprm = &(p->fprm);
  const NonbPrm *nonbprm = ForcePrm_nonbprm(fprm);
  const char *t;
  Tcl_Obj *a = NULL;
  int s;

  TEXT("nonbprm");
  ASSERT(nonbprm != NULL);
  if (objc != 1) return ERROR(ERR_EXPECT);
  t = Tcl_GetString(objv[0]);
  if (strcmp(t,"cutoff")==0) {
    const dreal r = nonbprm->cutoff;
    if ((s=NLEnergy_new_obj_dreal(interp, &a, r)) != OK) return ERROR(s);
  }
  else if (strcmp(t,"switching")==0) {
    const char *str = (nonbprm->switching ? "on" : "off");
    if ((s=NLEnergy_new_obj_string(interp, &a, str)) != OK) return ERROR(s);
  }
  else if (strcmp(t,"switchdist")==0) {
    const dreal r = nonbprm->switchdist;
    if ((s=NLEnergy_new_obj_dreal(interp, &a, r)) != OK) return ERROR(s);
  }
  else if (strcmp(t,"exclude")==0) {
    const char *str;
    switch (nonbprm->exclude) {
      case EXCL_NONE:      str = "none"; break;
      case EXCL_12:        str = "1-2";  break;
      case EXCL_13:        str = "1-3";  break;
      case EXCL_14:        str = "1-4";  break;
      case EXCL_SCALED14:  str = "scaled1-4";  break;
      default:  return ERROR(ERR_EXPECT);
    }
    if ((s=NLEnergy_new_obj_string(interp, &a, str)) != OK) return ERROR(s);
  }
  else if (strcmp(t,"dielectric")==0) {
    const dreal r = nonbprm->dielectric;
    if ((s=NLEnergy_new_obj_dreal(interp, &a, r)) != OK) return ERROR(s);
  }
  else if (strcmp(t,"1-4scaling")==0) {
    const dreal r = nonbprm->scaling14;
    if ((s=NLEnergy_new_obj_dreal(interp, &a, r)) != OK) return ERROR(s);
  }
  else if (strcmp(t,"fulldirect")==0) {
    const char *str = (p->fulldirect ? "on" : "off");
    if ((s=NLEnergy_new_obj_string(interp, &a, str)) != OK) return ERROR(s);
  }
  else if (strcmp(t,"fulldirectvdw")==0) {
    const char *str = (p->fulldirectvdw ? "on" : "off");
    if ((s=NLEnergy_new_obj_string(interp, &a, str)) != OK) return ERROR(s);
  }
  else return ERROR(ERR_EXPECT);
  if ((s=NLEnergy_set_obj_result(interp, a)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_get_cellbasis(NLEnergy *p, Tcl_Interp *interp,
    int objc, Tcl_Obj *const objv[]) {
  const Domain *domain = Coord_domain(&(p->coord));
  Tcl_Obj *alist = NULL;
  dvec bv;
  int s;

  TEXT("cellbasis");
  if (objc != 0) return ERROR(ERR_EXPECT);
  if ((s=new_list(interp, &alist)) != OK) return ERROR(s);
  bv = domain->basis[0];
  if (FALSE==domain->periodic_x) VECZERO(bv);
  if ((s=NLEnergy_list_append_dvec(interp, alist, &bv)) != OK) return ERROR(s);
  bv = domain->basis[1];
  if (FALSE==domain->periodic_y) VECZERO(bv);
  if ((s=NLEnergy_list_append_dvec(interp, alist, &bv)) != OK) return ERROR(s);
  bv = domain->basis[2];
  if (FALSE==domain->periodic_z) VECZERO(bv);
  if ((s=NLEnergy_list_append_dvec(interp, alist, &bv)) != OK) return ERROR(s);
  if ((s=set_obj_result(interp, alist)) != OK) return ERROR(s);
  return OK;
}
