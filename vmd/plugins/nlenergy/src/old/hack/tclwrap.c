#include "nlenergy/tclwrap.h"


int32 NLEnergy_atomid_from_obj(NLEnergy *p, Tcl_Interp *interp, Tcl_Obj *obj) {
  int n;
  const int32 *atomid = Array_data_const(&(p->atomid));
  if (TCL_OK != Tcl_GetIntFromObj(interp, obj, &n)) return ERROR(ERR_EXPECT);
  else if (n < p->firstid || n > p->lastid) return FAIL;
  return atomid[ n - p->firstid ];
}


int NLEnergy_new_obj_atomid(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj **pobj, int32 i) {
  const int32 *extatomid = Array_data_const(&(p->extatomid));
  const int32 extatomidlen = Array_length(&(p->extatomid));
  ASSERT(NULL==*pobj || (*pobj)->refCount == 1);
  if (i < 0 || i >= extatomidlen) return ERROR(ERR_EXPECT);
  if (NULL==(*pobj = Tcl_NewIntObj((int) extatomid[i]))) {
    return ERROR(ERR_EXPECT);
  }
  Tcl_IncrRefCount(*pobj);
  return OK;
}


int NLEnergy_list_append_atomid(NLEnergy *p, Tcl_Interp *interp,
    Tcl_Obj *list, int32 i) {
  const int32 *extatomid = Array_data_const(&(p->extatomid));
  const int32 extatomidlen = Array_length(&(p->extatomid));
  Tcl_Obj *obj = NULL;
  if (i < 0 || i >= extatomidlen) return ERROR(ERR_EXPECT);
  if (NULL==(obj = Tcl_NewIntObj((int) extatomid[i]))) {
    return ERROR(ERR_EXPECT);
  }
  if (TCL_OK != Tcl_ListObjAppendElement(interp, list, obj)) {
    return ERROR(ERR_EXPECT);
  }
  return OK;
}


int NLEnergy_new_obj_int32(Tcl_Interp *interp, Tcl_Obj **pobj, int32 n) {
  ASSERT(NULL==*pobj || (*pobj)->refCount == 1);
  if (NULL==(*pobj = Tcl_NewIntObj((int) n))) return ERROR(ERR_EXPECT);
  Tcl_IncrRefCount(*pobj);
  return OK;
}


int NLEnergy_new_obj_dreal(Tcl_Interp *interp, Tcl_Obj **pobj, dreal r) {
  ASSERT(NULL==*pobj || (*pobj)->refCount == 1);
  if (NULL==(*pobj = Tcl_NewDoubleObj((double) r))) return ERROR(ERR_EXPECT);
  Tcl_IncrRefCount(*pobj);
  return OK;
}


int NLEnergy_new_obj_string(Tcl_Interp *interp, Tcl_Obj **pobj,
    const char *str) {
  ASSERT(NULL==*pobj || (*pobj)->refCount == 1);
  if (NULL==(*pobj = Tcl_NewStringObj(str, -1))) return ERROR(ERR_EXPECT);
  Tcl_IncrRefCount(*pobj);
  return OK;
}


int NLEnergy_new_obj_dvec(Tcl_Interp *interp, Tcl_Obj **pobj, const dvec *v) {
  int s;
  ASSERT(NULL==*pobj || (*pobj)->refCount == 1);
  if ((s=new_list(interp, pobj)) != OK) return ERROR(s);
  if ((s=list_append_dreal(interp, *pobj, v->x)) != OK) return ERROR(s);
  if ((s=list_append_dreal(interp, *pobj, v->y)) != OK) return ERROR(s);
  if ((s=list_append_dreal(interp, *pobj, v->z)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_new_list(Tcl_Interp *interp, Tcl_Obj **plist) {
  ASSERT(NULL==*plist || (*plist)->refCount == 1);
  if (NULL==(*plist = Tcl_NewObj())) return ERROR(ERR_EXPECT);
  Tcl_IncrRefCount(*plist);
  return OK;
}


int NLEnergy_list_append_int32(Tcl_Interp *interp, Tcl_Obj *list, int32 n) {
  Tcl_Obj *obj = NULL;
  if (NULL==(obj = Tcl_NewIntObj((int) n))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_ListObjAppendElement(interp, list, obj)) {
    return ERROR(ERR_EXPECT);
  }
  return OK;
}


int NLEnergy_list_append_dreal(Tcl_Interp *interp, Tcl_Obj *list, dreal r) {
  Tcl_Obj *obj = NULL;
  if (NULL==(obj = Tcl_NewDoubleObj((double) r))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_ListObjAppendElement(interp, list, obj)) {
    return ERROR(ERR_EXPECT);
  }
  return OK;
}


int NLEnergy_list_append_string(Tcl_Interp *interp, Tcl_Obj *list,
    const char *str) {
  Tcl_Obj *obj = NULL;
  if (NULL==(obj = Tcl_NewStringObj(str, -1))) return ERROR(ERR_EXPECT);
  if (TCL_OK != Tcl_ListObjAppendElement(interp, list, obj)) {
    return ERROR(ERR_EXPECT);
  }
  return OK;
}


int NLEnergy_list_append_dvec(Tcl_Interp *interp, Tcl_Obj *list,
    const dvec *v) {
  Tcl_Obj *vobj = NULL;
  int s;
  if ((s=new_obj_dvec(interp, &vobj, v)) != OK) return ERROR(s);
  if ((s=list_append_obj(interp, list, vobj)) != OK) return ERROR(s);
  return OK;
}


int NLEnergy_list_append_obj(Tcl_Interp *interp, Tcl_Obj *list, Tcl_Obj *obj) {
  if (TCL_OK != Tcl_ListObjAppendElement(interp, list, obj)) {
    return ERROR(ERR_EXPECT);
  }
  Tcl_DecrRefCount(obj);
  return OK;
}


int NLEnergy_set_obj_result(Tcl_Interp *interp, Tcl_Obj *obj) {
  Tcl_SetObjResult(interp, obj);
  Tcl_DecrRefCount(obj);
  return OK;
}


/*
 * produce a sorted atomlist (array of int32) with unique indices,
 * elements bounded by natoms
 *
 * nesting:  0=list of int, 1=list of 1-lists, 2=list of 2-lists (bonds),
 * 3=list of 3-lists (angles), 4=list of 4-lists (diheds, imprs), etc.
 */
static int asclean(Array *psa, int retval) {
  Array_done(psa);
  return retval;
}

int NLEnergy_atomlist_obj(Array *a, Tcl_Interp *interp, Tcl_Obj *obj,
    int32 natoms, int32 nesting) {
  Array sa;  /* sort array */
  SortElem elem;
  SortElem *e;  /* points to sort array data */
  int32 *id;
  int32 len;
  int32 shift;
  int32 i, j;
  Tcl_Obj **aobjv;
  int aobjc;
  int n;
  int s;  /* error status */

  /* read tcl obj into sort array */
  if ((s=Array_init(&sa, sizeof(SortElem))) != OK) return ERROR(s);
  if (TCL_ERROR==Tcl_ListObjGetElements(interp, obj, &aobjc, &aobjv)) {
    return asclean(&sa,ERROR(ERR_EXPECT));
  }
  n = (0==nesting ? 1 : nesting);
  if ((s=Array_setbuflen(&sa, n * aobjc)) != OK) return ERROR(s);
  len = 0;
  for (i = 0;  i < aobjc;  i++) {
    if (nesting > 0) {
      Tcl_Obj **bobjv;
      int bobjc;
      if (TCL_ERROR==Tcl_ListObjGetElements(interp, aobjv[i], &bobjc, &bobjv)
          || bobjc != nesting) {
        return asclean(&sa,ERROR(ERR_EXPECT));
      }
      for (j = 0;  j < nesting;  j++) {
        if (TCL_ERROR==Tcl_GetIntFromObj(interp, bobjv[j], &n)
            || n < 0 || n >= natoms) {
          return asclean(&sa,ERROR(ERR_EXPECT));
        }
        elem.key = n;
        elem.value = len;
        if ((s=Array_append(&sa, &elem)) != OK) {
          return asclean(&sa,ERROR(ERR_EXPECT));
        }
        len++;
      }
    }
    else {
      if (TCL_ERROR==Tcl_GetIntFromObj(interp, aobjv[i], &n)
          || n < 0 || n >= natoms) {
        return asclean(&sa,ERROR(ERR_EXPECT));
      }
      elem.key = n;
      elem.value = len;
      if ((s=Array_append(&sa, &elem)) != OK) {
        return asclean(&sa,ERROR(ERR_EXPECT));
      }
      len++;
    }
  }

  /* sort the array based on atom index, remove duplicate entries */
  e = Array_data(&sa);
  ASSERT(Array_length(&sa) == len);
  if ((s=Sort_quick(e, len)) != OK) {
    return asclean(&sa,ERROR(ERR_EXPECT));
  }
  shift = 0;
  for (i = 1;  i < len;  i++) {
    if (e[i].key == e[i-1-shift].key) {
      shift++;
    }
    else if (shift > 0) {
      e[i-shift] = e[i];
    }
  }
  len -= shift;
  ASSERT(len <= natoms);

  /* copy into int32 array container held by caller */
  if ((s=Array_resize(a, len)) != OK) {
    return asclean(&sa,ERROR(ERR_EXPECT));
  }
  id = Array_data(a);
  for (i = 0;  i < len;  i++) {
    id[i] = e[i].key;
  }

  return asclean(&sa,OK);
}


int NLEnergy_shared_atoms(Array *a, Array *b) {
  const int32 *na = Array_data_const(a);
  const int32 alen = Array_length(a);
  const int32 *nb = Array_data_const(b);
  const int32 blen = Array_length(b);
  int32 min = (alen <= blen ? alen : blen);
  int32 i, j;
  ASSERT(alen > 0 && blen > 0);
  for (i = 0, j = 0;  i < min && j < min; ) {
    if (na[i] < nb[j]) i++;
    else if (na[i] > nb[j]) j++;
    else return TRUE;
  }
  return FALSE;
}
