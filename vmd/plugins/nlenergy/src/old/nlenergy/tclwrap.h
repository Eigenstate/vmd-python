/*
 * Copyright (C) 2008 by David J. Hardy.  All rights reserved.
 */

#ifndef NLENERGY_TCLWRAP_H
#define NLENERGY_TCLWRAP_H

#include "nlenergy/nlenergy.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* prototypes */
  int32 NLEnergy_atomid_from_obj(NLEnergy *, Tcl_Interp *, Tcl_Obj *obj);
  int NLEnergy_new_obj_atomid(NLEnergy *, Tcl_Interp *, Tcl_Obj **pobj, int32);
  int NLEnergy_list_append_atomid(NLEnergy *, Tcl_Interp *, Tcl_Obj *list,
      int32 i);

  int NLEnergy_new_obj_int32(Tcl_Interp *, Tcl_Obj **pobj, int32 n);
  int NLEnergy_new_obj_dreal(Tcl_Interp *, Tcl_Obj **pobj, dreal r);
  int NLEnergy_new_obj_string(Tcl_Interp *, Tcl_Obj **pobj, const char *str);
  int NLEnergy_new_obj_dvec(Tcl_Interp *, Tcl_Obj **pobj, const dvec *v);
  int NLEnergy_new_list(Tcl_Interp *, Tcl_Obj **plist);
  int NLEnergy_list_append_int32(Tcl_Interp *, Tcl_Obj *list, int32 n);
  int NLEnergy_list_append_dreal(Tcl_Interp *, Tcl_Obj *list, dreal r);
  int NLEnergy_list_append_string(Tcl_Interp *, Tcl_Obj *lis, const char *str);
  int NLEnergy_list_append_dvec(Tcl_Interp *, Tcl_Obj *list, const dvec *v);
  int NLEnergy_list_append_obj(Tcl_Interp *, Tcl_Obj *list, Tcl_Obj *obj);
  int NLEnergy_set_obj_result(Tcl_Interp *, Tcl_Obj *obj);
  int NLEnergy_atomlist_obj(Array *a, Tcl_Interp *interp, Tcl_Obj *obj,
      int32 natoms, int32 nesting);
  int NLEnergy_shared_atoms(Array *a, Array *b);

  /* convenient macro shorthand */
#define atomid_from_obj    NLEnergy_atomid_from_obj
#define new_obj_atomid     NLEnergy_new_obj_atomid
#define list_append_atomid NLEnergy_list_append_atomid

#define new_obj_int32      NLEnergy_new_obj_int32
#define new_obj_dreal      NLEnergy_new_obj_dreal
#define new_obj_string     NLEnergy_new_obj_string
#define new_obj_dvec       NLEnergy_new_obj_dvec
#define new_list           NLEnergy_new_list
#define list_append_int32  NLEnergy_list_append_int32
#define list_append_dreal  NLEnergy_list_append_dreal
#define list_append_string NLEnergy_list_append_string
#define list_append_dvec   NLEnergy_list_append_dvec
#define list_append_obj    NLEnergy_list_append_obj
#define set_obj_result     NLEnergy_set_obj_result
#define shared_atoms       NLEnergy_shared_atoms
#define atomlist_obj       NLEnergy_atomlist_obj

#ifdef __cplusplus
}
#endif

#endif /* NLENERGY_TCLWRAP_H */
