/*
 * Tcl bindings for the PME potential plugin
 *
 * $Id: tcl_pmepot.c,v 1.5 2014/02/21 20:46:21 jim Exp $
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "pmepot.h"

#include <tcl.h>

/* This function gets called if/when the Tcl interpreter is deleted. */
static void pmepot_deleteproc(ClientData cd, Tcl_Interp *interp) {
  /* pmepot_data *data = (pmepot_data *)cd; */
  /* free(data); */
}

void pmepot_data_delete_pointer(ClientData cd, Tcl_Interp *interp) {
  pmepot_data **dataptr = (pmepot_data **)cd;
  free(dataptr);
}

int tcl_pmepot_create(ClientData nodata, Tcl_Interp *interp,
			int objc, Tcl_Obj *const objv[]) {

  int dims_count, dims[3], i;
  Tcl_Obj **dims_list;
  double ewald_factor;
  char namebuf[128];
  int *countptr;
  pmepot_data *data;

  if ( objc != 3 ) {
    Tcl_SetResult(interp,"args: {na nb nc} ewald_factor",TCL_VOLATILE);
    return TCL_ERROR;
  }

  if ( Tcl_ListObjGetElements(interp,objv[1],&dims_count,&dims_list) != TCL_OK ) return TCL_ERROR;
  if ( dims_count != 3 ) {
    Tcl_SetResult(interp,"args: {na nb nc} ewald_factor",TCL_VOLATILE);
    return TCL_ERROR;
  }
  for ( i=0; i<3; ++i ) {
    if ( Tcl_GetIntFromObj(interp,dims_list[i],&dims[i]) != TCL_OK ) return TCL_ERROR;
    if ( dims[i] < 8 ) {
      Tcl_SetResult(interp,"each grid dimension must be at least 8",TCL_VOLATILE);
      return TCL_ERROR;
    }
  }
  if ( dims[2] % 2 ) {
    Tcl_SetResult(interp,"third grid dimension must be even",TCL_VOLATILE);
    return TCL_ERROR;
  }

  if ( Tcl_GetDoubleFromObj(interp,objv[2],&ewald_factor) != TCL_OK ) {
    return TCL_ERROR;
  }
  if ( ewald_factor <= 0. ) {
    Tcl_SetResult(interp,"ewald factor must be positive",TCL_VOLATILE);
    return TCL_ERROR;
  }

  countptr = Tcl_GetAssocData(interp, "Pmepot_count", 0);
  if ( ! countptr ) {
    Tcl_SetResult(interp,"Pmepot bug: Pmepot_count not initialized.",TCL_VOLATILE);
    return TCL_ERROR;
  }

  data = pmepot_create(dims, ewald_factor);
  if ( ! data ) {
    Tcl_SetResult(interp,"Pmepot bug: pmepot_create failed.",TCL_VOLATILE);
    return TCL_ERROR;
  }

  sprintf(namebuf,"Pmepot_%d",*countptr);
  Tcl_SetAssocData(interp,namebuf,pmepot_deleteproc,(ClientData)data);
  *countptr += 1;

  Tcl_SetResult(interp,namebuf,TCL_VOLATILE);
  return TCL_OK;
}

int tcl_pmepot_add(ClientData nodata, Tcl_Interp *interp,
			int objc, Tcl_Obj *const objv[]) {

  int cell_count, atom_count, sub_count, i, j;
  Tcl_Obj **cell_list, **atom_list, **sub_list;
  float cell[12], *atoms;
  double d;
  pmepot_data *data;
  if ( objc != 4 ) {
    Tcl_SetResult(interp,"args: handle {{o...} {a...} {b...} {c...}} {{x y z q}...}",TCL_VOLATILE);
    return TCL_ERROR;
  }
  data = Tcl_GetAssocData(interp, Tcl_GetString(objv[1]), 0);
  if ( ! data ) {
    Tcl_SetResult(interp,"Pmepot bug: unable to access handle.",TCL_VOLATILE);
    return TCL_ERROR;
  }

  if ( Tcl_ListObjGetElements(interp,objv[2],&cell_count,&cell_list) != TCL_OK ) return TCL_ERROR;
  if ( cell_count != 4 ) {
    Tcl_SetResult(interp,"cell format: {{ox oy oz} {ax ay az} {bx by bz} {cx cy cz}}",TCL_VOLATILE);
    return TCL_ERROR;
  }
  for ( i=0; i<4; ++i ) {
    if ( Tcl_ListObjGetElements(interp,cell_list[i],&sub_count,&sub_list) != TCL_OK ) return TCL_ERROR;
    if ( sub_count != 3 ) {
      Tcl_SetResult(interp,"cell format: {{ox oy oz} {ax ay az} {bx by bz} {cx cy cz}}",TCL_VOLATILE);
      return TCL_ERROR;
    }
    for ( j=0; j<3; ++j ) {
      if ( Tcl_GetDoubleFromObj(interp,sub_list[j],&d) != TCL_OK ) return TCL_ERROR;
      cell[3*i+j] = d;
    }
  }
  if ( Tcl_ListObjGetElements(interp,objv[3],&atom_count,&atom_list) != TCL_OK ) return TCL_ERROR;
  atoms = malloc(atom_count*4*sizeof(float));
  if ( ! atoms ) {
    Tcl_SetResult(interp,"Pmepot error: unable to allocate atom array.",TCL_VOLATILE);
    return TCL_ERROR;
  }
  for ( i=0; i<atom_count; ++i ) {
    if ( Tcl_ListObjGetElements(interp,atom_list[i],&sub_count,&sub_list) != TCL_OK ) { free(atoms); return TCL_ERROR; }
    if ( sub_count != 4 ) {
      Tcl_SetResult(interp,"atoms format: {{x y z q}...}",TCL_VOLATILE);
      free(atoms); return TCL_ERROR;
    }
    for ( j=0; j<4; ++j ) {
      if ( Tcl_GetDoubleFromObj(interp,sub_list[j],&d) != TCL_OK ) { free(atoms); return TCL_ERROR; }
      atoms[4*i+j] = d;
    }
  }

  if ( pmepot_add(data,cell,atom_count,atoms) ) {
    Tcl_SetResult(interp,"Pmepot bug: pmepot_add failed.",TCL_VOLATILE);
    free(atoms);
    return TCL_ERROR;
  }

  free(atoms);
  return TCL_OK;
}

int tcl_pmepot_writedx(ClientData nodata, Tcl_Interp *interp,
			int objc, Tcl_Obj *const objv[]) {

  pmepot_data *data;
  Tcl_DString fstring;
  char *fname;

  if ( objc != 3 ) {
    Tcl_SetResult(interp,"args: handle filename",TCL_VOLATILE);
    return TCL_ERROR;
  }
  data = Tcl_GetAssocData(interp, Tcl_GetString(objv[1]), 0);
  if ( ! data ) {
    Tcl_SetResult(interp,"Pmepot bug: unable to access handle.",TCL_VOLATILE);
    return TCL_ERROR;
  }

  fname = Tcl_TranslateFileName(interp,Tcl_GetString(objv[2]),&fstring);
  if ( 0 == fname ) {
    return TCL_ERROR;
  }

  if ( pmepot_writedx(data,fname) ) {
    Tcl_DStringFree(&fstring);
    Tcl_SetResult(interp,"Pmepot bug: unable to write file.",TCL_VOLATILE);
    return TCL_ERROR;
  }
  Tcl_DStringFree(&fstring);
  return TCL_OK;
}

int tcl_pmepot_destroy(ClientData nodata, Tcl_Interp *interp,
			int objc, Tcl_Obj *const objv[]) {

  pmepot_data *data;
  if ( objc != 2 ) {
    Tcl_SetResult(interp,"args: handle",TCL_VOLATILE);
    return TCL_ERROR;
  }
  data = Tcl_GetAssocData(interp, Tcl_GetString(objv[1]), 0);
  if ( ! data ) {
    Tcl_SetResult(interp,"Pmepot bug: unable to access handle.",TCL_VOLATILE);
    return TCL_ERROR;
  }

  pmepot_destroy(data);
  Tcl_DeleteAssocData(interp, Tcl_GetString(objv[1]));
  return TCL_OK;
}


static void count_delete_proc(ClientData data, Tcl_Interp *interp) {
  free(data);
}

#if defined(PMEPOTTCLDLL_EXPORTS) && defined(_WIN32)
#  undef TCL_STORAGE_CLASS
#  define TCL_STORAGE_CLASS DLLEXPORT

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Window s headers
#include <windows.h>

BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved
                                         )
{
    return TRUE;
}

EXTERN int Pmepot_Init(Tcl_Interp *interp) {

#else

int Pmepot_Init(Tcl_Interp *interp) {

#endif

  int *countptr;
  countptr = (int *)malloc(sizeof(int));
  if ( ! countptr ) {
    Tcl_SetResult(interp,"Pmepot error: unable to allocate count pointer.",TCL_VOLATILE);
    return TCL_ERROR;
  }
  Tcl_SetAssocData(interp, "Pmepot_count", count_delete_proc, 
				(ClientData)countptr);
  *countptr = 0;

  Tcl_CreateObjCommand(interp,"pmepot_create",tcl_pmepot_create,
	(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"pmepot_add",tcl_pmepot_add,
	(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"pmepot_writedx",tcl_pmepot_writedx,
	(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
  Tcl_CreateObjCommand(interp,"pmepot_destroy",tcl_pmepot_destroy,
	(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
 
  Tcl_PkgProvide(interp, "pmepot_core", "1.0.0");

  return TCL_OK;
}

