/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: TclCommands.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.50 $	$Date: 2016/11/28 03:05:05 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  prototypes for VMD<->Tcl functions
 *
 ***************************************************************************/
#ifndef TCLCOMMANDS_H
#define TCLCOMMANDS_H
//forward definition
class AtomSel;
class VMDApp;

// for the core VMD commands
int Vmd_Init(Tcl_Interp *);

// for 'molinfo'
int molecule_tcl(ClientData, Tcl_Interp *interp, int argc, const char *argv[]);

// for 'vec*' and 'trans*'
int Vec_Init(Tcl_Interp *);

// for 'atomselect'
int Atomsel_Init(Tcl_Interp *);

// get the atom selection associated with the given tcl selection
// (the one of the form 'atomselect%u')
AtomSel *tcl_commands_get_sel(Tcl_Interp *, const char *str);

// for accessing the graphics
int graphics_tcl(ClientData, Tcl_Interp *interp, int argc, const char *argv[]);

// for accessing the colors
int tcl_colorinfo(ClientData, Tcl_Interp *interp, int argc, const char *argv[]);

// for the measure commands
int obj_measure(ClientData, Tcl_Interp *, int, Tcl_Obj *const []);

#if 1
// for the mdff cc command
int obj_mdff_cc(ClientData cd, Tcl_Interp *interp, int argc, Tcl_Obj * const objv[]);
#endif

#if 0
// for the volgradient command
int obj_volgradient(ClientData cd, Tcl_Interp *interp, int argc, Tcl_Obj * const objv[]);
#endif

// for the volmap commands
int obj_volmap(ClientData, Tcl_Interp *, int, Tcl_Obj *const []);

// get a matrix from a string; 
// returns TCL_OK if good
// If bad, returns TCL_ERROR and sets the interp->result to the error message
// The name of the function should be passed in 'fctn' so the error message
// can be constructed correctly
int tcl_get_matrix(const char *fctn, Tcl_Interp *interp,
		   Tcl_Obj *s, float *mat);

/// converts a Tcl string into a usable array of weights for VMD
/// functions.
int tcl_get_weights(Tcl_Interp *interp, VMDApp *app, AtomSel *sel, 
                       Tcl_Obj *weight_obj, float *data);

// Get a vector from a string
int tcl_get_vector(const char *s, float *val, Tcl_Interp *interp);

// append the matrix information to the interp->result field
void tcl_append_matrix(Tcl_Interp *interp, const float *mat);

#if defined(VMDTKCON)
// set up and write console log messages
int tcl_vmdcon(ClientData nodata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
#endif

#endif

