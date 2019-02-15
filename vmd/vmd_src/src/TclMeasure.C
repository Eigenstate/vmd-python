/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: TclMeasure.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.171 $	$Date: 2019/01/23 22:28:10 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * These are essentially just Tcl wrappers for the measure commands in
 * Measure.C.
 *
 ***************************************************************************/

#include <stdlib.h>
#include <tcl.h>
#include <ctype.h>
#include <math.h>
#include "TclCommands.h"
#include "AtomSel.h"
#include "Matrix4.h"
#include "SymbolTable.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "utilities.h"
#include "config.h"
#include "Measure.h"
#include "MeasureSymmetry.h"
#include "SpatialSearch.h"
#include "Atom.h"
#include "Molecule.h"

// needed by VolInterior commands
#include "QuickSurf.h"
#include "MDFF.h"
#include "CUDAMDFF.h"
#include "MeasureVolInterior.h"


// Get weights from a Tcl obj.  Data must hold sel->selected items, or natoms.
// If there is no obj, or if it's "none", return a list of ones.
// If the obj matches an atom selection keyword/field, and the field returns 
// floating point data, return that data, otherwise return error.
// Otherwise, the obj must be a list of floats to use for the weights,
// with either a size == sel->selected,  or size == natoms.  
//
// NOTE: this routine cannot be used on selections that change between frames.
//
int tcl_get_weights(Tcl_Interp *interp, VMDApp *app, AtomSel *sel, 
                    Tcl_Obj *weight_obj, float *data) {
  char *weight_string = NULL;

  if (!sel) return MEASURE_ERR_NOSEL;
  if (!app->molecule_valid_id(sel->molid())) return MEASURE_ERR_NOMOLECULE;
  if (weight_obj)
    weight_string = Tcl_GetStringFromObj(weight_obj, NULL);

  if (!weight_string || !strcmp(weight_string, "none")) {
    for (int i=0; i<sel->selected; i++) {
      data[i] = 1.0;
    }
    return MEASURE_NOERR;
  }

  // if a selection string was given, check the symbol table
  SymbolTable *atomSelParser = app->atomSelParser; 

  // weights must return floating point values, so the symbol must not 
  // be a singleword, so macro is NULL.
  atomsel_ctxt context(atomSelParser, 
                       app->moleculeList->mol_from_id(sel->molid()), 
                       sel->which_frame, NULL);

  int fctn = atomSelParser->find_attribute(weight_string);
  if (fctn >= 0) {
    // the keyword exists, so get the data if it is type float, otherwise fail
    if (atomSelParser->fctns.data(fctn)->returns_a != SymbolTableElement::IS_FLOAT) {
      Tcl_AppendResult(interp, 
        "weight attribute must have floating point values", NULL);
      return MEASURE_ERR_BADWEIGHTPARM;  // can't understand weight parameter 
    }

    double *tmp_data = new double[sel->num_atoms];
    atomSelParser->fctns.data(fctn)->keyword_double(&context, 
                              sel->num_atoms, tmp_data, sel->on);

    for (int i=sel->firstsel, j=0; i<=sel->lastsel; i++) {
      if (sel->on[i])
        data[j++] = (float)tmp_data[i];
    }

    delete [] tmp_data;
    return MEASURE_NOERR;
  }

  // Determine if weights are a Tcl list with the right number of atoms
  int list_num;
  Tcl_Obj **list_data;
  if (Tcl_ListObjGetElements(interp, weight_obj, &list_num, &list_data) 
      != TCL_OK) {
    return MEASURE_ERR_BADWEIGHTPARM;
  }
  if (list_num != sel->selected && list_num != sel->num_atoms) 
    return MEASURE_ERR_BADWEIGHTNUM;
  
  for (int i=0, j=0; i<list_num; i++) {
    double tmp_data;

    if (Tcl_GetDoubleFromObj(interp, list_data[i], &tmp_data) != TCL_OK) 
      return MEASURE_ERR_NONNUMBERPARM;

    if (list_num == sel->selected) {
      data[i] = (float)tmp_data; // one weight list item per selected atom
    } else {
      if (sel->on[i]) {
        data[j++] = (float)tmp_data; // one weight list item for each atom
      }
    }
  }

  return MEASURE_NOERR;
}


int atomsel_default_weights(AtomSel *sel, float *weights) {
  if (sel->firstsel > 0)
    memset(&weights[0], 0, sel->firstsel * sizeof(float)); 

  for (int i=sel->firstsel; i<=sel->lastsel; i++) {
    // Use the standard "on" check instead of typecasting the array elements
    weights[i] = sel->on[i] ? 1.0f : 0.0f;
  }

  if (sel->lastsel < (sel->num_atoms - 1))
    memset(&weights[sel->lastsel+1], 0, ((sel->num_atoms - 1) - sel->lastsel) * sizeof(float)); 

  return 0;
}


int get_weights_from_tcl_list(Tcl_Interp *interp, VMDApp *app, AtomSel *sel,
                              Tcl_Obj *weights_obj, float *weights) {
  int list_num = 0;
  Tcl_Obj **list_elems = NULL;
  if (Tcl_ListObjGetElements(interp, weights_obj, &list_num, &list_elems)
      != TCL_OK) {
    return MEASURE_ERR_BADWEIGHTPARM;
  }
  if (list_num != sel->num_atoms) {
    return MEASURE_ERR_BADWEIGHTNUM;
  }
  for (int i = 0; i < sel->num_atoms; i++) {
    double tmp_data = 0.0;
    if (Tcl_GetDoubleFromObj(interp, list_elems[i], &tmp_data) != TCL_OK) {
      return TCL_ERROR;
    }
    weights[i] = static_cast<float>(tmp_data);
  }
  return 0;
}


// This routine eliminates the need to include SymbolTable.h,
// and allows the caller to determine if a per-atom attribute/field
// exists or not.
int get_attribute_index(VMDApp *app, char const *string) {
  SymbolTable *atomSelParser = app->atomSelParser;
  return atomSelParser->find_attribute(string);
}


int get_weights_from_attribute(VMDApp *app, AtomSel *sel,
                               char const *weights_string, float *weights) {
  SymbolTable *atomSelParser = app->atomSelParser;
  int fctn = atomSelParser->find_attribute(weights_string);
  // first, check to see that the function returns floats.
  // if it doesn't, it makes no sense to use it as a weight
  if (atomSelParser->fctns.data(fctn)->returns_a !=
      SymbolTableElement::IS_FLOAT) {
    return MEASURE_ERR_BADWEIGHTPARM;  // can't understand weight parameter
  }
  atomsel_ctxt context(atomSelParser,
                       app->moleculeList->mol_from_id(sel->molid()),
                       sel->which_frame, NULL);
  double *tmp_data = new double[sel->num_atoms];
  atomSelParser->fctns.data(fctn)->keyword_double(&context, sel->num_atoms,
                                                  tmp_data, sel->on);

  if (sel->firstsel > 0)
    memset(&weights[0], 0, sel->firstsel * sizeof(float)); 

  for (int i=sel->firstsel; i<=sel->lastsel; i++) {
    weights[i] = sel->on[i] ? static_cast<float>(tmp_data[i]) : 0.0;
  }

  if (sel->lastsel < (sel->num_atoms - 1))
    memset(&weights[sel->lastsel+1], 0, ((sel->num_atoms - 1) - sel->lastsel) * sizeof(float)); 

  delete [] tmp_data;
  return 0;
}


// get the  atom index re-ordering list for use by measure_fit
int tcl_get_orders(Tcl_Interp *interp, int selnum, 
                       Tcl_Obj *order_obj, int *data) {
  int list_num;
  Tcl_Obj **list_data;

  if (Tcl_ListObjGetElements(interp, order_obj, &list_num, &list_data)
      != TCL_OK) {
    return MEASURE_ERR_NOSEL;
  }

  // see if this is a Tcl list with the right number of atom indices
  if (list_num != selnum) return MEASURE_ERR_NOSEL;

  for (int i=0; i<list_num; i++) {
    if (Tcl_GetIntFromObj(interp, list_data[i], &data[i]) != TCL_OK)
      return MEASURE_ERR_NONNUMBERPARM;

    // order indices are 0-based
    if (data[i] < 0 || data[i] >= selnum)
      return MEASURE_ERR_BADORDERINDEX; // order index is out of range
  }

  return MEASURE_NOERR;
}


//
// Function:  vmd_measure_center
// Parameters:  <selection>               // computes with weight == 1
// Parameters:  <selection> weight [none | atom field | list] 
//   computes with the weights based on the following:
//       none       => weights all 1
//       atom field => value from atomSelParser (eg
//                     mass  => use weight based on mass
//                     index => use weight based on atom index (0 to n-1)
//             list => use list to get weights for each atom.  
//                     The list can have length == number of selected atoms,
//                     or it can have length == the total number of atoms
// 
//  Examples: 
//     vmd_measure_center atomselect12
//     vmd_measure_center atomselect12 weight mass
//     vmd_measure_center atomselect12 {12 13} [atomselect top "index 2 3"]
//  If no weight is given, no weight term is used (computes center of number)
//
static int vmd_measure_center(VMDApp *app, int argc, Tcl_Obj *const objv[], Tcl_Interp *interp) {
  if (argc != 2 && argc != 4 ) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<sel> [weight <weights>]");
    return TCL_ERROR;
  }
  if (argc == 4 && strcmp(Tcl_GetStringFromObj(objv[2],NULL), "weight")) {
    Tcl_SetResult(interp, (char *) "measure center: parameter can only be 'weight'", TCL_STATIC);
    return TCL_ERROR;
  }
  
  // get the selection
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_SetResult(interp, (char *) "measure center: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the weight
  float *weight = new float[sel->selected];
  int ret_val=0;
  if (argc == 2) {          // only from atom selection, so weight is 1
    ret_val = tcl_get_weights(interp, app, sel, NULL, weight);
  } else {
    ret_val = tcl_get_weights(interp, app, sel, objv[3], weight);
  }
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure center: ", measure_error(ret_val), NULL);
    delete [] weight;
    return TCL_ERROR;
  }

  // compute the center of "mass"
  float com[3];
  const float *framepos = sel->coordinates(app->moleculeList);
  ret_val = measure_center(sel, framepos, weight, com);
  delete [] weight;
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure center: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(com[0]));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(com[1]));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(com[2]));
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


static int vmd_measure_centerperresidue(VMDApp *app, int argc, Tcl_Obj *const objv[], Tcl_Interp *interp) {
  int ret_val = 0;

  // check argument counts for valid combinations
  if (argc != 2 && argc != 4 ) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<sel> [weight <weights>]");
    return TCL_ERROR;
  }

  // the only valid optional parameter is "weight"
  if (argc == 4 && strcmp(Tcl_GetStringFromObj(objv[2], NULL), "weight")) {
    Tcl_SetResult(interp, (char *) "measure centerperresidue: parameter can only be 'weight'", TCL_STATIC);
    return TCL_ERROR;
  }
  
  // get the selection
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1], NULL));
  if (!sel) {
    Tcl_SetResult(interp, (char *) "measure centerperresidue: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the weight
  float *weight = new float[sel->selected];
  if (argc == 2) {
    // only from atom selection, so weights are all set to 1
    ret_val = tcl_get_weights(interp, app, sel, NULL, weight);
  } else {
    // from a per-atom field or a list
    ret_val = tcl_get_weights(interp, app, sel, objv[3], weight);
  }
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure centerperresidue: ", 
                     measure_error(ret_val), NULL);
    delete [] weight;
    return TCL_ERROR;
  }

  // compute the center of "mass"
  float *com = new float[3*sel->selected];
  const float *framepos = sel->coordinates(app->moleculeList);
  ret_val = measure_center_perresidue(app->moleculeList, sel, 
                                      framepos, weight, com);
  delete [] weight;
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure centerperresidue: ", 
                     measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  // generate lists of CoMs from results
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (int i=0; i<ret_val; i++) {
    Tcl_Obj *m = Tcl_NewListObj(0, NULL);
    for (int j=0; j<3; j++) {
      Tcl_ListObjAppendElement(interp, m, Tcl_NewDoubleObj(com[3*i+j]));
    }
    Tcl_ListObjAppendElement(interp, tcl_result, m);
  }
  Tcl_SetObjResult(interp, tcl_result);
  delete [] com;

  return TCL_OK;
}


// measure sum of weights for selected atoms
static int vmd_measure_sumweights(VMDApp *app, int argc, Tcl_Obj *const objv[], Tcl_Interp *interp) {
  if (argc != 4) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<sel> weight <weights>");
    return TCL_ERROR;
  }
  if (strcmp(Tcl_GetStringFromObj(objv[2],NULL), "weight")) {
    Tcl_SetResult(interp, (char *) "measure sumweights: parameter can only be 'weight'", TCL_STATIC);
    return TCL_ERROR;
  }
 
  // get the selection
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL)
);
  if (!sel) {
    Tcl_SetResult(interp, (char *) "measure sumweights: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the weight
  float *weight = new float[sel->selected];
  int ret_val = 0;
  ret_val = tcl_get_weights(interp, app, sel, objv[3], weight);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure center: ", measure_error(ret_val), NULL);
    delete [] weight;
    return TCL_ERROR;
  }

  // compute the sum of the weights
  float weightsum=0;
  ret_val = measure_sumweights(sel, sel->selected, weight, &weightsum);
  delete [] weight;
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure center: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(weightsum));
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


// Function: vmd_measure_avpos <selection> first <first> last <last> step <step>
//  Returns: the average position of the selected atoms over the selected frames
//  Example: measure avpos atomselect76 0 20 1
static int vmd_measure_avpos(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  // start with first frame by default
  int last = -1;  // finish with last frame by default
  int step = 1;   // use all frames by default

  if (argc < 2 || argc > 8) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> [first <first>] [last <last>] [step <step>]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL)
);
  if (!sel) {
    Tcl_AppendResult(interp, "measure avpos: no atom selection", NULL);
    return TCL_ERROR;
  }

  int i;
  for (i=2; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "measure avpos: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "measure avpos: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "step", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &step) != TCL_OK) {
        Tcl_AppendResult(interp, "measure avpos: bad frame step value", NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "measure avpos: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }

  float *avpos = new float[3L*sel->selected];
  int ret_val = measure_avpos(sel, app->moleculeList, first, last, step, avpos);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure avpos: ", measure_error(ret_val), NULL);
    delete [] avpos;
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (i=0; i<sel->selected; i++) {
    Tcl_Obj *atom = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, atom, Tcl_NewDoubleObj(avpos[i*3L    ]));
    Tcl_ListObjAppendElement(interp, atom, Tcl_NewDoubleObj(avpos[i*3L + 1]));
    Tcl_ListObjAppendElement(interp, atom, Tcl_NewDoubleObj(avpos[i*3L + 2]));
    Tcl_ListObjAppendElement(interp, tcl_result, atom);
  }

  Tcl_SetObjResult(interp, tcl_result);
  delete [] avpos;

  return TCL_OK;
}


// Function: vmd_measure_dipole <selection>
//  Returns: the dipole moment for the selected atoms
static int vmd_measure_dipole(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  const char *opt;
  int unitsdebye=0; // default units are elementary charges/Angstrom
  int usecenter=1;  // remove net charge at the center of mass (-1), geometrical center (1), don't (0)

  if ((argc < 2) || (argc > 4)) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> [-elementary|-debye] [-geocenter|-masscenter|-origincenter]");
    return TCL_ERROR;
  }
  AtomSel *sel;
  sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1], NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure dipole: no atom selection", NULL);
    return TCL_ERROR;
  }

  int i;
  for (i=0; i < (argc-2); ++i) {
    opt = Tcl_GetStringFromObj(objv[2+i], NULL);
    if (!strcmp(opt, "-debye"))
      unitsdebye=1; 
    if (!strcmp(opt, "-elementary"))
      unitsdebye=0; 

    if (!strcmp(opt, "-geocenter"))
      usecenter=1; 
    if (!strcmp(opt, "-masscenter"))
      usecenter=-1; 
    if (!strcmp(opt, "-origincenter"))
      usecenter=0; 
  }

  float dipole[3];
  int ret_val = measure_dipole(sel, app->moleculeList, dipole, unitsdebye, usecenter);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure dipole: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(dipole[0]));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(dipole[1]));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(dipole[2]));
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}



// Function: vmd_measure_dihed {4 atoms as {<atomid> ?<molid>?}} ?molid <default molid>?
//                             ?frame [f|all|last]? | ?first <first>? ?last <last>?
//  Returns: the dihedral angle for the specified atoms
static int vmd_measure_dihed(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first=-1, last=-1, frame=-1;
  if (argc < 2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "{{<atomid1> [<molid1>]} {<atomid2> [<molid2>]} {<atomid3> [<molid3>]} {<atomid4> [<molid4>]}} [molid <default molid>] [frame <frame|all|last> | first <first> last <last>]");
    return TCL_ERROR;
  }

  int molid[4], atmid[4], defmolid = -1;
  bool allframes = false;

  // Get the geometry type dihed/imprp
  char *geomname = Tcl_GetStringFromObj(objv[0],NULL);

  // Read the atom list
  int numatms;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &numatms, &data) != TCL_OK) {
    Tcl_AppendResult(interp, " measure ", geomname, ": bad syntax", NULL);
    Tcl_AppendResult(interp, " Usage: measure ", geomname, " {{<atomid1> [<molid1>]} {<atomid2> [<molid2>]} {<atomid3> [<molid3>]} {<atomid4> [<molid4>]}} [molid <default molid>] [frame <frame|all|last> | first <first> last <last>]", NULL);
    return TCL_ERROR;
  }

  if (numatms != 4) {
    Tcl_AppendResult(interp, " measure dihed: must specify exactly four atoms in a list", NULL);
    return TCL_ERROR;
  }

  if (argc > 3) {
    int i;
    for (i=2; i<argc; i+=2) {
      char *argvcur = Tcl_GetStringFromObj(objv[i], NULL);
      if (!strupncmp(argvcur, "molid", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &defmolid) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure ", geomname, ": bad molid", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "first", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure ", geomname, ": bad first frame value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "last", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure ", geomname, ": bad last frame value", NULL);
	  return TCL_ERROR;
	}
     } else if (!strupncmp(argvcur, "frame", CMDLEN)) {
	if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "all")) {
          allframes = true;
        } else if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "last")) {
          frame=-2;
        } else if (Tcl_GetIntFromObj(interp, objv[i+1], &frame) != TCL_OK) {
          Tcl_AppendResult(interp, " measure ", geomname, ": bad frame value", NULL);
          return TCL_ERROR;
        }
      } else {
        Tcl_AppendResult(interp, "measure ", geomname, ": invalid syntax, no such keyword: ", argvcur, NULL);
        return TCL_ERROR;
      }
    }
  }

  if ((allframes || frame>=0) && (first>=0 || last>=0)) {
    Tcl_AppendResult(interp, "measure ", geomname, ": Ambiguous syntax: You cannot specify a frame AND a frame range (using first or last).", NULL);
    Tcl_AppendResult(interp, "\nUsage:\nmeasure ", geomname, " {<atomid1> [<molid1>]} {<atomid2> [<molid2>]} {<atomid3> [<molid3>]} {<atomid4> [<molid4>]}} [molid <default molid>] [frame <frame|all|last> | first <first> last <last>]", NULL);
    return TCL_ERROR;    
  }

  if (allframes) first=0;

  // If no molecule was specified use top as default
  if (defmolid<0) defmolid = app->molecule_top();

  // Assign atom IDs and molecule IDs
  int i,numelem;
  Tcl_Obj **atmmol;
  for (i=0; i<numatms; i++) {
    if (Tcl_ListObjGetElements(interp, data[i], &numelem, &atmmol) != TCL_OK) {
      return TCL_ERROR;
    }

    if (!numelem) {
      Tcl_AppendResult(interp, " measure ", geomname, ": empty atom index", NULL);
      return TCL_ERROR;
    }

    if (Tcl_GetIntFromObj(interp, atmmol[0], atmid+i) != TCL_OK) {
      Tcl_AppendResult(interp, " measure ", geomname, ": bad atom index", NULL);
      return TCL_ERROR;
    }
    
    if (numelem==2) {
      if (Tcl_GetIntFromObj(interp, atmmol[1], molid+i) != TCL_OK) {
	Tcl_AppendResult(interp, " measure ", geomname, ": bad molid", NULL);
	return TCL_ERROR;
      }
    } else molid[i] = defmolid;
  }

  // Compute the value
  ResizeArray<float> gValues(1024);
  int ret_val;
  ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, 
                         frame, first, last, defmolid, MEASURE_DIHED);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  int numvalues = gValues.num();
  for (int count = 0; count < numvalues; count++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(gValues[count]));
  }
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


// Function: vmd_measure_angle {3 atoms as {<atomid> ?<molid>?}} ?molid <default molid>? 
//                             ?frame [f|all|last]? | ?first <first>? ?last <last>?
//  Returns: the bond angle for the specified atoms
static int vmd_measure_angle(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first=-1, last=-1, frame=-1;

  if(argc<2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "{{<atomid1> [<molid1>]} {<atomid2> [<molid2>]} {<atomid3> [<molid3>]}} [molid <default molid>] [frame <frame|all|last> | first <first> last <last>]");
    return TCL_ERROR;
  }

  int molid[3], atmid[3], defmolid = -1;
  bool allframes = false;

  // Read the atom list
  int numatms;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &numatms, &data) != TCL_OK) {
    Tcl_AppendResult(interp, " measure bond: bad syntax", NULL);
    Tcl_AppendResult(interp, " Usage: measure angle {{<atomid1> [<molid1>]} {<atomid2> [<molid2>]} {<atomid3> [<molid3>]}} [molid <default molid>] [frame <frame|all|last> | first <first> last <last>]", NULL);
    return TCL_ERROR;
  }

  if (numatms != 3) {
    Tcl_AppendResult(interp, " measure angle: must specify exactly three atoms in a list", NULL);
    return TCL_ERROR;
  }

  if (argc > 3) {
    for (int i=2; i<argc; i+=2) {
      char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
      if (!strupncmp(argvcur, "molid", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &defmolid) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure angle: bad molid", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "first", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure angle: bad first frame value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "last", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure angle: bad last frame value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "frame", CMDLEN)) {
	if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "all")) {
	  allframes = true;
	} else if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "last")) {
	  frame=-2;
	} else if (Tcl_GetIntFromObj(interp, objv[i+1], &frame) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure angle: bad frame value", NULL);
	  return TCL_ERROR;
	}
      } else {
	Tcl_AppendResult(interp, " measure angle: invalid syntax, no such keyword: ", argvcur, NULL);
	return TCL_ERROR;
      }
    }
  }

  if ((allframes || frame>=0) && (first>=0 || last>=0)) {
    Tcl_AppendResult(interp, "measure angle: Ambiguous syntax: You cannot specify a frame AND a frame range (using first or last).", NULL);
    Tcl_AppendResult(interp, "\nUsage:\nmeasure angle {<atomid1> [<molid1>]} {<atomid2> [<molid2>]} {<atomid3> [<molid3>]}} [molid <default molid>] [frame <frame|all|last> | first <first> last <last>]", NULL);
    return TCL_ERROR;    
  }

  if (allframes) first=0;

  // If no molecule was specified use top as default
  if (defmolid<0) defmolid = app->molecule_top();

  // Assign atom IDs and molecule IDs
  int i,numelem;
  Tcl_Obj **atmmol;
  for (i=0; i<numatms; i++) {
    if (Tcl_ListObjGetElements(interp, data[i], &numelem, &atmmol) != TCL_OK) {
      return TCL_ERROR;
    }

    if (!numelem) {
      Tcl_AppendResult(interp, " measure angle: empty atom index", NULL);
      return TCL_ERROR;
    }

    if (Tcl_GetIntFromObj(interp, atmmol[0], atmid+i) != TCL_OK) {
      Tcl_AppendResult(interp, " measure angle: bad atom index", NULL);
      return TCL_ERROR;
    }
    
    if (numelem==2) {
      if (Tcl_GetIntFromObj(interp, atmmol[1], molid+i) != TCL_OK) {
	Tcl_AppendResult(interp, " measure angle: bad molid", NULL);
	return TCL_ERROR;
      }
    } else molid[i] = defmolid;
  }

  // Compute the value
  ResizeArray<float> gValues(1024);
  int ret_val;
  ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, 
                         frame, first, last, defmolid, MEASURE_ANGLE);
  if (ret_val<0) {
    printf("ERROR\n %s\n", measure_error(ret_val));
    Tcl_AppendResult(interp, measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  int numvalues = gValues.num();
  for (int count = 0; count < numvalues; count++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(gValues[count]));
  }
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


// vmd_measure_bond {2 atoms as {<atomid> ?<molid>?}} ?molid <molid>? ?frame [f|all|last]? | ?first <first>? ?last <last>?
// Returns the bond length for the specified atoms
static int vmd_measure_bond(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first=-1, last=-1, frame=-1;

  if (argc<2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "{{<atomid1> [<molid1>]} {<atomid2> [<molid2>]}} [molid <default molid>] [frame <frame|all|last> | first <first> last <last>]]");
    return TCL_ERROR;
  }
  
  int molid[2], atmid[2], defmolid = -1;
  bool allframes = false;

  // Read the atom list
  int numatms=0;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[1], &numatms, &data) != TCL_OK) {
    Tcl_AppendResult(interp, " measure bond: bad syntax", NULL);
    Tcl_AppendResult(interp, " Usage: measure bond {{<atomid1> [<molid1>]} {<atomid2> [<molid2>]}} [molid <default>] [frame <frame|all|last> | first <first> last <last>]]", NULL);
    return TCL_ERROR;
  }

  if (numatms != 2) {
    Tcl_AppendResult(interp, " measure bond: must specify exactly two atoms in a list", NULL);
    return TCL_ERROR;
  }

  if (argc > 3) {
    for (int i=2; i<argc; i+=2) {
      char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
      if (!strupncmp(argvcur, "molid", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &defmolid) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure bond: bad molid", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "first", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure bond: bad first frame value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "last", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure bond: bad last frame value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "frame", CMDLEN)) {
	if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "all")) {
	  allframes = true;
	} else if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "last")) {
	  frame=-2;
	} else if (Tcl_GetIntFromObj(interp, objv[i+1], &frame) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure bond: bad frame value", NULL);
	  return TCL_ERROR;
	}
      } else {
	Tcl_AppendResult(interp, " measure bond: invalid syntax, no such keyword: ", argvcur, NULL);
	return TCL_ERROR;
      }
    }
  }

  if ((allframes || frame>=0) && (first>=0 || last>=0)) {
    Tcl_AppendResult(interp, "measure bond: Ambiguous syntax: You cannot specify a frame AND a frame range (using first or last).", NULL);
    Tcl_AppendResult(interp, "\nUsage:\nmeasure bond {{<atomid1> [<molid1>]} {<atomid2> [<molid2>]}} [molid <default>] [frame <frame|all|last> | first <first> last <last>]", NULL);
    return TCL_ERROR;    
  }

  if (allframes) first=0;

  // If no molecule was specified use top as default
  if (defmolid<0) defmolid = app->molecule_top();

  // Assign atom IDs and molecule IDs
  int i, numelem;
  Tcl_Obj **atmmol;
  for (i=0; i<numatms; i++) {
    if (Tcl_ListObjGetElements(interp, data[i], &numelem, &atmmol) != TCL_OK) {
      return TCL_ERROR;
    }

    if (!numelem) {
      Tcl_AppendResult(interp, " measure bond: empty atom index", NULL);
      return TCL_ERROR;
    }

    if (Tcl_GetIntFromObj(interp, atmmol[0], atmid+i) != TCL_OK) {
      Tcl_AppendResult(interp, " measure bond: bad atom index", NULL);
      return TCL_ERROR;
    }
    
    if (numelem==2) {
      if (Tcl_GetIntFromObj(interp, atmmol[1], molid+i) != TCL_OK) {
	Tcl_AppendResult(interp, " measure bond: bad molid", NULL);
	return TCL_ERROR;
      }
    } else molid[i] = defmolid;
  }

  // Compute the value
  ResizeArray<float> gValues(1024);
  int ret_val;
  ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, 
                         frame, first, last, defmolid, MEASURE_BOND);
  if (ret_val < 0) {
    printf("ERROR\n %s\n", measure_error(ret_val));
    Tcl_AppendResult(interp, measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  int numvalues = gValues.num();
  for (int count = 0; count < numvalues; count++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(gValues[count]));
  }
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


// vmd_measure_rmsfperresidue <selection> first <first> last <last> step <step>
// Returns: position variance of the selected atoms over the selected frames, 
// returning one value per residue in the selection.
// Example: measure rmsfperresidue atomselect76 0 20 1
static int vmd_measure_rmsfperresidue(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  // start with first frame by default
  int last = -1;  // finish with last frame by default
  int step = 1;   // use all frames by default

  if (argc < 2 || argc > 8) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<sel> [first <first>] [last <last>] [step <step>]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL)
);
  if (!sel) {
    Tcl_AppendResult(interp, "measure rmsfperresidue: no atom selection", NULL);
    return TCL_ERROR;
  }
 
  int i;
  for (i=2; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsfperresidue: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsfperresidue: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "step", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &step) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsfperresidue: bad frame step value", NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "measure rmsfperresidue: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }

  float *rmsf = new float[sel->selected];
  int ret_val = measure_rmsf_perresidue(sel, app->moleculeList, first, last, step, rmsf);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rmsfperresidue: ", measure_error(ret_val), NULL);
    delete [] rmsf;
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (i=0; i<ret_val; i++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(rmsf[i]));
  }
  Tcl_SetObjResult(interp, tcl_result);
  delete [] rmsf;

  return TCL_OK;
}


// Function: vmd_measure_rmsf <selection> first <first> last <last> step <step>
//  Returns: position variance of the selected atoms over the selected frames
//  Example: measure rmsf atomselect76 0 20 1
static int vmd_measure_rmsf(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  // start with first frame by default
  int last = -1;  // finish with last frame by default
  int step = 1;   // use all frames by default

  if (argc < 2 || argc > 8) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<sel> [first <first>] [last <last>] [step <step>]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL)
);
  if (!sel) {
    Tcl_AppendResult(interp, "measure rmsf: no atom selection", NULL);
    return TCL_ERROR;
  }
 
  int i;
  for (i=2; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsf: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsf: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "step", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &step) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsf: bad frame step value", NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "measure rmsf: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }

  float *rmsf = new float[sel->selected];
  int ret_val = measure_rmsf(sel, app->moleculeList, first, last, step, rmsf);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rmsf: ", measure_error(ret_val), NULL);
    delete [] rmsf;
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (i=0; i<sel->selected; i++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(rmsf[i]));
  }
  Tcl_SetObjResult(interp, tcl_result);

  delete [] rmsf;
  return TCL_OK;
}


// measure radius of gyration for selected atoms
static int vmd_measure_rgyr(VMDApp *app, int argc, Tcl_Obj *const objv[], Tcl_Interp *interp) {
  if (argc != 2 && argc != 4) {
    Tcl_WrongNumArgs(interp, 2, objv-1,
                     (char *)"<selection> [weight <weights>]");
    return TCL_ERROR;
  }
  if (argc == 4 && strcmp(Tcl_GetStringFromObj(objv[2],NULL), "weight")) {
    Tcl_SetResult(interp, (char *) "measure rgyr: parameter can only be 'weight'", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the selection
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_SetResult(interp, (char *) "measure rgyr: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the weight
  float *weight = new float[sel->selected];
  int ret_val = 0;
  if (argc == 2) {          // only from atom selection, so weight is 1
    ret_val = tcl_get_weights(interp, app, sel, NULL, weight);
  } else {
    ret_val = tcl_get_weights(interp, app, sel, objv[3], weight);
  }
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rgyr: ", measure_error(ret_val), NULL);
    delete [] weight;
    return TCL_ERROR;
  }

  float rgyr;
  ret_val = measure_rgyr(sel, app->moleculeList, weight, &rgyr);
  delete [] weight;
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rgyr: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }
  Tcl_SetObjResult(interp, Tcl_NewDoubleObj(rgyr));
  return TCL_OK;
}


// Function: vmd_measure_minmax <selection>
//  Returns: the cartesian range of a selection (min/max){x,y,z}
//  Example: vmd_measure_minmax atomselect76
//     {-5 0 0} {15 10 11.2}
static int vmd_measure_minmax(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  const float *radii = NULL;
  if (argc != 2 && argc != 3) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<selection> [-withradii]");
    return TCL_ERROR;
  }
  if (argc == 3 && strcmp(Tcl_GetStringFromObj(objv[2],NULL), "-withradii")) {
    Tcl_SetResult(interp, (char *) "measure minmax: parameter can only be '-withradii'", TCL_STATIC);
    return TCL_ERROR;
  }

  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure minmax: no atom selection", NULL);
    return TCL_ERROR;
  }

  float min_coord[3], max_coord[3];
  const float *framepos = sel->coordinates(app->moleculeList);
  if (!framepos) return TCL_ERROR;

  // get atom radii if requested
  if (argc == 3) {
    Molecule *mol = app->moleculeList->mol_from_id(sel->molid());
    radii = mol->extraflt.data("radius");
  } 

  int ret_val = measure_minmax(sel->num_atoms, sel->on, framepos, radii, 
                               min_coord, max_coord);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure minmax: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *list1 = Tcl_NewListObj(0, NULL);
  Tcl_Obj *list2 = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);

  Tcl_ListObjAppendElement(interp, list1, Tcl_NewDoubleObj(min_coord[0]));
  Tcl_ListObjAppendElement(interp, list1, Tcl_NewDoubleObj(min_coord[1]));
  Tcl_ListObjAppendElement(interp, list1, Tcl_NewDoubleObj(min_coord[2]));

  Tcl_ListObjAppendElement(interp, list2, Tcl_NewDoubleObj(max_coord[0]));
  Tcl_ListObjAppendElement(interp, list2, Tcl_NewDoubleObj(max_coord[1]));
  Tcl_ListObjAppendElement(interp, list2, Tcl_NewDoubleObj(max_coord[2]));

  Tcl_ListObjAppendElement(interp, tcl_result, list1);
  Tcl_ListObjAppendElement(interp, tcl_result, list2);
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


static int vmd_measure_rmsdperresidue(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc !=3 && argc != 5) {
    Tcl_WrongNumArgs(interp, 2, objv-1, 
                     (char *)"<sel1> <sel2> [weight <weights>]");
    return TCL_ERROR;
  }

  // get the selections
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  AtomSel *sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2],NULL));
  if (!sel1 || !sel2) {
    Tcl_AppendResult(interp, "measure rmsd: no atom selection", NULL);
    return TCL_ERROR;
  }

  if (sel1->selected != sel2->selected) {
    Tcl_AppendResult(interp, "measure rmsd: selections must have the same number of atoms", NULL);
    return TCL_ERROR;
  }
  if (!sel1->selected) {
    Tcl_AppendResult(interp, "measure rmsd: no atoms selected", NULL);
    return TCL_ERROR;
  }
  float *weight = new float[sel1->selected];
  int ret_val = 0;
  if (argc == 3) {
    ret_val = tcl_get_weights(interp, app, sel1, NULL, weight);
  } else {
    ret_val = tcl_get_weights(interp, app, sel1, objv[4], weight);
  }
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rmsd: ", measure_error(ret_val), NULL);
    delete [] weight;
    return TCL_ERROR;
  }

  // compute the rmsd
  float *rmsd = new float[sel1->selected];
  int rc = measure_rmsd_perresidue(sel1, sel2, app->moleculeList, 
                                   sel1->selected, weight, rmsd);
  if (rc < 0) {
    Tcl_AppendResult(interp, "measure rmsd: ", measure_error(rc), NULL);
    delete [] weight;
    delete [] rmsd;
    return TCL_ERROR;
  }
  delete [] weight;
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  int i;
  for (i=0; i<rc; i++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(rmsd[i]));
  }
  delete [] rmsd;
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}

// Function: vmd_measure_rmsd <selection1> <selection2> 
//                     {weight [none|atom value|string array]}
//
// Returns the RMSD between the two selection, taking the weight (if
// any) into account.  If number of elements in the weight != num_atoms
// in sel1 then (num in weight = num selected in sel1 = num selected in
// sel2) else (num in weight = total num in sel1 = total num in sel2).
// The weights are taken from the FIRST selection, if needed
// 
// Examples:
//   set sel1 [atomselect 0 all]
//   set sel2 [atomselect 1 all]
//   measure rmsd $sel1 $sel2
//   measure rmsd $sel1 $sel2 weight mass
//   set sel3 [atomselect 0 "index 3 4 5"]
//   set sel4 [atomselect 1 "index 8 5 9"]    # gets turned to 5 8 9
//   measure rmsd $sel3 $sel4 weight occupancy
//
static int vmd_measure_rmsd(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp)
{
  if (argc !=3 && argc != 5) {
    Tcl_WrongNumArgs(interp, 2, objv-1, 
      (char *)"<sel1> <sel2> [weight <weights>]");
    return TCL_ERROR;
  }

  // get the selections
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  AtomSel *sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2],NULL));
  if (!sel1 || !sel2) {
    Tcl_AppendResult(interp, "measure rmsd: no atom selection", NULL);
    return TCL_ERROR;
  }

  if (sel1->selected != sel2->selected) {
    Tcl_AppendResult(interp, "measure rmsd: selections must have the same number of atoms", NULL);
    return TCL_ERROR;
  }
  if (!sel1->selected) {
    Tcl_AppendResult(interp, "measure rmsd: no atoms selected", NULL);
    return TCL_ERROR;
  }

  float *weight = new float[sel1->selected];
  int ret_val = 0;
  if (argc == 3) {
    ret_val = tcl_get_weights(interp, app, sel1, NULL, weight);
  } else {
    ret_val = tcl_get_weights(interp, app, sel1, objv[4], weight);
  }
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rmsd: ", measure_error(ret_val), NULL);
    delete [] weight;
    return TCL_ERROR;
  }

  // compute the rmsd
  float rmsd = 0;
  const float *x = sel1->coordinates(app->moleculeList);
  const float *y = sel2->coordinates(app->moleculeList);
  if (!x || !y) {
    delete [] weight;
    return TCL_ERROR;
  }

  ret_val = measure_rmsd(sel1, sel2, sel1->selected, x, y, weight, &rmsd);
  delete [] weight;
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rmsd: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }
  Tcl_SetObjResult(interp, Tcl_NewDoubleObj(rmsd));

  return TCL_OK;
}


//////////////////////////////////////////////
// measure rmsd_qcp $sel1 $sel2 [weight <weights>]
static int vmd_measure_rmsd_qcp(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc !=3 && argc != 5) {
    Tcl_WrongNumArgs(interp, 2, objv-1, 
      (char *)"<sel1> <sel2> [weight <weights>]");
    return TCL_ERROR;
  }
  // get the selections
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  AtomSel *sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2],NULL));
  if (!sel1 || !sel2) {
    Tcl_AppendResult(interp, "measure rmsd: no atom selection", NULL);
    return TCL_ERROR;
  }

  if (sel1->selected != sel2->selected) {
    Tcl_AppendResult(interp, "measure rmsd: selections must have the same number of atoms", NULL);
    return TCL_ERROR;
  }
  if (!sel1->selected) {
    Tcl_AppendResult(interp, "measure rmsd: no atoms selected", NULL);
    return TCL_ERROR;
  }
  float *weight = new float[sel1->selected];
  {
    int ret_val;
    if (argc == 3) {
      ret_val = tcl_get_weights(interp, app, sel1, NULL, weight);
    } else {
      ret_val = tcl_get_weights(interp, app, sel1, objv[4], weight);
    }
    if (ret_val < 0) {
      Tcl_AppendResult(interp, "measure rmsd: ", measure_error(ret_val),
		       NULL);
      delete [] weight;
      return TCL_ERROR;
    }
  }

  // compute the rmsd
  {
    float rmsd = 0;
    const float *x = sel1->coordinates(app->moleculeList);
    const float *y = sel2->coordinates(app->moleculeList);
    if (!x || !y) {
      delete [] weight;
      return TCL_ERROR;
    }
    int ret_val = measure_rmsd_qcp(app, sel1, sel2, sel1->selected, x, y, weight, &rmsd);
    delete [] weight;
    if (ret_val < 0) {
      Tcl_AppendResult(interp, "measure rmsd: ", measure_error(ret_val),
		       NULL);
      return TCL_ERROR;
    }
    Tcl_SetObjResult(interp, Tcl_NewDoubleObj(rmsd));
  }
  return TCL_OK;
}


//////////////////////////////////////////////
// measure rmsdmat_qcp $sel1 [weight <weights>] start s  end e  step s
static int vmd_measure_rmsdmat_qcp(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  // start with first frame by default
  int last = -1;  // finish with last frame by default
  int step = 1;   // use all frames by default

  if (argc < 2 || argc > 9) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> [weight <weights>]  [first <first>] [last <last>] [step <step>]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL)
);
  if (!sel) {
    Tcl_AppendResult(interp, "measure rmsdmat_qcp: no atom selection", NULL);
    return TCL_ERROR;
  }

  int i;
  for (i=2; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsdmat_qcp: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsdmat_qcp: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "step", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &step) != TCL_OK) {
        Tcl_AppendResult(interp, "measure rmsdmat_qcp: bad frame step value", NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "measure avpos: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }

  float *weight = NULL;
  if (0)  {
    weight = new float[sel->selected];
    int ret_val;
    if (argc == 2) {
      ret_val = tcl_get_weights(interp, app, sel, NULL, weight);
    } else {
      ret_val = tcl_get_weights(interp, app, sel, objv[3], weight);
    }

    if (ret_val < 0) {
      Tcl_AppendResult(interp, "measure rmsdmat_qcp: ", measure_error(ret_val),
		       NULL);
      delete [] weight;
      return TCL_ERROR;
    }
  }


  int framecount = (last - first + 1) / step;
  float *rmsdmat = (float *) calloc(1, framecount * framecount * sizeof(float));

  // compute the rmsd matrix
  int ret_val = measure_rmsdmat_qcp(app, sel, app->moleculeList, 
                                    sel->selected, weight, 
                                    first, last, step, rmsdmat);
  delete [] weight;

  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure rmsdmat_qcp: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  long j;
  for (j=0; j<framecount; j++) {
    Tcl_Obj *rmsdlist = Tcl_NewListObj(0, NULL);
    for (i=0; i<framecount; i++) {
      Tcl_ListObjAppendElement(interp, rmsdlist, 
                               Tcl_NewDoubleObj(rmsdmat[j*framecount + i]));
    }
    Tcl_ListObjAppendElement(interp, tcl_result, rmsdlist);
  }
  Tcl_SetObjResult(interp, tcl_result);

  free(rmsdmat);

  return TCL_OK;
}




//////////////////////////////////////////////
// measure fit $sel1 $sel2 [weight <weights>][
static int vmd_measure_fit(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp)
{
  AtomSel *sel1, *sel2;
  int *order = NULL;
  float *weight = NULL;
  int rc;

  if (argc != 3 && argc != 5 
      && argc != 7) {
    Tcl_WrongNumArgs(interp, 2, objv-1, 
       (char *)"<sel1> <sel2> [weight <weights>] [order <index list>]");
    return TCL_ERROR;
  } else if (argc == 5
             && strcmp("weight", Tcl_GetStringFromObj(objv[3], NULL)) 
             && strcmp("order", Tcl_GetStringFromObj(objv[3], NULL))) {
    Tcl_WrongNumArgs(interp, 2, objv-1, 
       (char *)"<sel1> <sel2> [weight <weights>] [order <index list>]");
    return TCL_ERROR;
  } else if (argc == 7 && 
             (strcmp("weight", Tcl_GetStringFromObj(objv[3], NULL)) || 
              strcmp("order", Tcl_GetStringFromObj(objv[5], NULL)))) {
    Tcl_WrongNumArgs(interp, 2, objv-1, 
       (char *)"<sel1> <sel2> [weight <weights>] [order <index list>]");
    return TCL_ERROR;
  }

  // get the selections
  sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2],NULL));

  if (!sel1 || !sel2) {
    Tcl_AppendResult(interp, "measure fit: no atom selection", NULL);
    return TCL_ERROR;
  }

  int num = sel1->selected;
  if (!num) {
    Tcl_AppendResult(interp, "measure fit: no atoms selected", NULL);
    return TCL_ERROR;
  }
  if (num != sel2->selected) {
    Tcl_AppendResult(interp, "measure fit: selections must have the same number of atoms", NULL);
    return TCL_ERROR;
  }

  // get the weights
  weight = new float[num]; 
  if (argc > 3 && !strcmp("weight", Tcl_GetStringFromObj(objv[3], NULL))) {
    // get user weight parameter
    rc = tcl_get_weights(interp, app, sel1, objv[4], weight);
  } else {
    // default to weights of 1.0
    rc = tcl_get_weights(interp, app, sel1, NULL, weight);
  }
  if (rc < 0) {
    Tcl_AppendResult(interp, "measure fit: ", measure_error(rc), NULL);
    delete [] weight;
    return TCL_ERROR;
  }


  int orderparam = 0;
  if (argc == 5 && !strcmp("order", Tcl_GetStringFromObj(objv[3], NULL))) {
    orderparam = 4;
  } else if (argc == 7 && !strcmp("order", Tcl_GetStringFromObj(objv[5], NULL))) {
    orderparam = 6;
  }

  if (orderparam != 0) {
    // get the atom order
    order = new int[num];
    rc = tcl_get_orders(interp, num, objv[orderparam], order);
    if (rc < 0) {
      Tcl_AppendResult(interp, "measure fit: ", measure_error(rc), NULL);
      delete [] order;
      return TCL_ERROR;
    }
  }

  // compute the transformation matrix
  Matrix4 T;
  const float *x = sel1->coordinates(app->moleculeList);
  const float *y = sel2->coordinates(app->moleculeList);

  int ret_val = MEASURE_ERR_NOMOLECULE;
  if (x && y) 
    ret_val = measure_fit(sel1, sel2, x, y, weight, order, &T);

  delete [] weight;

  if (order != NULL)
    delete [] order;

  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure fit: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  // and return the matrix
  tcl_append_matrix(interp, T.mat);
  return TCL_OK;
}

//////////////////////////////////
// measure inverse 4x4matrix
static int vmd_measure_inverse(int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  // make there there is exactly one matrix
  if (argc != 2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<matrix>");
    return TCL_ERROR;
  }
  // Get the first matrix
  Matrix4 inv;
  if (tcl_get_matrix("measure inverse: ",interp,objv[1],inv.mat) != TCL_OK) {
    return TCL_ERROR;
  }
  if (inv.inverse()) {
    Tcl_AppendResult(interp, "Singular Matrix; inverse not computed", NULL);
    return TCL_ERROR;
  }
  tcl_append_matrix(interp, inv.mat);
  return TCL_OK;
}

// Find all atoms p in sel1 and q in sel2 within the cutoff.  
static int vmd_measure_contacts(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  
  // Cutoff, and either one or two atom selections
  if (argc != 3 && argc != 4) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<cutoff> <sel1> [<sel2>]");
    return TCL_ERROR;
  }
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2],NULL));
  if (!sel1) {
    Tcl_AppendResult(interp, "measure contacts: no atom selection", NULL);
    return TCL_ERROR;
  }
  AtomSel *sel2 = NULL;
  if (argc == 4) {
    sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[3],NULL));
    if (!sel2) {
      Tcl_AppendResult(interp, "measure contacts: no atom selection", NULL);
      return TCL_ERROR;
    }
  }
  if (!sel2) sel2 = sel1;
  
  double cutoff;
  if (Tcl_GetDoubleFromObj(interp, objv[1], &cutoff) != TCL_OK)
    return TCL_ERROR;

  const float *pos1 = sel1->coordinates(app->moleculeList);
  const float *pos2 = sel2->coordinates(app->moleculeList);
  if (!pos1 || !pos2) {
    Tcl_AppendResult(interp, "measure contacts: error, molecule contains no coordinates", NULL);
    return TCL_ERROR;
  }
  Molecule *mol1 = app->moleculeList->mol_from_id(sel1->molid());
  Molecule *mol2 = app->moleculeList->mol_from_id(sel2->molid());

  GridSearchPair *pairlist = vmd_gridsearch3(pos1, sel1->num_atoms, sel1->on, pos2, sel2->num_atoms, sel2->on, (float) cutoff, -1, (sel1->num_atoms + sel2->num_atoms) * 27L);
  GridSearchPair *p, *tmp;
  Tcl_Obj *list1 = Tcl_NewListObj(0, NULL);
  Tcl_Obj *list2 = Tcl_NewListObj(0, NULL);
  for (p=pairlist; p != NULL; p=tmp) {
    // throw out pairs that are already bonded
    MolAtom *a1 = mol1->atom(p->ind1);
    if (mol1 != mol2 || !a1->bonded(p->ind2)) {
      Tcl_ListObjAppendElement(interp, list1, Tcl_NewIntObj(p->ind1));
      Tcl_ListObjAppendElement(interp, list2, Tcl_NewIntObj(p->ind2));
    }
    tmp = p->next;
    free(p);
  }
  Tcl_Obj *result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, result, list1);
  Tcl_ListObjAppendElement(interp, result, list2);
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}


// measure g(r) for two selections, with delta, rmax, usepbc, first/last/step 
// frame parameters the code will compute the normalized histogram.
static int vmd_measure_gofr(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int i;
  // initialize optional arguments to default values
  double rmax=10.0;
  double delta=0.1;
  int usepbc=0;
  int selupdate=0;
  int first=-1, last=-1, step=1;
  int rc;

  // argument error message
  const char *argerrmsg = "<sel1> <sel2> [delta <value>] [rmax <value>] [usepbc <bool>] [selupdate <bool>] [first <first>] [last <last>] [step <step>]";

  // Two atom selections and optional keyword/value pairs.
  if ((argc < 3) || (argc > 17) || (argc % 2 == 0) )  {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
    return TCL_ERROR;
  }

  // check atom selections
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1], NULL));
  if (!sel1) {
    Tcl_AppendResult(interp, "measure gofr: invalid first atom selection", NULL);
    return TCL_ERROR;
  }

  AtomSel *sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2], NULL));
  if (!sel2) {
    Tcl_AppendResult(interp, "measure gofr: invalid second atom selection", NULL);
    return TCL_ERROR;
  }

  // parse optional arguments
  for (i=3; i<argc; i+=2) {
    const char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (i==(argc-1)) {
      Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
      return TCL_ERROR;
    }
    if (!strcmp(opt, "delta")) {
      if (Tcl_GetDoubleFromObj(interp, objv[i+1], &delta) != TCL_OK)
        return TCL_ERROR;
      if (delta <= 0.0) {
        Tcl_AppendResult(interp, "measure gofr: invalid 'delta' value", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "rmax")) {
      if (Tcl_GetDoubleFromObj(interp, objv[i+1], &rmax) != TCL_OK)
        return TCL_ERROR;
      if (rmax <= 0.0) {
        Tcl_AppendResult(interp, "measure gofr: invalid 'rmax' value", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "usepbc")) {
      if (Tcl_GetBooleanFromObj(interp, objv[i+1], &usepbc) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "selupdate")) {
      if (Tcl_GetBooleanFromObj(interp, objv[i+1], &selupdate) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "first")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "last")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "step")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &step) != TCL_OK)
        return TCL_ERROR;
    } else { // unknown keyword.
      Tcl_AppendResult(interp, "unknown keyword '", opt, "'. usage: measure gofr ", argerrmsg, NULL);
      return TCL_ERROR;
    }
  }

  // allocate and initialize histogram arrays
  int    count_h = (int)(rmax / delta + 1.0);
  double *gofr   = new double[count_h];
  double *numint = new double[count_h];
  double *histog = new double[count_h];
  int *framecntr = new int[3];

  // do the gofr calculation
  rc = measure_gofr(sel1, sel2, app->moleculeList,
               count_h, gofr, numint, histog,
               (float) delta,
               first, last, step, framecntr,
               usepbc, selupdate);

  // XXX: this needs a 'case' structure to provide more meaninful error messages.
  if (rc != MEASURE_NOERR) { 
    Tcl_AppendResult(interp, "measure gofr: error during g(r) calculation.", NULL);
    return TCL_ERROR;
  }

  // convert the results of the lowlevel call to tcl lists
  // and build a list from them as return value.
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_rlist  = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_gofr   = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_numint = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_histog = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_frames = Tcl_NewListObj(0, NULL);

  // build lists with results ready for plotting
  for (i=0; i<count_h; i++) { 
    Tcl_ListObjAppendElement(interp, tcl_rlist,  Tcl_NewDoubleObj(delta * ((double)i + 0.5)));
    Tcl_ListObjAppendElement(interp, tcl_gofr,   Tcl_NewDoubleObj(gofr[i]));
    Tcl_ListObjAppendElement(interp, tcl_numint, Tcl_NewDoubleObj(numint[i]));
    Tcl_ListObjAppendElement(interp, tcl_histog, Tcl_NewDoubleObj(histog[i]));
  }

  // build list with number of frames: 
  // total, skipped and processed (one entry for each algorithm).
  Tcl_ListObjAppendElement(interp, tcl_frames, Tcl_NewIntObj(framecntr[0]));
  Tcl_ListObjAppendElement(interp, tcl_frames, Tcl_NewIntObj(framecntr[1]));
  Tcl_ListObjAppendElement(interp, tcl_frames, Tcl_NewIntObj(framecntr[2]));

  // build final list-of-lists as return value
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_rlist);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_gofr);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_numint);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_histog);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_frames);
  Tcl_SetObjResult(interp, tcl_result);

  delete [] gofr;
  delete [] numint;
  delete [] histog;
  delete [] framecntr;
  return TCL_OK;
}


// measure g(r) for two selections, with delta, rmax, usepbc, first/last/step 
// frame parameters the code will compute the normalized histogram.
static int vmd_measure_rdf(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int i;
  // initialize optional arguments to default values
  double rmax=10.0;
  double delta=0.1;
  int usepbc=0;
  int selupdate=0;
  int first=-1, last=-1, step=1;
  int rc;

  // argument error message
  const char *argerrmsg = "<sel1> <sel2> [delta <value>] [rmax <value>] [usepbc <bool>] [selupdate <bool>] [first <first>] [last <last>] [step <step>]";

  // Two atom selections and optional keyword/value pairs.
  if ((argc < 3) || (argc > 17) || (argc % 2 == 0) )  {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
    return TCL_ERROR;
  }

  // check atom selections
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1], NULL));
  if (!sel1) {
    Tcl_AppendResult(interp, "measure rdf: invalid first atom selection", NULL);
    return TCL_ERROR;
  }

  AtomSel *sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2], NULL));
  if (!sel2) {
    Tcl_AppendResult(interp, "measure rdf: invalid second atom selection", NULL);
    return TCL_ERROR;
  }

  // parse optional arguments
  for (i=3; i<argc; i+=2) {
    const char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (i==(argc-1)) {
      Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
      return TCL_ERROR;
    }
    if (!strcmp(opt, "delta")) {
      if (Tcl_GetDoubleFromObj(interp, objv[i+1], &delta) != TCL_OK)
        return TCL_ERROR;
      if (delta <= 0.0) {
        Tcl_AppendResult(interp, "measure rdf: invalid 'delta' value", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "rmax")) {
      if (Tcl_GetDoubleFromObj(interp, objv[i+1], &rmax) != TCL_OK)
        return TCL_ERROR;
      if (rmax <= 0.0) {
        Tcl_AppendResult(interp, "measure rdf: invalid 'rmax' value", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "usepbc")) {
      if (Tcl_GetBooleanFromObj(interp, objv[i+1], &usepbc) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "selupdate")) {
      if (Tcl_GetBooleanFromObj(interp, objv[i+1], &selupdate) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "first")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "last")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "step")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &step) != TCL_OK)
        return TCL_ERROR;
    } else { // unknown keyword.
      Tcl_AppendResult(interp, "unknown keyword '", opt, "'. usage: measure rdf ", argerrmsg, NULL);
      return TCL_ERROR;
    }
  }

  // allocate and initialize histogram arrays
  int    count_h = (int)(rmax / delta + 1.0);
  double *gofr   = new double[count_h];
  double *numint = new double[count_h];
  double *histog = new double[count_h];
  int *framecntr = new int[3];

  // do the gofr calculation
  rc = measure_rdf(app, sel1, sel2, app->moleculeList,
                   count_h, gofr, numint, histog,
                   (float) delta,
                   first, last, step, framecntr,
                   usepbc, selupdate);

  // XXX: this needs a 'case' structure to provide more meaninful error messages.
  if (rc != MEASURE_NOERR) { 
    Tcl_AppendResult(interp, "measure rdf: error during rdf calculation.", NULL);
    return TCL_ERROR;
  }

  // convert the results of the lowlevel call to tcl lists
  // and build a list from them as return value.
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_rlist  = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_gofr   = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_numint = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_histog = Tcl_NewListObj(0, NULL);
  Tcl_Obj *tcl_frames = Tcl_NewListObj(0, NULL);

  // build lists with results ready for plotting
  for (i=0; i<count_h; i++) { 
    Tcl_ListObjAppendElement(interp, tcl_rlist,  Tcl_NewDoubleObj(delta * ((double)i + 0.5)));
    Tcl_ListObjAppendElement(interp, tcl_gofr,   Tcl_NewDoubleObj(gofr[i]));
    Tcl_ListObjAppendElement(interp, tcl_numint, Tcl_NewDoubleObj(numint[i]));
    Tcl_ListObjAppendElement(interp, tcl_histog, Tcl_NewDoubleObj(histog[i]));
  }

  // build list with number of frames: 
  // total, skipped and processed (one entry for each algorithm).
  Tcl_ListObjAppendElement(interp, tcl_frames, Tcl_NewIntObj(framecntr[0]));
  Tcl_ListObjAppendElement(interp, tcl_frames, Tcl_NewIntObj(framecntr[1]));
  Tcl_ListObjAppendElement(interp, tcl_frames, Tcl_NewIntObj(framecntr[2]));

  // build final list-of-lists as return value
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_rlist);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_gofr);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_numint);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_histog);
  Tcl_ListObjAppendElement(interp, tcl_result, tcl_frames);
  Tcl_SetObjResult(interp, tcl_result);

  delete [] gofr;
  delete [] numint;
  delete [] histog;
  delete [] framecntr;
  return TCL_OK;
}


/// do cluster analysis for one selection.
static int vmd_measure_cluster(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int i,j;
  // initialize optional arguments to default values
  int algorithm=0; // will be MEASURE_CLUSTER_QT when finished
  int likeness=MEASURE_DIST_FITRMSD;
  int numcluster=5;
  double cutoff=1.0;
  float *weights=NULL;
  int selupdate=0;
  int first=0, last=-1, step=1;
  int rc;

  // argument error message
  const char *argerrmsg = "<sel> [num <#clusters>] [distfunc <flag>] "
    "[cutoff <cutoff>] [first <first>] [last <last>] [step <step>] "
    "[selupdate <bool>] [weight <weights>]";

  // Two atom selections and optional keyword/value pairs.
  if ((argc < 2) || (argc > 19) || ((argc-1) % 2 == 0) )  {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
    return TCL_ERROR;
  }

  // check atom selection
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1], NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure cluster: invalid atom selection", NULL);
    return TCL_ERROR;
  }
  if (!app->molecule_valid_id(sel->molid())) return MEASURE_ERR_NOMOLECULE;

  // parse optional arguments
  for (i=2; i<argc; i+=2) {
    const char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (i==(argc-1)) {
      Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
      return TCL_ERROR;
    }
    if (!strcmp(opt, "num")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &numcluster) != TCL_OK)
        return TCL_ERROR;
      if (numcluster < 1) {
        Tcl_AppendResult(interp, "measure cluster: invalid 'num' value (cannot be smaller than 1)", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "cutoff")) {
      if (Tcl_GetDoubleFromObj(interp, objv[i+1], &cutoff) != TCL_OK)
        return TCL_ERROR;
      if (cutoff <= 0.0) {
        Tcl_AppendResult(interp, "measure cluster: invalid 'cutoff' value (should be larger than 0.0)", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "distfunc")) {
      char *argstr = Tcl_GetStringFromObj(objv[i+1], NULL);
      if (!strcmp(argstr,"rmsd")) {
        likeness = MEASURE_DIST_RMSD;
      } else if (!strcmp(argstr,"fitrmsd")) {
        likeness = MEASURE_DIST_FITRMSD;
      } else if (!strcmp(argstr,"rgyrd")) {
        likeness = MEASURE_DIST_RGYRD;
      } else {
        Tcl_AppendResult(interp, "measure cluster: unknown distance function (supported are 'rmsd', 'rgyrd' and 'fitrmsd')", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "selupdate")) {
      if (Tcl_GetBooleanFromObj(interp, objv[i+1], &selupdate) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "weight")) {
      // NOTE: we cannot use tcl_get_weights here, since we have to 
      // get a full (all atoms) list of weights as we may be updating
      // the selection in the process. Also we don't support explicit
      // lists of weights for now.
      const char *weight_string = Tcl_GetStringFromObj(objv[i+1], NULL);
      weights = new float[sel->num_atoms];

      if (!weight_string || !strcmp(weight_string, "none")) {
        for (j=0; j<sel->num_atoms; j++) weights[j]=1.0f;
      } else {
        // if a selection string was given, check the symbol table
        SymbolTable *atomSelParser = app->atomSelParser; 
        // weights must return floating point values, so the symbol must not 
        // be a singleword, so macro is NULL.
        atomsel_ctxt context(atomSelParser,
                             app->moleculeList->mol_from_id(sel->molid()), 
                             sel->which_frame, NULL);

        int fctn = atomSelParser->find_attribute(weight_string);
        if (fctn >= 0) {
          // the keyword exists, so get the data
          // first, check to see that the function returns floats.
          // if it doesn't, it makes no sense to use it as a weight
          if (atomSelParser->fctns.data(fctn)->returns_a != SymbolTableElement::IS_FLOAT) {
            Tcl_AppendResult(interp, "weight attribute must have floating point values", NULL);
            delete [] weights;
            return MEASURE_ERR_BADWEIGHTPARM;  // can't understand weight parameter 
          }

          double *tmp_data = new double[sel->num_atoms];
          int *all_on = new int[sel->num_atoms];
          for (j=0; j<sel->num_atoms; j++) all_on[j]=1;

          atomSelParser->fctns.data(fctn)->keyword_double(
            &context, sel->num_atoms, tmp_data, all_on);
          
          for (j=0; j<sel->num_atoms; j++) weights[j] = (float)tmp_data[j];
          
          // clean up.
          delete [] tmp_data;
          delete [] all_on;
        }
      }
    } else if (!strcmp(opt, "first")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "last")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "step")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &step) != TCL_OK)
        return TCL_ERROR;
    } else { // unknown keyword.
      Tcl_AppendResult(interp, "unknown keyword '", opt, "'. usage: measure cluster ", argerrmsg, NULL);
      return TCL_ERROR;
    }
  }

  // set default for weights if not already defined
  if (!weights) {
    weights = new float[sel->num_atoms];
    for (j=0; j<sel->num_atoms; j++) weights[j]=1.0f;
  }
  
  // Allocate temporary result storage. we add one more cluster 
  // slot for collecting unclustered frames in an additional "cluster".
  // NOTE: the individual cluster lists are going to
  //       allocated in the ancilliary code.
  int  *clustersize = new int  [numcluster+1];
  int **clusterlist = new int *[numcluster+1];
  
  // do the cluster analysis
  rc = measure_cluster(sel, app->moleculeList, numcluster, algorithm, likeness, cutoff, 
                       clustersize, clusterlist, first, last, step, selupdate, weights);

  if (weights) delete [] weights;

  // XXX: this needs a 'case' structure to provide more meaninful error messages.
  if (rc != MEASURE_NOERR) { 
    Tcl_AppendResult(interp, "measure cluster: error during cluster analysis calculation.", NULL);
    return TCL_ERROR;
  }

  // convert the results of the lowlevel call to tcl lists
  // and build a list from them as return value.
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);

  for (i=0; i <= numcluster; ++i) { 
    int j;
    Tcl_Obj *tcl_clist  = Tcl_NewListObj(0, NULL);
    for (j=0; j < clustersize[i]; ++j) {
      Tcl_ListObjAppendElement(interp, tcl_clist, Tcl_NewIntObj(clusterlist[i][j]));
    }
    Tcl_ListObjAppendElement(interp, tcl_result, tcl_clist);
  }
  Tcl_SetObjResult(interp, tcl_result);

  // free temporary result storage
  for (i=0; i <= numcluster; ++i)
    delete[] clusterlist[i];

  delete[] clusterlist;
  delete[] clustersize;

  return TCL_OK;
}

/// compute cluster size distribution for a given selection of atoms
static int vmd_measure_clustsize(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int i;
  // initialize optional arguments to default values
  double cutoff=3.0; // arbitrary. took hbond cutoff.
  char *storesize=NULL;
  char *storenum=NULL;
  int usepbc=0;
  int minsize=2;
  int numshared=1;
  int rc;

  // argument error message
  const char *argerrmsg = "<sel> [cutoff <float>] [minsize <num>] [numshared <num>] "
    "[usepbc <bool>] [storesize <fieldname>] [storenum <fieldname>]";

  // Two atom selections and optional keyword/value pairs.
  if ((argc < 2) || (argc > 13) || ((argc-1) % 2 == 0) )  {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
    return TCL_ERROR;
  }

  // check atom selection
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1], NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure clustsize: invalid atom selection", NULL);
    return TCL_ERROR;
  }
  if (!app->molecule_valid_id(sel->molid())) return MEASURE_ERR_NOMOLECULE;

  // parse optional arguments
  for (i=2; i<argc; i+=2) {
    const char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (i==(argc-1)) {
      Tcl_WrongNumArgs(interp, 2, objv-1, (char *)argerrmsg);
      return TCL_ERROR;
    } else if (!strcmp(opt, "cutoff")) {
      if (Tcl_GetDoubleFromObj(interp, objv[i+1], &cutoff) != TCL_OK)
        return TCL_ERROR;
      if (cutoff <= 0.0) {
        Tcl_AppendResult(interp, "measure clustsize: invalid 'cutoff' value", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "minsize")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &minsize) != TCL_OK)
        return TCL_ERROR;
      if (minsize < 2) {
        Tcl_AppendResult(interp, "measure clustsize: invalid 'minsize' value", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "numshared")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &numshared) != TCL_OK)
        return TCL_ERROR;
      if (numshared < 0) {
        Tcl_AppendResult(interp, "measure clustsize: invalid 'numshared' value", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "usepbc")) {
      if (Tcl_GetBooleanFromObj(interp, objv[i+1], &usepbc) != TCL_OK)
        return TCL_ERROR;
    } else if (!strcmp(opt, "storenum")) {
      storenum = Tcl_GetStringFromObj(objv[i+1], NULL);
    } else if (!strcmp(opt, "storesize")) {
      storesize = Tcl_GetStringFromObj(objv[i+1], NULL);
    } else { // unknown keyword.
      Tcl_AppendResult(interp, "unknown keyword '", opt, "'. usage: measure clustsize ", argerrmsg, NULL);
      return TCL_ERROR;
    }
  }

  if (usepbc) { 
    Tcl_AppendResult(interp, "measure clustsize: does not support periodic boundaries yet.", NULL);
    return TCL_ERROR;
  }

  // allocate temporary result storage
  // NOTE: the individual cluster lists are going to
  //       allocated in the ancilliary code.
  int num_selected=sel->selected;
  int *clustersize = new int[num_selected];
  int *clusternum= new int [num_selected];
  int *clusteridx= new int [num_selected];
  for (i=0; i < num_selected; i++) {
    clustersize[i] = 0;
    clusternum[i]  = -1;
    clusteridx[i]  = -1;
  }
  
  // do the cluster analysis
  rc = measure_clustsize(sel, app->moleculeList, cutoff, 
                         clustersize, clusternum, clusteridx,
                         minsize, numshared, usepbc);

  // XXX: this needs a 'case' structure to provide more meaninful error messages.
  if (rc != MEASURE_NOERR) { 
    Tcl_AppendResult(interp, "measure clustsize: error during cluster size analysis calculation.", NULL);
    return TCL_ERROR;
  }


  if (storenum || storesize) {
    // field names were given to store the results. check the keywords and so on.
    SymbolTable *atomSelParser = app->atomSelParser; 
    atomsel_ctxt context(atomSelParser, 
                         app->moleculeList->mol_from_id(sel->molid()), 
                         sel->which_frame, NULL);

    // the keyword exists, set the data
    if (storenum) {
      int fctn = atomSelParser->find_attribute(storenum);
      if (fctn >= 0) {
        if (atomSelParser->fctns.data(fctn)->returns_a == SymbolTableElement::IS_FLOAT) {
          double *tmp_data = new double[sel->num_atoms];
          int j=0;
          for (int i=sel->firstsel; i<=sel->lastsel; i++) {
            if (sel->on[i])
              tmp_data[i] = (double) clusternum[j++];
          }
          atomSelParser->fctns.data(fctn)->set_keyword_double(&context, 
                                                              sel->num_atoms,
                                                              tmp_data, sel->on);
          delete[] tmp_data;
          
        } else if (atomSelParser->fctns.data(fctn)->returns_a == SymbolTableElement::IS_INT) {
          int *tmp_data = new int[sel->num_atoms];
          int j=0;
          for (int i=sel->firstsel; i<=sel->lastsel; i++) {
            if (sel->on[i])
              tmp_data[i] = clusternum[j++];
          }
          atomSelParser->fctns.data(fctn)->set_keyword_int(&context, 
                                                           sel->num_atoms,
                                                           tmp_data, sel->on);
          delete[] tmp_data;
        } else {
          Tcl_AppendResult(interp, "measure clustsize: storenum field must accept numbers", NULL);
          return TCL_ERROR;
        }
      } else {
        Tcl_AppendResult(interp, "measure clustsize: invalid field name for storenum", NULL);
        return TCL_ERROR;
      }
    }

    // the keyword exists, set the data
    if (storesize) {
      int fctn = atomSelParser->find_attribute(storesize);
      if (fctn >= 0) {
        if (atomSelParser->fctns.data(fctn)->returns_a == SymbolTableElement::IS_FLOAT) {
          double *tmp_data = new double[sel->num_atoms];
          int j=0;
          for (int i=sel->firstsel; i<=sel->lastsel; i++) {
            if (sel->on[i])
              tmp_data[i] = (double) clustersize[j++];
          }
          atomSelParser->fctns.data(fctn)->set_keyword_double(&context, 
                                                              sel->num_atoms,
                                                              tmp_data, sel->on);
          delete[] tmp_data;
          
        } else if (atomSelParser->fctns.data(fctn)->returns_a == SymbolTableElement::IS_INT) {
          int *tmp_data = new int[sel->num_atoms];
          int j=0;
          for (int i=sel->firstsel; i<=sel->lastsel; i++) {
            if (sel->on[i])
              tmp_data[i] = clustersize[j++];
          }
          atomSelParser->fctns.data(fctn)->set_keyword_int(&context, 
                                                           sel->num_atoms,
                                                           tmp_data, sel->on);
          delete[] tmp_data;
        } else {
          Tcl_AppendResult(interp, "measure clustsize: storenum field must accept numbers", NULL);
          return TCL_ERROR;
        }
      } else {
        Tcl_AppendResult(interp, "measure clustsize: invalid field name for storesize", NULL);
        return TCL_ERROR;
      }
    }
  } else {
    // convert the results of the lowlevel call to tcl lists
    // and build a list from them as return value.
    Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);

    Tcl_Obj *tcl_ilist  = Tcl_NewListObj(0, NULL);
    Tcl_Obj *tcl_clist  = Tcl_NewListObj(0, NULL);
    Tcl_Obj *tcl_nlist  = Tcl_NewListObj(0, NULL);
    for (i=0; i<num_selected; i++) { 
      Tcl_ListObjAppendElement(interp, tcl_ilist, Tcl_NewIntObj(clusteridx[i]));
      Tcl_ListObjAppendElement(interp, tcl_clist, Tcl_NewIntObj(clustersize[i]));
      Tcl_ListObjAppendElement(interp, tcl_nlist, Tcl_NewIntObj(clusternum[i]));
    }
    Tcl_ListObjAppendElement(interp, tcl_result, tcl_ilist);
    Tcl_ListObjAppendElement(interp, tcl_result, tcl_clist);
    Tcl_ListObjAppendElement(interp, tcl_result, tcl_nlist);
    Tcl_SetObjResult(interp, tcl_result);
  }
  
  delete[] clustersize;
  delete[] clusternum;
  delete[] clusteridx;

  return TCL_OK;
}

static int vmd_measure_hbonds(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  
  // Cutoff, angle, and either one or two atom selections
  if (argc != 4 && argc != 5) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<cutoff> <angle> <selection1> [<selection2>]");
    return TCL_ERROR;
  }
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[3],NULL));
  if (!sel1) {
    Tcl_AppendResult(interp, "measure hbonds: invalid first atom selection", NULL);
    return TCL_ERROR;
  }

  AtomSel *sel2 = NULL;
  if (argc == 5) {
    sel2 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[4],NULL));
    if (!sel2) {
      Tcl_AppendResult(interp, "measure hbonds: invalid second atom selection", NULL);
      return TCL_ERROR;
    }
  }
  if (sel2 && sel2->molid() != sel1->molid()) {
    Tcl_AppendResult(interp, "measure hbonds: error, atom selections must come from same molecule.", NULL);
    return TCL_ERROR;
  }
  double cutoff;
  if (Tcl_GetDoubleFromObj(interp, objv[1], &cutoff) != TCL_OK) 
    return TCL_ERROR;

  double maxangle;
  if (Tcl_GetDoubleFromObj(interp, objv[2], &maxangle) != TCL_OK) 
    return TCL_ERROR;
  
  const float *pos = sel1->coordinates(app->moleculeList);
  if (!pos) {
    Tcl_AppendResult(interp, "measure bondsearch: error, molecule contains no coordinates", NULL);
    return TCL_ERROR;
  }

  // XXX the actual code for measuring hbonds doesn't belong here, it should
  //     be moved into Measure.[Ch] where it really belongs.  This file
  //     only implements the Tcl interface, and should not be doing the
  //     hard core math, particularly if we want to expose the same
  //     feature via other scripting interfaces.  Also, having a single
  //     implementation avoids having different Tcl/Python bugs in the 
  //     long-term.  Too late to do anything about this now, but should be
  //     addressed for the next major version when time allows.

  // XXX This code is close, but not identical to the HBonds code in 
  //     DrawMolItem.  Is there any good reason they aren't identical?
  //     This version does a few extra tests that the other does not.

  Molecule *mol = app->moleculeList->mol_from_id(sel1->molid());

  int *donlist, *hydlist, *acclist;
  int maxsize = 2 * sel1->num_atoms; //This heuristic is based on ice, where there are < 2 hydrogen bonds per atom if hydrogens are in the selection, and exactly 2 if hydrogens are not considered.
  donlist = new int[maxsize];
  hydlist = new int[maxsize];
  acclist = new int[maxsize];
  int rc = measure_hbonds(mol, sel1, sel2, cutoff, maxangle, donlist, hydlist, acclist, maxsize);
  if (rc > maxsize) {
    delete [] donlist;
    delete [] hydlist;
    delete [] acclist;
    maxsize = rc;
    donlist = new int[maxsize];
    hydlist = new int[maxsize];
    acclist = new int[maxsize];
    rc = measure_hbonds(mol, sel1, sel2, cutoff, maxangle, donlist, hydlist, acclist, maxsize);
  }
  if (rc < 0) {
    Tcl_AppendResult(interp, "measure hbonds: internal error to measure_hbonds", NULL);
    return TCL_ERROR;
  }
  
  Tcl_Obj *newdonlist = Tcl_NewListObj(0, NULL);
  Tcl_Obj *newhydlist = Tcl_NewListObj(0, NULL);
  Tcl_Obj *newacclist = Tcl_NewListObj(0, NULL);
  for (int k = 0; k < rc; k++) {
    Tcl_ListObjAppendElement(interp, newdonlist, Tcl_NewIntObj(donlist[k]));
    Tcl_ListObjAppendElement(interp, newhydlist, Tcl_NewIntObj(hydlist[k]));
    Tcl_ListObjAppendElement(interp, newacclist, Tcl_NewIntObj(acclist[k]));
  }
  delete [] donlist;
  delete [] hydlist;
  delete [] acclist;
  Tcl_Obj *result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, result, newdonlist);
  Tcl_ListObjAppendElement(interp, result, newacclist);
  Tcl_ListObjAppendElement(interp, result, newhydlist);
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

  
static int vmd_measure_sasa(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {

  int i;
  // srad and one atom selection, plus additional options
  if (argc < 3) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<srad> <sel> [-points <varname>] [-restrict <restrictedsel>] [-samples <numsamples>]");
    return TCL_ERROR;
  }
  // parse options
  Tcl_Obj *ptsvar = NULL;
  AtomSel *restrictsel = NULL;
  int nsamples = -1;
  int *sampleptr = NULL;
  for (i=3; i<argc-1; i+=2) {
    const char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-points")) {
      ptsvar = objv[i+1];
    } else if (!strcmp(opt, "-restrict")) {
      restrictsel = tcl_commands_get_sel(interp, 
          Tcl_GetStringFromObj(objv[i+1], NULL));
      if (!restrictsel) {
        Tcl_AppendResult(interp, "measure sasa: invalid restrict atom selection", NULL);
        return TCL_ERROR;
      }
    } else if (!strcmp(opt, "-samples")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &nsamples) != TCL_OK)
        return TCL_ERROR;
      sampleptr = &nsamples;
    } else {
      Tcl_AppendResult(interp, "measure sasa: unknown option '", opt, "'", 
          NULL);
      return TCL_ERROR;
    }
  }

  double srad;
  if (Tcl_GetDoubleFromObj(interp, objv[1], &srad) != TCL_OK) 
    return TCL_ERROR;
  AtomSel *sel1 = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2],NULL));
  if (!sel1) {
    Tcl_AppendResult(interp, "measure sasa: invalid first atom selection", NULL);
    return TCL_ERROR;
  }

  const float *pos = sel1->coordinates(app->moleculeList);
  if (!pos) {
    Tcl_AppendResult(interp, "measure sasa: error, molecule contains no coordinates", NULL);
    return TCL_ERROR;
  }
  Molecule *mol = app->moleculeList->mol_from_id(sel1->molid());
  const float *radius = mol->extraflt.data("radius");

  ResizeArray<float> sasapts;
  float sasa = 0;
  int rc = measure_sasa(sel1, pos, radius, (float) srad, &sasa, &sasapts, 
                        restrictsel, sampleptr);
  if (rc < 0) {
    Tcl_AppendResult(interp, "measure: sasa: ", measure_error(rc), NULL);
    return TCL_ERROR;
  }
  Tcl_SetObjResult(interp, Tcl_NewDoubleObj(sasa));
  if (ptsvar) {
    // construct list from sasapts
    Tcl_Obj *listobj = Tcl_NewListObj(0, NULL);
    i=0;
    while (i<sasapts.num()) {
      Tcl_Obj *elem = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, elem, Tcl_NewDoubleObj(sasapts[i++]));
      Tcl_ListObjAppendElement(interp, elem, Tcl_NewDoubleObj(sasapts[i++]));
      Tcl_ListObjAppendElement(interp, elem, Tcl_NewDoubleObj(sasapts[i++]));
      Tcl_ListObjAppendElement(interp, listobj, elem);
    }
    Tcl_ObjSetVar2(interp, ptsvar, NULL, listobj, 0);
  }
  return TCL_OK;
}


#if 1

static int vmd_measure_sasalist(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {

  int i;
  // srad and one atom selection, plus additional options
  if (argc < 3) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *)"<srad> <sel list> [-samples <numsamples>]");
    return TCL_ERROR;
  }

  // parse options
  int nsamples = -1;
  int *sampleptr = NULL;
  for (i=3; i<argc-1; i+=2) {
    const char *opt = Tcl_GetStringFromObj(objv[i], NULL);

    if (!strcmp(opt, "-samples")) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &nsamples) != TCL_OK)
        return TCL_ERROR;
      sampleptr = &nsamples;
    } else {
      Tcl_AppendResult(interp, "measure sasa: unknown option '", opt, "'", NULL);
      return TCL_ERROR;
    }
  }

  double srad;
  if (Tcl_GetDoubleFromObj(interp, objv[1], &srad) != TCL_OK) 
    return TCL_ERROR;

  int numsels;
  Tcl_Obj **sel_list;
  if (Tcl_ListObjGetElements(interp, objv[2], &numsels, &sel_list) != TCL_OK) {
    Tcl_AppendResult(interp, "measure sasalist: bad selection list", NULL);
    return TCL_ERROR;
  }

#if 0
printf("measure sasalist: numsels %d\n", numsels);
#endif

  AtomSel **asels = (AtomSel **) calloc(1, numsels * sizeof(AtomSel *));
  float *sasalist = (float *) calloc(1, numsels * sizeof(float));

  int s;
  for (s=0; s<numsels; s++) {
    asels[s] = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(sel_list[s], NULL));
    if (!asels[s]) {
printf("measure sasalist: invalid selection %d\n", s);
      Tcl_AppendResult(interp, "measure sasalist: invalid atom selection list element", NULL);
      return TCL_ERROR;
    }
  }

  int rc = measure_sasalist(app->moleculeList, (const AtomSel **) asels, 
                            numsels, (float) srad, sasalist, sampleptr);
  free(asels);
  if (rc < 0) {
    Tcl_AppendResult(interp, "measure: sasalist: ", measure_error(rc), NULL);
    return TCL_ERROR;
  }

  // construct list from sasa values
  Tcl_Obj *listobj = Tcl_NewListObj(0, NULL);
  for (i=0; i<numsels; i++) {
    Tcl_ListObjAppendElement(interp, listobj, Tcl_NewDoubleObj(sasalist[i]));
  }
  Tcl_SetObjResult(interp, listobj);
  free(sasalist);

  return TCL_OK;
}

#endif


// Function: vmd_measure_energy 
// Returns: the energy for the specified bond/angle/dihed/imprp/vdw/elec
// FIXME -- usage doesn't match user guide
static int vmd_measure_energy(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {

  if (argc<3) {
    Tcl_WrongNumArgs(interp, 2, objv-1,
		     (char *) "bond|angle|dihed|imprp|vdw|elec {{<atomid1> ?<molid1>?} {<atomid2> ?<molid2>?}} ?molid <default molid>? [?frame <frame|all>? | ?first <first>? ?last <last>?]");
    return TCL_ERROR;
  }

  int geomtype, reqatoms;
  char *geomname = Tcl_GetStringFromObj(objv[1],NULL);
  if        (!strncmp(geomname, "bond", 4)) {
    reqatoms = 2; geomtype = MEASURE_BOND;
  } else if (!strncmp(geomname, "angl", 4)) {
    reqatoms = 3; geomtype = MEASURE_ANGLE;
  } else if (!strncmp(geomname, "dihe", 4)) {
    reqatoms = 4; geomtype = MEASURE_DIHED;
  } else if (!strncmp(geomname, "impr", 4)) {
    reqatoms = 4; geomtype = MEASURE_IMPRP;
  } else if (!strncmp(geomname, "vdw",  3)) {
    reqatoms = 2; geomtype = MEASURE_VDW;
  } else if (!strncmp(geomname, "elec", 4)) {
    reqatoms = 2; geomtype = MEASURE_ELECT;
  } else {
    Tcl_AppendResult(interp, " measure energy: bad syntax (must specify bond|angle|dihed|imprp|vdw|elec)", NULL);
    return TCL_ERROR;
  }

  int molid[4];
  int atmid[4];
  int defmolid = -1;
  bool allframes = false;
  char errstring[200];

  // Read the atom list
  int numatms;
  Tcl_Obj **data;
  if (Tcl_ListObjGetElements(interp, objv[2], &numatms, &data) != TCL_OK) {
    Tcl_AppendResult(interp, " measure energy: bad syntax", NULL);
    Tcl_AppendResult(interp, " Usage: measure energy bond|angle|dihed|imprp|vdw|elec {{<atomid1> ?<molid1>?} {<atomid2> ?<molid2>?} ...} ?molid <default molid>? [?frame <frame|all>? | ?first <first>? ?last <last>?]", NULL);
    return TCL_ERROR;
  }

  if (numatms!=reqatoms) {
    sprintf(errstring, " measure energy %s: must specify exactly %i atoms in list", geomname, reqatoms);
    Tcl_AppendResult(interp, errstring, NULL);
    return TCL_ERROR;
  }
    
  int first=-1, last=-1, frame=-1;
  double params[6];
  memset(params, 0, 6L*sizeof(double));

  if (argc>4) {
    int i;
    for (i=3; i<argc-1; i+=2) {
      char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
      if (!strupncmp(argvcur, "molid", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &defmolid) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad molid", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "k", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad force constant value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "x0", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+1) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad equilibrium value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "kub", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+2) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad Urey-Bradley force constant value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "s0", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+3) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad Urey-Bradley equilibrium distance", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "n", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+1) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad dihedral periodicity", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "delta", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+2) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad dihedral phase shift", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "eps1", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad vdw well depth", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "rmin1", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+1) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad vdw equilibrium distance", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "eps2", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+2) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad vdw well depth", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "rmin2", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+3) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad vdw equilibrium distance", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "q1", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad charge value", NULL);
	  return TCL_ERROR;
	}
	params[2]=1.0;
      } else if (!strupncmp(argvcur, "q2", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+1) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad charge value", NULL);
	  return TCL_ERROR;
	}
	params[3]=1.0;
      } else if (!strupncmp(argvcur, "cutoff", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+4) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad electrostatic cutoff value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "switchdist", CMDLEN)) {
	if (Tcl_GetDoubleFromObj(interp, objv[i+1], params+5) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad switching distance value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "first", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad first frame value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "last", CMDLEN)) {
	if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad last frame value", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "frame", CMDLEN)) {
	if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "all")) {
	  allframes = true;
	} else if (Tcl_GetIntFromObj(interp, objv[i+1], &frame) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure energy: bad frame value", NULL);
	  return TCL_ERROR;
	}
      } else {
	Tcl_AppendResult(interp, " measure energy: invalid syntax, no such keyword: ", argvcur, NULL);
	return TCL_ERROR;
      }
    }
  }

  if ((allframes || frame>=0) && (first>=0 || last>=0)) {
    Tcl_AppendResult(interp, "measure energy: Ambiguous syntax: You cannot specify a frame AND a frame range (using first or last).", NULL);
    Tcl_AppendResult(interp, "\nUsage:\nmeasure bond <molid1>/<atomid1> <molid2>/<atomid2> [?frame <frame|all>? | ?first <first>? ?last <last>?]", NULL);
    return TCL_ERROR;    
  }

  if (allframes) first=0;

  // If no molecule was specified use top as default
  if (defmolid<0) defmolid = app->molecule_top();

  // Assign atom IDs and molecule IDs
  int i,numelem;
  Tcl_Obj **atmmol;
  for (i=0; i<numatms; i++) {
    if (Tcl_ListObjGetElements(interp, data[i], &numelem, &atmmol) != TCL_OK) {
      return TCL_ERROR;
    }

    if (Tcl_GetIntFromObj(interp, atmmol[0], atmid+i) != TCL_OK) {
      Tcl_AppendResult(interp, " measure energy: bad atom index", NULL);
      return TCL_ERROR;
    }
    
    if (numelem==2) {
      if (Tcl_GetIntFromObj(interp, atmmol[1], molid+i) != TCL_OK) {
	Tcl_AppendResult(interp, " measure energy: bad molid", NULL);
	return TCL_ERROR;
      }
    } else molid[i] = defmolid;
  }
  

  // Compute the value
  ResizeArray<float> gValues(1024);
  int ret_val;
  ret_val = measure_energy(app->moleculeList, molid, atmid, reqatoms, &gValues, frame, first, last,
			 defmolid, params, geomtype);
  if (ret_val<0) {
    printf("ERROR\n %s\n", measure_error(ret_val));
    Tcl_AppendResult(interp, measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  int numvalues = gValues.num();
  for (int count = 0; count < numvalues; count++) {
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(gValues[count]));
  }
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


//
// Function: vmd_measure_surface <selection> <gridsize> <radius> <depth>
//
static int vmd_measure_surface(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc!=5) {
    Tcl_WrongNumArgs(interp, 2, objv-1,
                     (char *) "<sel> <gridsize> <radius> <depth>");
    return TCL_ERROR;
  }
   
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure surface: no atom selection", NULL);
    return TCL_ERROR;
  }

  const float *framepos = sel->coordinates(app->moleculeList);
  if (!framepos) return TCL_ERROR;

  double gridsz;
  if (Tcl_GetDoubleFromObj(interp, objv[2], &gridsz) != TCL_OK) 
     return TCL_ERROR;

  double radius;
  if (Tcl_GetDoubleFromObj(interp, objv[3], &radius) != TCL_OK) 
     return TCL_ERROR;

  double depth;
  if (Tcl_GetDoubleFromObj(interp, objv[4], &depth) != TCL_OK) 
     return TCL_ERROR;
  
  int *surface;
  int n_surf;
  
  int ret_val = measure_surface(sel, app->moleculeList, framepos,
                               gridsz, radius, depth, &surface, &n_surf);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure surface: ", measure_error(ret_val));
    return TCL_ERROR;
  }

  Tcl_Obj *surf_list = Tcl_NewListObj(0, NULL);

  int i;
  for(i=0; i < n_surf; i++) {
    Tcl_ListObjAppendElement(interp, surf_list, Tcl_NewIntObj(surface[i]));
  }
  Tcl_SetObjResult(interp, surf_list);
  delete [] surface;
  
  return TCL_OK;
}


// Function: vmd_measure_pbc2onc <center> ?molid <default>? ?frame <frame|last>?
//  Returns: the transformation matrix to wrap a nonorthogonal pbc unicell
//           into an orthonormal cell.
static int vmd_measure_pbc2onc_transform(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<center> [molid <default>] [frame <frame|last>]");
    return TCL_ERROR;
  }

  // Read the center
  int ndim;
  Tcl_Obj **centerObj;
  if (Tcl_ListObjGetElements(interp, objv[1], &ndim, &centerObj) != TCL_OK) {
    Tcl_AppendResult(interp, " measure pbc2onc: bad syntax", NULL);
    Tcl_AppendResult(interp, " Usage: measure pbc2onc <center> [molid <default>] [frame <frame|last>]", NULL);
    return TCL_ERROR;
  }

  if (ndim!=3) {
    Tcl_AppendResult(interp, " measure pbc2onc: need three numbers for a vector", NULL);
    return TCL_ERROR;
  }
    
  int i;
  double tmp;
  float center[3];
  for (i=0; i<3; i++) {
    if (Tcl_GetDoubleFromObj(interp, centerObj[i], &tmp) != TCL_OK) {
      Tcl_AppendResult(interp, " measure pbc2onc: non-numeric in center", NULL);
      return TCL_ERROR;
    }
    center[i] = (float)tmp;
  }

  int molid = app->molecule_top();
  int frame = -2;
  if (argc>3) {
    for (i=2; i<argc; i+=2) {
      char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
      if (!strupncmp(argvcur, "molid", CMDLEN)) {
	if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "top")) {
	  // top is already default
	} else if (Tcl_GetIntFromObj(interp, objv[i+1], &molid) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure pbc2onc: bad molid", NULL);
	  return TCL_ERROR;
	}
      } else if (!strupncmp(argvcur, "frame", CMDLEN)) {
	if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "last")) {
	  frame=-1;
	} else if (Tcl_GetIntFromObj(interp, objv[i+1], &frame) != TCL_OK) {
	  Tcl_AppendResult(interp, " measure pbc2onc: bad frame value", NULL);
	  return TCL_ERROR;
	}
      } else {
	Tcl_AppendResult(interp, " measure pbc2onc: invalid syntax, no such keyword: ", argvcur, NULL);
	return TCL_ERROR;
      }
    }
  }


  Matrix4 transform;
  int ret_val = measure_pbc2onc(app->moleculeList, molid, frame, center, transform);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure pbc2onc: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  for (i=0; i<4; i++) {
    Tcl_Obj *rowListObj = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(transform.mat[i+0]));
    Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(transform.mat[i+4]));
    Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(transform.mat[i+8]));
    Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(transform.mat[i+12]));
    Tcl_ListObjAppendElement(interp, tcl_result, rowListObj);
  }
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}

// Function: vmd_measure_pbc_neighbors <center> <cutoff>
//  Returns: All image atoms that are within cutoff Angstrom of the pbc unitcell.
//           Two lists are returned. The first list holds the atom coordinates while
//           the second one is an indexlist mapping the image atoms to the atoms in
//           the unitcell.

//   Input:  Since the pbc cell <center> is not stored in DCDs and cannot be set in
//           VMD it must be provided by the user as the first argument.
//           The second argument <cutoff> is the maximum distance (in Angstrom) from
//           the PBC unit cell for atoms to be considered.

// Options:  ?sel <selection>? :
//           If an atomselection is provided after the keyword 'sel' then only those
//           image atoms are returned that are within cutoff of the selected atoms
//           of the main cell. In case cutoff is a vector the largest value will be
//           used.

//           ?align <matrix>? :
//           In case the molecule was aligned you can supply the alignment matrix
//           which is then used to correct for the rotation and shift of the pbc cell.

//           ?boundingbox PBC | {<mincoord> <maxcoord>}?
//           With this option the atoms are wrapped into a rectangular bounding box.
//           If you provide "PBC" as an argument then the bounding box encloses the
//           PBC box but then the cutoff is added to the bounding box. Negative 
//           values for the cutoff dimensions are allowed and lead to a smaller box.
//           Instead you can also provide a custom bounding box in form of the 
//           minmax coordinates (list containing two coordinate vectors such as 
//           returned by the measure minmax command). Here, again, the cutoff is
//           added to the bounding box.
static int vmd_measure_pbc_neighbors(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc < 3) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<center> <cutoff> ?sel <selection>? ?align <matrix>? ?molid <default>? ?frame <frame|last>? ?boundingbox PBC|{<mincoord> <maxcoord>}?");
    return TCL_ERROR;
  }

  // Read the center
  int ndim;
  Tcl_Obj **centerObj;
  if (Tcl_ListObjGetElements(interp, objv[1], &ndim, &centerObj) != TCL_OK) {
    Tcl_AppendResult(interp, " measure pbcneighbors: bad syntax", NULL);
    Tcl_AppendResult(interp, " Usage: measure pbcneighbors <center> <cutoff> [<sel <sel>] [align <matrix>] [molid <default>] [frame <frame|last>] [boundingbox <PBC|{<mincoord> <maxcoord>}>]", NULL);
    return TCL_ERROR;
  }

  if (ndim!=3) {
    Tcl_AppendResult(interp, " measure pbcneighbors: need three numbers for a vector", NULL);
    return TCL_ERROR;
  }
    
  int i;
  double tmp;
  float center[3];
  for (i=0; i<3; i++) {
    if (Tcl_GetDoubleFromObj(interp, centerObj[i], &tmp) != TCL_OK) {
      Tcl_AppendResult(interp, " measure pbcneighbors: non-numeric in center", NULL);
      return TCL_ERROR;
    }
    center[i] = float(tmp);
  }

  // Read the cutoff
  Tcl_Obj **cutoffObj;
  if (Tcl_ListObjGetElements(interp, objv[2], &ndim, &cutoffObj) != TCL_OK) {
    Tcl_AppendResult(interp, " measure pbcneighbors: bad syntax", NULL);
    Tcl_AppendResult(interp, " Usage: measure pbcneighbors <center> <cutoff> [<sel <sel>] [align <matrix>] [molid <default>] [frame <frame|last>] [boundingbox <PBC|{<mincoord> <maxcoord>}>]", NULL);
    return TCL_ERROR;
  }

  if (ndim!=3 && ndim!=1) {
    Tcl_AppendResult(interp, " measure pbcneighbors: need either one or three numbers for cutoff", NULL);
    return TCL_ERROR;
  }

  float cutoff[3];
  for (i=0; i<ndim; i++) {
    if (Tcl_GetDoubleFromObj(interp, cutoffObj[i], &tmp) != TCL_OK) {
      Tcl_AppendResult(interp, " measure pbcneighbors: non-numeric in cutoff", NULL);
      return TCL_ERROR;
    }
    cutoff[i] = float(tmp);
  }

  if (ndim==1) { cutoff[2] = cutoff[1] = cutoff[0]; }

  bool molidprovided=0;
  float *boxminmax=NULL;
  int molid = app->molecule_top();
  int frame = -2;
  AtomSel *sel = NULL;
  Matrix4 alignment;
  if (argc>4) {
    for (i=3; i<argc; i+=2) {
      char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
      if (!strupncmp(argvcur, "sel", CMDLEN)) {
        // Read the atom selection
        sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[i+1],NULL));
        if (!sel) {
          Tcl_AppendResult(interp, "measure pbcneighbors: invalid atom selection", NULL);
          return TCL_ERROR;
        }
        if (!app->molecule_valid_id(sel->molid())) {
          Tcl_AppendResult(interp, "measure pbcneighbors: ",
                           measure_error(MEASURE_ERR_NOMOLECULE), NULL);
          return TCL_ERROR;
        }
        if (!sel->selected) {
          Tcl_AppendResult(interp, "measure pbcneighbors: selection contains no atoms.", NULL);
          return TCL_ERROR;
        }
      } else if (!strupncmp(argvcur, "molid", CMDLEN)) {
        if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "top")) {
          // top is already default
        } else if (Tcl_GetIntFromObj(interp, objv[i+1], &molid) != TCL_OK) {
          Tcl_AppendResult(interp, " measure pbcneighbors: bad molid", NULL);
          return TCL_ERROR;
        }
        molidprovided = 1;
      } else if (!strupncmp(argvcur, "frame", CMDLEN)) {
        if (!strupcmp(Tcl_GetStringFromObj(objv[i+1],NULL), "last")) {
          frame=-1;
        } else if (Tcl_GetIntFromObj(interp, objv[i+1], &frame) != TCL_OK) {
          Tcl_AppendResult(interp, " measure pbcneighbors: bad frame value", NULL);
          return TCL_ERROR;
        }
      } else if (!strupncmp(argvcur, "align", CMDLEN)) {
        // Get the alignment matrix (as returned by 'measure fit')
        if (tcl_get_matrix("measure pbcneighbors: ", interp, objv[i+1], alignment.mat) != TCL_OK) {
          return TCL_ERROR;
        }
      } else if (!strupncmp(argvcur, "boundingbox", CMDLEN)) {
        // Determine if the atoms shall be wrapped to the smallest rectangular
        // bounding box that still encloses the unitcell plus the given cutoff.
        char *argv2 = Tcl_GetStringFromObj(objv[i+1],NULL);
        if (!strupncmp(argv2, "on", CMDLEN) || !strupncmp(argv2, "pbc", CMDLEN)) {
          boxminmax = new float[6];
          compute_pbcminmax(app->moleculeList, molid, frame, center, &alignment,
                            boxminmax, boxminmax+3);
        } else {
          // Read the bounding box
          int j, k, ncoor;
          Tcl_Obj **boxListObj;
          if (Tcl_ListObjGetElements(interp, objv[i+1], &ncoor, &boxListObj) != TCL_OK) {
            Tcl_AppendResult(interp, " measure pbcneighbors: invalid bounding box parameter", NULL);
            return TCL_ERROR;
          }
          if (ncoor!=2) {
            Tcl_AppendResult(interp, " measure pbcneighbors: need 2 points for bounding box", NULL);
            return TCL_ERROR;
          }
          int ndim = 0;
          double tmp;
          Tcl_Obj **boxObj;
          boxminmax = new float[6];
          for (j=0; j<2; j++) {
            if (Tcl_ListObjGetElements(interp, boxListObj[j], &ndim, &boxObj) != TCL_OK) {
              Tcl_AppendResult(interp, " measure pbcneighbors: bad syntax in boundingbox", NULL);
              return TCL_ERROR;
            }

            for (k=0; k<3; k++) {
              if (Tcl_GetDoubleFromObj(interp, boxObj[k], &tmp) != TCL_OK) {
                Tcl_AppendResult(interp, " measure pbcneighbors: non-numeric in boundingbox", NULL);
                return TCL_ERROR;
              }
              boxminmax[3L*j+k] = (float)tmp;
            }
          }
        }

      } else {
        Tcl_AppendResult(interp, " measure pbcneighbors: invalid syntax, no such keyword: ", argvcur, NULL);
        return TCL_ERROR;
      }
    }
  }

  // If no molid was provided explicitely but a selection was given
  // then use that molid.
  if (sel && !molidprovided) {
    molid = sel->molid();
  }

  ResizeArray<float> extcoord_array;
  ResizeArray<int> indexmap_array;

  int ret_val = measure_pbc_neighbors(app->moleculeList, sel, molid, frame, &alignment, center, cutoff,
				      boxminmax, &extcoord_array, &indexmap_array);
  delete [] boxminmax;

  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure pbcneighbors: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }
  printf("measure pbcneighbors: %ld neighbor atoms found\n", 
         long(indexmap_array.num()));

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_Obj *coorListObj = Tcl_NewListObj(0, NULL);
  Tcl_Obj *indexListObj = Tcl_NewListObj(0, NULL);
  for (i=0; i<indexmap_array.num(); i++) {
    Tcl_Obj *rowListObj = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(extcoord_array[3L*i]));
    Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(extcoord_array[3L*i+1]));
    Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(extcoord_array[3L*i+2]));
    Tcl_ListObjAppendElement(interp, coorListObj, rowListObj);
    Tcl_ListObjAppendElement(interp, indexListObj, Tcl_NewIntObj(indexmap_array[i]));
  }
  Tcl_ListObjAppendElement(interp, tcl_result, coorListObj);
  Tcl_ListObjAppendElement(interp, tcl_result, indexListObj);
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}



// Function: vmd_measure_inertia <selection> [moments] [eigenvals]
//  Returns: The center of mass and the principles axes of inertia for the 
//           selected atoms. 
//  Options: -moments -- also return the moments of inertia tensor
//           -eigenvals -- also return the corresponding eigenvalues
static int vmd_measure_inertia(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  bool moments = FALSE;
  bool eigenvals = FALSE;
  if (argc < 2 || argc > 4) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<selection> [moments] [eigenvals]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure inertia: no atom selection", NULL);
    return TCL_ERROR;
  }
  
  if (!app->molecule_valid_id(sel->molid())) {
    Tcl_AppendResult(interp, "measure inertia: ",
                     measure_error(MEASURE_ERR_NOMOLECULE), NULL);
    return TCL_ERROR;
  }
  
  if (argc>2) {
    int i;
    for (i=2; i<argc; i++) {
      char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
      // Allow syntax with and without leading dash
      if (argvcur[0]=='-') argvcur++;

      if (!strupncmp(argvcur, "moments", CMDLEN)) {
        moments = TRUE; 
      }
      else if (!strupncmp(argvcur, "eigenvals", CMDLEN)) {
        eigenvals = TRUE; 
      }
      else {
        Tcl_AppendResult(interp, " measure inertia: unrecognized option\n", NULL);
        Tcl_AppendResult(interp, " Usage: measure inertia <selection> [moments] [eigenvals]", NULL);
        return TCL_ERROR;
      }
    }
  }

  float priaxes[3][3];
  float itensor[4][4];
  float evalue[3], rcom[3];
  int ret_val = measure_inertia(sel, app->moleculeList, NULL, rcom, priaxes, itensor, evalue);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure inertia: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  // return COM and list of 3 principal axes
  int i;
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);

  Tcl_Obj *rcomObj = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, rcomObj, Tcl_NewDoubleObj(rcom[0]));
  Tcl_ListObjAppendElement(interp, rcomObj, Tcl_NewDoubleObj(rcom[1]));
  Tcl_ListObjAppendElement(interp, rcomObj, Tcl_NewDoubleObj(rcom[2]));
  Tcl_ListObjAppendElement(interp, tcl_result, rcomObj);

  Tcl_Obj *axesListObj = Tcl_NewListObj(0, NULL);
  for (i=0; i<3; i++) {
    Tcl_Obj *axesObj = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(priaxes[i][0]));
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(priaxes[i][1]));
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(priaxes[i][2]));
    Tcl_ListObjAppendElement(interp, axesListObj, axesObj);
  }
  Tcl_ListObjAppendElement(interp, tcl_result, axesListObj);

  if (moments) {
    Tcl_Obj *momListObj = Tcl_NewListObj(0, NULL);
    for (i=0; i<3; i++) {
      Tcl_Obj *momObj = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, momObj, Tcl_NewDoubleObj(itensor[i][0]));
      Tcl_ListObjAppendElement(interp, momObj, Tcl_NewDoubleObj(itensor[i][1]));
      Tcl_ListObjAppendElement(interp, momObj, Tcl_NewDoubleObj(itensor[i][2]));
      Tcl_ListObjAppendElement(interp, momListObj, momObj);
    }
    Tcl_ListObjAppendElement(interp, tcl_result, momListObj);
  }

  if (eigenvals) {
      Tcl_Obj *eigvListObj = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, eigvListObj, Tcl_NewDoubleObj(evalue[0]));
      Tcl_ListObjAppendElement(interp, eigvListObj, Tcl_NewDoubleObj(evalue[1]));
      Tcl_ListObjAppendElement(interp, eigvListObj, Tcl_NewDoubleObj(evalue[2]));
      Tcl_ListObjAppendElement(interp, tcl_result, eigvListObj);
  }
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}


// measure symmetry <sel> [plane|I|C<n>|S<n> [<vector>]] [-tol <value>]
//                  [-nobonds] [-verbose <level>]
//
// This function evaluates the molecular symmetry of an atom selection.
// The underlying algorithm finds the symmetry elements such as 
// inversion center, mirror planes, rotary axes and rotary reflections.
// Based on the found elements it guesses the underlying point group.
// The guess is fairly robust and can handle molecules whose
// coordinatesthat deviate to a certain extent from the ideal symmetry.
// The closest match with the highest symmetry will be returned.
//
// Options:
// --------
// -tol <value>
//    Allows one to control tolerance of the algorithm when
//    considering wether something is symmetric or not.
//    A smaller value signifies a lower tolerance, the default
//    is 0.1.
// -nobonds 
//    If this flag is set then the bond order and orientation
//    are not considered when comparing structures.
// -verbose <level>
//    Controls the amount of console output.
//    A level of 0 means no output, 1 gives some statistics at the
//    end of the search (default). Level 2 gives additional info
//    about each stage, level 3 more output for each iteration
//    and 3, 4 yield yet more additional info.
// -I
//    Instead of guessing the symmetry pointgroup of the selection
//    determine if the selection's center off mass represents an
//    inversion center. The returned value is a score between 0
//    and 1 where 1 denotes a perfect match.
// -plane <vector>
//    Instead of guessing the symmetry pointgroup of the selection
//    determine if the plane with the defined by its normal
//    <vector> is a mirror plane of the selection. The
//    returned value is a score between 0 and 1 where 1 denotes
//    a perfect match.
// -Cn | -Sn  <vector>
//    Instead of guessing the symmetry pointgroup of the selection
//    determine if the rotation or rotary reflection axis Cn/Sn
//    with order n defined by <vector> exists for the
//    selection. E.g., if you want to query wether the Y-axis
//    has a C3 rotational symmetry you specify "C3 {0 1 0}".
//    The returned value is a score between 0 and 1 where 1
//    denotes a perfect match.
//
// Result:
// -------
// The return value is a TCL list of pairs consisting of a label
// string and a value or list. For each label the data following
// it are described below:
//
// * [pointgroup] The guessed pointgroup:
// * [order] Point group order (the n from above)
// * [elements] Summary of found elements.
//     For instance {(i) (C3) 3*(C2) (S6) 3*(sigma)} for D3d.
// * [missing] Elements missing with respect to ideal number of
//     elements (same format as above). If this is not an empty
//     list then something has gone awfully wrong with the symmetry
//     finding algorithm.
// * [additional] Additional elements  that would not be expected
//     for this point group (same format as above). If this is not
//     an empty list then something has gone awfully wrong with the
//     symmetry finding algorithm.
// * [com] Center of mass of the selection based on the idealized
//     coordinates.
// * [inertia] List of the three axes of inertia, the eigenvalues
//     of the moments of inertia tensor and a list of three 0/1 flags 
//     specifying for each axis wether it is unique or not.
// * [inversion] Flag 0/1 signifying if there is an inversion center.
// * [axes] Normalized vectors defining rotary axes
// * [rotreflect] Normalized vectors defining rotary reflections
// * [planes] Normalized vectors defining mirror planes.
// * [ideal]  Idealized symmetric coordinates for all atoms of
//     the selection. The coordinates are listed in the order of 
//     increasing atom indices (same order asa returned by the
//     atomselect command ``get {x y z}''). Thus you can use the list
//     to set the atoms of your selection to the ideal coordinates
// * [unique] Index list defining a set of atoms with unique
//     coordinates
// * [orient] Matrix that aligns molecule with GAMESS standard
//     orientation
//
// If a certain item is not present (e.g. no planes or no axes)
// then the corresponding value is an empty list.
// The pair format allows to use the result as a TCL array for
// convenient access of the different return items.

static int vmd_measure_symmetry(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc<2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> [element [<vector>]] [-tol <value>] [-nobonds] [-verbose <level>]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure symmetry: no atom selection", NULL);
    return TCL_ERROR;
  }
  if (!app->molecule_valid_id(sel->molid())) {
    Tcl_AppendResult(interp, "measure symmetry: ",
                     measure_error(MEASURE_ERR_NOMOLECULE), NULL);
    return TCL_ERROR;
  }

  double sigma = 0.1;
  float axis[3];
  int checkelement = 0;
  int checkbonds = 1;
  int order = 0;
  int verbose = 1;
  int impose = 0;
  int imposeinvers = 0;
  int numimposeplan = 0;
  int numimposeaxes = 0;
  int numimposerref = 0;
  float *imposeplan = NULL;
  float *imposeaxes = NULL;
  float *imposerref = NULL;
  int *imposeaxesord = NULL;
  int *imposerreford = NULL;
  AtomSel *idealsel = NULL;

  if (argc>2) {
    int bailout = 0;
    int i;
    for (i=2; (i<argc && !bailout); i++) {
      char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
      // Allow syntax with and without leading dash
      if (argvcur[0]=='-') argvcur++;
      
      if (!strupncmp(argvcur, "tol", CMDLEN)) {
        if (Tcl_GetDoubleFromObj(interp, objv[i+1], &sigma) != TCL_OK) {
          Tcl_AppendResult(interp, " measure symmetry: bad tolerance value", NULL);
          bailout = 1; continue;
        }	
        i++;
      }
      else if (!strupncmp(argvcur, "verbose", CMDLEN)) {
        if (Tcl_GetIntFromObj(interp, objv[i+1], &verbose) != TCL_OK) {
          Tcl_AppendResult(interp, " measure symmetry: bad verbosity level value", NULL);
          bailout = 1; continue;
        }	
        i++;
      }
      else if (!strupncmp(argvcur, "nobonds", CMDLEN)) {
        checkbonds = 0;
      }
      else if (!strupcmp(argvcur, "I")) {
        checkelement = 1;
      }
      else if (!strupncmp(argvcur, "plane", CMDLEN)) {
        if (tcl_get_vector(Tcl_GetStringFromObj(objv[i+1],NULL), axis, interp)!=TCL_OK) {
          bailout = 1; continue;
        }
        checkelement = 2;
        i++;
      }
      else if (!strupncmp(argvcur, "C", 1) || !strupncmp(argvcur, "S", 1)) {
        char *begptr = argvcur+1;
        char *endptr;
        order = strtol(begptr, &endptr, 10);
        if (endptr==begptr || *endptr!='\0') {
          Tcl_AppendResult(interp, " measure symmetry: bad symmetry element format (must be I, C*, S*, plane, where * is the order). ", NULL);
          bailout = 1; continue;
        }

        if (tcl_get_vector(Tcl_GetStringFromObj(objv[i+1],NULL), axis, interp)!=TCL_OK) {
          bailout = 1; continue;
        }
        
        if (!strupncmp(argvcur, "C", 1)) checkelement = 3;
        if (!strupncmp(argvcur, "S", 1)) checkelement = 4;
        i++;
      }
      else if (!strupncmp(argvcur, "idealsel", CMDLEN)) {
        idealsel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[i+1],NULL));
        if (!sel) {
          Tcl_AppendResult(interp, "measure symmetry: no atom selection for idealized coordinates", NULL);
          bailout = 1; continue;
        }
        if (idealsel->molid()!=sel->molid()) {
          Tcl_AppendResult(interp, "measure symmetry: selection and idealsel must be from the same molecule", NULL);
          bailout = 1; continue;
        }
        if (idealsel->selected<sel->selected) {
          Tcl_AppendResult(interp, "measure symmetry: selection must be a subset of idealsel", NULL);
          bailout = 1; continue;
        }
        // make sure sel is subset of idealsel
        int j;
        for (j=0; j<sel->num_atoms; j++) {
          if (sel->on[j] && !idealsel->on[j]) {
            Tcl_AppendResult(interp, "measure symmetry: selection must be a subset of idealsel", NULL);
            bailout = 1; continue;
          }
        }

        i++;
      }
      else if (!strupncmp(argvcur, "imposeinversion", CMDLEN)) {
        imposeinvers = 1;
        impose = 1;
      }
      else if (!strupncmp(argvcur, "imposeplanes", CMDLEN)) {
        // Format: list of normal vectors {{x y z} {x y z} ...}
        int nelem;
        Tcl_Obj **elemListObj;
        if (i+1>=argc ||
            Tcl_ListObjGetElements(interp, objv[i+1], &nelem, &elemListObj) != TCL_OK) {
          Tcl_AppendResult(interp, " measure symmetry: bad syntax for imposeplanes option", NULL);
          bailout = 1; continue;
        }
        float *elem = new float[3L*nelem];
        int k;
        for (k=0; k<nelem; k++) {
          int nobj;
          Tcl_Obj **vecObj;
          if (Tcl_ListObjGetElements(interp, elemListObj[k], &nobj, &vecObj) != TCL_OK) {
            delete [] elem;
            Tcl_AppendResult(interp, " measure symmetry: bad syntax for imposeplanes option", NULL);
            bailout = 1; continue;
          }
          if (nobj!=3) {
            delete [] elem;
            Tcl_AppendResult(interp, " measure symmetry imposeplanes: vector must have 3 elements", NULL);
            bailout = 1; continue;
          }
          
          int m;
          for (m=0; m<3; m++) {
            double d;
            if (Tcl_GetDoubleFromObj(interp, vecObj[m], &d) != TCL_OK) {
              delete [] elem;
              bailout = 1; continue;
            }
            elem[3L*k+m] = (float)d;
          }
        }
        if (imposeplan) delete [] imposeplan;
        imposeplan = elem;
        numimposeplan = nelem;
        impose = 1;
        i++;
      }
      else if (!strupncmp(argvcur, "imposeaxes", CMDLEN) ||
               !strupncmp(argvcur, "imposerotref", CMDLEN)) {
        // Format: list of axes and orders {{x y z} 3 {x y z} 2 ...}
        int nelem;
        Tcl_Obj **elemListObj;
        if (i+1>=argc ||
            Tcl_ListObjGetElements(interp, objv[i+1], &nelem, &elemListObj) != TCL_OK ||
            nelem%2) {
          Tcl_AppendResult(interp, " measure symmetry: bad syntax for imposeaxes/imposerotref option", NULL);
          bailout = 1; continue;
        }
        nelem /= 2;
     
        if (nelem<=0) {
          i++;
          continue;
        }
        float *elem = new float[3L*nelem];
        int *axorder = new int[nelem];
        int k;
        for (k=0; k<nelem; k++) {
          int nobj;
          Tcl_Obj **vecObj;
          if (Tcl_ListObjGetElements(interp, elemListObj[2L*k], &nobj, &vecObj) != TCL_OK) {
            delete [] elem;
            delete [] axorder;
            Tcl_AppendResult(interp, " measure symmetry impose: bad syntax for axis vector", NULL);
            bailout = 1; continue;
          }
          if (Tcl_GetIntFromObj(interp, elemListObj[2L*k+1], &axorder[k]) != TCL_OK) {
            delete [] elem;
            delete [] axorder;
            bailout = 1; continue;
          }
          if (nobj!=3) {
            delete [] elem;
            delete [] axorder;
            Tcl_AppendResult(interp, " measure symmetry impose: axis vector must have 3 elements", NULL);
            bailout = 1; continue;
          }
          
          int m;
          for (m=0; m<3; m++) {
            double d;
            if (Tcl_GetDoubleFromObj(interp, vecObj[m], &d) != TCL_OK) {
              delete [] elem;
              delete [] axorder;
              bailout = 1; continue;
            }
            elem[3L*k+m] = (float)d;
          }
        }
        if (!strupncmp(argvcur, "imposeaxes", CMDLEN)) {
          if (imposeaxes)    delete [] imposeaxes;
          if (imposeaxesord) delete [] imposeaxesord;
          imposeaxes = elem;
          imposeaxesord = axorder;
          numimposeaxes = nelem;
        } else if (!strupncmp(argvcur, "imposerotref", CMDLEN)) {
          if (imposerref)    delete [] imposerref;
          if (imposerreford) delete [] imposerreford;
          imposerref = elem;
          imposerreford = axorder;
          numimposerref = nelem;
        }
      
        impose = 1;
        i++;
      }
      else {
        Tcl_AppendResult(interp, " measure symmetry: unrecognized option ", NULL);
        Tcl_AppendResult(interp, argvcur);
        Tcl_AppendResult(interp, ".\n Usage: measure symmetry <selection> [element [<vector>]] [-tol <value>] [-nobonds] [-verbose <level>]", NULL);
        bailout = 1; continue;
      }
    }

    if (bailout) {
      if (imposeplan) delete [] imposeplan;
      if (imposeaxes) delete [] imposeaxes;
      if (imposerref) delete [] imposerref;
      if (imposeaxesord) delete [] imposeaxesord;
      if (imposerreford) delete [] imposerreford;
      return TCL_ERROR;
    }
  }

  // Initialize
  Symmetry sym = Symmetry(sel, app->moleculeList, verbose);

  // Set tolerance for atomic overlapping
  sym.set_overlaptol(float(sigma));

  // Wether to include bonds into the analysis
  sym.set_checkbonds(checkbonds);

  if (checkelement) {
    // We are interested only in the score for a specific
    // symmetry element.
    float overlap = 0.0;
    if (checkelement==1) {
      overlap = sym.score_inversion();
    }
    else if (checkelement==2) {
      overlap = sym.score_plane(axis);
    }
    else if (checkelement==3) {
      overlap = sym.score_axis(axis, order);
    }
    else if (checkelement==4) {
      overlap = sym.score_rotary_reflection(axis, order);
    }

    Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(overlap));
    Tcl_SetObjResult(interp, tcl_result);
    return TCL_OK;
  }

  if (impose) {
    sym.impose(imposeinvers, numimposeplan, imposeplan,
               numimposeaxes, imposeaxes, imposeaxesord,
               numimposerref, imposerref, imposerreford);
    if (imposeplan) delete [] imposeplan;
    if (imposeaxes) delete [] imposeaxes;
    if (imposerref) delete [] imposerref;
    if (imposeaxesord) delete [] imposeaxesord;
    if (imposerreford) delete [] imposerreford;
  }

  int ret_val = sym.guess(float(sigma));
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure symmetry: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  int natoms = sel->selected;
  Symmetry *s = &sym;

  if (idealsel) {
    Symmetry *isym = new Symmetry(idealsel, app->moleculeList, verbose);
    isym->set_overlaptol(float(sigma));
    isym->set_checkbonds(checkbonds);
    int j;
    float *plane = new float[3L*sym.numplanes()];
    for (j=0; j<sym.numplanes(); j++) {
      vec_copy(&plane[3L*j], sym.plane(j));
    }
    int *axisorder = new int[sym.numaxes()];
    float *axis = new float[3L*sym.numaxes()];
    for (j=0; j<sym.numaxes(); j++) {
      axisorder[j] = sym.get_axisorder(j);
      vec_copy(&axis[3L*j], sym.axis(j));
    }
    int *rrorder = new int[sym.numrotreflect()];
    float *rraxis = new float[3L*sym.numrotreflect()];
    for (j=0; j<sym.numrotreflect(); j++) {
      rrorder[j] = sym.get_rotreflectorder(j);
      vec_copy(&rraxis[3L*j], sym.rotreflect(j));
    }
    // XXX must check if sel is subset of idealsel
    int k=0, m=0;
    for (j=0; j<sel->num_atoms; j++) {
      if (idealsel->on[j]) {
        if (sel->on[j]) {
          vec_copy(isym->idealpos(k), sym.idealpos(m));
          m++;
        }
        k++;
      }
    }
    isym->impose(sym.has_inversion(),
                 sym.numplanes(), plane,
                 sym.numaxes(),   axis, axisorder,
                 sym.numrotreflect(), rraxis, rrorder);

    ret_val = isym->guess(float(sigma));
    if (ret_val < 0) {
      Tcl_AppendResult(interp, "measure symmetry: ", measure_error(ret_val), NULL);
      return TCL_ERROR;
    }

    natoms = idealsel->selected;
    s = isym;
  }

  int pgorder;
  char pointgroup[6];
  s->get_pointgroup(pointgroup, &pgorder);

  int i;
  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("pointgroup", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(pointgroup, -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("order", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewIntObj(pgorder));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("rmsd", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(s->get_rmsd()));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("elements", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(s->get_element_summary(), -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("missing", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(s->get_missing_elements(), -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("additional", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(s->get_additional_elements(), -1));

  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("com", -1));
  Tcl_Obj *invObj  = Tcl_NewListObj(0, NULL);
  float *com = s->center_of_mass();
  Tcl_ListObjAppendElement(interp, invObj, Tcl_NewDoubleObj(com[0]));
  Tcl_ListObjAppendElement(interp, invObj, Tcl_NewDoubleObj(com[1]));
  Tcl_ListObjAppendElement(interp, invObj, Tcl_NewDoubleObj(com[2]));
  Tcl_ListObjAppendElement(interp, tcl_result, invObj);

  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("inertia", -1));
  Tcl_Obj *inertListListObj  = Tcl_NewListObj(0, NULL);
  float *inertia  = s->get_inertia_axes();
  float *eigenval = s->get_inertia_eigenvals();
  int   *unique   = s->get_inertia_unique();
  for (i=0; i<3; i++) {
    Tcl_Obj *inertObj  = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, inertObj, Tcl_NewDoubleObj(inertia[3L*i]));
    Tcl_ListObjAppendElement(interp, inertObj, Tcl_NewDoubleObj(inertia[3L*i+1]));
    Tcl_ListObjAppendElement(interp, inertObj, Tcl_NewDoubleObj(inertia[3L*i+2]));
    Tcl_Obj *inertListObj  = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, inertListObj, inertObj);
    Tcl_ListObjAppendElement(interp, inertListObj, Tcl_NewDoubleObj(eigenval[i]));
    Tcl_ListObjAppendElement(interp, inertListObj, Tcl_NewIntObj(unique[i]));
    Tcl_ListObjAppendElement(interp, inertListListObj, inertListObj);
  }
  Tcl_ListObjAppendElement(interp, tcl_result, inertListListObj);

 
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("inversion", -1));
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewIntObj(s->has_inversion()));


  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("axes", -1));
  Tcl_Obj *axesListListObj  = Tcl_NewListObj(0, NULL);
  for (i=0; i<s->numaxes(); i++) {
    Tcl_Obj *axesObj  = Tcl_NewListObj(0, NULL);
    //printf("Tcl %i: %.2f %.2f %.2f\n", i, s->axis(i)[0], s->axis(i)[1], s->axis(i)[2]);
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(s->axis(i)[0]));
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(s->axis(i)[1]));
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(s->axis(i)[2]));
    Tcl_Obj *axesListObj  = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, axesListObj, axesObj);
    Tcl_ListObjAppendElement(interp, axesListObj, Tcl_NewIntObj(s->get_axisorder(i)));
    int axistype = s->get_axistype(i);
    if (axistype & PRINCIPAL_AXIS && 
        !(!s->is_spherical_top() && (axistype & PERPENDICULAR_AXIS))) {
      Tcl_ListObjAppendElement(interp, axesListObj, Tcl_NewStringObj("principal", -1));
    } else if (!s->is_spherical_top() && (axistype & PERPENDICULAR_AXIS)) {
      Tcl_ListObjAppendElement(interp, axesListObj, Tcl_NewStringObj("perpendicular", -1));
    } else {
      Tcl_ListObjAppendElement(interp, axesListObj, Tcl_NewListObj(0, NULL));
    }
    Tcl_ListObjAppendElement(interp, axesListListObj, axesListObj);
  }
  Tcl_ListObjAppendElement(interp, tcl_result, axesListListObj);


  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("rotreflect", -1));
  Tcl_Obj *rraxesListListObj  = Tcl_NewListObj(0, NULL);
  for (i=0; i<s->numrotreflect(); i++) {
    Tcl_Obj *axesObj  = Tcl_NewListObj(0, NULL);
    //printf("Tcl %i: %.2f %.2f %.2f\n", i, s->axis(i)[0], s->axis(i)[1], s->axis(i)[2]);
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(s->rotreflect(i)[0]));
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(s->rotreflect(i)[1]));
    Tcl_ListObjAppendElement(interp, axesObj, Tcl_NewDoubleObj(s->rotreflect(i)[2]));
    Tcl_Obj *rraxesListObj  = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, rraxesListObj, axesObj);
    Tcl_ListObjAppendElement(interp, rraxesListObj, Tcl_NewIntObj(s->get_rotreflectorder(i)));
    Tcl_ListObjAppendElement(interp, rraxesListObj, Tcl_NewIntObj(s->get_rotrefltype(i)));
    Tcl_ListObjAppendElement(interp, rraxesListListObj, rraxesListObj);
  }
  Tcl_ListObjAppendElement(interp, tcl_result, rraxesListListObj);


  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("planes", -1));
  Tcl_Obj *planesListListObj = Tcl_NewListObj(0, NULL);
  for (i=0; i<s->numplanes(); i++) {
    Tcl_Obj *planesObj = Tcl_NewListObj(0, NULL);
    //printf("Tcl %i: %.2f %.2f %.2f\n", i, s->plane(i)[0], s->plane(i)[1], s->plane(i)[2]);
    Tcl_ListObjAppendElement(interp, planesObj, Tcl_NewDoubleObj(s->plane(i)[0]));
    Tcl_ListObjAppendElement(interp, planesObj, Tcl_NewDoubleObj(s->plane(i)[1]));
    Tcl_ListObjAppendElement(interp, planesObj, Tcl_NewDoubleObj(s->plane(i)[2]));
    Tcl_Obj *planesListObj = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, planesListObj, planesObj);
    switch (s->get_planetype(i)) {
    case 1:
      Tcl_ListObjAppendElement(interp, planesListObj, Tcl_NewStringObj("vertical", -1));
      break;
    case 3:
      Tcl_ListObjAppendElement(interp, planesListObj, Tcl_NewStringObj("dihedral", -1));
      break;
    case 4:
      Tcl_ListObjAppendElement(interp, planesListObj, Tcl_NewStringObj("horizontal", -1));
      break;
    default:
      Tcl_ListObjAppendElement(interp, planesListObj, Tcl_NewListObj(0, NULL));
    }
    Tcl_ListObjAppendElement(interp, planesListListObj, planesListObj);
  }
  Tcl_ListObjAppendElement(interp, tcl_result, planesListListObj);


  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("ideal", -1));
  Tcl_Obj *idealcoorListObj = Tcl_NewListObj(0, NULL);
  for (i=0; i<natoms; i++) {
    Tcl_Obj *idealcoorObj = Tcl_NewListObj(0, NULL);
    //printf("Tcl %i: %.2f %.2f %.2f\n", i, s->plane(i)[0], s->plane(i)[1], s->plane(i)[2]);
    Tcl_ListObjAppendElement(interp, idealcoorObj, Tcl_NewDoubleObj(s->idealpos(i)[0]));
    Tcl_ListObjAppendElement(interp, idealcoorObj, Tcl_NewDoubleObj(s->idealpos(i)[1]));
    Tcl_ListObjAppendElement(interp, idealcoorObj, Tcl_NewDoubleObj(s->idealpos(i)[2]));
    Tcl_ListObjAppendElement(interp, idealcoorListObj, idealcoorObj);
  }
  Tcl_ListObjAppendElement(interp, tcl_result, idealcoorListObj);

  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("unique", -1));
  Tcl_Obj *uniquecoorListObj = Tcl_NewListObj(0, NULL);
  for (i=0; i<natoms; i++) {
    if (!s->get_unique_atom(i)) continue; 
    Tcl_ListObjAppendElement(interp, uniquecoorListObj, Tcl_NewIntObj(i));
  }
  Tcl_ListObjAppendElement(interp, tcl_result, uniquecoorListObj);

  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj("orient", -1));
  Matrix4 *orient = s->get_orientation();
  Tcl_Obj *matrixObj = Tcl_NewListObj(0, NULL);
  for (i=0; i<4; i++) {
    Tcl_Obj *rowObj = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, rowObj, Tcl_NewDoubleObj(orient->mat[i+0]));
    Tcl_ListObjAppendElement(interp, rowObj, Tcl_NewDoubleObj(orient->mat[i+4]));
    Tcl_ListObjAppendElement(interp, rowObj, Tcl_NewDoubleObj(orient->mat[i+8]));
    Tcl_ListObjAppendElement(interp, rowObj, Tcl_NewDoubleObj(orient->mat[i+12]));
    Tcl_ListObjAppendElement(interp, matrixObj, rowObj);
  }
  Tcl_ListObjAppendElement(interp, tcl_result, matrixObj);

  Tcl_SetObjResult(interp, tcl_result);
  if (idealsel) {
    delete s;
  }
  return TCL_OK;

}

// Function: vmd_measure_transoverlap <selection>
//  Returns: The structural overlap of a selection with a copy of itself
//           that is transformed according to a given transformation matrix.
//           The normalized sum over all gaussian function values of the pair
//           distances between atoms in the original and the transformed
//           selection.
//    Input: the atom selection and the transformation matrix.
//   Option: with -sigma you can specify the sigma value of the overlap 
//           gaussian function. The default is 0.1 Angstrom.
static int vmd_measure_trans_overlap(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  if (argc!=3 && argc!=5) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> <matrix> [-sigma <value>]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "measure transoverlap: no atom selection", NULL);
    return TCL_ERROR;
  }
  if (!app->molecule_valid_id(sel->molid())) {
    Tcl_AppendResult(interp, "measure transoverlap: ",
		     measure_error(MEASURE_ERR_NOMOLECULE), NULL);
    return TCL_ERROR;
  }

  // Get the transformation matrix
  Matrix4 trans;
  if (tcl_get_matrix("measure transoverlap: ",interp,objv[2], trans.mat) != TCL_OK) {
    return TCL_ERROR;
  }

  double sigma = 0.1;
  if (argc==5) {
    if (!strupncmp(Tcl_GetStringFromObj(objv[3],NULL), "-sigma", CMDLEN)) {
      if (Tcl_GetDoubleFromObj(interp, objv[4], &sigma) != TCL_OK) {
	Tcl_AppendResult(interp, " measure transoverlap: bad sigma value", NULL);
	return TCL_ERROR;
      }	
    } else {
      Tcl_AppendResult(interp, " measure transoverlap: unrecognized option\n", NULL);
      Tcl_AppendResult(interp, " Usage: measure transoverlap <sel> <matrix> [-sigma <value>]", NULL);
      return TCL_ERROR;
    }
  }

  int maxnatoms = 100;// XXX
  float overlap;
  int ret_val = measure_trans_overlap(sel, app->moleculeList, &trans,
                                      float(sigma), NOSKIP_IDENTICAL, 
                                      maxnatoms, overlap);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "measure transoverlap: ", measure_error(ret_val), NULL);
    return TCL_ERROR;
  }

  Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
  Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewDoubleObj(overlap));
  Tcl_SetObjResult(interp, tcl_result);

  return TCL_OK;
}



//
// Raycasting brute-force approach by Juan R. Perilla <juan@perilla.me>
//
int vmd_measure_volinterior(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int verbose = 0;
  if (argc < 3) {
     // "     options: --allframes (average over all frames)\n"
    Tcl_SetResult(interp, (char *) "usage: volinterior "
 		  " <selection1> [options]\n"
		  "-res (default 10.0)\n"
		  "-spacing (default res/3)\n"
		  "-loadmap (load synth map)"
		  "-mol molid [default: top]\n"
		  "-vol volid [0]\n"
		  "-nrays number of rays to cast [6]\n"
		  "-isovalue [float: 1.0]\n"
		  "-verbose\n"
		  "-overwrite volid\n"
,
      TCL_STATIC);
    return TCL_ERROR;
  }

  //atom selection
  AtomSel *sel = NULL;
  sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_AppendResult(interp, "volinterior: no atom selection.", NULL);
    return TCL_ERROR;
  }
  if (!sel->selected) {
    Tcl_AppendResult(interp, "volinterior: no atoms selected.", NULL);
    return TCL_ERROR;
  }
  if (!app->molecule_valid_id(sel->molid())) {
    Tcl_AppendResult(interp, "invalid mol id.", NULL);
    return TCL_ERROR;
  }
  int loadsynthmap = 0;
  int molid = -1;
  int volid = -1;
  int volidOverwrite = -1;
  int overwrite=0;
  int nrays=6;
  long Nout = 0;
  float isovalue=1.0;
  float radscale;
  double resolution = 10.0;
  double gspacing;
  MoleculeList *mlist = app->moleculeList;
  Molecule *mymol = mlist->mol_from_id(sel->molid());
//  const char *outputmap = NULL;

  for (int i=0; i < argc; i++) {
    char *opt = Tcl_GetStringFromObj(objv[i], NULL);
    if (!strcmp(opt, "-mol")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-vol")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volid) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt, "-overwrite")) {
      if (i == argc-1){
        Tcl_AppendResult(interp, "No volume id specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &volidOverwrite) != TCL_OK) {
        Tcl_AppendResult(interp, "\n volume id incorrectly specified",NULL);
        return TCL_ERROR;
      }
      overwrite=1;
    }

    // Calculate synthmap
    if (!strcmp(opt, "-loadmap")) {
      loadsynthmap = 1;
    }
    if (!strcmp(opt, "-mol")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No molid specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &molid) != TCL_OK) {
        Tcl_AppendResult(interp, "\n molid incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
   if (!strcmp(opt, "-res")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No resolution specified",NULL);
        return TCL_ERROR;
      } else if (Tcl_GetDoubleFromObj(interp, objv[i+1], &resolution) != TCL_OK) { 
        Tcl_AppendResult(interp, "\nResolution incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }
    if (!strcmp(opt, "-spacing")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No spacing specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetDoubleFromObj(interp, objv[i+1], &gspacing) != TCL_OK) {
        Tcl_AppendResult(interp, "\nspacing incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt,"-nrays")) {
      if (i == argc-1) {
        Tcl_AppendResult(interp, "No number of rays specified",NULL);
        return TCL_ERROR;
      } else if ( Tcl_GetIntFromObj(interp, objv[i+1], &nrays) != TCL_OK) {
        Tcl_AppendResult(interp, "\n number of rays incorrectly specified",NULL);
        return TCL_ERROR;
      }
    }

    if (!strcmp(opt,"-isovalue")) {
      double tmp;
      if (i == argc-1) {
	Tcl_AppendResult(interp,"No isovalue specified",NULL);
	return TCL_ERROR;
      } else if (Tcl_GetDoubleFromObj(interp,objv[i+1],&tmp) != TCL_OK) {
	Tcl_AppendResult(interp,"\n Isovalue incorrectly specified", NULL);
	  return TCL_ERROR;
      }
      isovalue=(float) tmp;
    }

//    if (!strcmp(opt, "-o")) {
//      if (i == argc-1) {
//	//	Tcl_AppendResult(interp, "No output file specified",NULL);
//	//        return TCL_ERROR;
//	if (verbose) 
//	  printf("No output file specified.");
//      } else {
//        outputmap = Tcl_GetStringFromObj(objv[i+1], NULL);
//      }
//    }

    if (!strcmp(opt, "-verbose") || (getenv("VMDVOLINVERBOSE") != NULL)) {
      verbose = 1;
    }
  }


  Molecule *volmolOverwrite = NULL;
  VolumetricData *volmapOverwrite = NULL;
  if (overwrite==1) {
    volmolOverwrite = mlist->mol_from_id(molid);
    volmapOverwrite = volmolOverwrite->modify_volume_data(volidOverwrite);
    if(volmapOverwrite==NULL) {
      Tcl_AppendResult(interp, "\n no overwrite volume correctly specified",NULL);
      return TCL_ERROR;
    }
  }
  
  VolumetricData *target = NULL;
  VolumetricData *volmapA = NULL;
  const float *framepos = sel->coordinates(app->moleculeList);
  const float *radii = mymol->radius();
  radscale=.2*resolution;
  if (gspacing == 0) {
    gspacing=resolution*0.33;
  }
  int quality=0;
  QuickSurf *qs = new QuickSurf();
  if (resolution >= 9)
    quality = 0;
  else
    quality = 3;
  if (molid != -1 && volid != -1 ) {
    Molecule *volmol = mlist->mol_from_id(molid);
    volmapA = (VolumetricData *) volmol->get_volume_data(volid);
    if(volmapA==NULL) {
      Tcl_AppendResult(interp, "\n no target volume correctly specified",NULL);
      return TCL_ERROR;
    }
  } else {
    if (verbose) printf("\n Calculating grid ... \n");
    int cuda_err=-1;
#if defined(VMDCUDA)
    cuda_err = vmd_cuda_calc_density(sel,app->moleculeList,quality,radscale,gspacing,&volmapA, NULL, NULL, verbose);
#endif
    if (cuda_err == -1 ) {
      if (verbose) printf("Using CPU version of the code ... \n");
	volmapA = qs->calc_density_map(sel,mymol,framepos,radii,
				       quality,(float)radscale, (float) gspacing);
      }
    if (verbose) printf("Done.\n");
  }

  if (volmapA == NULL) {
    Tcl_AppendResult(interp, "\n no test volume or molid correctly specified",NULL);
    return TCL_ERROR;
  }
  target=CreateEmptyGrid(volmapA);    
  // Normalize dir vector 
  // TODO: Let user define direction vectors

  //vec_normalize(rayDir);
  
  // Create Grid
#if 0
  float rayDir[3] = {1,0,0};
  Nout += RaycastGrid((const) volmapA,target,isovalue,rayDir);
  rayDir[0] = -1; rayDir[1]=0; rayDir[2]=0;
  Nout += RaycastGrid((const VolumetricData *) volmapA,target,isovalue,rayDir);
  rayDir[0] = 0; rayDir[1]=1; rayDir[2]=0;
  Nout += RaycastGrid((const VolumetricData *) volmapA,target,isovalue,rayDir);
  rayDir[0] = 0; rayDir[1]=-1; rayDir[2]=0;
  Nout += RaycastGrid((const VolumetricData *) volmapA,target,isovalue,rayDir);
  rayDir[0] = 0; rayDir[1]=0; rayDir[2]=1;
  Nout += RaycastGrid((const VolumetricData *) volmapA,target,isovalue,rayDir);
  rayDir[0] = 0; rayDir[1]=0; rayDir[2]=-1;
  Nout += RaycastGrid((const VolumetricData *) volmapA,target,isovalue,rayDir);
#endif
#if 0
  if (verbose) printf("Marking down boundary.\n");
  long nVoxIso=markIsoGrid((const VolumetricData *) volmapA, target,isovalue);
  if (verbose) printf("Casting rays ...\n");
  float rayDir[3] = {0.1f,0.1f,1.0f};
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 1 %ld \n",Nout);
  rayDir[0] = -0.1f; rayDir[1]=-0.1f; rayDir[2]=-1.0f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 2 %ld \n",Nout);
  rayDir[0] = 0.1f; rayDir[1]=1.0f; rayDir[2]=0.1f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 3 %ld \n",Nout);
  rayDir[0] = -0.1f; rayDir[1]=-1.0f; rayDir[2]=-0.1f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 4 %ld \n",Nout);
  rayDir[0] = 1.0f; rayDir[1]=0.1f; rayDir[2]=0.1f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 5 %ld \n",Nout);
  rayDir[0] = -1.0f; rayDir[1]=-0.1f; rayDir[2]=-0.1f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 6 %ld \n",Nout);
  rayDir[0] = -0.5f; rayDir[1]=-0.5f; rayDir[2]=-0.5f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 7 %ld \n",Nout);
  rayDir[0] = 0.5f; rayDir[1]=0.5f; rayDir[2]=0.5f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 9 %ld \n",Nout);
  rayDir[0] = -0.5f; rayDir[1]=0.5f; rayDir[2]=0.5f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 10 %ld \n",Nout);
  rayDir[0] = 0.5f; rayDir[1]=-0.5f; rayDir[2]=0.5f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 11 %ld \n",Nout);
  rayDir[0] = 0.5f; rayDir[1]=0.5f; rayDir[2]=-0.5f;
  Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
  if (verbose) printf("Dir 12 %ld \n",Nout);
#endif
  if (verbose) printf("Marking down boundary.\n");
  long nVoxIso=markIsoGrid((const VolumetricData *) volmapA, target,isovalue);
  if (verbose) printf("Casting rays ...\n");
  float rayDir[3] = {0.1f,0.1f,1.0f};

  static const float RAND_MAX_INV = 1.0f/VMD_RAND_MAX;
  // Seed the random number generator before each calculation.  
  vmd_srandom(103467829);
  // Cast nrays uniformly over the sphere
  for (int ray=0; ray < nrays; ray++) { 
    float u1 = (float) vmd_random();
    float u2 = (float) vmd_random();
    float z = 2.0f*u1*RAND_MAX_INV -1.0f;
    float phi = (float) (2.0f*VMD_PI*u2*RAND_MAX_INV);
    float R = sqrtf(1.0f-z*z);
    rayDir[0] = R*cosf(phi);
    rayDir[1] = R*sinf(phi);
    rayDir[2] = z;
    Nout += volin_threaded((const VolumetricData *) volmapA,target,isovalue,rayDir);
    if (verbose) printf("Ray(%d)  (%4.3f %4.3f %4.3f). Voxels : %ld \n",ray+1,rayDir[0],rayDir[1],rayDir[2],Nout);    
  }
  if (verbose) printf("Done.\n");


  if (verbose) printf("Counting voxels above isovalue ...\n");
  
  Nout=countIsoGrids((const VolumetricData *) target,5.0f);
  long nIn=countIsoGrids((const VolumetricData *) target,0.0f);
  
  if (loadsynthmap == 1 ) {
    app->molecule_add_volumetric(molid, volmapA->name, volmapA->origin, 
				 volmapA->xaxis, volmapA->yaxis, volmapA->zaxis,
				 volmapA->xsize, volmapA->ysize, volmapA->zsize, volmapA->data);
    volmapA->compute_volume_gradient();
    volmapA->data = NULL; // prevent destruction of density array;  
  }
  if (overwrite != 1 ) {
    app->molecule_add_volumetric(molid, target->name, target->origin, 
				 target->xaxis, target->yaxis, target->zaxis,
				 target->xsize, target->ysize, target->zsize, target->data);
    target->compute_volume_gradient();
    target->data = NULL; // prevent destruction of density array;  
  } else {
    if (verbose) printf("Overwriting volume ... \n");
    volmapOverwrite->name = target -> name;
    volmapOverwrite->xsize = target->xsize;
    volmapOverwrite->ysize = target->ysize;
    volmapOverwrite->zsize = target->zsize;
    volmapOverwrite->data = target->data;
    target->data=NULL;
    volmapOverwrite->compute_volume_gradient();
    if (verbose) printf("Freeing up memory.\n ");
    //    delete(tmp);
    if (verbose) printf("Done.\n ");
  }

  if (molid != -1 && volid != -1 ) {
    target->data = NULL; // prevent destruction of density array;  
  } else {
    delete qs;
    delete volmapA;
  }
  printf("VolIn: %ld external voxels.\n",Nout);

  long Ntotal = target->xsize * target->ysize * target->zsize;
  Tcl_Obj *tcl_result = Tcl_NewListObj(0,NULL);
  Tcl_ListObjAppendElement(interp,tcl_result,Tcl_NewLongObj(Ntotal));
  Tcl_ListObjAppendElement(interp,tcl_result,Tcl_NewLongObj(Nout));
  Tcl_ListObjAppendElement(interp,tcl_result,Tcl_NewLongObj(nIn));
  Tcl_ListObjAppendElement(interp,tcl_result,Tcl_NewLongObj(nVoxIso));
  Tcl_SetObjResult(interp,tcl_result);
  return TCL_OK;
}



int obj_measure(ClientData cd, Tcl_Interp *interp, int argc,
                            Tcl_Obj * const objv[]) {

  if (argc < 2) {
    Tcl_SetResult(interp, 
      (char *) "usage: measure <command> [args...]\n"
      "\nMeasure Commands:\n"
      "  avpos <sel> [first <first>] [last <last>] [step <step>] -- average position\n"
      "  center <sel> [weight <weights>]          -- geometrical (or weighted) center\n"
      "  centerperresidue <sel> [weight <weights>]  -- geometrical center for every residue in sel\n"
      "  cluster <sel> [num <#clusters>] [distfunc <flag>] [cutoff <cutoff>]\n"
      "          [first <first>] [last <last>] [step <step>] [selupdate <bool>]\n"
      "          [weight <weights>]\n"
      "     -- perform a cluster analysis (cluster similar timesteps)\n"
      "  clustsize <sel> [cutoff <float>] [minsize <num>] [numshared <num>]\n"
      "            [usepbc <bool>] [storesize <fieldname>] [storenum <fieldname>]\n"
      "     -- perform a cluster size analysis (find clusters of atoms)\n"
      "  contacts <cutoff> <sel1> [<sel2>]        -- list contacts\n" 
      "  dipole <sel> [-elementary|-debye] [-geocenter|-masscenter|-origincenter]\n"
      "     -- dipole moment\n"
      "  fit <sel1> <sel2> [weight <weights>] [order <index list>]\n"
      "     -- transformation matrix from selection 1 to 2\n"
      "  gofr <sel1> <sel2> [delta <value>] [rmax <value>] [usepbc <bool>]\n"
      "     [selupdate <bool>] [first <first>] [last <last>] [step <step>]\n"
      "     -- atomic pair distribution function g(r)\n"
      "  hbonds <cutoff> <angle> <sel1> [<sel2>]\n"
      "     -- list donors, acceptors, hydrogens involved in hydrogen bonds\n"
      "  inverse <matrix>                         -- inverse matrix\n"
      "  inertia <sel> [-moments] [-eigenvals]    -- COM and principle axes of inertia\n"
      "  minmax <sel> [-withradii]                -- bounding box\n"
      "  rgyr <sel> [weight <weights>]            -- radius of gyration\n"
      "  rmsd <sel1> <sel2> [weight <weights>]    -- RMS deviation\n"
      "  rmsdperresidue <sel1> <sel2> [weight <weights>] -- deviation per residue\n"
      "  rmsf <sel> [first <first>] [last <last>] [step <step>] -- RMS fluctuation\n"
      "  rmsfperresidue <sel> [first <first>] [last <last>] [step <step>] -- fluctuation per residue\n"
      "  sasa <srad> <sel> [-points <varname>] [-restrict <restrictedsel>]\n"
      "     [-samples <numsamples>]               -- solvent-accessible surface area\n"
      "  sumweights <sel> weight <weights>        -- sum of selected weights\n"
      "  bond {{<atomid1> [<molid1>]} {<atomid2> [<molid2>]}}\n"
      "     [molid <default mol>] [frame <frame|all|last> | first <first> last <last>]\n"
      "     -- bond length between atoms 1 and 2\n"
      "  angle {{<atomid1> [<molid1>]} {<atomid2> [<molid2>]} {<atomid3> [<molid3>]}}\n"
      "     [molid <default mol>] [frame <frame|all|last> | first <first> last <last>]\n"
      "     -- angle between atoms 1-3\n"
      "  dihed {{<atomid1> [<molid1>]} ... {<atomid4> [<molid4>]}}\n"
      "     [molid <default mol>] [frame <frame|all|last> | first <first> last <last>]\n"
      "      -- dihedral angle between atoms 1-4\n"
      "  imprp {{<atomid1> [<molid1>]} ... {<atomid4> [<molid4>]}}\n"
      "     [molid <default mol>] [frame <frame|all|last> | first <first> last <last>]\n"
      "     -- improper angle between atoms 1-4\n"
      // FIXME: Complete 'measure energy' usage info here?
      "  energy bond|angle|dihed|impr|vdw|elec     -- compute energy\n"
      "  surface <sel> <gridsize> <radius> <thickness> -- surface of selection\n"
      "  pbc2onc <center> [molid <default>] [frame <frame|last>]\n"
      "     --  transformation matrix to wrap a nonorthogonal PBC unit cell\n"
      "         into an orthonormal cell\n"
      "  pbcneighbors <center> <cutoff> [sel <sel>] [align <matrix>] [molid <default>]\n"
      "     [frame <frame|last>] [boundingbox <PBC|{<mincoord> <maxcoord>}>]\n"
      "     -- all image atoms that are within cutoff Angstrom of the pbc unit cell\n"
      "  symmetry <sel> [element [<vector>]] [-tol <value>] [-nobonds] [-verbose <level>]\n"
      "  transoverlap <sel> <matrix> [-sigma <value>]\n"
      "     -- overlap of a structure with a transformed copy of itself\n",
      TCL_STATIC);
    return TCL_ERROR;
  }
  VMDApp *app = (VMDApp *)cd;
  char *argv1 = Tcl_GetStringFromObj(objv[1],NULL);
  if (!strupncmp(argv1, "avpos", CMDLEN))
    return vmd_measure_avpos(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "center", CMDLEN))
    return vmd_measure_center(app, argc-1, objv+1, interp);
#if 1
  // XXX in test
  else if (!strupncmp(argv1, "centerperresidue", CMDLEN))
    return vmd_measure_centerperresidue(app, argc-1, objv+1, interp);
#endif
  else if (!strupncmp(argv1, "cluster", CMDLEN))
    return vmd_measure_cluster(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "clustsize", CMDLEN))
    return vmd_measure_clustsize(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "contacts", CMDLEN))
    return vmd_measure_contacts(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "dipole", CMDLEN))
    return vmd_measure_dipole(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "fit", CMDLEN)) 
    return vmd_measure_fit(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "minmax", CMDLEN))
    return vmd_measure_minmax(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "gofr", CMDLEN))
    return vmd_measure_gofr(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "rdf", CMDLEN))
    return vmd_measure_rdf(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "hbonds", CMDLEN))
    return vmd_measure_hbonds(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "inverse", CMDLEN)) 
    return vmd_measure_inverse(argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "inertia", CMDLEN)) 
    return vmd_measure_inertia(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "rgyr", CMDLEN))
    return vmd_measure_rgyr(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "rmsd", CMDLEN))
    return vmd_measure_rmsd(app, argc-1, objv+1, interp);
#if 1
  // XXX in test
  else if (!strupncmp(argv1, "rmsdperresidue", CMDLEN))
    return vmd_measure_rmsdperresidue(app, argc-1, objv+1, interp);
#endif
#if 1
  else if (!strupncmp(argv1, "rmsd_qcp", CMDLEN))
    return vmd_measure_rmsd_qcp(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "rmsdmat_qcp", CMDLEN))
    return vmd_measure_rmsdmat_qcp(app, argc-1, objv+1, interp);
#endif
  else if (!strupncmp(argv1, "rmsf", CMDLEN))
    return vmd_measure_rmsf(app, argc-1, objv+1, interp);
#if 1
  // XXX in test
  else if (!strupncmp(argv1, "rmsfperresidue", CMDLEN))
    return vmd_measure_rmsfperresidue(app, argc-1, objv+1, interp);
#endif
  else if (!strupncmp(argv1, "sasa", CMDLEN))
    return vmd_measure_sasa(app, argc-1, objv+1, interp);
#if 1
  else if (!strupncmp(argv1, "sasalist", CMDLEN))
    return vmd_measure_sasalist(app, argc-1, objv+1, interp);
#endif
  else if (!strupncmp(argv1, "sumweights", CMDLEN))
    return vmd_measure_sumweights(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "imprp", CMDLEN))
    return vmd_measure_dihed(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "dihed", CMDLEN))
    return vmd_measure_dihed(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "angle", CMDLEN))
    return vmd_measure_angle(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "bond", CMDLEN))
    return vmd_measure_bond(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "energy", CMDLEN))
    return vmd_measure_energy(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "pbc2onc", CMDLEN))
    return vmd_measure_pbc2onc_transform(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "pbcneighbors", CMDLEN))
    return vmd_measure_pbc_neighbors(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "surface", CMDLEN))
    return vmd_measure_surface(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "transoverlap", CMDLEN))
    return vmd_measure_trans_overlap(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "symmetry", CMDLEN))
    return vmd_measure_symmetry(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "volinterior", CMDLEN))
    return vmd_measure_volinterior(app, argc-1, objv+1, interp);

  Tcl_SetResult(interp, (char *) "Type 'measure' for summary of usage\n", TCL_VOLATILE);
  return TCL_OK;
}



