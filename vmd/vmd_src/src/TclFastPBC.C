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
 *      $RCSfile: TclFastPBC.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Fast PBC wrapping code
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <tcl.h>
// #include <ctype.h>
#include "TclCommands.h"
#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "utilities.h"
#include "config.h"
#include "Atom.h"
#include "Molecule.h"
#include "Measure.h"
#include "FastPBC.h"

//Compound is the type of compounding: 1=fragment, 2=residue.
//mol is the molecule we are operating on. Should come from app->moleculeList->mol_from_id(sel->molid());
//sel is the atomselection we are working on.
//indexlist and fragmentmap are the output pointers. NOTE: It is the caller's responsibility to delete indexlist and fragmentmap.
static int compoundmapper (int compound, Molecule *mol, AtomSel *sel, int *&indexlist, int *&fragmentmap) {
  indexlist = new int[sel->selected];
  switch (compound) {
    case 1:
    fragmentmap = new int[mol->nFragments+1]; break;
    case 2:
    fragmentmap = new int[mol->nResidues+1]; break;
    default:
    //This should never happen. Ever.
    printf("This should have never happened. Josh is sorry.\n Compound was passed with argument %d\n", compound);
    return TCL_ERROR;
  }
  int *idxptr = indexlist;
  int *frag = fragmentmap;
  *frag = 0;
  int i, j, k, l;
  int count;
  if (compound == 1) {
    for (i = 0; (i < mol->nFragments); i++) {
      count = 0;
      Fragment *f = mol->fragment(i);
      for (j = 0; j < f->residues.num(); j++) {
        Residue *r = mol->residue(f->residues[j]);
        for (k = 0; k < r->atoms.num(); k++) {
          l = r->atoms[k];
          if (sel->on[l]) {
            count++;
            *idxptr = l;
            idxptr++;
          }
        }
      }
      if (count > 0) {
        frag++;
        *frag = count + *(frag-1);
      }
    }
  } else if (compound == 2) {
    for (j = 0; j < mol->nResidues; j++) {
      count = 0;
      Residue *r = mol->residue(j);
      for (k = 0; k < r->atoms.num(); k++) {
        l = r->atoms[k];
        if (sel->on[l]) {
          count++;
          *idxptr = l;
          idxptr++;
        }
      }
      if (count > 0) {
        frag++;
        *frag = count + *(frag-1);
      }
    }
  }
  return (frag - fragmentmap);
}

/**
 * **Wrap all the molecule within the unit cell**.
 *
 * ## User Usage
 *
 @verbatim
 fpbc wrap <atom selection> [first <first frame ID>] [last <last frame ID>] [center [list x y z]]
      [centersel <atom selection>] [compound "residue"/"fragment"]
 @endverbatim
 *
 * ### User Options
 * | Keyword                |  Type   | Value
 * | --------------------   |:------: | --------------------------------------------------------------|
 * | \<**atom selection**\> | object  | An atom selection object created VMD **atomselect** command.  |
 * | **first**              |   int   | The staring frame ID                                          |
 * | **last**               | int     |  The last frame ID                                            |
 * | **center**             | list of float| The 3D coordinate of the point around which everything is wrapped.|
 * | **centersel**          | object  | An atom selection object created VMD **atomselect** command. <br> Note: if the atom selection represents a range (e.g. "within 5 of index 0"), <br> this atom selection will not be updated inside **fpbc wrap**.|
*  | **compound**           | string  | Either  "residue" or "fragment". <br> "residue" means the wrapping will be done by residue. <br> "fragment" means the wrapping will be donw by fragment <br> (i.e. a group of particles chemically bounded together). |
 *
 *
 * @param app VMD application object that contains all the information about the current molecule object
 *        such as atomic coordinates.
 * @param argc Total number of arguments, i.e. the length of array objv[]
 * @param objv[] An array containging user-specified options with even number of elements (i.e. keyword-value pairs)
 *              Note: objv[0] should be one the fpbc keywords, e.g. wrap, join.
 * @param interp Tcl interpreter object
 * @return **TCL_OK**: (Tcl's C enum type) a flag meaning everything was okay;\n
 *         **TCL_ERROR**: (Tcl's C enum type) a flag indicating the case where user specified illegal options
 */
static int fpbc_wrap(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  //start with first frame by default
  int last = -1;  //finish with last frame by default
  int i, f,j;
  bool gpu = true;
  int compound = 0;
  float center[3] = {0,0,0};
  float boxsize[3] = {0,0,0};
  //Parse arguments!
  if (argc < 2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> [first <first>] [last <last>] [center [list x y z]]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  AtomSel *centersel = NULL;
  if (!sel) {
    Tcl_SetResult(interp, (char *) "fastpbc wrap: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }

  for (i=2; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc wrap: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc wrap: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "center", CMDLEN)) {
      int num_vect;
      Tcl_Obj **vec;
      Tcl_Obj *vecobj = objv[i+1];
      if (Tcl_ListObjGetElements(interp, vecobj, &num_vect, &vec) != TCL_OK) {
        return TCL_ERROR;
      }
      if (num_vect != 3) {
        Tcl_SetResult(interp, (char *) "fastpbc wrap: center vector can only be of length 3", TCL_STATIC);
        return TCL_ERROR;
      }
      for (j=0; j<3; j++) {
        double tmp;
        if (Tcl_GetDoubleFromObj(interp, vec[j], &tmp) != TCL_OK) {
          Tcl_SetResult(interp, (char *)"fastpbc wrap: non-numeric in vector", TCL_STATIC);
          return TCL_ERROR;
        }
        center[j] = (float)tmp;
      }
    } else if (!strupncmp(argvcur, "centersel", CMDLEN)) {
      centersel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[i+1],NULL));
      if (!centersel) {
        Tcl_SetResult(interp, (char *) "fastpbc wrap: no atom selection for centersel", TCL_STATIC);
        return TCL_ERROR;
      }

    } else if (!strupncmp(argvcur, "nogpu", CMDLEN)) {
      i--;
      gpu = false;
    } else if (!strupncmp(argvcur, "compound", CMDLEN)) {
      char *compoundarg = Tcl_GetStringFromObj(objv[i+1],NULL);
      if (!strupncmp(compoundarg, "fragment", CMDLEN)) {
        compound=1;
      } else if (!strupncmp(compoundarg, "residue", CMDLEN)) {
        compound=2;
      } else {
        Tcl_AppendResult(interp, "fastpbc wrap: invalid compound syntax, no such keyword: ", compoundarg, NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "fastpbc wrap: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }
  Molecule *mol = app->moleculeList->mol_from_id(sel->molid());
  int maxframes = mol->numframes();
  if (last < 0)
    last = maxframes-1;

  if (maxframes == 0 || first < 0 || first > last ||
      last >= maxframes) {
    Tcl_AppendResult(interp, "fastpbc wrap: invalid frame range: ", first, " to ", last, NULL);
      return TCL_ERROR;
  }
  Timestep *ts;
  for (f=first; f<=last; f++) {
    ts = mol->get_frame(f);
    boxsize[0] = ts->a_length;
    boxsize[1] = ts->b_length;
    boxsize[2] = ts->c_length;
    if (boxsize[0] <= 0 || boxsize[1] <= 0 || boxsize[2] <= 0) {
      Tcl_AppendResult(interp, "fastpbc wrap: frame : ", f, " has no periodic box defined", NULL);
      return TCL_ERROR;
    }
  }
  float *weight = NULL;
  if (centersel != NULL) {
    if (centersel->molid() != sel->molid()) {
       Tcl_AppendResult(interp, "fastpbc wrap: selections are from different molecules", NULL);
      return TCL_ERROR;
    }
    //Find the center for this timestep and atomselection.
    weight = new float[centersel->selected];
    {
      int ret_val = tcl_get_weights(interp, app, centersel, Tcl_NewStringObj("mass\0",-1), weight);
      if (ret_val < 0) {
        Tcl_AppendResult(interp, "fastpbc wrap: ", measure_error(ret_val),
             NULL);
        delete [] weight;
        return TCL_ERROR;
      }
    }
  }
  //End parse arguments.
  int *indexlist = NULL;
  int *compoundmap = NULL;
  int fnum = 0;
  if (compound) {
    fnum = compoundmapper (compound, mol, sel, indexlist, compoundmap);
  }
  // if compound, moving will require mass weighting.  find the array storing mass values
  float *massarr;
  if (compound) { massarr = mol->mass(); }
  if (compound) {
    if (gpu)
      fpbc_exec_wrapcompound(mol, first, last, fnum, compoundmap, sel->selected, indexlist, weight, centersel, center, massarr);
    else
      fpbc_exec_wrapcompound_cpu(mol, first, last, fnum, compoundmap, sel->selected, indexlist, weight, centersel, center, massarr);
  }
  else {
    //Create the indexlist
    indexlist = new int[sel->selected];
    j = 0;
    for (i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        indexlist[j++] = i;
      }
    }
    if (gpu)
      fpbc_exec_wrapatomic(mol, first, last, sel->selected, indexlist, weight, centersel, center);
    else
      fpbc_exec_wrapatomic_cpu(mol, first, last, sel->selected, indexlist, weight, centersel, center);
  }
  if (weight != NULL) {
    delete [] weight;
  }
  if (indexlist != NULL) {
    delete [] indexlist;
  }
  if (compoundmap != NULL) {
    delete [] compoundmap;
  }
  //Force a redraw since the atoms have moved.
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  return TCL_OK;
}






/**
 * **Unwrap the molecule coordinates such that there are no jumps in the trajectory**.
 *
 * ## User Usage
 *
 \verbatim
  unwrap <atom selection> [first <first frame ID>] [last <last frame ID>]
 \endverbatim
 *
 * ### User Options
 * | Keyword                |  Type   | Value
 * | --------------------   |:------: | --------------------------------------------------------------|
 * | \<**atom selection**\> | object  | An atom selection object created VMD **atomselect** command.  |
 * | **first**              |   int   | The staring frame ID                                          |
 * | **last**               | int     |  The last frame ID                                            |
 *  | **compound**           | string  | Either  "residue" or "fragment". <br> "residue" means the wrapping will be done by residue. <br> "fragment" means the wrapping will be donw by fragment <br> (i.e. a group of particles chemically bounded together). |
 *
 * @param app VMD application object that contains all the information about the current molecule object
 *        such as atomic coordinates.
 * @param argc Total number of arguments, i.e. the length of array objv[]
 * @param objv[] An array containging user-specified options with even number of elements (i.e. keyword-value pairs)
 *              Note: objv[0] should be one the fpbc keywords, e.g. wrap, join.
 * @param interp Tcl interpreter object
 * @return **TCL_OK**: (Tcl's C enum type) a flag meaning everything was okay;\n
 *         **TCL_ERROR**: (Tcl's C enum type) a flag indicating the case where user specified illegal options
 */
static int fpbc_unwrap(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  // start with first frame by default
  int last = -1;  // finish with last frame by default
  int i, f,j;
  bool gpu = true;
  float boxsize[3] = {0,0,0};
  //parameter handling
  if (argc < 2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> [first <first>] [last <last>]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_SetResult(interp, (char *) "fastpbc unwrap: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }

  for (i=2; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc unwrap: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc unwrap: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "nogpu", CMDLEN)) {
      i--;
      gpu = false;
    } else {
      Tcl_AppendResult(interp, "fastpbc unwrap: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }
  Molecule *mol = app->moleculeList->mol_from_id(sel->molid());
  int maxframes = mol->numframes();
  if (last < 0)
    last = maxframes-1;
  if (maxframes == 0 || first < 0 || first > last ||
      last >= maxframes) {
    Tcl_AppendResult(interp, "fastpbc unwrap: invalid frame range: ", first, " to ", last, NULL);
      return TCL_ERROR;
  }
  if (last == first) {
  	Tcl_AppendResult(interp, "fastpbc unwrap: first and last frames must be distinct", NULL);
      return TCL_ERROR;
  }
  Timestep *ts;
  for (f=first; f<=last; f++) {
    ts = mol->get_frame(f);
    boxsize[0] = ts->a_length;
    boxsize[1] = ts->b_length;
    boxsize[2] = ts->c_length;
    if (boxsize[0] <= 0 || boxsize[1] <= 0 || boxsize[2] <= 0) {
      Tcl_AppendResult(interp, "fastpbc unwrap: frame : ", f, " has no periodic box defined", NULL);
      return TCL_ERROR;
    }
  }
	int *indexlist = new int[sel->selected];
  j = 0;
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      indexlist[j++] = i;
    }
  }
  //Begin unwrapping!
  //Keep the current and previous timesteps, and make sure that every atom jumps by no more than half the boxlength.
  if (gpu)
    fpbc_exec_unwrap(mol, first, last, sel->selected, indexlist);
  else
    fpbc_exec_unwrap_cpu(mol, first, last, sel->selected, indexlist);
  if (indexlist != NULL) {
    delete [] indexlist;
  }
  //Force a redraw since the atoms have moved.
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  return TCL_OK;
}

/**
 * Join molecules/residues that were splitted due to PBC wrapping
 *
 * ## User Usage
 \verbatim
 join <atom selection> [first <first frame ID>] [last <last frame ID>] [compound residue/fragment]
 \endverbatim
 *
 * ### User Options
 *
 * | Keyword                |  Type   | Value
 * | --------------------   |:------: | --------------------------------------------------------------|
 * | \<**atom selection**\> | object  | An atom selection object created VMD **atomselect** command.  |
 * | **first**              |   int   | The staring frame ID                                          |
 * | **last**               | int     |  The last frame ID                                            |
 *
 */
static int fpbc_join(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  // start with first frame by default
  int last = -1;  // finish with last frame by default
  int i, f;
  bool gpu = true;
  int compound = 1; //Default to fragment.
  float boxsize[3] = {0,0,0};
  //Deal with input.
  if (argc < 2) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> [first <first>] [last <last>] [compound residue/fragment]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_SetResult(interp, (char *) "fastpbc join: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }

  for (i=2; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc join: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc join: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "nogpu", CMDLEN)) {
      i--;
      gpu = false;
    } else if (!strupncmp(argvcur, "compound", CMDLEN)) {
      char *compoundarg = Tcl_GetStringFromObj(objv[i+1],NULL);
      if (!strupncmp(compoundarg, "fragment", CMDLEN)) {
        compound=1;
      } else if (!strupncmp(compoundarg, "residue", CMDLEN)) {
        compound=2;
      } else {
        Tcl_AppendResult(interp, "fastpbc join: invalid compound syntax, no such keyword: ", compoundarg, NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "fastpbc join: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }
  Molecule *mol = app->moleculeList->mol_from_id(sel->molid());
  int maxframes = mol->numframes();
  if (last < 0)
    last = maxframes-1;
  if (maxframes == 0 || first < 0 || first > last ||
      last >= maxframes) {
    Tcl_AppendResult(interp, "fastpbc join: invalid frame range: ", first, " to ", last, NULL);
      return TCL_ERROR;
  }
  int *indexlist = NULL;
  int *compoundmap = NULL;
  int fnum = 0;
  if (compound) {
    fnum = compoundmapper (compound, mol, sel, indexlist, compoundmap);
  }
  Timestep *ts;
  for (f=first; f<=last; f++) {
    ts = mol->get_frame(f);
    boxsize[0] = ts->a_length;
    boxsize[1] = ts->b_length;
    boxsize[2] = ts->c_length;
    if (boxsize[0] <= 0 || boxsize[1] <= 0 || boxsize[2] <= 0) {
      Tcl_AppendResult(interp, "fastpbc join: frame : ", f, " has no periodic box defined", NULL);
      return TCL_ERROR;
    }
  }
  if (gpu)
    fpbc_exec_join(mol, first, last, fnum, compoundmap, sel->selected, indexlist);
  else
    fpbc_exec_join_cpu(mol, first, last, fnum, compoundmap, sel->selected, indexlist);
  if (indexlist != NULL) {
    delete [] indexlist;
  }
  if (compoundmap != NULL) {
    delete [] compoundmap;
  }
  //Force a redraw since the atoms have moved.
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  return TCL_OK;
}
//recenter <sel> <centersel> [first <first>] [last <last>] [compound residue/fragment]
static int fpbc_recenter(VMDApp *app, int argc, Tcl_Obj * const objv[], Tcl_Interp *interp) {
  int first = 0;  // start with first frame by default
  int last = -1;  // finish with last frame by default
  int i, j, f;
  bool gpu = true;
  int compound = 0; //Default to atomic wrap.
  float boxsize[3] = {0,0,0};
  //Deal with input.
  if (argc < 3) {
    Tcl_WrongNumArgs(interp, 2, objv-1, (char *) "<sel> <centersel> [first <first>] [last <last>] [compound residue/fragment]");
    return TCL_ERROR;
  }
  AtomSel *sel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[1],NULL));
  if (!sel) {
    Tcl_SetResult(interp, (char *) "fastpbc recenter: no atom selection", TCL_STATIC);
    return TCL_ERROR;
  }
  AtomSel *csel = tcl_commands_get_sel(interp, Tcl_GetStringFromObj(objv[2],NULL));
  if (!sel) {
    Tcl_SetResult(interp, (char *) "fastpbc recenter: no center atom selection", TCL_STATIC);
    return TCL_ERROR;
  }
  for (i=3; i<argc; i+=2) {
    char *argvcur = Tcl_GetStringFromObj(objv[i],NULL);
    if (!strupncmp(argvcur, "first", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &first) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc recenter: bad first frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "last", CMDLEN)) {
      if (Tcl_GetIntFromObj(interp, objv[i+1], &last) != TCL_OK) {
        Tcl_AppendResult(interp, "fastpbc recenter: bad last frame value", NULL);
        return TCL_ERROR;
      }
    } else if (!strupncmp(argvcur, "nogpu", CMDLEN)) {
      i--;
      gpu = false;
    } else if (!strupncmp(argvcur, "compound", CMDLEN)) {
      char *compoundarg = Tcl_GetStringFromObj(objv[i+1],NULL);
      if (!strupncmp(compoundarg, "fragment", CMDLEN)) {
        compound=1;
      } else if (!strupncmp(compoundarg, "residue", CMDLEN)) {
        compound=2;
      } else {
        Tcl_AppendResult(interp, "fastpbc recenter: invalid compound syntax, no such keyword: ", compoundarg, NULL);
        return TCL_ERROR;
      }
    } else {
      Tcl_AppendResult(interp, "fastpbc recenter: invalid syntax, no such keyword: ", argvcur, NULL);
      return TCL_ERROR;
    }
  }

  Molecule *mol = app->moleculeList->mol_from_id(sel->molid());
  int maxframes = mol->numframes();
  if (last < 0)
    last = maxframes-1;

  if (maxframes == 0 || first < 0 || first > last ||
      last >= maxframes) {
    Tcl_AppendResult(interp, "fastpbc recenter: invalid frame range: ", first, " to ", last, NULL);
      return TCL_ERROR;
  }
  Timestep *ts;
  for (f=first; f<=last; f++) {
    ts = mol->get_frame(f);
    boxsize[0] = ts->a_length;
    boxsize[1] = ts->b_length;
    boxsize[2] = ts->c_length;
    if (boxsize[0] <= 0 || boxsize[1] <= 0 || boxsize[2] <= 0) {
      Tcl_AppendResult(interp, "fastpbc recenter: frame : ", f, " has no periodic box defined", NULL);
      return TCL_ERROR;
    }
  }
  if (csel->molid() != sel->molid()) {
    Tcl_AppendResult(interp, "fastpbc recenter: selections are from different molecules", NULL);
    return TCL_ERROR;
  }
  float *weight = new float[csel->selected];
  int ret_val = tcl_get_weights(interp, app, csel, Tcl_NewStringObj("mass\0",-1), weight);
  if (ret_val < 0) {
    Tcl_AppendResult(interp, "fastpbc recenter: ", measure_error(ret_val),NULL);
    delete [] weight;
    return TCL_ERROR;
  }
  int *indexlist = NULL;
  int *centeridxlist = new int[csel->selected];
  j = 0;
  for (i=csel->firstsel; i<=csel->lastsel; i++) {
    if (csel->on[i]) {
      centeridxlist[j++] = i;
    }
  }
  // if compound, moving will require mass weighting.  find the array storing mass values
  float *massarr = NULL;
  if (compound) { massarr = mol->mass(); }
  int *compoundmap = NULL;
  int fnum = 0;
  if (compound) {
    fnum = compoundmapper (compound, mol, sel, indexlist, compoundmap);
  }
  else {
    //Create the indexlist
    indexlist = new int[sel->selected];
    j = 0;
    for (i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
        indexlist[j++] = i;
      }
    }
  }
  if (gpu)
    fpbc_exec_recenter(mol, first, last, csel->selected, centeridxlist, fnum, compoundmap, sel->selected, indexlist, weight, csel, massarr);
  else
    fpbc_exec_recenter_cpu(mol, first, last, csel->selected, centeridxlist, fnum, compoundmap, sel->selected, indexlist, weight, csel, massarr);
  if (weight != NULL) {
    delete [] weight;
  }
  if (indexlist != NULL) {
    delete [] indexlist;
  }
  if (centeridxlist != NULL) {
    delete [] centeridxlist;
  }
  if (compoundmap != NULL) {
    delete [] compoundmap;
  }
  //Force a redraw since the atoms have moved.
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  return TCL_OK;
}
/**
  * The interface between user and Fast PBC module (works as a controller)
  *
  * Function obj_fastpbc() trys to find the content of the first
  * user argument given to fpbc (i.e. objv[0]) and decide which sub-function
  * to call e.g. "wrap", "unwrap", "join".
  *
  * ## Implementation
  * Call VMD's function **strupncmp()** (case-insensitive string comparison) to
  * find which sub-function the user want to call.
  */
int obj_fastpbc(ClientData cd, Tcl_Interp *interp, int argc,
                            Tcl_Obj * const objv[]) {
  if (argc < 2) {
    Tcl_SetResult(interp,
      (char *) "usage: fastpbc <command> [args...]\n"
      "\nFastPBC Commands:\n"
      "  wrap <sel> [first <first>] [last <last>] [center [list x y z]] [centersel <sel>]\n"
      "      [compound residue/fragment]-- (re)wraps trajectory\n"
      "  unwrap <sel> [first <first>] [last <last>] -- prevents atoms from \"jumping\" during\n"
      "      a trajectory across a periodic boundary\n"
      "  join <sel> [first <first>] [last <last>] [compound residue/fragment]\n"
      "      -- Eliminates long bonds within a compound (by default, by fragment)\n"
      "  recenter <sel> <centersel> [first <first>] [last <last>] [compound residue/fragment]\n"
      "      --Identical to an unwrap followed by a wrap. The unwrap is applied only to centersel,\n"
      "      and the wrap is also centered on centersel\n",
      TCL_STATIC);
    return TCL_ERROR;
  }
  VMDApp *app = (VMDApp *)cd;
  char *argv1 = Tcl_GetStringFromObj(objv[1],NULL);
  if (!strupncmp(argv1, "wrap", CMDLEN))
    return fpbc_wrap(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "unwrap", CMDLEN))
    return fpbc_unwrap(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "join", CMDLEN))
    return fpbc_join(app, argc-1, objv+1, interp);
  else if (!strupncmp(argv1, "recenter", CMDLEN))
    return fpbc_recenter(app, argc-1, objv+1, interp);
  Tcl_SetResult(interp, (char *) "Type 'fastpbc' for summary of usage\n", TCL_VOLATILE);
  return TCL_OK;
}



