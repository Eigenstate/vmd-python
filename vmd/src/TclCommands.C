/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: TclCommands.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.163 $	$Date: 2011/06/15 05:17:39 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Tcl <--> VMD interface commands used for the analysis and 
 * manipulation of structures
 *
 ***************************************************************************/

#include <stdlib.h> 
#include <string.h>
#include <errno.h>
#include "tcl.h"
#include "MoleculeList.h"
#include "TclCommands.h"
#include "SymbolTable.h"
#include "VMDApp.h"

#include "config.h"
#if defined(VMDTKCON)
#include "JString.h"
#include "vmdconsole.h"
#endif

#include "Inform.h"
#include "MolFilePlugin.h"
#include "CommandQueue.h"
#include "Measure.h"

////////////////////////////////////////////////////////////////////////
// given a string, return the indicated molecule.
// String can be a number or 'top'

static Molecule *find_molecule(Tcl_Interp *interp, MoleculeList *mlist, const char *text)
{
  int molid = -1;
  if (!strcmp(text, "top")) {
    if (mlist->top()) {
      molid = mlist->top()->id();
    } else {
      Tcl_AppendResult(interp, "There is no 'top' molecule ", NULL);
      return NULL;
    }
  } else {
    if (Tcl_GetInt(interp, text, &molid) != TCL_OK) {
      Tcl_AppendResult(interp, "Not valid molecule id ", text, NULL);
      return NULL;
    }
  }
  // here I have 'molid', so get the given molecule 
  Molecule *mol = mlist-> mol_from_id(molid);  
  if (!mol) {
    Tcl_AppendResult(interp, "Cannot find molecule ", text, NULL);
  }
  return mol;
}

///// tcl interface to the AtomSel object

// forward definitions
static int access_tcl_atomsel(ClientData my_data, Tcl_Interp *interp,
		       int argc, const char *argv[]);
static int access_tcl_atomsel_obj(ClientData my_data, Tcl_Interp *interp,
		       int argc, Tcl_Obj * const argv[]);
static void remove_tcl_atomsel(ClientData my_data);

// given the interpreter and attribute string, construct the array
// mapping from attribute to atomSelParser index
static int split_tcl_atomsel_info(Tcl_Interp *interp, SymbolTable *parser,
                                  const char *opts, 
				  int *num, int **mapping) 
{
  *num = 0;
  *mapping = NULL;

  // make the list of attributes
  const char **attribs;
  int num_attribs;
  if (Tcl_SplitList(interp, opts, &num_attribs, &attribs) != TCL_OK) {
    Tcl_AppendResult(interp, "cannot split attributes list", NULL);
    return TCL_ERROR;
  }

  // verify that each attrib is a valid KEYWORD or SINGLEWORD
  // in the parser
  int *info_index = new int[num_attribs];
  for (int i=0; i<num_attribs; i++) {
    // search for a match to the attribute
    int j = parser->find_attribute(attribs[i]);

    if (j == -1) { // the name wasn't found, so complain
      Tcl_AppendResult(interp, "cannot find attribute '", 
		       attribs[i], "'", NULL);
      delete [] info_index;
      ckfree((char *)attribs); // free of tcl data
      return TCL_ERROR;
    }
    // make sure this is a KEYWORD or SINGLEWORD
    if (parser->fctns.data(j)->is_a != SymbolTableElement::KEYWORD &&
        parser->fctns.data(j)->is_a != SymbolTableElement::SINGLEWORD) {
      Tcl_AppendResult(interp, "'", attribs[i], 
		       "' is not a keyword or singleword", NULL);
      delete [] info_index;
      ckfree((char *)attribs); // free of tcl data
      return TCL_ERROR;
    }
    info_index[i] = j; // make the mapping from attrib to atomSelParser index
  }

  ckfree((char *)attribs); // free of tcl data
  *mapping = info_index; // return the mapping
  *num = num_attribs;
  return TCL_OK;
}
				    
// the Tcl command is "atomselect".  It generates 'local' (with upproc)
// functions which return information about the AtomSel selection
// Format is: atomselect <molecule id> <text>
static int make_tcl_atomsel(ClientData cd, Tcl_Interp *interp, int argc, const char *argv[])
{

  VMDApp *app = (VMDApp *)cd;
  MoleculeList *mlist = app->moleculeList; 
  SymbolTable *atomSelParser = app->atomSelParser; 

  if (argc == 4 && !strcmp(argv[1], "macro")) {
    if (atomSelParser->add_custom_singleword(argv[2], argv[3])) {
      // XXX log command ourselves; should define a VMDApp method to do it.
      app->commandQueue->runcommand(new CmdAddAtomSelMacro(argv[2], argv[3]));
      return TCL_OK;
    }
    Tcl_AppendResult(interp, "Unable to create macro for '",argv[2],"'", NULL);
    return TCL_ERROR;
  }
  if (argc == 3 && !strcmp(argv[1], "macro")) {
    const char *macro = atomSelParser->get_custom_singleword(argv[2]);
    if (!macro) {
      Tcl_AppendResult(interp, "No macro exists for '",argv[2], "'", NULL);
      return TCL_ERROR;
    }
    Tcl_AppendResult(interp, (char *)macro, NULL);
    return TCL_OK;
  }
  if (argc == 2 && !strcmp(argv[1], "macro")) {
    for (int i=0; i<atomSelParser->num_custom_singleword(); i++) {
      const char *macro = atomSelParser->custom_singleword_name(i);
      if (macro && strlen(macro) > 0)
        Tcl_AppendElement(interp, (char *)macro);
    }
    return TCL_OK;
  }
  if (argc == 3 && !strcmp(argv[1], "delmacro")) {
    if (!atomSelParser->remove_custom_singleword(argv[2])) {
      Tcl_AppendResult(interp, "Unable to delete macro '", argv[2], "'", NULL);
      return TCL_ERROR;
    }
    // XXX log command ourselves; should define a VMDApp method to do it.
    app->commandQueue->runcommand(new CmdDelAtomSelMacro(argv[2]));
    return TCL_OK;
  }
  
  // return a list of all the undeleted selection
  //
  // XXX since atomselection names are practially always stored in 
  // a variable and thus the name itself does not matter, we could
  // consider to change the original code to generate symbols of 
  // the kind  __atomselect## or even __vmd_atomselect##.
  if (argc == 2 && !strcmp(argv[1], "list")) {
    char script[] = "info commands {atomselect[0-9]*}"; 
    return Tcl_Eval(interp, script);
  }

  // return a list of the available keywords in the form
  if (argc == 2 && !strcmp(argv[1], "keywords")) {
    for (int i=0; i<atomSelParser->fctns.num(); i++) {
      Tcl_AppendElement(interp, atomSelParser->fctns.name(i));
    }
    return TCL_OK;
  }

  // return all the symbol table information for the available keywords
  // in the form  {visiblename regex is takes}, where
  //   "is" is one of "int", "float", "string"
  //   "takes" is one of "keyword", "function", "boolean", "sfunction"
  if (argc == 2 && !strcmp(argv[1], "symboltable")) {
    char *pis, *ptakes;
    // go through the parser, one by one
    for (int i=0; i< atomSelParser->fctns.num(); i++) {
      Tcl_AppendResult(interp, i==0?"":" ", "{", NULL);
      // what kind of function is this?
      switch (atomSelParser->fctns.data(i) -> is_a) {
      case SymbolTableElement::KEYWORD: ptakes = (char *) "keyword"; break;
      case SymbolTableElement::FUNCTION: ptakes = (char *) "function"; break;
      case SymbolTableElement::SINGLEWORD: ptakes = (char *) "boolean"; break;
      case SymbolTableElement::STRINGFCTN: ptakes = (char *) "sfunction"; break;
      default: ptakes = (char *) "unknown"; break;
      }
      // what does it return?
      switch (atomSelParser->fctns.data(i) -> returns_a) {
      case SymbolTableElement::IS_INT : pis = (char *) "int"; break;
      case SymbolTableElement::IS_FLOAT : pis = (char *) "float"; break;
      case SymbolTableElement::IS_STRING : pis = (char *) "string"; break;
      default: pis = (char *) "unknown"; break;
      }
      // append to the result string
      Tcl_AppendElement(interp, atomSelParser->fctns.name(i));
      Tcl_AppendElement(interp, atomSelParser->fctns.name(i));
      Tcl_AppendElement(interp, pis);
      Tcl_AppendElement(interp, ptakes);
      Tcl_AppendResult(interp, "}", NULL);
    }
    return TCL_OK;
  }

  if (!((argc == 3) || (argc == 5 && !strcmp(argv[3], "frame")))) {
    Tcl_SetResult(interp, 
      (char *) "usage: atomselect <command> [args...]\n"
      "\nCreating an Atom Selection:\n"
      "  <molId> <selection text> [frame <n>]  -- creates an atom selection function\n"
      "  list                         -- list existing atom selection functions\n"
      "  (type an atomselection function to see a list of commands for it)\n"
      "\nGetting Info about Keywords:\n"      
      "  keywords                     -- keywords for selection's get/set commands\n"
      "  symboltable                  -- list keyword function and return types\n"
      "\nAtom Selection Text Macros:\n"        
      "  macro <name> <definition>    -- define a new text macro\n"
      "  delmacro <name>              -- delete a text macro definition\n"
      "  macro [<name>]               -- list all (or named) text macros\n",
      TCL_STATIC);
    return TCL_ERROR;
  }
  int frame = AtomSel::TS_NOW;
  if (argc == 5) { // get the frame number
    int val;
    if (AtomSel::get_frame_value(argv[4], &val) != 0) {
      Tcl_SetResult(interp, 
        (char *) "atomselect: bad frame number in input, must be "
	"'first', 'last', 'now', or a non-negative number",
        TCL_STATIC);
      return TCL_ERROR;
    }
    frame = val;
  }
      
  // get the molecule id
  Molecule *mol = find_molecule(interp, mlist, argv[1]);
  if (!mol) {
    Tcl_AppendResult(interp, " in atomselect's 'molId'", NULL);
    return TCL_ERROR;
  }
  // do the selection 
  AtomSel *atomSel = new AtomSel(atomSelParser, mol->id());
  atomSel -> which_frame = frame;
  if (atomSel->change(argv[2], mol) == AtomSel::NO_PARSE) {
    Tcl_AppendResult(interp, "atomselect: cannot parse selection text: ",
		     argv[2], NULL);
    return TCL_ERROR;
  }
  // At this point the data is okay so construct the new function

  // make the name
  char newname[30];
  int *num = (int *)Tcl_GetAssocData(interp, (char *)"AtomSel", NULL);
  sprintf(newname, "atomselect%d", *num);
  (*num)++;

  // make the new proc
  Tcl_CreateObjCommand(interp, newname, access_tcl_atomsel_obj, 
		    (ClientData) atomSel, 
		    (Tcl_CmdDeleteProc *) remove_tcl_atomsel);

  // here I need to change the context ...
  Tcl_VarEval(interp, "upproc 0 ", newname, NULL);

  // return the new function name and return it
  Tcl_AppendElement(interp, newname);
  return TCL_OK;
}

// given the tcl variable string, get the selection
AtomSel *tcl_commands_get_sel(Tcl_Interp *interp, const char *str) {
  Tcl_CmdInfo info;
  if (Tcl_GetCommandInfo(interp, (char *)str, &info) != 1)
    return NULL;

  return (AtomSel *)(info.objClientData); 
}

// improve the speed of 'move' and 'moveby'
// needs a selection and a matrix
//  Applies the matrix to the coordinates of the selected atoms
static int atomselect_move(Tcl_Interp *interp, AtomSel *sel, const char *mattext) { 
  int molid = sel->molid();
  VMDApp *app = (VMDApp *)Tcl_GetAssocData(interp, (char *)"VMDApp", NULL);
  MoleculeList *mlist = app->moleculeList;
  Molecule *mol = mlist->mol_from_id(molid);
  if (!mol) {
    Tcl_SetResult(interp, (char *) "atomselection move: molecule was deleted",
                  TCL_STATIC);
    return TCL_ERROR;
  }

  // get the frame
  float *framepos = sel->coordinates(mlist);
  if (!framepos) {
    Tcl_SetResult(interp, (char *) "atomselection move: invalid/ no coordinates in selection", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the matrix
  Matrix4 mat;
  Tcl_Obj *matobj = Tcl_NewStringObj(mattext, -1);
  if (tcl_get_matrix("atomselection move:", interp, 
                     matobj , mat.mat) != TCL_OK) {
    Tcl_DecrRefCount(matobj); 
    return TCL_ERROR;
  }
  Tcl_DecrRefCount(matobj); 

  // and apply it to the coordinates
  int err;
  if ((err = measure_move(sel, framepos, mat)) != MEASURE_NOERR) {
    Tcl_SetResult(interp, (char *)measure_error(err), TCL_STATIC);
    return TCL_ERROR;
  }
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  return TCL_OK;
}


// and the same for the vector offset
//  Applies the vector to the coordinates of the selected atoms
static int atomselect_moveby(Tcl_Interp *interp, AtomSel *sel, const char *vectxt) { 
  int i;
  int molid = sel->molid();
  VMDApp *app = (VMDApp *)Tcl_GetAssocData(interp, (char *)"VMDApp", NULL);
  MoleculeList *mlist = app->moleculeList;
  Molecule *mol = mlist->mol_from_id(molid);
  if (!mol) {
    Tcl_SetResult(interp, (char *) "atomselection moveby: molecule was deleted", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the frame
  float *framepos = sel->coordinates(mlist);
  if (!framepos) {
    Tcl_SetResult(interp, (char *) "atomselection moveby: invalid/ no coordinates in selection", TCL_STATIC);
    return TCL_ERROR;
  }

  // get the vector
  int num_vect;
  Tcl_Obj **vec;
  Tcl_Obj *vecobj = Tcl_NewStringObj(vectxt, -1);
  if (Tcl_ListObjGetElements(interp, vecobj, &num_vect, &vec) != TCL_OK) {
    Tcl_DecrRefCount(vecobj); // free translation vector
    return TCL_ERROR;
  }
  if (num_vect != 3) {
    Tcl_SetResult(interp, (char *) "atomselection moveby: translation vector can only be of length 3", TCL_STATIC);
    Tcl_DecrRefCount(vecobj); // free translation vector
    return TCL_ERROR;
  }
  float vect[3];
  for (i=0; i<3; i++) {
    double tmp; 
    if (Tcl_GetDoubleFromObj(interp, vec[i], &tmp) != TCL_OK) {
      Tcl_SetResult(interp, (char *)"atomselect moveby: non-numeric in vector", TCL_STATIC);
      Tcl_DecrRefCount(vecobj); // free translation vector
      return TCL_ERROR;
    }
    vect[i] = (float)tmp;
  }

  // and apply it to the coordinates
  for (i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      vec_add(framepos + 3*i, framepos + 3*i, vect);
    }
  }

  Tcl_DecrRefCount(vecobj); // free translation vector

  // notify molecule that coordinates changed.
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  return TCL_OK;
}


#define ATOMSEL_SET_BAD_DATA(x) \
do { \
  char buf[80];  \
  sprintf(buf, "atomsel: set: bad data in %dth element", x); \
  Tcl_AppendResult(interp, buf, NULL); \
  delete [] data; \
  delete [] atomon; \
  delete [] elems; \
} while (0)

#define ATOMSEL_SET_BADDATA2(x) \
do { \
  char buf[80];  \
  sprintf(buf, "atomsel: set: bad data in %dth element", x);\
  Tcl_AppendResult(interp, buf, NULL); \
  delete [] data; \
  delete [] atomon; \
  delete [] elems; \
} while (0)

static int atomsel_set(ClientData my_data, Tcl_Interp *interp,
    int argc, Tcl_Obj * const objv[]) {

  AtomSel *atomSel = (AtomSel *)my_data;
  VMDApp *app = (VMDApp *)Tcl_GetAssocData(interp, "VMDApp", NULL);
  {
    // check that the molecule exists
    Molecule *mol = app->moleculeList->mol_from_id(atomSel -> molid());
    if (!mol) {
      char tmpstring[1024];
      sprintf(tmpstring, "atomsel: get: was molecule %d deleted?",
	      atomSel->molid());
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_ERROR;
    }
  }
  SymbolTable *atomSelParser = app->atomSelParser;
  if (atomSel == NULL) {
    Tcl_SetResult(interp, (char *) "atomselect access without data!", TCL_STATIC);
    return TCL_ERROR;
  } 

  int i, num_mapping;
  Tcl_Obj **attrs;
  // Get the list of attributes we want to set
  if (Tcl_ListObjGetElements(interp, objv[2], &num_mapping, &attrs))
    return TCL_ERROR;

  // Get the list of data elements
  int num_outerlist;
  Tcl_Obj **outerlist;
  if (Tcl_ListObjGetElements(interp, objv[3], &num_outerlist, &outerlist))
    return TCL_ERROR;

  // Check that all the attributes are writable
  SymbolTableElement **elems = new SymbolTableElement *[num_mapping];
  for (i=0; i<num_mapping; i++) {
    const char *attrname = Tcl_GetStringFromObj(attrs[i], NULL);
    int id = atomSelParser->find_attribute(attrname);
    if (id <  0) {
      delete [] elems;
      Tcl_AppendResult(interp, "cannot find attribute '", attrname, "'", NULL);
      return TCL_ERROR;
    }
    SymbolTableElement *elem = atomSelParser->fctns.data(id);
    if (elem->is_a != SymbolTableElement::KEYWORD || !elem->set_fctn) {
      delete [] elems;
      Tcl_AppendResult(interp, "atomsel object: set: data not modifiable: ",
          attrname, NULL);
      return TCL_ERROR;
    }
    elems[i] = elem;
  }
  atomsel_ctxt context(atomSelParser, 
                       app->moleculeList->mol_from_id(atomSel->molid()),
                         atomSel->which_frame, NULL);

  // Make list of the atom indices that are on
  int *atomon = new int[atomSel->selected];
  int ind = 0;
  for (i=atomSel->firstsel; i<=atomSel->lastsel; i++) 
    if (atomSel->on[i])
      atomon[ind++] = i;

  // If there is only one attribute, then outerlist must be either a
  // single element or contain one element for each selected atom.
  // If there is more than one attribute, then outerlist must be
  // a list of scalars or lists, one for each attribute.

  if (num_mapping == 1) {
    if (num_outerlist != 1 && num_outerlist != atomSel->selected) {
      char tmpstring[1024];
      sprintf(tmpstring,
          "atomselect set: %d data items doesn't match %d selected atoms.",
          num_outerlist, atomSel->selected);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      delete [] elems;
      delete [] atomon;
      return TCL_ERROR;
    }
    SymbolTableElement *elem = elems[0];
    switch (elem->returns_a) {
      case SymbolTableElement::IS_INT:
      {
        int val;
        int *data = new int[atomSel->num_atoms];
        if (num_outerlist == 1) {
          if (Tcl_GetIntFromObj(NULL, outerlist[0], &val) != TCL_OK) {
            // try to convert to double instead
            double dval;
            if (Tcl_GetDoubleFromObj(NULL, outerlist[0], &dval) == TCL_OK) {
              val = (int)dval;
            } else {
              ATOMSEL_SET_BAD_DATA(0);
              return TCL_ERROR;
            }
          }
          for (int i=0; i<atomSel->selected; i++) data[atomon[i]] = val;
        } else if (num_outerlist == atomSel->selected) {
          for (int i=0; i<num_outerlist; i++) {
            if (Tcl_GetIntFromObj(NULL, outerlist[i], &val) != TCL_OK) {

              // try to convert to double instead
              double dval;
              if (Tcl_GetDoubleFromObj(NULL, outerlist[i], &dval) == TCL_OK) {
                val = (int)dval;
              } else {
                ATOMSEL_SET_BAD_DATA(i);
                return TCL_ERROR;
              }
            }
            data[atomon[i]] = val;
          }
        }
        elem->set_keyword_int(&context, atomSel->num_atoms, data, atomSel->on);
        delete [] data;
      }
      break;
      case SymbolTableElement::IS_FLOAT:
      {
        double val;
        double *data = new double[atomSel->num_atoms];
        if (num_outerlist == 1) {
          if (Tcl_GetDoubleFromObj(NULL,outerlist[0],&val) != TCL_OK) {
            ATOMSEL_SET_BAD_DATA(0);
            return TCL_ERROR;
          }
          for (int i=0; i<atomSel->selected; i++) data[atomon[i]] = val;
        } else if (num_outerlist == atomSel->selected) {
          for (int i=0; i<num_outerlist; i++) {
            if (Tcl_GetDoubleFromObj(NULL, outerlist[i], &val) != TCL_OK) {
              ATOMSEL_SET_BAD_DATA(i);
              return TCL_ERROR;
            }
            data[atomon[i]] = val;
          }
        }
        elem->set_keyword_double(&context, atomSel->num_atoms, data, atomSel->on);
        delete [] data;
      }
      break;
      case SymbolTableElement::IS_STRING:
      {
        const char *val;
        const char **data = new const char *[atomSel->num_atoms];
        if (num_outerlist == 1) {
          val = Tcl_GetStringFromObj(outerlist[0], NULL);
          for (int i=0; i<atomSel->selected; i++) data[atomon[i]] = val;
        } else if (num_outerlist == atomSel->selected) {
          for (int i=0; i<num_outerlist; i++) {
            data[atomon[i]] = Tcl_GetStringFromObj(outerlist[i], NULL);
          }
        }
        elem->set_keyword_string(&context, atomSel->num_atoms, data, atomSel->on);
        delete [] data;
      }
      break;
    }
  } else {
    // something like "$sel set {mass beta} {{1 0} {2 1} {3 1} {3 2}}"
    if (num_outerlist != atomSel->selected) {
      char tmpstring[1024];
      sprintf(tmpstring, 
          "atomselect: set: %d data items doesn't match %d selected atoms.", 
          num_outerlist, atomSel->selected);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      delete [] elems;
      delete [] atomon;
      return TCL_ERROR;
    }
    Tcl_Obj ***objdata = new Tcl_Obj **[num_outerlist];
    for (i=0; i<num_outerlist; i++) {
      int itemsize;
      Tcl_Obj **itemobjs;
      if (Tcl_ListObjGetElements(interp, outerlist[i], &itemsize, &itemobjs)
          != TCL_OK) {
        delete [] objdata;
        delete [] atomon;
        delete [] elems;
        return TCL_ERROR;
      }
      if (itemsize != num_mapping) {
        char tmpstring[1024];
        delete [] objdata;
        delete [] atomon;
        delete [] elems;
        sprintf(tmpstring, 
            "atomselect: set: data element %d has %d terms (instead of %d)", 
            i, itemsize, num_mapping);
        Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
        return TCL_ERROR;
      }
      objdata[i] = itemobjs;
    }

    // Now go back through the elements and extract their data values
    for (i=0; i<num_mapping; i++) {
      SymbolTableElement *elem = elems[i];
      switch (elem->returns_a) {
      case (SymbolTableElement::IS_INT): {
        int *data = new int[atomSel->num_atoms];
        for (int j=0; j<num_outerlist; j++) {
          int val;
          if (Tcl_GetIntFromObj(NULL, objdata[j][i], &val) != TCL_OK) {
            // try to get double
            double dval;
            if (Tcl_GetDoubleFromObj(NULL, objdata[j][i], &dval) == TCL_OK) {
              val = (int)dval;
            } else {
              ATOMSEL_SET_BADDATA2(j);
              return TCL_ERROR;
            }
          }
          data[atomon[j]] = val;
        }
        elem->set_keyword_int(&context, atomSel->num_atoms,
                              data, atomSel->on);
        delete [] data;
      }
      break;

      case (SymbolTableElement::IS_FLOAT): {
        double *data = new double[atomSel->num_atoms];
        for (int j=0; j<num_outerlist; j++) {
          double val;
          if (Tcl_GetDoubleFromObj(NULL, objdata[j][i], &val) != TCL_OK) {
            ATOMSEL_SET_BADDATA2(j);
            return TCL_ERROR;
          }
          data[atomon[j]] = val;
        }
        elem->set_keyword_double(&context, atomSel->num_atoms,
            data, atomSel->on);
        delete [] data;
      }
      break;
      case (SymbolTableElement::IS_STRING): {
        const char **data = new const char *[atomSel->num_atoms];
        for (int j=0; j<num_outerlist; j++)
          data[atomon[j]] = Tcl_GetStringFromObj(objdata[j][i], NULL);
        elem->set_keyword_string(&context, atomSel->num_atoms,
            data, atomSel->on);
        delete [] data;
      }
      break;
      }
    } 
    delete [] objdata;
  }
  delete [] atomon;
  delete [] elems;

  // Recompute the color assignments if certain atom attributes are changed.
  for (i=0; i<num_mapping; i++) {
    const char *attr = Tcl_GetStringFromObj(attrs[i], NULL);
    if (!strcmp(attr, "name") ||
        !strcmp(attr, "element") ||
        !strcmp(attr, "atomicnumber") ||
        !strcmp(attr, "type") ||
        !strcmp(attr, "resname") ||
        !strcmp(attr, "chain") ||
        !strcmp(attr, "segid") ||
        !strcmp(attr, "segname")) {
      app->moleculeList->add_color_names(atomSel->molid());
      break;
    }
  }

  // This call to force_recalc is potentially expensive; 
  // When reps have to be updated, it amounts to about 25% of the 
  // time for a 13,000 atom system on a 1.1 GHz Athlon.  It's
  // here so that changing atom values immediately updates the screen.
  // For better performance, we set dirty bits and do the update only 
  // when the next screen redraw occurs.
  Molecule *mol = app->moleculeList->mol_from_id(atomSel->molid());
  mol->force_recalc(DrawMolItem::SEL_REGEN | DrawMolItem::COL_REGEN); 
  return TCL_OK;
}

// methods related to a selection
//0  num       -- number of atoms selected
//1  list      -- list of atom indicies
//2  molid     -- id of the molecule used
//3  text      -- the selection text
//4  get {options}  -- return a list of the listed data for each atom
//6  type      -- returns "atomselect"
//20 frame     -- returns the value of the frame (or 'now' or 'last')
//21 frame <num> -- sets the frame value given the name or number
///// these are defered to other Tcl functions
//7  moveby {x y z}    -- move by a given {x y z} offset
//8  lmoveby {{x y z}} -- move by a list of {x y z} offsets, 1 per atom
//9  moveto {x y z}    -- move to a given {x y z} offset
//10 lmoveto {{x y z}  -- same as 'set {x y z}'
/////
//11 move {transformation}   -- takes a 4x4 transformation matrix
/////
//12 delete    -- same as 'rename $sel {}'
//13 global    -- same as 'upproc #0 $argv[0]'
//14 uplevel L -- same as 'upproc $argv[1] $argv[0]'
#define CHECK_MATCH(string,val) if(!strcmp(argv[1],string)){option=val;break;}

int access_tcl_atomsel_obj(ClientData my_data, Tcl_Interp *interp, 
    int argc, Tcl_Obj * const objv[]) {

  if (argc > 1) {
    const char *argv1 = Tcl_GetStringFromObj(objv[1], NULL);
    if (argc == 4 && !strcmp(argv1, "set")) 
      return atomsel_set(my_data, interp, argc, objv);
  }
  const char **argv = new const char *[argc];
  for (int i=0; i<argc; i++) argv[i] = Tcl_GetStringFromObj(objv[i], NULL);
  int rc = access_tcl_atomsel(my_data, interp, argc, argv);
  delete [] argv;
  return rc;
}

int access_tcl_atomsel(ClientData my_data, Tcl_Interp *interp,
		       int argc, const char *argv[]) {

  VMDApp *app = (VMDApp *)Tcl_GetAssocData(interp, (char *)"VMDApp", NULL);
  AtomSel *atomSel = (AtomSel *)my_data; 
  MoleculeList *mlist = app->moleculeList; 
  SymbolTable *atomSelParser = app->atomSelParser;
  int i;
 
  if (atomSel == NULL) {
    Tcl_SetResult(interp, (char *) "atomselect access without data!", TCL_STATIC);
    return TCL_ERROR;
  }
  // We don't have a singleword defined yet, so macro is NULL.
  atomsel_ctxt context(atomSelParser, mlist->mol_from_id(atomSel->molid()), 
               atomSel->which_frame, NULL);

  int option = -1;
  const char *outfile_name = NULL;  // for 'writepdb'
  while (1) {
    if (argc == 2) {
      CHECK_MATCH("num", 0);
      CHECK_MATCH("list", 1);
      CHECK_MATCH("molindex", 2);
      CHECK_MATCH("molid", 2);
      CHECK_MATCH("text", 3);
      CHECK_MATCH("type", 6);
      CHECK_MATCH("delete", 12);
      CHECK_MATCH("global", 13);
      CHECK_MATCH("frame", 20);
      CHECK_MATCH("getbonds", 24);
      CHECK_MATCH("update", 26);
      CHECK_MATCH("getbondorders", 27);
      CHECK_MATCH("getbondtypes", 29);
    } else if (argc == 3) {
      CHECK_MATCH("get", 4);
      CHECK_MATCH("moveby", 7);   // these now pass via the "extended"
      CHECK_MATCH("lmoveby", 8);  // Tcl functionality
      CHECK_MATCH("moveto", 9);
      CHECK_MATCH("lmoveto", 10);
      CHECK_MATCH("move", 11);
      CHECK_MATCH("uplevel", 14);
      CHECK_MATCH("frame", 21);
      CHECK_MATCH("setbonds", 25);
      CHECK_MATCH("setbondorders", 28);
      CHECK_MATCH("setbondtypes", 30);
      if (!strncmp(argv[1],"write", 5)) { option = 23; break; }
    }
    if (argc != 1) {
      // gave some wierd keyword
      Tcl_AppendResult(interp, "atomselection: improper method: ", argv[1],
		       "\n", NULL);
    }
    // Now list the available options
    Tcl_AppendResult(interp, 
       "usage: <atomselection> <command> [args...]\n"
       "\nCommands for manipulating atomselection metadata:\n",
       "  frame [new frame value]      -- get/set frame\n",
       "  molid|molindex               -- get selection's molecule id\n",
       "  text                         -- get selection's text\n",
       "  delete                       -- delete atomselection (to free memory)\n",
       "  global                       -- move atomselection to global scope\n",
       "  update                       -- recalculate selection\n",
       "\nCommands for getting/setting attributes:\n",
       "  num                          -- number of atoms\n",
       "  list                         -- get atom indices\n",
       "  get <list of attributes>     -- for attributes use 'atomselect keywords'\n",
       "  set <list of attributes> <nested list of values>\n",
       "  getbonds                     -- get list of bonded atoms\n",
       "  setbonds <bondlists>\n",
       "  getbondorders                -- get list of bond orders\n",
       "  setbondorders <bondlists>\n",
       "  getbondtypes                 -- get list of bond types\n",
       "  setbondtypes  <bondlists>\n",
       "  moveto|moveby <3 vector>     -- change atomic coordinates\n",
       "  lmoveto|lmoveby <x> <y> <z>\n",
       "  move <4x4 transforamtion matrix>\n",
       "\nCommands for writing to a file:\n",
       "  writepdb <filename>          -- write sel to PDB file\n",
       "  writeXXX <filename>          -- write sel to XXX file (if XXX is a known format)\n",
		     NULL);
    return TCL_ERROR;
  }

  switch(option) {
  case 0: { // num
    char tmpstring[64];
    sprintf(tmpstring, "%d", atomSel->selected);
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    return TCL_OK;
  }
  case 1: { // list
    char tmpstring[64];
    for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
      if (atomSel->on[i]) {
	sprintf(tmpstring, "%d", i);
	Tcl_AppendElement(interp, tmpstring);
      } 
    }
    return TCL_OK;
  }
  case 2: { // molid
    char tmpstring[64];
    sprintf(tmpstring, "%d", atomSel->molid());
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE); 
    return TCL_OK;
  }
  case 3: { // text
    Tcl_SetResult(interp, atomSel->cmdStr, TCL_VOLATILE);
    return TCL_OK;
  }
  case 20: { // frame
    char tmpstring[1024];
    switch (atomSel->which_frame) {
      case AtomSel::TS_LAST: sprintf(tmpstring, "last"); break;
      case AtomSel::TS_NOW : sprintf(tmpstring, "now"); break;
      default:
	sprintf(tmpstring, "%d", atomSel->which_frame);
    }
    Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
    return TCL_OK;
  }
  case 21: { // frame <num>
    int val;
    if (AtomSel::get_frame_value(argv[2], &val) != 0) {
      Tcl_AppendResult(interp, "atomsel: frame '", argv[2], "' invalid; ",
	 "please use a number >=0 or 'first', 'last', or 'now'", NULL);
      return TCL_ERROR;
    }
    atomSel -> which_frame = val;
    return TCL_OK;
  }
  case 4: { // get
    // check that the molecule exists
    Molecule *mol = mlist->mol_from_id(atomSel -> molid());
    if (!mol) {
      char tmpstring[1024];
      sprintf(tmpstring, "atomsel: get: was molecule %d deleted?",
	      atomSel->molid());
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_ERROR;
    }

    // get the mapping
    int *mapping;
    int num_mapping;
    if (split_tcl_atomsel_info(interp, atomSelParser,argv[2], &num_mapping, 
			       &mapping) != TCL_OK) {
      Tcl_AppendResult(interp, ": in atomsel: get:", NULL);
      return TCL_ERROR;
    }

    // get the requested information
    Tcl_Obj *result = Tcl_NewListObj(0,NULL);
    if (num_mapping == 1) {
      // special case for only one property - don't have to build sublists
      // for data elements, resulting in large speedup.
      SymbolTableElement *elem = atomSelParser->fctns.data(mapping[0]);
      if (elem->is_a == SymbolTableElement::SINGLEWORD) {
        // Set the singleword, in case this is a macro.
        context.singleword = atomSelParser->fctns.name(mapping[0]);
        // get the boolean state
        int *flgs = new int[atomSel->num_atoms]; 
        memcpy(flgs, atomSel->on, atomSel->num_atoms * sizeof(int));
        elem->keyword_single(&context, atomSel->num_atoms, flgs);
        for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
          if (atomSel->on[j])
            Tcl_ListObjAppendElement(interp, result, Tcl_NewIntObj(flgs[j]));
        }
        delete [] flgs;
      } else { // then this is a keyword, and I already have routines to use
        switch(elem->returns_a) {
          case (SymbolTableElement::IS_STRING):
            {
              const char **tmp = new const char *[atomSel->num_atoms]; 
              elem->keyword_string(&context, atomSel->num_atoms, tmp, atomSel->on);
              for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
                if (atomSel->on[j])
                  Tcl_ListObjAppendElement(interp, result,
                                   Tcl_NewStringObj((char *)tmp[j], -1));
              }
              delete [] tmp;
            }
            break;
          case (SymbolTableElement::IS_INT):
            {
              int *tmp = new int[atomSel->num_atoms]; 
              elem->keyword_int(&context, atomSel->num_atoms, tmp, atomSel->on);
              for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
                if (atomSel->on[j])
                  Tcl_ListObjAppendElement(interp, result,
                                           Tcl_NewIntObj(tmp[j]));
              }
              delete [] tmp;
            }
            break; 
          case (SymbolTableElement::IS_FLOAT):
            {
              double *tmp = new double[atomSel->num_atoms]; 
              elem->keyword_double(&context, atomSel->num_atoms, tmp, atomSel->on);
              for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
                if (atomSel->on[j])
                  Tcl_ListObjAppendElement(interp, result,
                                           Tcl_NewDoubleObj(tmp[j]));
              }
              delete [] tmp;
            } 
            break;
          default: ;
        }  // switch
      }
    } else {
      // construct sublists each atom; each sublist will contain the
      // requested properties for each atom.
      for (i=0; i<atomSel->selected; i++) {
        Tcl_ListObjAppendElement(interp, result, Tcl_NewListObj(0,NULL));
      } 
      // Get the array of sublists for efficient access.
      Tcl_Obj **arr;
      int dum;
      Tcl_ListObjGetElements(interp, result, &dum, &arr);

      for (i=0; i<num_mapping; i++) {
        SymbolTableElement *elem = atomSelParser->fctns.data(mapping[i]);
        if (elem->is_a == SymbolTableElement::SINGLEWORD) {
          // Set the singleword, in case this is a macro.
          context.singleword = atomSelParser->fctns.name(mapping[i]);
          // get the boolean state
          int *flgs = new int[atomSel->num_atoms]; 
          memcpy(flgs, atomSel->on, atomSel->num_atoms * sizeof(int));
          elem->keyword_single(&context, atomSel->num_atoms, flgs);
          int k=0; 
          for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
            if (atomSel->on[j])
              Tcl_ListObjAppendElement(interp, arr[k++], 
                                       Tcl_NewIntObj(flgs[j]));
          }
          delete [] flgs;
        } else { // then this is a keyword, and I already have routines to use
          switch(elem->returns_a) {
            case (SymbolTableElement::IS_STRING):
              {
                const char **tmp = new const char *[atomSel->num_atoms]; 
                elem->keyword_string(&context, atomSel->num_atoms, tmp, atomSel->on);
                int k=0;
                for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
                  if (atomSel->on[j])
                    Tcl_ListObjAppendElement(interp, arr[k++],
                                          Tcl_NewStringObj((char *)tmp[j], -1));
                }
                delete [] tmp;
              }
              break;
            case (SymbolTableElement::IS_INT):
              {
                int *tmp = new int[atomSel->num_atoms]; 
                elem->keyword_int(&context, atomSel->num_atoms, tmp, atomSel->on);
                int k=0;
                for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
                  if (atomSel->on[j])
                    Tcl_ListObjAppendElement(interp, arr[k++],
                                             Tcl_NewIntObj(tmp[j]));
                }
                delete [] tmp;
              }
              break; 
            case (SymbolTableElement::IS_FLOAT):
              {
                double *tmp = new double[atomSel->num_atoms]; 
                elem->keyword_double(&context, atomSel->num_atoms, tmp, atomSel->on);
                int k=0;
                for (int j=atomSel->firstsel; j<=atomSel->lastsel; j++) {
                  if (atomSel->on[j])
                    Tcl_ListObjAppendElement(interp, arr[k++],
                                             Tcl_NewDoubleObj(tmp[j]));
                }
                delete [] tmp;
              } 
              break;
            default: ;
          }  // switch
        }    // else (singleword)
      }      // loop over mappings
    }        // if (num_mapping)
    delete [] mapping;
    Tcl_SetObjResult(interp, result);
    return TCL_OK;
  }
  case 6: // type
    Tcl_SetResult(interp, (char *) "atomselect", TCL_STATIC);
    return TCL_OK;

  case 7: // moveby
    return atomselect_moveby(interp, atomSel, argv[2]);

  case 8: // lmoveby
    return Tcl_VarEval(interp, "vmd_atomselect_lmoveby {", argv[0], 
                               (char *)"} {",
                               argv[2], "}", NULL); 

  case 9: // moveto
    return Tcl_VarEval(interp, "vmd_atomselect_moveto {", argv[0], 
                               (char *)"} {",
                               argv[2], "}", NULL); 

  case 10: // lmoveto
    return Tcl_VarEval(interp, "vmd_atomselect_lmoveto {", argv[0], 
                               (char *)"} {",
                               argv[2], "}", NULL); 

  case 11: // move {transformation}
    return atomselect_move(interp, atomSel, argv[2]);

  case 12: // delete
    return Tcl_VarEval(interp, "unset upproc_var_", argv[0], NULL);
  case 13: // global
    return Tcl_VarEval(interp, "upproc #0 ", argv[0], NULL);
  case 14: // uplevel
    return Tcl_VarEval(interp, "upproc ", argv[1], " ", argv[0], NULL);

  case 23: {   // writeXXX <name>
    const char *filetype = argv[1]+5;
    outfile_name = argv[2];
    // check that the molecule exists
    int molid = atomSel->molid();
    if (!app->molecule_valid_id(molid)) {
      char buf[512];
      sprintf(buf, "atomsel: writeXXX: was molecule %d deleted?", molid);
      Tcl_SetResult(interp, buf, TCL_VOLATILE);
      return TCL_ERROR;
    }
    // parse the selected frame and check for valid range
    int frame=-1;
    switch (atomSel -> which_frame) {
      case AtomSel::TS_NOW:  frame = app->molecule_frame(molid); break;
      case AtomSel::TS_LAST: frame = app->molecule_numframes(molid)-1; break;
      default:               frame = atomSel->which_frame; break;
    }
    if (frame < 0 || frame >= app->molecule_numframes(molid)) {
      char tmpstring[1024];
      sprintf(tmpstring, "atomsel: frame %d out of range for molecule %d", 
              frame, molid);
      Tcl_SetResult(interp, tmpstring, TCL_VOLATILE);
      return TCL_ERROR;
    }
    // Write the requested atoms to the file
    FileSpec spec;
    spec.first = frame;           // write current frame only
    spec.last = frame;            // write current frame only
    spec.stride = 1;              // write all selected frames
    spec.waitfor = -1;            // wait for all frames to be written
    spec.selection = atomSel->on; // write only selected atoms
    if (!app->molecule_savetrajectory(molid, outfile_name, filetype, &spec)) {
      Tcl_AppendResult(interp, "atomsel: ", argv[1], " failed.", NULL);
        return TCL_ERROR;
    }
    return TCL_OK;
  }
   
  case 24:  // getbonds
  {
    Molecule *mol = mlist->mol_from_id(atomSel->molid());
    if (!mol) {
      Tcl_AppendResult(interp, "atomsel : getbonds: was molecule deleted", 
        NULL);
      return TCL_ERROR;
    }
    Tcl_Obj *result = Tcl_NewListObj(0,NULL);
    for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
      if (atomSel->on[i]) {
        Tcl_Obj *bondlist = Tcl_NewListObj(0,NULL);
        const MolAtom *atom = mol->atom(i);
        for (int j=0; j<atom->bonds; j++) {
          Tcl_ListObjAppendElement(interp, bondlist, 
            Tcl_NewIntObj(atom->bondTo[j]));
        } 
        Tcl_ListObjAppendElement(interp, result, bondlist); 
      }
    }
    Tcl_SetObjResult(interp, result);
    return TCL_OK;
  }
  break;

  case 25:  // setbonds:
  {
    Molecule *mol = mlist->mol_from_id(atomSel->molid());
    if (!mol) {
      Tcl_AppendResult(interp, "atomsel : setbonds: was molecule deleted",
        NULL);
      return TCL_ERROR;
    }
    int num;
    const char **bondlists;
    if (Tcl_SplitList(interp, argv[2], &num, &bondlists) != TCL_OK) {
      Tcl_AppendResult(interp, "atomsel : setbonds: invalid bondlists", NULL);
      return TCL_ERROR;
    }
    if (num != atomSel->selected) {
      Tcl_AppendResult(interp, "atomsel : setbonds: Need one bondlist for ",
        "each selected atom", NULL);
      return TCL_ERROR;
    }

    // when user sets data fields they are marked as valid data in BaseMolecule
    mol->set_dataset_flag(BaseMolecule::BONDS);

    int ii = 0;
    mol->force_recalc(DrawMolItem::MOL_REGEN); // XXX many reps ignore bonds
    for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
      if (!atomSel->on[i]) 
        continue;
      int numbonds;
      const char **atomids;
      if (Tcl_SplitList(interp, bondlists[ii], &numbonds, &atomids) != TCL_OK) {
        Tcl_AppendResult(interp, "atomsel: setbonds: Unable to parse bondlist",
          NULL);
        Tcl_Free((char *)bondlists);
        return TCL_ERROR;
      }
      if (numbonds > MAXATOMBONDS) {
        Tcl_AppendResult(interp, 
          "atomsel: setbonds: too many bonds in bondlist: ", bondlists[ii],
          "\n", NULL);
        char buf[8];
        sprintf(buf, "%d", MAXATOMBONDS);
        Tcl_AppendResult(interp, "Maximum of ", buf, " bonds\n", NULL);
        Tcl_Free((char *)atomids);
        Tcl_Free((char *)bondlists);
        return TCL_ERROR;
      }
      MolAtom *atom = mol->atom(i);
      int k=0; 
      for (int j=0; j<numbonds; j++) {
        int id;
        if (Tcl_GetInt(interp, atomids[j], &id) != TCL_OK) {
          Tcl_Free((char *)atomids);
          Tcl_Free((char *)bondlists);
          return TCL_ERROR;
        }
        if (id >= 0 && id < mol->nAtoms) {
          atom->bondTo[k++] = id;
        } else {
          Tcl_AppendResult(interp,
            "atomsel: setbonds: warning, ignoring invalid atom id: ",  
            atomids[j], "\n", NULL);
        } 
      }
      atom->bonds = k;
      Tcl_Free((char *)atomids);
      ii++; 
    }
    Tcl_Free((char *)bondlists);
    return TCL_OK;
  } 
  break; 

  case 26:  // update
  {
    Molecule *mol = mlist->mol_from_id(atomSel->molid());
    if (!mol) {
      Tcl_AppendResult(interp, "atomsel : update: was molecule deleted?",
        NULL);
      return TCL_ERROR;
    }
    int retval = atomSel->change(NULL, mol);
    if (retval == AtomSel::NO_PARSE) {
      Tcl_AppendResult(interp, "atomsel : update: invalid selection",
        NULL);
      return TCL_ERROR;
    }
    return TCL_OK;
  }

  case 27:  // getbondorders
  {
    Molecule *mol = mlist->mol_from_id(atomSel->molid());
    if (!mol) {
      Tcl_AppendResult(interp, "atomsel : getbondorders: was molecule deleted", NULL);
      return TCL_ERROR;
    }
    Tcl_Obj *result = Tcl_NewListObj(0,NULL);
    for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
      if (atomSel->on[i]) {
        Tcl_Obj *bondlist = Tcl_NewListObj(0,NULL);
        const MolAtom *atom = mol->atom(i);
        for (int j=0; j<atom->bonds; j++) {
          Tcl_ListObjAppendElement(interp, bondlist, 
            Tcl_NewDoubleObj(mol->getbondorder(i, j)));
        } 
        Tcl_ListObjAppendElement(interp, result, bondlist); 
      }
    }
    Tcl_SetObjResult(interp, result);
    return TCL_OK;
  }
  break;

  case 28:  // setbondorders:
  {
    Molecule *mol = mlist->mol_from_id(atomSel->molid());
    if (!mol) {
      Tcl_AppendResult(interp, "atomsel : setbondorders: was molecule deleted",
        NULL);
      return TCL_ERROR;
    }
    int num;
    const char **bondlists;
    if (Tcl_SplitList(interp, argv[2], &num, &bondlists) != TCL_OK) {
      Tcl_AppendResult(interp, "atomsel : setbondorders: invalid bond order lists", NULL);
      return TCL_ERROR;
    }
    if (num != atomSel->selected) {
      Tcl_AppendResult(interp, "atomsel : setbondorders: Need one bond order list for ", "each selected atom", NULL);
      return TCL_ERROR;
    }

    // when user sets data fields they are marked as valid data in BaseMolecule
    mol->set_dataset_flag(BaseMolecule::BONDORDERS);

    int ii = 0;
    mol->force_recalc(DrawMolItem::MOL_REGEN); // XXX many reps ignore bonds
    for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
      if (!atomSel->on[i]) 
        continue;
      int numbonds;
      const char **atomids;
      if (Tcl_SplitList(interp, bondlists[ii], &numbonds, &atomids) != TCL_OK) {
        Tcl_AppendResult(interp, "atomsel: setbondorders: Unable to parse bond order list",
          NULL);
        Tcl_Free((char *)bondlists);
        return TCL_ERROR;
      }
      if (numbonds > MAXATOMBONDS || numbonds > mol->atom(i)->bonds) {
        Tcl_AppendResult(interp, 
          "atomsel: setbondorders: too many items in bond order list: ", bondlists[ii],
          "\n", NULL);
        char buf[8];
        sprintf(buf, "%d", MAXATOMBONDS);
        Tcl_AppendResult(interp, "Maximum of ", buf, " bonds\n", NULL);
        Tcl_Free((char *)atomids);
        Tcl_Free((char *)bondlists);
        return TCL_ERROR;
      }
      int k=0; 
      for (int j=0; j<numbonds; j++) {
        double order;
        if (Tcl_GetDouble(interp, atomids[j], &order) != TCL_OK) {
          Tcl_Free((char *)atomids);
          Tcl_Free((char *)bondlists);
          return TCL_ERROR;
        }
        mol->setbondorder(i, k++, (float) order);
      }
      Tcl_Free((char *)atomids);
      ii++; 
    }
    Tcl_Free((char *)bondlists);
    return TCL_OK;
  }
  break;
    
  case 29:  // getbondtypes
  {
    Molecule *mol = mlist->mol_from_id(atomSel->molid());
    if (!mol) {
      Tcl_AppendResult(interp, "atomsel : getbondtypes: was molecule deleted", NULL);
      return TCL_ERROR;
    }
    Tcl_Obj *result = Tcl_NewListObj(0,NULL);
    for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
      if (atomSel->on[i]) {
        Tcl_Obj *bondlist = Tcl_NewListObj(0,NULL);
        const MolAtom *atom = mol->atom(i);
        for (int j=0; j<atom->bonds; j++) {
          Tcl_ListObjAppendElement(interp, bondlist, 
              Tcl_NewStringObj(mol->bondTypeNames.name(mol->getbondtype(i, j)),-1));
        } 
        Tcl_ListObjAppendElement(interp, result, bondlist); 
      }
    }
    Tcl_SetObjResult(interp, result);
    return TCL_OK;
  }
  break;

  case 30:  // setbondtypes:
  {
    Molecule *mol = mlist->mol_from_id(atomSel->molid());
    if (!mol) {
      Tcl_AppendResult(interp, "atomsel : setbondtypes: was molecule deleted",
        NULL);
      return TCL_ERROR;
    }
    int num;
    const char **bondlists;
    if (Tcl_SplitList(interp, argv[2], &num, &bondlists) != TCL_OK) {
      Tcl_AppendResult(interp, "atomsel : setbondtypes: invalid bond type lists", NULL);
      return TCL_ERROR;
    }
    if (num != atomSel->selected) {
      Tcl_AppendResult(interp, "atomsel : setbondtypes: Need one bond type list for ", "each selected atom", NULL);
      return TCL_ERROR;
    }

    // when user sets data fields they are marked as valid data in BaseMolecule
    mol->set_dataset_flag(BaseMolecule::BONDTYPES);

    int ii = 0;
    for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
      if (!atomSel->on[i]) 
        continue;
      int numbonds;
      const char **atomids;
      if (Tcl_SplitList(interp, bondlists[ii], &numbonds, &atomids) != TCL_OK) {
        Tcl_AppendResult(interp, "atomsel: setbondtypes: Unable to parse bond type list",
          NULL);
        Tcl_Free((char *)bondlists);
        return TCL_ERROR;
      }
      if (numbonds > MAXATOMBONDS || numbonds > mol->atom(i)->bonds) {
        Tcl_AppendResult(interp, 
          "atomsel: setbondtypes: too many items in bond type list: ", bondlists[ii],
          "\n", NULL);
        char buf[8];
        sprintf(buf, "%d", MAXATOMBONDS);
        Tcl_AppendResult(interp, "Maximum of ", buf, " bonds\n", NULL);
        Tcl_Free((char *)atomids);
        Tcl_Free((char *)bondlists);
        return TCL_ERROR;
      }
      int k=0; 
      for (int j=0; j<numbonds; j++) {
        int type = mol->bondTypeNames.add_name(atomids[j], 0);
        mol->setbondtype(i, k++, type);
      }
      Tcl_Free((char *)atomids);
      ii++; 
    }
    Tcl_Free((char *)bondlists);
    return TCL_OK;
  } 
  break; 
  default:
    break;
  }

  Tcl_SetResult(interp, (char *) "atomselect: error: major weirdness!", TCL_STATIC);
  return TCL_ERROR;
}


// an "atomselect%u" is to be deleted
void remove_tcl_atomsel(ClientData my_data) {
  delete (AtomSel *)my_data;
}

// callback for when the interpreter gets deleted.
static void Atomsel_Delete(ClientData cd, Tcl_Interp *) {
  free(cd);
}

int Atomsel_Init(Tcl_Interp *interp) {
  VMDApp *app = (VMDApp *)Tcl_GetAssocData(interp, (char *)"VMDApp", NULL);
 
  Tcl_CreateCommand(interp, (char *) "atomselect", make_tcl_atomsel,
                      (ClientData) app, (Tcl_CmdDeleteProc *) NULL);

  int *num = (int *)malloc(sizeof(int)); 
  *num = 0;
  Tcl_SetAssocData(interp, (char *)"AtomSel", Atomsel_Delete, num);
  return TCL_OK;
}

#if defined(VMDTKCON)
// tk based console glue code.
#ifndef CONST
#define CONST
#endif

/* provides a vmdcon command */
int tcl_vmdcon(ClientData nodata, Tcl_Interp *interp,
               int objc, Tcl_Obj *const objv[]) {

    int newline, objidx, loglvl;
    CONST char *txt;
    
    newline=1;
    objidx=1;

    /* handle -nonewline */
    if (objidx < objc) {
        txt = Tcl_GetString(objv[objidx]);
        if (strcmp(txt, "-nonewline") == 0) {
            ++objidx;
            newline=0;
        }
    }

    /* handle -register/-unregister/-info/-warn/-error */
    if (objidx < objc) {
        txt = Tcl_GetString(objv[objidx]);
        // register a text widget as a console
        if (strcmp(txt, "-register") == 0) {
            ++objidx;
            newline=0;
            if (objidx < objc) {
                CONST char *mark="end";
                txt = Tcl_GetString(objv[objidx]);
                ++objidx;
                if (objidx < objc) {
                    mark = Tcl_GetString(objv[objidx]);
                }
                vmdcon_register(txt, mark, (void *)interp);
                return TCL_OK;
            } else {
                Tcl_WrongNumArgs(interp, 1, objv, "-register widget_path ?mark?");
                return TCL_ERROR;
            }
        }
        // unregister the current text widget as console
        // NOTE: this will keep a history buffer which will
        // be displayed on the next registered text widget.
        if (strcmp(txt, "-unregister") == 0) {
            vmdcon_register(NULL, NULL, (void *)interp);
            return TCL_OK;
        }

        // connect console output back to the calling terminal
        if (strcmp(txt, "-textmode") == 0) {
            vmdcon_use_text((void *)interp);
            return TCL_OK;
        }
        // connect console output to the registered text widget
        if (strcmp(txt, "-widgetmode") == 0) {
            vmdcon_use_widget((void *)interp);
            return TCL_OK;
        }

        // reprint recent console messages.
        if (strcmp(txt, "-dmesg") == 0) {
            vmdcon_showlog();
            return TCL_OK;
        }

        // report console status
        if (strcmp(txt, "-status") == 0) {
            Tcl_Obj *result;
            switch (vmdcon_get_status()) {
              case VMDCON_UNDEF:   
                  result = Tcl_NewStringObj("undefined",-1);
                  break;
                  
              case VMDCON_NONE:   
                  result = Tcl_NewStringObj("none",-1);
                  break;
                  
              case VMDCON_TEXT:   
                  result = Tcl_NewStringObj("text",-1);
                  break;
                  
              case VMDCON_WIDGET: 
                  result = Tcl_NewStringObj("widget",-1);
                  break;
                  
              default: 
                  Tcl_AppendResult(interp, 
                                   "vmdcon: unknown console status", 
                                   NULL);
                  return TCL_ERROR; 
            }
            Tcl_SetObjResult(interp, result);
            return TCL_OK;
        }

        // report console status
        if (strcmp(txt, "-loglevel") == 0) {
            ++objidx;
            if (objidx < objc) {
                txt = Tcl_GetString(objv[objidx]);
                if (strcmp(txt,"all")==0) {
                    vmdcon_set_loglvl(VMDCON_ALL);
                } else if (strcmp(txt,"info")==0) {
                    vmdcon_set_loglvl(VMDCON_INFO);
                } else if (strcmp(txt,"warn")==0) {
                    vmdcon_set_loglvl(VMDCON_WARN);
                } else if (strcmp(txt,"err")==0)  {
                    vmdcon_set_loglvl(VMDCON_ERROR);
                } else {
                    Tcl_AppendResult(interp, "vmdcon: unkown log level: ",
                                     txt, NULL);
                    return TCL_ERROR;
                }
                return TCL_OK;
            } else {
                Tcl_Obj *result;
                switch (vmdcon_get_loglvl()) {
                  case VMDCON_ALL:   
                      result = Tcl_NewStringObj("all",-1);
                      break;
                      
                  case VMDCON_INFO:   
                      result = Tcl_NewStringObj("info",-1);
                      break;
                      
                  case VMDCON_WARN:   
                      result = Tcl_NewStringObj("warn",-1);
                      break;
                      
                  case VMDCON_ERROR: 
                      result = Tcl_NewStringObj("err",-1);
                      break;
                      
                  default: 
                      Tcl_AppendResult(interp, 
                                       "vmdcon: unknown log level.", 
                                       NULL);
                      return TCL_ERROR; 
                }
                Tcl_SetObjResult(interp, result);
                return TCL_OK;
            }
        }

        // print a help message
        if (strcmp(txt, "-help") == 0) {
            Tcl_AppendResult(interp, 
                             "usage: vmdcon ?-nonewline? ?options? [arguments]\n",
                             "       print data to the VMD console or change console behavior\n\n",
                             "Output options:\n",
                             "  with no options 'vmdcon' copies all arguments to the current console\n",
                             "  -info      -- prepend output with 'Info) '\n",
                             "  -warn      -- prepend output with 'Warning) '\n",
                             "  -err       -- prepend output with 'ERROR) '\n",
                             "  -nonewline -- don't append a newline to the output\n",
                             "Console mode options:\n",
                             "  -register <widget_path> ?<mark>?  -- register a tk text widget as console\n",
                             "    optionally provide a mark as reference for insertions. otherwise 'end' is used\n",
                             "  -unregister                       -- unregister the currently registered console widget\n",
                             "  -textmode                         -- switch to text mode console (using stdio)\n",
                             "  -widgetmode                       -- switch to tk (registered) text widget as console\n\n",
                             "  -loglevel ?all|info|warn|err?     -- get or set console log level (output to console only at that level or higher)\n",
                             "General options:\n",
                             "  -status   -- report current console status (text|widget|none)\n",
                             "  -dmesg    -- (re)print recent console messages\n",
                             "  -help     -- print this help message\n",
                             NULL);

            return TCL_OK;
        }

        // from here on we assume that the intent is to send output

        // prepend the final output with "urgency" indicators
        // XXX: ideally, there would be no vmdcon without any 
        // loglevel argument, but for the time being we tolerate 
        // it and promote it to the highest loglevel.
        loglvl=VMDCON_ALWAYS;
        
        if (strcmp(txt, "-info") == 0) {
            loglvl=VMDCON_INFO;
            vmdcon_append(loglvl, "Info) ", 6);
            ++objidx;
        } else if (strncmp(txt, "-warn", 5) == 0) {
            loglvl=VMDCON_WARN;
            vmdcon_append(loglvl, "Warning) ", 9);
            ++objidx;
        } else if (strncmp(txt, "-err", 4) == 0) {
            loglvl=VMDCON_ERROR;
            vmdcon_append(loglvl, "ERROR) ", 7);
            ++objidx;
        }
    }

    if (objidx < objc) {
        txt = Tcl_GetString(objv[objidx]);
        vmdcon_append(loglvl, txt, -1);
        ++objidx;
    }

    if(newline==1) {
        vmdcon_append(loglvl, "\n", 1);
    }
    vmdcon_purge();

    if (objidx < objc) {
        Tcl_WrongNumArgs(interp, 1, objv, "?-nonewline? ?-info|-warn|-err? string");
        return TCL_ERROR;
    }
    
    return TCL_OK;
}

// we use c bindings, so the subroutines can be
// exported to c code (plugins!) as well.
const char *tcl_vmdcon_insert(void *interp, const char *w_path, 
                              const char *mark, const char *text)
{
    // do: .path.to.text insert <mark> <text> ;  .path.to.text see end
    JString cmd;         
    cmd  = w_path;
    cmd += " insert ";
    cmd += mark;
    cmd += " {";
    cmd += text;
    cmd += "}; ";
    cmd += w_path;
    cmd += " see end;"; 

    if (Tcl_Eval((Tcl_Interp *)interp,(char *)(const char *)cmd) != TCL_OK) {
        return Tcl_GetStringResult((Tcl_Interp *)interp);
    }
    return NULL;
}

void tcl_vmdcon_set_status_var(void *interp, int status) 
{
    if (interp != NULL) {
        Tcl_ObjSetVar2((Tcl_Interp *)interp, 
                       Tcl_NewStringObj("vmd_console_status", -1),
                       NULL, Tcl_NewIntObj(status),
                       TCL_GLOBAL_ONLY|TCL_LEAVE_ERR_MSG);
    }
}

#endif /* VMDTKCON */
