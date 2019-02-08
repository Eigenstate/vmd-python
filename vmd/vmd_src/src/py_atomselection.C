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
 *      $RCSfile: py_atomselection.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.53 $       $Date: 2019/01/23 23:03:33 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python atom selection interface.
 ***************************************************************************/

#include "py_commands.h"
#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "SymbolTable.h"
#include "Measure.h"
#include "SpatialSearch.h"

static char create_doc[] = "create(molid, frame, selection) -> tuple\nFind atoms in atom selection.";
static PyObject *create(PyObject *self, PyObject *args) {

  int molid = 0;
  int frame = 0;
  char *sel = 0;

  if (!PyArg_ParseTuple(args, (char *)"iis", &molid, &frame, &sel))
    return NULL;

  VMDApp *app = get_vmdapp();

  DrawMolecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, "invalid molid");
    return NULL;
  }

  AtomSel *atomSel = new AtomSel(app->atomSelParser, mol->id());
  atomSel->which_frame = frame;
  if (atomSel->change(sel, mol) == AtomSel::NO_PARSE) {
    PyErr_SetString(PyExc_ValueError, "cannot parse atom selection text");
    delete atomSel;
    return NULL;
  }
  
  // construct a Python tuple to return
  PyObject *newlist = PyTuple_New(atomSel->selected);
  int j=0;
  for (int i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
    if (atomSel->on[i])
      PyTuple_SET_ITEM(newlist, j++, PyInt_FromLong(i));
  }
  delete atomSel;
  return newlist;
}

static char get_doc[] = "get(molid, frame, tuple, attribute) -> value list\nGet selected atom values for the given attribute.";
static PyObject *get(PyObject *self, PyObject *args) {
  int i, molid, frame;
  PyObject *selected;
  int num_selected;
  char *attr = 0;

  //
  // get molid, list, and attribute
  //
  if (!PyArg_ParseTuple(args, (char *)"iiO!s", 
                        &molid, &frame, &PyTuple_Type, &selected, &attr))
    return NULL;  // bad args

  //
  // check molecule
  //
  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, "molecule no longer exists");
    return NULL;
  }
  const int num_atoms = mol->nAtoms;
 
  // 
  // Check for a valid attribute
  //
  SymbolTable *table = app->atomSelParser;
  int attrib_index = table->find_attribute(attr);
  if (attrib_index == -1) {
    PyErr_SetString(PyExc_ValueError, "unknown atom attribute");
    return NULL;
  }
  SymbolTableElement *elem = table->fctns.data(attrib_index);
  if (elem->is_a != SymbolTableElement::KEYWORD &&
      elem->is_a != SymbolTableElement::SINGLEWORD) {
    PyErr_SetString(PyExc_ValueError, "attribute is not a keyword or singleword");
    return NULL;
  }
 
  // 
  // fetch the data
  //

  atomsel_ctxt context(table, mol, frame, attr);
  num_selected = PyTuple_Size(selected);
  PyObject *newlist = PyList_New(num_selected); 

  // XXX should check that selected contains valid indices
  int *flgs = new int[num_atoms];
  memset(flgs,0,num_atoms*sizeof(int));
  for (i=0; i<num_selected; i++) 
    flgs[PyInt_AsLong(PyTuple_GET_ITEM(selected,i))] = 1;

  if (elem->is_a == SymbolTableElement::SINGLEWORD) {
    int *tmp = new int[num_atoms];
    memcpy(tmp, flgs, num_atoms*sizeof(int));
    elem->keyword_single(&context, num_atoms, tmp);
    int j=0;
    for (i=0; i<num_atoms; i++) {
      if (flgs[i]) {
        if (tmp[i]) {
          PyList_SET_ITEM(newlist, j++, PyInt_FromLong(1));       
        } else {
          PyList_SET_ITEM(newlist, j++, PyInt_FromLong(0));       
        }
      }
    }
    delete [] tmp;
  } else {
    switch(table->fctns.data(attrib_index)->returns_a) {
      case (SymbolTableElement::IS_STRING):
      {
        const char **tmp= new const char *[num_atoms];
        elem->keyword_string(&context, num_atoms, tmp, flgs);
        int j=0;
        for (int i=0; i<num_atoms; i++) {
          if (flgs[i]) {
            PyList_SET_ITEM(newlist, j++, PyString_FromString(tmp[i]));
          }
        }
        delete [] tmp;
      }
      break;
      case (SymbolTableElement::IS_INT):
      {
        int *tmp = new int[num_atoms];
        elem->keyword_int(&context, num_atoms, tmp, flgs);
        int j=0;
        for (int i=0; i<num_atoms; i++) {
          if (flgs[i]) {
            PyList_SET_ITEM(newlist, j++, PyInt_FromLong(tmp[i]));
          }
        }
        delete [] tmp;
      }
      break;
      case (SymbolTableElement::IS_FLOAT):
      {
        double *tmp = new double[num_atoms];
        elem->keyword_double(&context, num_atoms, tmp, flgs);
        int j=0;
        for (int i=0; i<num_atoms; i++) {
          if (flgs[i])  
            PyList_SET_ITEM(newlist, j++, PyFloat_FromDouble(tmp[i]));
        }
        delete [] tmp;
      }
      break;
    } // end switch
  }   // end else
  delete [] flgs;
  return newlist;
}

static char set_doc[] = "set(molid, frame, tuple, attribute, values) -> None\nSe attributes for selected atoms using values.";
static PyObject *set(PyObject *self, PyObject *args) {
  int i, molid, frame;
  PyObject *selected, *val;
  char *attr = 0;

  //
  // get molid, frame, list, attribute, and value
  //
  if (!PyArg_ParseTuple(args, (char *)"iiO!sO!", &molid, &frame,
                        &PyTuple_Type, &selected, 
                        &attr, &PyTuple_Type, &val ))
    return NULL;  // bad args

  // 
  // check that we have been given either one value or one for each selected
  // atom
  //
  int num_selected = PyTuple_Size(selected);
  int tuplesize = PyTuple_Size(val);
  if (tuplesize != 1 && tuplesize != num_selected) {
    PyErr_SetString(PyExc_ValueError, "wrong number of items");
    return NULL; 
  }
 
  //
  // check molecule
  //
  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, "molecule no longer exists");
    return NULL;
  }
  const int num_atoms = mol->nAtoms;

  //
  // Check for a valid attribute
  //
  SymbolTable *table = app->atomSelParser;
  int attrib_index = table->find_attribute(attr);
  if (attrib_index == -1) {
    PyErr_SetString(PyExc_ValueError, "unknown atom attribute");
    return NULL;
  }
  SymbolTableElement *elem = table->fctns.data(attrib_index);
  if (elem->is_a != SymbolTableElement::KEYWORD &&
      elem->is_a != SymbolTableElement::SINGLEWORD) {
    PyErr_SetString(PyExc_ValueError, "attribute is not a keyword or singleword");
    return NULL;
  }
  if (!table->is_changeable(attrib_index)) {
    PyErr_SetString(PyExc_ValueError, "attribute is not modifiable");
    return NULL; 
  }

  // 
  // convert the list of selected atoms into an array of integer flags
  //
  // XXX should check that selected contains valid indices
  int *flgs = new int[num_atoms];
  memset(flgs,0,num_atoms*sizeof(int));
  for (i=0; i<num_selected; i++)
    flgs[PyInt_AsLong(PyTuple_GET_ITEM(selected,i))] = 1;
 
  //  
  // set the data
  //

  // singlewords can never be set, so macro is NULL.
  atomsel_ctxt context(table, mol, frame, NULL);
  if (elem->returns_a == SymbolTableElement::IS_INT) {
    int *list = new int[num_atoms];
    if (tuplesize > 1) {
      int j=0;
      for (int i=0; i<num_atoms; i++) {
        if (flgs[i])
          list[i] = PyInt_AsLong(PyTuple_GET_ITEM(val, j++));
      }
    } else {
      for (int i=0; i<num_atoms; i++) {
        if (flgs[i])
          list[i] = PyInt_AsLong(PyTuple_GET_ITEM(val, 0));
      }
    }
    elem->set_keyword_int(&context, num_atoms, list, flgs);
    delete [] list;

  } else if (elem->returns_a == SymbolTableElement::IS_FLOAT) {
    double *list = new double[num_atoms];
    if (tuplesize > 1) { 
      int j=0;
      for (int i=0; i<num_atoms; i++) { 
        if (flgs[i])
          list[i] = PyFloat_AsDouble(PyTuple_GET_ITEM(val, j++));
      }
    } else {
      for (int i=0; i<num_atoms; i++) {
        if (flgs[i])
          list[i] = PyFloat_AsDouble(PyTuple_GET_ITEM(val, 0));
      }
    }
    elem->set_keyword_double(&context, num_atoms, list, flgs);
    delete [] list;


  } else if (elem->returns_a == SymbolTableElement::IS_STRING) {

    const char **list = new const char *[num_atoms];
    if (tuplesize > 1) { 
      int j=0;
      for (int i=0; i<num_atoms; i++) { 
        if (flgs[i])
          list[i] = PyString_AsString(PyTuple_GET_ITEM(val, j++));
      }
    } else {
      for (int i=0; i<num_atoms; i++) {
        if (flgs[i])
          list[i] = PyString_AsString(PyTuple_GET_ITEM(val, 0));
      }
    }
    elem->set_keyword_string(&context, num_atoms, list, flgs);
    delete [] list;
  }

  // Recompute the color assignments if certain atom attributes are changed.
  if (!strcmp(attr, "name") ||
      !strcmp(attr, "type") ||
      !strcmp(attr, "resname") ||
      !strcmp(attr, "chain") ||
      !strcmp(attr, "segid") ||
      !strcmp(attr, "segname")) 
    app->moleculeList->add_color_names(molid);

  mol->force_recalc(DrawMolItem::SEL_REGEN | DrawMolItem::COL_REGEN); 
  delete [] flgs;
  Py_INCREF(Py_None);
  return Py_None;
}

// getbonds(molid, atomlist)
static char getbonds_doc[] = "getbonds(molid, tuple) -> bondlists\nReturn list of bonds for each atom id in tuple.";
static PyObject *getbonds(PyObject *self, PyObject *args) {
  int molid;
  PyObject *atomlist;
   
  if (!PyArg_ParseTuple(args, (char *)"iO!:getbonds", &molid, 
                        &PyTuple_Type, &atomlist)) 
    return NULL;  // bad args
 
  Molecule *mol = get_vmdapp()->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, "molecule no longer exists");
    return NULL;
  }
  int num_atoms = mol->nAtoms;
  int num_selected = PyTuple_Size(atomlist);
  PyObject *newlist = PyList_New(num_selected);
  for (int i=0; i< num_selected; i++) {
    int id = PyInt_AsLong(PyTuple_GET_ITEM(atomlist, i));
    if (PyErr_Occurred()) {
      Py_DECREF(newlist);
      return NULL;
    }
    if (id < 0 || id >= num_atoms) {
      PyErr_SetString(PyExc_ValueError, (char *)"invalid atom id found");
      Py_DECREF(newlist);
      return NULL;
    }
    const MolAtom *atom = mol->atom(id);
    PyObject *bondlist = PyList_New(atom->bonds);
    for (int j=0; j<atom->bonds; j++) {
      PyList_SET_ITEM(bondlist, j, PyInt_FromLong(atom->bondTo[j]));
    }
    PyList_SET_ITEM(newlist, i, bondlist);
  }
  return newlist;
}

// setbonds(molid, atomlist, bondlist)
static char setbonds_doc[] = "setbonds(molid, tuple, bondlist) -> None\nSet bonds for each atom in tuple using bondlist.";
static PyObject *setbonds(PyObject *self, PyObject *args) {
  int molid;
  PyObject *atomlist, *bondlist; 

  if (!PyArg_ParseTuple(args, (char *)"iO!O!:setbonds", &molid, 
                        &PyTuple_Type, &atomlist, &PyList_Type, &bondlist)) 
    return NULL;  // bad args
 
  Molecule *mol = get_vmdapp()->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, "molecule no longer exists");
    return NULL;
  }
  int num_atoms = mol->nAtoms;
  int num_selected = PyTuple_Size(atomlist);
  if (PyList_Size(bondlist) != num_selected) {
    PyErr_SetString(PyExc_ValueError, 
      (char *)"setbonds: atomlist and bondlist must have the same size");
    return NULL;
  }
  mol->force_recalc(DrawMolItem::MOL_REGEN); // many reps ignore bonds
  for (int i=0; i<num_selected; i++) {
    int id = PyInt_AsLong(PyTuple_GET_ITEM(atomlist, i));
    if (PyErr_Occurred()) {
      return NULL;
    }
    if (id < 0 || id >= num_atoms) {
      PyErr_SetString(PyExc_ValueError, (char *)"invalid atom id found");
      return NULL;
    }
    MolAtom *atom = mol->atom(id);
   
    PyObject *atomids = PyList_GET_ITEM(bondlist, i);
    if (!PyList_Check(atomids)) {
      PyErr_SetString(PyExc_TypeError, 
        (char *)"bondlist must contain lists");
      return NULL;
    }
    int numbonds = PyList_Size(atomids);
    int k=0;
    for (int j=0; j<numbonds; j++) {
      int bond = PyInt_AsLong(PyList_GET_ITEM(atomids, j));
      if (PyErr_Occurred())
        return NULL;
      if (bond >= 0 && bond < mol->nAtoms) {
        atom->bondTo[k++] = bond;
      } else {
        char buf[40];
        sprintf(buf, "Invalid atom id in bondlist: %d", bond);
        PyErr_SetString(PyExc_ValueError, buf);
        return NULL;
      }
    }
    atom->bonds = k;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// macro(name, selection)
static char macro_doc[] = "macro(name=None, selection=None) -> macro information\nIf both name and selection are None, return list of macro names.\nIf selection is None, return definition for name.\nIf both name and selection are given, define new macro.\n";
static PyObject *macro(PyObject *self, PyObject *args, PyObject *keywds) {
  char *name = NULL, *selection = NULL;
  static char *kwlist[] = {
    (char *)"name", (char *)"selection", NULL
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"|ss:atomselection.macro", kwlist, &name, &selection))
    return NULL;

  if (selection && !name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Must specify name for macro");
    return NULL;
  }
  SymbolTable *table = get_vmdapp()->atomSelParser;
  if (!name && !selection) {
    // return list of defined macros
    PyObject *macrolist = PyList_New(0);
    for (int i=0; i<table->num_custom_singleword(); i++) {
      const char *s = table->custom_singleword_name(i);
      if (s && strlen(s))
        PyList_Append(macrolist, PyString_FromString(s));
    }
    return macrolist;
  }
  if (name && !selection) {
    // return definition of macro
    const char *s = table->get_custom_singleword(name);
    if (!s) {
      PyErr_SetString(PyExc_ValueError, (char *)"No macro for given name");
      return NULL;
    }
    return PyString_FromString(s);
  }
  // must have both and selection.  Define a new macro.
  if (!table->add_custom_singleword(name, selection)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to create new macro");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// delmacro(name)
static char delmacro_doc[] = "delmacro(name) -> None\nDelete macro with given name.";
static PyObject *delmacro(PyObject *self, PyObject *args) {
  char *name;
  if (!PyArg_ParseTuple(args, (char *)"s:atomselection.delmacro", &name))
    return NULL;
  if (!get_vmdapp()->atomSelParser->remove_custom_singleword(name)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to remove macro");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}
      
static AtomSel *sel_from_py(int molid, int frame, PyObject *selected, VMDApp *app) {
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, "molecule no longer exists");
    return NULL;
  }
  if (mol->nAtoms == 0) {
    PyErr_SetString(PyExc_ValueError, "Molecule in atom selection contains no atoms");
    return NULL;
  }
  AtomSel *sel = new AtomSel(app->atomSelParser, mol->id());
  sel->on = new int[mol->nAtoms];
	sel->num_atoms = mol->nAtoms;
  if (!selected) {
    // turn them all on
    for (int i=0; i<sel->num_atoms; i++) sel->on[i] = 1;
    sel->selected = sel->num_atoms;
  } else {
    memset(sel->on, 0, mol->nAtoms*sizeof(int));
    const int num_selected = PyTuple_Size(selected);
	  sel->selected = num_selected;
    for (int i=0; i<num_selected; i++) {
      int ind = PyInt_AsLong(PyTuple_GET_ITEM(selected,i));
      if (ind < 0 || ind >= mol->nAtoms) {
        delete sel;
        PyErr_SetString(PyExc_ValueError, "Invalid atom id in selection");
        return NULL;
      }
      sel->on[ind] = 1;
    }
  }
  sel->which_frame = frame;
  return sel;
}

// utility routine for parsing weight values.  Uses the sequence protocol
// so that sequence-type structure (list, tuple) will be accepted.
static float *parse_weight(AtomSel *sel, PyObject *wtobj) {
  float *weight = new float[sel->selected];
  if (!wtobj || wtobj == Py_None) {
    for (int i=0; i<sel->selected; i++) weight[i] = 1.0f;
    return weight;
  }

  PyObject *seq = PySequence_Fast(wtobj, (char *)"weight must be a sequence.");
  if (!seq) return NULL;
  if (PySequence_Size(seq) != sel->selected) {
    Py_DECREF(seq);
    PyErr_SetString(PyExc_ValueError, "weight must be same size as selection.");
    delete [] weight;
    return NULL;
  }
  for (int i=0; i<sel->selected; i++) {
    double tmp = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "non-floating point value found in weight.");
      Py_DECREF(seq);
      delete [] weight;
      return NULL;
    }
    weight[i] = (float)tmp;
  }
  Py_DECREF(seq);
  return weight;
}

static char minmax_doc[] = "minmax(molid, frame, tuple) -> (min, max)\nReturn minimum and maximum coordinates for selected atoms.";
static PyObject *minmax(PyObject *self, PyObject *args) {
  int molid, frame;
  PyObject *selected;

  if (!PyArg_ParseTuple(args, (char *)"iiO!", 
                        &molid, &frame, &PyTuple_Type, &selected))
    return NULL;  // bad args

  VMDApp *app = get_vmdapp();
  AtomSel *sel = sel_from_py(molid, frame, selected, app);
  if (!sel) return NULL;
  const Timestep *ts = app->moleculeList->mol_from_id(molid)->get_frame(frame); 
  if (!ts) {
    PyErr_SetString(PyExc_ValueError, "No coordinates in selection");
    delete sel;
    return NULL;
  }
  float min[3], max[3];
  int rc = measure_minmax(sel->num_atoms, sel->on, ts->pos, NULL, min, max);
  delete sel;
  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }
  PyObject *mintuple, *maxtuple, *result;
  mintuple = PyTuple_New(3);
  maxtuple = PyTuple_New(3);
  result = PyTuple_New(2);
  for (int i=0; i<3; i++) {
    PyTuple_SET_ITEM(mintuple, i, PyFloat_FromDouble(min[i]));
    PyTuple_SET_ITEM(maxtuple, i, PyFloat_FromDouble(max[i]));
  }
  PyTuple_SET_ITEM(result, 0, mintuple);
  PyTuple_SET_ITEM(result, 1, maxtuple);
  return result;
}

static char center_doc[] = "center(molid, frame, tuple, weight) -> (x, y, z)\nReturn a tuple corresponding to the center of the selection,\npossibly weighted by weight.";
static PyObject *center(PyObject *self, PyObject *args) {
  int molid, frame;
  PyObject *selected;
  PyObject *weightobj = NULL;
  AtomSel *sel;
  // parse arguments
  if (!PyArg_ParseTuple(args, (char *)"iiO!|O",
        &molid, &frame, &PyTuple_Type, &selected, &weightobj))
    return NULL;
  VMDApp *app = get_vmdapp();
  // get selection
  if (!(sel = sel_from_py(molid, frame, selected, app)))
    return NULL;
  // get weight
  float *weight = parse_weight(sel, weightobj);
  if (!weight) return NULL;
  float cen[3];
  // compute center
  int ret_val = measure_center(sel, sel->coordinates(app->moleculeList),
      weight, cen);
  delete [] weight;
  delete sel;
  if (ret_val < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(ret_val));
    return NULL;
  }
  // return as (x, y, z)
  PyObject *cenobj = PyTuple_New(3);
  for (int i=0; i<3; i++)
    PyTuple_SET_ITEM(cenobj, i, PyFloat_FromDouble(cen[i]));
  return cenobj;
}

static PyObject *py_align(PyObject *self, PyObject *args) {
  int selmol, selframe, refmol, refframe, movemol, moveframe;
  PyObject *selobj, *refobj, *moveobj, *weightobj = NULL;
  if (!PyArg_ParseTuple(args, (char *)"iiO!iiO!iiO!O:atomselection.align",
        &selmol, &selframe, &PyTuple_Type, &selobj,
        &refmol, &refframe, &PyTuple_Type, &refobj,
        &movemol, &moveframe, &PyTuple_Type, &moveobj,
        &weightobj))
    return NULL;

  // check if movemol is -1.  If so, use the sel molecule and timestep instead
  if (movemol == -1) {
    movemol = selmol;
    moveobj = NULL;
  }
  VMDApp *app = get_vmdapp();
  AtomSel *sel=NULL, *ref=NULL, *move=NULL;
  if (!(sel = sel_from_py(selmol, selframe, selobj, app)) ||
      !(ref = sel_from_py(refmol, refframe, refobj, app)) ||
      !(move = sel_from_py(movemol, moveframe, moveobj, app))) {
    delete sel;
    delete ref;
    delete move;
    return NULL;
  }
  const float *selts, *refts;
  float *movets;
  if (!(selts = sel->coordinates(app->moleculeList)) ||
      !(refts = ref->coordinates(app->moleculeList)) || 
      !(movets = move->coordinates(app->moleculeList))) {
    delete sel;
    delete ref;
    delete move;
    PyErr_SetString(PyExc_ValueError, "No coordinates in selection");
    return NULL;
  }
  float *weight = parse_weight(sel, weightobj);
  if (!weight) {
    delete sel;
    delete ref;
    delete move;
    return NULL;
  }
  // Find the matrix that aligns sel with ref.  Apply the transformation to
  // the atoms in move.
  // XXX need to add support for the "order" parameter as in Tcl.
  Matrix4 mat;
  int rc = measure_fit(sel, ref, selts, refts, weight, NULL, &mat);
  delete [] weight;
  delete sel;
  delete ref;
  if (rc < 0) {
    delete move;
    PyErr_SetString(PyExc_ValueError, (char *)measure_error(rc));
    return NULL;
  }
  for (int i=move->firstsel; i<=move->lastsel; i++) {
    if (move->on[i]) {
      float *pos = movets+3*i;
      mat.multpoint3d(pos, pos);
    }
  }
  Molecule *mol = app->moleculeList->mol_from_id(move->molid());
  mol->force_recalc(DrawMolItem::MOL_REGEN);
  delete move;
  Py_INCREF(Py_None);
  return Py_None;
}

static char rmsd_doc[] = "rmsd(mol1, frame1, tuple1, mol2, frame2, tuple2, weights) -> rms distance.\nWeight must be None or list of same size as tuples.";
static PyObject *py_rmsd(PyObject *self, PyObject *args) {
  int mol1, frame1, mol2, frame2;
  PyObject *selected1, *selected2, *weightobj = NULL;
  if (!PyArg_ParseTuple(args, (char *)"iiO!iiO!O:atomselection.rmsd",
        &mol1, &frame1, &PyTuple_Type, &selected1,
        &mol2, &frame2, &PyTuple_Type, &selected2,
				&weightobj))
    return NULL;
  VMDApp *app = get_vmdapp();
  AtomSel *sel1 = sel_from_py(mol1, frame1, selected1, app);
  AtomSel *sel2 = sel_from_py(mol2, frame2, selected2, app);
  if (!sel1 || !sel2) {
    delete sel1;
    delete sel2;
    return NULL;
  }
  const Timestep *ts1 =app->moleculeList->mol_from_id(mol1)->get_frame(frame1); 
  const Timestep *ts2 =app->moleculeList->mol_from_id(mol2)->get_frame(frame2); 
  if (!ts1 || !ts2) {
    PyErr_SetString(PyExc_ValueError, "No coordinates in selection");
    delete sel1;
    delete sel2;
    return NULL;
  }
  float *weight = parse_weight(sel1, weightobj);
  if (!weight) {
    delete sel1;
    delete sel2;
    return NULL;
  }
  float rmsd;
  int rc = measure_rmsd(sel1, sel2, sel1->selected, ts1->pos, ts2->pos,
      weight, &rmsd);
  delete sel1;
  delete sel2;
  delete [] weight;
  if (rc < 0) {
    PyErr_SetString(PyExc_ValueError, measure_error(rc));
    return NULL;
  }
  return PyFloat_FromDouble(rmsd);
}

// Find all atoms p in sel1 and q in sel2 within the cutoff.  
static char contacts_doc[] = "contacts(mol1, frame1, tuple1, mol2, frame2, tuple2, cutoff) -> contact pairs\nReturn pairs of atoms, one from each selection, within cutoff of each other.";
static PyObject *contacts(PyObject *self, PyObject *args) {
  
  int mol1, frame1, mol2, frame2;
  PyObject *selected1, *selected2;
  float cutoff;
  if (!PyArg_ParseTuple(args, (char *)"iiO!iiO!f:atomselection.contacts",
        &mol1, &frame1, &PyTuple_Type, &selected1,
        &mol2, &frame2, &PyTuple_Type, &selected2,
        &cutoff))
    return NULL;
  VMDApp *app = get_vmdapp();
  AtomSel *sel1 = sel_from_py(mol1, frame1, selected1, app);
  AtomSel *sel2 = sel_from_py(mol2, frame2, selected2, app);
  if (!sel1 || !sel2) {
    delete sel1;
    delete sel2;
    return NULL;
  }
  const float *ts1 = sel1->coordinates(app->moleculeList);
  const float *ts2 = sel2->coordinates(app->moleculeList);
  if (!ts1 || !ts2) {
    PyErr_SetString(PyExc_ValueError, "No coordinates in selection");
    delete sel1;
    delete sel2;
    return NULL;
  }
  Molecule *mol = app->moleculeList->mol_from_id(mol1);

  GridSearchPair *pairlist = vmd_gridsearch3(
      ts1, sel1->num_atoms, sel1->on,
      ts2, sel2->num_atoms, sel2->on,
      cutoff, -1, (sel1->num_atoms + sel2->num_atoms) * 27);

  delete sel1;
  delete sel2;
  GridSearchPair *p, *tmp;
  PyObject *list1 = PyList_New(0);
  PyObject *list2 = PyList_New(0);
  PyObject *tmp1;
  PyObject *tmp2;
  for (p=pairlist; p != NULL; p=tmp) {
    // throw out pairs that are already bonded
    MolAtom *a1 = mol->atom(p->ind1);
    if (mol1 != mol2 || !a1->bonded(p->ind2)) {
      // Needed to avoid a memory leak. Append increments the refcount 
      // of whatever gets added to it, but so does PyInt_FromLong.
      // Without a decref, the integers created never have their refcount
      //  go to zero, and you leak memory.
      tmp1 = PyInt_FromLong(p->ind1);
      tmp2 = PyInt_FromLong(p->ind2);
      PyList_Append(list1, tmp1);
      PyList_Append(list2, tmp2);
      Py_DECREF(tmp1);
      Py_DECREF(tmp2);
    }
    tmp = p->next;
    free(p);
  }
  PyObject *result = PyList_New(2);
  PyList_SET_ITEM(result, 0, list1);
  PyList_SET_ITEM(result, 1, list2);
  return result;
}

static char sasa_doc[] = "sasa(srad, sel, samples=500, points=None, restrict=None)\npoints must be a list; surface points will be appended to the list\nin the order xyzxyzxyz (i.e. a flat list)";

static PyObject *sasa(PyObject *self, PyObject *args, PyObject *keywds) {
  int molid = -1, frame = -1;
  float srad = 0;
  PyObject *selobj = NULL, *restrictobj = NULL;
  int samples = -1;
  const int *sampleptr = NULL;
  PyObject *pointsobj = NULL;

  static char *kwlist[] = {
    (char *)"srad", (char *)"molid", (char *)"frame", (char *)"selected",
    (char *)"samples", (char *)"points", (char *)"restrict"
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywds, 
        (char *)"fiiO!|iO!O!:atomselection.sasa", kwlist, 
        &srad, &molid, &frame, &PyTuple_Type, &selobj, 
        &samples, &PyList_Type, &pointsobj, &PyTuple_Type, &restrictobj))
    return NULL;

  // validate srad
  if (srad < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"atomselect.sasa: srad must be non-negative.");
    return NULL;
  }

  // validate selection
  VMDApp *app = get_vmdapp();
  AtomSel *sel = sel_from_py(molid, frame, selobj, app);
  if (!sel) return NULL;

  // fetch the radii and coordinates
  const float *radii = 
    app->moleculeList->mol_from_id(sel->molid())->extraflt.data("radius");
  const float *coords = sel->coordinates(app->moleculeList);

  // if samples was given and is valid, use it
  if (samples > 1) sampleptr = &samples;

  // if restrict is given, validate it
  AtomSel *restrictsel = NULL;
  if (restrictobj) {
    if (!(restrictsel = sel_from_py(molid, frame, restrictobj, app))) {
      delete sel;
      return NULL;
    }
  }

  // if points are requested, fetch them
  ResizeArray<float> sasapts;
  ResizeArray<float> *sasaptsptr = pointsobj ? &sasapts : NULL;
 
  // go!
  float sasa = 0;
  int rc = measure_sasa(sel, coords, radii, srad, &sasa, 
        sasaptsptr, restrictsel, sampleptr);
  delete sel;
  delete restrictsel;
  if (rc) {
    PyErr_SetString(PyExc_ValueError, (char *)measure_error(rc));
    return NULL;
  }

  // append surface points to the provided list object.
  if (pointsobj) {
    for (int i=0; i<sasapts.num(); i++) {
      PyList_Append(pointsobj, PyFloat_FromDouble(sasapts[i]));
    }
  }

  // return the total SASA.
  return PyFloat_FromDouble(sasa);
}

static PyMethodDef AtomselectionMethods[] = {
  {(char *)"create", (vmdPyMethod)create, METH_VARARGS, create_doc },
  {(char *)"get", (vmdPyMethod)get, METH_VARARGS, get_doc },
  {(char *)"set", (vmdPyMethod)set, METH_VARARGS, set_doc },
  {(char *)"getbonds", (vmdPyMethod)getbonds, METH_VARARGS, getbonds_doc },
  {(char *)"setbonds", (vmdPyMethod)setbonds, METH_VARARGS, setbonds_doc },
  {(char *)"macro", (PyCFunction)macro, METH_VARARGS | METH_KEYWORDS, macro_doc},
  {(char *)"delmacro", (vmdPyMethod)delmacro, METH_VARARGS, delmacro_doc },
  {(char *)"minmax", (vmdPyMethod)minmax, METH_VARARGS, minmax_doc },
  {(char *)"center", (vmdPyMethod)center, METH_VARARGS, center_doc },
  {(char *)"rmsd", (vmdPyMethod)py_rmsd, METH_VARARGS, rmsd_doc },
  {(char *)"align", (vmdPyMethod)py_align, METH_VARARGS},
  {(char *)"contacts", (vmdPyMethod)contacts, METH_VARARGS, contacts_doc },
  {(char *)"sasa", (vmdPyMethod)sasa, METH_VARARGS | METH_KEYWORDS, sasa_doc },
  {NULL, NULL}
};

void initatomselection() {
  (void) Py_InitModule((char *)"atomselection", AtomselectionMethods);
}

  
  
