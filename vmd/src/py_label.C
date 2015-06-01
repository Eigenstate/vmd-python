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
 *      $RCSfile: py_label.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $       $Date: 2011/02/01 18:59:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface to labelling functions
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"
#include "GeometryList.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "config.h"

// helper function to turn a GeometryMol object into a dictionary
static PyObject *geom2dict(GeometryMol *g) {
  
  PyObject *newdict = PyDict_New();
  {
    int n = g->items();
    PyObject *molid_tuple = PyTuple_New(n);
    PyObject *atomid_tuple = PyTuple_New(n);
    for (int i=0; i<n; i++) {
      PyTuple_SET_ITEM(molid_tuple, i, PyInt_FromLong(g->obj_index(i)));
      PyTuple_SET_ITEM(atomid_tuple, i, PyInt_FromLong(g->com_index(i)));
    } 
    PyDict_SetItemString(newdict, (char *)"molid", molid_tuple);
    PyDict_SetItemString(newdict, (char *)"atomid", atomid_tuple);
  }
  {
    PyObject *value;
    if (g->ok())
      value = PyFloat_FromDouble(g->calculate());
    else
      value = Py_None;
    PyDict_SetItemString(newdict, (char *)"value", value);
  }
  {
    PyObject *on = PyInt_FromLong(g->displayed() ? 1 : 0);
    PyDict_SetItemString(newdict, (char *)"on", on);
  }
  if (PyErr_Occurred()) {
    Py_DECREF(newdict);
    return NULL;
  } 
  return newdict; 
}

// listall(category) - return a list of labels for the given label category. 
// labels will be returned as dictionary objects with the following keys:
// molid, atomid, value, on.  molid and atomid will be tuples, value will
// be either a float or PyNone, and on will be 1 or 0.
static PyObject *listall(PyObject *self, PyObject *args) {
  char *type;
  if (!PyArg_ParseTuple(args, (char *)"s:label.listall", &type))
    return NULL;

  VMDApp *app = get_vmdapp();
  int cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unknown label category");
    return NULL;
  }
  GeomListPtr glist = app->geometryList->geom_list(cat);
  int gnum = glist->num(); 
  PyObject *newlist = PyList_New(gnum);
  for (int i=0; i<gnum; i++) {
    PyObject *obj = geom2dict((*glist)[i]);
    if (obj == NULL) {
      Py_DECREF(newlist);
      return NULL;
    }
    PyList_SET_ITEM(newlist, i, geom2dict((*glist)[i]));
  }
  return newlist;
} 

// add(category, (molids), (atomids))
static PyObject *label_add(PyObject *self, PyObject *args) {
  char *type;
  PyObject *molids, *atomids;
  int i;
  if (!PyArg_ParseTuple(args, (char *)"sO!O!:label.add", 
    &type, &PyTuple_Type, &molids, &PyTuple_Type, &atomids))
    return NULL;

  VMDApp *app = get_vmdapp();
  MoleculeList *mlist = app->moleculeList;
  int cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unknown label category");
    return NULL;
  }
  int numitems;
  if (PyTuple_Size(molids) == 1 && PyTuple_Size(atomids) == 1)
    numitems = 1;
  else if (PyTuple_Size(molids) == 2 && PyTuple_Size(atomids) == 2)
    numitems = 2; 
  else if (PyTuple_Size(molids) == 3 && PyTuple_Size(atomids) == 3)
    numitems = 3;
  else if (PyTuple_Size(molids) == 4 && PyTuple_Size(atomids) == 4)
    numitems = 4;
  else {
    PyErr_SetString(PyExc_TypeError, (char *)"label.add: 2nd and 3rd arguments"
      " must be tuples of size 1, 2, 3, or 4");
    return NULL;
  }
  int m[4], a[4];
  for (i=0; i<numitems; i++) {
    m[i] = PyInt_AsLong(PyTuple_GET_ITEM(molids, i));
    a[i] = PyInt_AsLong(PyTuple_GET_ITEM(atomids, i));
    if (PyErr_Occurred())
      return NULL;
    Molecule *mol = mlist->mol_from_id(m[i]);
    if (!mol) {
      PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
      return NULL;
    }
    if (a[i] < 0 || a[i] >= mol->nAtoms) {
      PyErr_SetString(PyExc_ValueError, (char *)"Invalid atom id");
      return NULL;
    }
  } 
  // Add the label, but don't toggle the on/off status.  
  int ind = app->label_add(type, numitems, m, a, NULL, 0.0f, 0);
  if (ind < 0) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  // Get the dict corresponding to this label.  The dict we return
  // corresponds to either the label we just created, or to an existing
  // label whose state we have not changed.  This makes it safe to use
  // label.add to get a proxy to a VMD label.
  GeomListPtr glist = app->geometryList->geom_list(cat);
  return geom2dict((*glist)[ind]);
}

// returns true if the given label object matches the given geometry object
static int dict2geom(PyObject *dict, GeometryMol *g) {
  
  PyObject *molid = PyDict_GetItemString(dict, (char *)"molid");
  PyObject *atomid = PyDict_GetItemString(dict, (char *)"atomid");
  if (molid == NULL || atomid == NULL || 
      !PyTuple_Check(molid) || !PyTuple_Check(atomid)) {
    return 0;
  }

  int numitems = PyTuple_Size(molid);
  if (numitems != PyTuple_Size(atomid) || numitems != g->items()) {
    return 0;
  }

  for (int i=0; i<numitems; i++) {
    int m = PyInt_AsLong(PyTuple_GET_ITEM(molid, i));
    int a = PyInt_AsLong(PyTuple_GET_ITEM(atomid, i));
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return 0;
    }
    if (m != g->obj_index(i) || a != g->com_index(i)) {
      return 0;
    }
  }
  return 1;
}
   
// show(category, labeldict)
static PyObject *label_show(PyObject *self, PyObject *args) {
  char *type;
  PyObject *labeldict;
  if (!PyArg_ParseTuple(args, (char *)"sO!:label.show", 
    &type, &PyDict_Type, &labeldict))
    return NULL;

  VMDApp *app = get_vmdapp();
  int cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unknown label category");
    return NULL;
  }
  GeomListPtr glist = app->geometryList->geom_list(cat);
  int gnum = glist->num();
  for (int i=0; i<gnum; i++) {
    if (dict2geom(labeldict, (*glist)[i])) {
      app->label_show(type, i, 1); // XXX check return code
      Py_INCREF(Py_None);
      return Py_None;
    }
  }
  PyErr_SetString(PyExc_ValueError, "Invalid labeldict.");
  return NULL;
} 
 
// hide(category, labeldict)
// XXX cut 'n paste from show...
static PyObject *label_hide(PyObject *self, PyObject *args) {
  char *type;
  PyObject *labeldict;
  if (!PyArg_ParseTuple(args, (char *)"sO!:label.hide", 
    &type, &PyDict_Type, &labeldict))
    return NULL;

  VMDApp *app = get_vmdapp();
  int cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unknown label category");
    return NULL;
  }
  GeomListPtr glist = app->geometryList->geom_list(cat);
  int gnum = glist->num();
  for (int i=0; i<gnum; i++) {
    if (dict2geom(labeldict, (*glist)[i])) {
      app->label_show(type, i, 0); // XXX check return code
      Py_INCREF(Py_None);
      return Py_None;
    }
  }
  PyErr_SetString(PyExc_ValueError, "Invalid labeldict.");
  return NULL;
} 

// delete(category, labeldict)
// XXX cut 'n paste from show...
static PyObject *label_delete(PyObject *self, PyObject *args) {
  char *type;
  PyObject *labeldict;
  if (!PyArg_ParseTuple(args, (char *)"sO!:label.delete", 
    &type, &PyDict_Type, &labeldict))
    return NULL;

  VMDApp *app = get_vmdapp();
  int cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unknown label category");
    return NULL;
  }
  GeomListPtr glist = app->geometryList->geom_list(cat);
  int gnum = glist->num();
  for (int i=0; i<gnum; i++) {
    if (dict2geom(labeldict, (*glist)[i])) {
      app->label_delete(type, i); // XXX check return code
      Py_INCREF(Py_None);
      return Py_None;
    }
  }
  PyErr_SetString(PyExc_ValueError, "Invalid labeldict.");
  return NULL;
} 

// return Python list of values for this label.  Return None if this label
// has no values (e.g. Atom labels), or NULL on error.
static PyObject *getvalues(GeometryMol *g) {
  if (!g->has_value()) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  ResizeArray<float> gValues(1024);
  if (!g->calculate_all(gValues)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid label");
    return NULL;
  }
  const int n = gValues.num();
  PyObject *newlist = PyList_New(n);
  for (int i=0; i<n; i++) {
    PyList_SET_ITEM(newlist, i, PyFloat_FromDouble(gValues[i]));
  }
  return newlist;
}

// getvalues(category, labeldict) -> list
// returns list containing value of label for every frame in the label
// if the label contains atoms from more than one molecule, only the first
// molecule is cycled (this the behavior of GeometryMol).
// XXX Note: this command is bad: the GeometryMol::calculate_all method 
// twiddles the frame of the molecule, so this command isn't read only as 
// its semantics would imply.  But it should be possible to fix this 
// in GeometryMol.
static PyObject *label_getvalues(PyObject *self, PyObject *args) {
  char *type;
  PyObject *labeldict;
  if (!PyArg_ParseTuple(args, (char *)"sO!:label.getvalues",
    &type, &PyDict_Type, &labeldict))
    return NULL;

  VMDApp *app = get_vmdapp();
  int cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unknown label category");
    return NULL;
  }
  GeomListPtr glist = app->geometryList->geom_list(cat);
  int gnum = glist->num();
  for (int i=0; i<gnum; i++) {
    if (dict2geom(labeldict, (*glist)[i])) 
      return getvalues((*glist)[i]);
  }
  PyErr_SetString(PyExc_ValueError, (char *)"Invalid label");
  return NULL;
}

static PyObject *label_textsize(PyObject *self, PyObject *args) {
  float newsize = -1;
  if (!PyArg_ParseTuple(args, (char *)"|f:label.textsize",
        &newsize))
    return NULL;
  VMDApp *app = get_vmdapp();
  if (newsize > 0) {
    app->label_set_text_size(newsize);
  }
  return Py_BuildValue("%f", app->label_get_text_size());
}

static PyObject *label_textthickness(PyObject *self, PyObject *args) {
  float newthick = -1;
  if (!PyArg_ParseTuple(args, (char *)"|f:label.textthickness",
        &newthick))
    return NULL;
  VMDApp *app = get_vmdapp();
  if (newthick > 0) {
    app->label_set_text_thickness(newthick);
  }
  return Py_BuildValue("%f", app->label_get_text_thickness());
}


static PyMethodDef LabelMethods[] = {
  {(char *)"listall", (vmdPyMethod)listall, METH_VARARGS },
  {(char *)"add", (vmdPyMethod)label_add, METH_VARARGS },
  {(char *)"show", (vmdPyMethod)label_show, METH_VARARGS },
  {(char *)"hide", (vmdPyMethod)label_hide, METH_VARARGS },
  {(char *)"delete", (vmdPyMethod)label_delete, METH_VARARGS },
  {(char *)"getvalues", (vmdPyMethod)label_getvalues, METH_VARARGS },
  {(char *)"textsize", (vmdPyMethod)label_textsize, METH_VARARGS },
  {(char *)"textthickness", (vmdPyMethod)label_textthickness, METH_VARARGS },
  {NULL, NULL}
};

void initlabel() {
  PyObject *m = Py_InitModule((char *)"label", LabelMethods);
  PyModule_AddStringConstant(m, (char *)"ATOM", (char *)"Atoms"); 
  PyModule_AddStringConstant(m, (char *)"BOND", (char *)"Bonds"); 
  PyModule_AddStringConstant(m, (char *)"ANGLE", (char *)"Angles"); 
  PyModule_AddStringConstant(m, (char *)"DIHEDRAL", (char *)"Dihedrals"); 
}
 
