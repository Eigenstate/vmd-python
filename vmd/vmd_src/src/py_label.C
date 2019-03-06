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
 *      $RCSfile: py_label.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.24 $       $Date: 2019/01/17 21:21:03 $
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
static PyObject* geom2dict(GeometryMol *g)
{
  PyObject *newdict, *molid_tuple, *atomid_tuple, *value, *on;
  int n = g->items();

  newdict = PyDict_New();

  // Populate molid and atomid tuples
  molid_tuple = PyTuple_New(n);
  atomid_tuple = PyTuple_New(n);
  for (int i=0; i<n; i++) {
    PyTuple_SET_ITEM(molid_tuple, i, as_pyint(g->obj_index(i)));
    PyTuple_SET_ITEM(atomid_tuple, i, as_pyint(g->com_index(i)));
  }
  PyDict_SetItemString(newdict, (char *)"molid", molid_tuple);
  PyDict_SetItemString(newdict, (char *)"atomid", atomid_tuple);

  // Value is a float or None
  value = g->ok() ? PyFloat_FromDouble(g->calculate()) : Py_None;
  PyDict_SetItemString(newdict, (char *)"value", value);

  // On is if it's displayed
  on = g->displayed() ? Py_True : Py_False;
  PyDict_SetItemString(newdict, (char *)"on", on);
  Py_INCREF(on);

  if (PyErr_Occurred()) {
    Py_DECREF(newdict);
    return NULL;
  }
  return newdict;
}

// helper function to check if a label dictionary matches a GeometryMol object
static int dict2geom(PyObject *dict, GeometryMol *g) {

  PyObject *molid, *atomid;
  int m, a, numitems;

  if (!PyDict_Check(dict)) {
    PyErr_SetString(PyExc_RuntimeError, "Non-dict passed to dict2geom");
    return 0;
  }

  molid = PyDict_GetItemString(dict, (char *)"molid");
  atomid = PyDict_GetItemString(dict, (char *)"atomid");

  if (!molid || !atomid || !PyTuple_Check(molid) || !PyTuple_Check(atomid))
    return 0;

  numitems = PyTuple_Size(molid);
  if (numitems != PyTuple_Size(atomid) || numitems != g->items()) {
    return 0;
  }

  for (int i=0; i<numitems; i++) {
    m = as_int(PyTuple_GET_ITEM(molid, i));
    a = as_int(PyTuple_GET_ITEM(atomid, i));

    if (PyErr_Occurred()) {
      PyErr_Clear();
      return 0;
    }

    if (m != g->obj_index(i) || a != g->com_index(i))
      return 0;
  }

  return 1;
}

static const char listall_doc[] =
"Get labels in the given label category.\n\n"
"Args:\n"
"    category (str): Label category to list, in ['atoms','bonds','dihedrals']\n"
"Returns:\n"
"    (dict): All labels, with keys 'molid', 'atomid', 'value', and 'on'.\n"
"        Molid and atomid will be tuples, value will be a float or None, and\n"
"        on will be True or False.";
static PyObject* py_listall(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"category", NULL};
  PyObject *newlist, *obj;
  GeomListPtr glist;
  int cat, gnum;
  VMDApp *app;
  char *type;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:label.listall",
                                   (char**) kwlist, &type))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_Format(PyExc_ValueError, "Unknown label category '%s'", type);
    return NULL;
  }

  glist = app->geometryList->geom_list(cat);
  gnum = glist->num();

  newlist = PyList_New(gnum);

  for (int i=0; i<gnum; i++) {
    obj = geom2dict((*glist)[i]);

    if (!obj) {
      Py_DECREF(newlist);
      return NULL;
    }
    PyList_SET_ITEM(newlist, i, obj);
  }
  return newlist;
}

// add(category, (molids), (atomids))
static const char label_add_doc[] =
"Add a label of the given type. If label already exists, no action is \n"
"performed. The number of elements in the molids and atomids tuple is \n"
"determined by the label type.\n\nArgs:\n"
"    category (str): Label category to add. Must be one of the following:\n"
"        'Atoms', 'Bonds', 'Angles', 'Dihedrals', 'Springs'\n"
"    molids (list or tuple): Molids to label. Length as required to describe\n"
"        label category-- 1 for atoms, 2 bonds, etc.\n"
"    atomids (list or tuple): Atom IDs to label. Must be same length as molids\n"
"Returns:\n"
"    (dict) Dictionary describing the label, can be used as input to other\n"
"        label module functions";
static PyObject* py_label_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"category", "molids", "atomids",  NULL};
  PyObject *molseq = NULL, *atomseq = NULL;
  int cat, numitems, ind, i, m[4], a[4];
  PyObject *molids, *atomids;
  GeomListPtr glist;
  Molecule *mol;
  VMDApp *app;
  char *type;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOO:label.add", (char**)
                                   kwlist, &type, &molids, &atomids))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Get label category and check it is valid
  cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_Format(PyExc_ValueError, "Unknown label category '%s'", type);
    return NULL;
  }

  // PySequence_Fast type checks molids and atomids are a sequence
  if (!(molseq = PySequence_Fast(molids, "molids must be a list or tuple")))
    goto failure;
  if (!(atomseq = PySequence_Fast(atomids, "atomids must be a list or tuple")))
    goto failure;

  // Validate length of molids and atomids tuples
  numitems = PySequence_Fast_GET_SIZE(molseq);
  if (numitems != PySequence_Fast_GET_SIZE(atomseq)) {
    PyErr_SetString(PyExc_ValueError, "atomids and molids must be same length");
    return NULL;
  }

  // Check appropriate num items for type
  if (!strcmp(type, "Atoms") && numitems != 1) {
    PyErr_Format(PyExc_ValueError, "Atom labels require 1 item, got %d",
                 numitems);
    return NULL;
  } else if (!strcmp(type, "Bonds") && numitems != 2) {
    PyErr_Format(PyExc_ValueError, "Bond labels require 2 items, got %d",
                 numitems);
    return NULL;
  } else if (!strcmp(type, "Angles") && numitems != 3) {
    PyErr_Format(PyExc_ValueError, "Angle labels require 3 items, got %d",
                 numitems);
    return NULL;
  } else if (!strcmp(type, "Dihedrals") && numitems != 4) {
    PyErr_Format(PyExc_ValueError, "Dihedral labels require 4 items, got %d",
                 numitems);
    return NULL;
  } else if (!strcmp(type, "Springs") && numitems != 4) {
    PyErr_Format(PyExc_ValueError, "Spring labels require 4 items, got %d",
                 numitems);
    return NULL;
  }

  // Unpack label tuples, checking molid and atomid for validity
  for (i = 0; i < numitems; i++) {

    m[i] = as_int(PySequence_Fast_GET_ITEM(molseq, i));
    a[i] = as_int(PySequence_Fast_GET_ITEM(atomseq, i));

    if (PyErr_Occurred())
      goto failure;

    // Check molid is correct
    mol = app->moleculeList->mol_from_id(m[i]);
    if (!mol) {
      PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", m[i]);
      goto failure;
    }

    // Check atomid is correct
    if (a[i] < 0 || a[i] >= mol->nAtoms) {
      PyErr_Format(PyExc_ValueError, "Invalid atom id '%d'", a[i]);
      goto failure;
    }
  }

  // Add the label, but don't toggle the on/off status.
  ind = app->label_add(type, numitems, m, a, NULL, 0.0f, 0);
  if (ind < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Could not add label. Is the number "
                    "of atoms correct for the label type?");
    goto failure;
  }

  // Get the dict corresponding to this label. The dict we return
  // corresponds to either the label we just created, or to an existing
  // label whose state we have not changed.  This makes it safe to use
  // label.add to get a proxy to a VMD label.
  glist = app->geometryList->geom_list(cat);
  return geom2dict((*glist)[ind]);

failure:
  Py_XDECREF(molseq);
  Py_XDECREF(atomseq);
  return NULL;
}

static const char label_visible_doc[] =
"Sets a label to be visible\n\n"
"Args:\n"
"    category (str): Label category, in ['atoms','bonds','dihedrals']\n"
"    label (dict): Label to show, from output of label.add or similar\n"
"    visible (bool): True to show, False to hide the label";
static PyObject* py_label_visible(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"category", "label", "visible", NULL};
  int shown = 1, rc = 0;
  PyObject *labeldict;
  GeomListPtr glist;
  VMDApp *app;
  char *type;
  int cat;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO!O&:label.show", (char**)
                                   kwlist, &type, &PyDict_Type, &labeldict,
                                   convert_bool, &shown))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Get label category
  cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_Format(PyExc_ValueError, "Unknown label category '%s'", type);
    return NULL;
  }

  glist = app->geometryList->geom_list(cat);

  // Look through all labels in category for one that matches
  for (int i=0; i<glist->num(); i++) {
    if (dict2geom(labeldict, (*glist)[i])) {
      rc = app->label_show(type, i, shown);
      break;
    }
  }

  // If no label matched or show failed, rc will be FALSE
  if (!rc) {
    PyErr_SetString(PyExc_ValueError, "Could not show/hide label");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char label_del_doc[] =
"Delete a label\n\nArgs:\n"
"    label (dict): Label to delete. Dictionary format generated by label.add\n";
static PyObject* py_label_delete(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"label", NULL};
  PyObject *labeldict;
  GeomListPtr glist;
  VMDApp *app;
  char *type;
  int rc = 0;
  int cat;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO!:label.delete", (char**)
                                   kwlist, &type, &PyDict_Type, &labeldict))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Get category
  cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unknown label category");
    return NULL;
  }

  // Look for our label
  glist = app->geometryList->geom_list(cat);
  for (int i = 0; i < glist->num(); i++) {
    if (dict2geom(labeldict, (*glist)[i])) {
      rc = app->label_delete(type, i);
      break;
    }
  }

  // rc will be TRUE only if deletion was successful
  if (!rc) {
      PyErr_SetString(PyExc_ValueError, "Could not delete label");
      return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

// return Python list of values for this label.  Return None if this label
// has no values (e.g. Atom labels), or NULL on error.
static PyObject* getvalues(GeometryMol *g) {

  if (!g->has_value()) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  ResizeArray<float> gValues(1024);
  if (!g->calculate_all(gValues)) {
    PyErr_SetString(PyExc_ValueError, "Cannot calculate label values");
    return NULL;
  }

  const int n = gValues.num();
  PyObject *newlist = PyList_New(n);
  for (int i=0; i<n; i++) {
    PyList_SET_ITEM(newlist, i, PyFloat_FromDouble(gValues[i]));
  }

  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, "Problem getting label values");
    Py_DECREF(newlist);
    return NULL;
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
static const char label_values_doc[] =
"Get label values for every frame in the molecule. 'value' refers to the "
"quantity the label measures-- for example, bond labels give distance.\n\n"
"Args:\n"
"    category (str): Label category to list, in ['atoms','bonds','dihedrals']\n"
"    label (dict): Label to query, from output of label.add\n"
"Returns:\n"
"    (list of float) Label values for each frame in label, or None if\n"
"        label has no values, like atom labels.";
static PyObject* py_label_getvalues(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  const char* kwlist[] = {"category", "label", NULL};
  PyObject *result = NULL;
  PyObject *labeldict;
  GeomListPtr glist;
  VMDApp *app;
  char *type;
  int cat;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO!:label:get_values",
                                   (char**) kwlist, &type, &PyDict_Type,
                                   &labeldict))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  cat = app->geometryList->geom_list_index(type);
  if (cat < 0) {
    PyErr_Format(PyExc_ValueError, "Unknown label category '%s'", type);
    return NULL;
  }

  glist = app->geometryList->geom_list(cat);
  for (int i=0; i<glist->num(); i++) {
    if (dict2geom(labeldict, (*glist)[i])) {
      result = getvalues((*glist)[i]);
      break;
    }
  }

  if (!result) {
      PyErr_SetString(PyExc_ValueError, "Could not find label");
      return NULL;
  }
  return result;
}

static const char label_size_doc[] =
"Sets text size for all labels.\n\n"
"Args:\n"
"    size (float): Text size, optional. Must be greater than zero.\n"
"        If None, just returns size\n"
"Returns:\n"
"    (float) Text size";
static PyObject* py_label_textsize(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {

  const char *kwlist[] = {"size", NULL};
  PyObject *newobj = NULL;
  float newsize;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:label.text_size",
                                   (char**) kwlist, &newobj))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (newobj) {
      if (!PyFloat_Check(newobj)) {
        PyErr_SetString(PyExc_TypeError, "Text size must be a float");
        return NULL;
      }

      newsize = PyFloat_AsDouble(newobj);

      if (PyErr_Occurred() || newsize <= 0) {
        PyErr_SetString(PyExc_ValueError, "Text size must be > 0");
        return NULL;
      }

    app->label_set_text_size(newsize);
  }
  return PyFloat_FromDouble(app->label_get_text_size());
}

static const char label_thick_doc[] =
"Sets text thickness for all labels.\n\n"
"Args:\n"
"    thickness (float): Thickness, optional. Must be greater than zero.\n"
"        If None, just returns thickness.\n"
"Returns:\n"
"    (float) Text thickness";
static PyObject *py_label_textthickness(PyObject *self, PyObject *args,
                                        PyObject *kwargs) {

  const char *kwlist[] = {"thickness", NULL};
  PyObject *newobj = NULL;
  float newthick;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:label.text_thickness",
                                   (char**) kwlist, &newobj))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (newobj) {
      if (!PyFloat_Check(newobj)) {
        PyErr_SetString(PyExc_TypeError, "Text thickness must be a float");
        return NULL;
      }

      newthick = PyFloat_AsDouble(newobj);

      if (PyErr_Occurred() || newthick <= 0) {
        PyErr_SetString(PyExc_ValueError, "Text thickness must be > 0");
        return NULL;
      }

      app->label_set_text_thickness(newthick);
  }
  return PyFloat_FromDouble(app->label_get_text_thickness());
}


static PyMethodDef LabelMethods[] = {
  {"listall", (PyCFunction)py_listall, METH_VARARGS | METH_KEYWORDS, listall_doc },
  {"add", (PyCFunction)py_label_add, METH_VARARGS | METH_KEYWORDS, label_add_doc },
  {"set_visible", (PyCFunction)py_label_visible, METH_VARARGS | METH_KEYWORDS, label_visible_doc},
  {"delete", (PyCFunction)py_label_delete, METH_VARARGS | METH_KEYWORDS, label_del_doc },
  {"get_values", (PyCFunction)py_label_getvalues, METH_VARARGS | METH_KEYWORDS, label_values_doc},
  {"text_size", (PyCFunction)py_label_textsize, METH_VARARGS | METH_KEYWORDS, label_size_doc},
  {"text_thickness", (PyCFunction)py_label_textthickness, METH_VARARGS | METH_KEYWORDS, label_thick_doc},
  {NULL, NULL}
};

static const char label_moddoc[] =
"Methods to control the visibility and style of molecule labels";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef labeldef = {
  PyModuleDef_HEAD_INIT,
  "label",
  label_moddoc,
  -1,
  LabelMethods,
};
#endif

PyObject* initlabel(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&labeldef);
#else
  PyObject *m = Py_InitModule3("label", LabelMethods, label_moddoc);
#endif

  PyModule_AddStringConstant(m, "ATOM", "Atoms");
  PyModule_AddStringConstant(m, "BOND", "Bonds");
  PyModule_AddStringConstant(m, "ANGLE", "Angles");
  PyModule_AddStringConstant(m, "DIHEDRAL", "Dihedrals");
  return m;
}

