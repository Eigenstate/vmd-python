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
 *      $RCSfile: py_material.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.25 $       $Date: 2015/05/20 20:23:48 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface for managing materials.
 ***************************************************************************/

#include "py_commands.h"
#include "CmdMaterial.h"
#include "utilities.h"
#include "MaterialList.h"
#include "CommandQueue.h"
#include "VMDApp.h"
#include <stdlib.h>
#include <ctype.h>

// listall(): list all materials
static PyObject *listall(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  MaterialList *mlist = get_vmdapp()->materialList;
  PyObject *newlist = PyList_New(0);
  for (int i=0; i<mlist->num(); i++) 
    PyList_Append(newlist, PyString_FromString(mlist->material_name(i)));

  return newlist;
}

// settings(name): return a dictionary object with the material settings 
static PyObject *settings(PyObject *self, PyObject *args) {
  char *name;
  if (!PyArg_ParseTuple(args, (char *)"s", &name))
    return NULL;

  MaterialList *mlist = get_vmdapp()->materialList;
  int ind = mlist->material_index(name);
  if (ind < 0) {
    PyErr_SetString(PyExc_ValueError, "No such material");
    return NULL;
  }
  
  PyObject *dict = PyDict_New();
  PyDict_SetItemString(dict, (char *)"ambient", 
    PyFloat_FromDouble(mlist->get_ambient(ind)));
  PyDict_SetItemString(dict, (char *)"specular", 
    PyFloat_FromDouble(mlist->get_specular(ind)));
  PyDict_SetItemString(dict, (char *)"diffuse", 
    PyFloat_FromDouble(mlist->get_diffuse(ind)));
  PyDict_SetItemString(dict, (char *)"shininess", 
    PyFloat_FromDouble(mlist->get_shininess(ind)));
  PyDict_SetItemString(dict, (char *)"mirror", 
    PyFloat_FromDouble(mlist->get_mirror(ind)));
  PyDict_SetItemString(dict, (char *)"opacity", 
    PyFloat_FromDouble(mlist->get_opacity(ind)));
  PyDict_SetItemString(dict, (char *)"outline", 
    PyFloat_FromDouble(mlist->get_outline(ind)));
  PyDict_SetItemString(dict, (char *)"outlinewidth", 
    PyFloat_FromDouble(mlist->get_outlinewidth(ind)));
  PyDict_SetItemString(dict, (char *)"transmode", 
    PyFloat_FromDouble(mlist->get_transmode(ind)));
 
  return dict;
}

static PyObject *set_default(PyObject *self, PyObject *args) {
  int ind;
  if (!PyArg_ParseTuple(args, (char *)"i:material.default", &ind))
    return NULL;
  if (!get_vmdapp()->material_restore_default(ind)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Material has no default settings");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// add('name=None', copy=None): create a new material with the given name.
// optionally copy the given name.  If no name is given, make up a new name.

static PyObject *add(PyObject *self, PyObject *args) {
  char *name = NULL;
  char *copy = NULL;
  if (!PyArg_ParseTuple(args, (char *)"|ss:material.add", &name, &copy))
    return NULL;

  VMDApp *app = get_vmdapp();
  MaterialList *mlist = app->materialList;
  if (name) {
    int ind = mlist->material_index(name);
    if (ind >= 0) { // material already exists
      PyErr_SetString(PyExc_ValueError, (char *)"Material already exists");
      return NULL;
    }
  }
  if (copy) {
    int ind = mlist->material_index(copy);
    if (ind < 0) { // material doesn't exist
      PyErr_SetString(PyExc_ValueError, (char *)"Material to copy doesn't exist");
      return NULL;
    }
  }
  const char *result = app->material_add(name, copy);
  if (!result) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to add material.");
    return NULL;
  }
  return PyString_FromString((char *)result);
}

// delete('name').
static PyObject *matdelete(PyObject *self, PyObject *args) {
  char *name = NULL;
  if (!PyArg_ParseTuple(args, (char *)"s:material.delete", &name)) 
    return NULL;
  VMDApp *app = get_vmdapp();
  if (!app->material_delete(name)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to delete material.");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// rename(oldname, newname): change the name of an existing material

static PyObject *rename(PyObject *self, PyObject *args) {
  char *oldname, *newname;
  if (!PyArg_ParseTuple(args, (char *)"ss:material.rename",&oldname, &newname))
    return NULL;

  VMDApp *app = get_vmdapp();
  MaterialList *mlist = app->materialList;
  int ind = mlist->material_index(oldname);
  if (ind < 0) {
    PyErr_SetString(PyExc_ValueError, "Material to change does not exist");
    return NULL;
  }
  if (mlist->material_index(newname) >= 0) {
    PyErr_SetString(PyExc_ValueError, "New name already exists");
    return NULL;
  }
  if (!app->material_rename(oldname, newname)) {
    PyErr_SetString(PyExc_ValueError, "Could not change material name");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// change(name, ambient, specular, diffuse, shininess, mirror, opacity, 
//        outline, outlinewidth, transmode)
static PyObject *change(PyObject *self, PyObject *args, PyObject *keywds) {
  char *name;
  float ambient, specular, diffuse, shininess, mirror, opacity;
  float outline, outlinewidth, transmode;

  static char *kwlist[] = {
    (char *)"name",
    (char *)"ambient",
    (char *)"specular",
    (char *)"diffuse",
    (char *)"shininess",
    (char *)"mirror",
    (char *)"opacity",
    (char *)"outline",
    (char *)"outlinewidth",
    (char *)"transmode", 
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"s|fffffffff", kwlist,
      &name, &ambient, &specular, &diffuse, &shininess, &mirror, &opacity, 
      &outline, &outlinewidth, &transmode))
    return NULL;

  VMDApp *app = get_vmdapp();
  MaterialList *mlist = app->materialList;
  int ind = mlist->material_index(name);
  if (ind < 0) {
    PyErr_SetString(PyExc_ValueError, "Material to change does not exist");
    return NULL;
  }

  if (PyDict_GetItemString(keywds, (char *)"ambient"))
    app->material_change(name, MAT_AMBIENT, ambient);
  if (PyDict_GetItemString(keywds, (char *)"specular"))
    app->material_change(name, MAT_SPECULAR, specular);
  if (PyDict_GetItemString(keywds, (char *)"diffuse"))
    app->material_change(name, MAT_DIFFUSE, diffuse);
  if (PyDict_GetItemString(keywds, (char *)"shininess"))
    app->material_change(name, MAT_SHININESS, shininess);
  if (PyDict_GetItemString(keywds, (char *)"mirror"))
    app->material_change(name, MAT_MIRROR, mirror);
  if (PyDict_GetItemString(keywds, (char *)"opacity"))
    app->material_change(name, MAT_OPACITY, opacity);
  if (PyDict_GetItemString(keywds, (char *)"outline"))
    app->material_change(name, MAT_OUTLINE, outline);
  if (PyDict_GetItemString(keywds, (char *)"outlinewidth"))
    app->material_change(name, MAT_OUTLINEWIDTH, outlinewidth);
  if (PyDict_GetItemString(keywds, (char *)"transmode"))
    app->material_change(name, MAT_TRANSMODE, transmode);
  
  Py_INCREF(Py_None);
  return Py_None;
}
   
// default(index) - restore default of material with given index
static PyMethodDef methods[] = {
  {(char *)"listall", (vmdPyMethod)listall, METH_VARARGS },
  {(char *)"settings", (vmdPyMethod)settings, METH_VARARGS },
  {(char *)"add", (vmdPyMethod)add, METH_VARARGS },
  {(char *)"delete", (vmdPyMethod)matdelete, METH_VARARGS },
  {(char *)"rename", (vmdPyMethod)rename, METH_VARARGS },
  {(char *)"change", (PyCFunction)change, METH_VARARGS | METH_KEYWORDS },
  {(char *)"default", (vmdPyMethod)set_default, METH_VARARGS },
  {NULL, NULL}
};

void initmaterial() {
  (void) Py_InitModule((char *)"material", methods);
}

 


