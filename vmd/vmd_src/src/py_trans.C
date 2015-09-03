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
 *      $RCSfile: py_trans.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.15 $       $Date: 2010/12/16 04:08:57 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface for transforming molecules (scale/trans/rot, etc)
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"
#include "CommandQueue.h"
#include "MoleculeList.h"
#include "Molecule.h"

// rotate(axis, angle)
static PyObject *rotate(PyObject *self, PyObject *args) {
  char axis;
  float angle;

  if (!PyArg_ParseTuple(args, (char *)"cf:trans.rotate", &axis, &angle)) 
    return NULL;

  if (axis != 'x' && axis != 'y' && axis != 'z') {
    PyErr_SetString(PyExc_ValueError, (char *)"axis must be 'x', 'y', or 'z'");
    return NULL;
  }
  VMDApp *app = get_vmdapp();
  app->scene_rotate_by(angle, axis);

  Py_INCREF(Py_None);
  return Py_None;
}

// translate(x,y,z)
static PyObject *translate(PyObject *self, PyObject *args) {
  float x,y,z;
  
  if (!PyArg_ParseTuple(args, (char *)"fff:trans.translate", &x, &y, &z))
    return NULL;

  VMDApp *app = get_vmdapp(); 
  app->scene_translate_by(x,y,z);
  
  Py_INCREF(Py_None);
  return Py_None;
}

// scale(factor)
static PyObject *scale(PyObject *self, PyObject *args) {
  float factor;
  if (!PyArg_ParseTuple(args, (char *)"f:trans.scale", &factor))
    return NULL;

  VMDApp *app = get_vmdapp();
  app->scene_scale_by(factor);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *get_center(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:trans.get_center", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  PyObject *newlist = PyList_New(3);
  for (int i=0; i<3; i++) 
    PyList_SET_ITEM(newlist, i, PyFloat_FromDouble(mol->centt[i]));

  return newlist;
}

// get_scale(molid)
static PyObject *get_scale(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:trans.get_scale", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  return PyFloat_FromDouble(mol->scale);
}

// get_trans(molid)
static PyObject *get_trans(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:trans.get_trans", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  PyObject *newlist = PyList_New(3);
  for (int i=0; i<3; i++) 
    PyList_SET_ITEM(newlist, i, PyFloat_FromDouble(mol->globt[i]));

  return newlist;
}

//get_rotation(molid)
static PyObject *get_rotation(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:trans.get_rotation", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  PyObject *mat = PyList_New(16);
  Matrix4 &rotm = mol->rotm;
  for (int i=0; i<16; i++) {
    PyList_SET_ITEM(mat, i, PyFloat_FromDouble(rotm.mat[i]));
  }
  return mat;
}

// set_center(molid)
static PyObject *set_center(PyObject *self, PyObject *args) {
  int i, molid;
  float c[3];
  PyObject *pylist;
  if (!PyArg_ParseTuple(args, (char *)"iO!:trans.set_center", &molid, &PyList_Type, &pylist))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (PyList_GET_SIZE(pylist) != 3) {
    PyErr_SetString(PyExc_ValueError, (char *)"List must be of length 3");
    return NULL;
  }
  
  for (i=0; i<3; i++) {
    c[i] = PyFloat_AsDouble(PyList_GET_ITEM(pylist, i));
    if (PyErr_Occurred())
      return NULL;
  }
  mol->set_cent_trans(c[0], c[1], c[2]);

  Py_INCREF(Py_None);
  return Py_None;
}

// set_scale(molid)
static PyObject *set_scale(PyObject *self, PyObject *args) {
  int molid;
  float scale;
  if (!PyArg_ParseTuple(args, (char *)"if:trans.set_scale", &molid, &scale)) 
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  mol->set_scale(scale);

  Py_INCREF(Py_None);
  return Py_None;
}
  
// set_trans(molid)
static PyObject *set_trans(PyObject *self, PyObject *args) {
  int i, molid;
  float c[3];
  PyObject *pylist;
  if (!PyArg_ParseTuple(args, (char *)"iO!:trans.set_trans", &molid, &PyList_Type, &pylist))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (PyList_GET_SIZE(pylist) != 3) {
    PyErr_SetString(PyExc_ValueError, (char *)"List must be of length 3");
    return NULL;
  }
  
  for (i=0; i<3; i++) {
    c[i] = PyFloat_AsDouble(PyList_GET_ITEM(pylist, i));
    if (PyErr_Occurred())
      return NULL;
  }
  mol->set_glob_trans(c[0], c[1], c[2]); 

  Py_INCREF(Py_None);
  return Py_None;
}

// set_rotation(molid)
static PyObject *set_rotation(PyObject *self, PyObject *args) {
  int i, molid;
  float c[16];
  PyObject *pylist;
  if (!PyArg_ParseTuple(args, (char *)"iO!:trans.set_rotation", &molid, &PyList_Type, &pylist))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (PyList_GET_SIZE(pylist) != 16) {
    PyErr_SetString(PyExc_ValueError, (char *)"List must be of length 16");
    return NULL;
  }
  
  for (i=0; i<16; i++) {
    c[i] = PyFloat_AsDouble(PyList_GET_ITEM(pylist, i));
    if (PyErr_Occurred())
      return NULL;
  }
  mol->set_rot(c);

  Py_INCREF(Py_None);
  return Py_None;
}

// resetview(molid)
// centers the view around the given molecule
// Actually, this sets the top molecule, then does a resetview, then restores
// the previous top molecule.
// If the molid is invalid, raise a ValueError
static PyObject *resetview(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:trans.resetview", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  MoleculeList *mlist = app->moleculeList;
  Molecule *mol = mlist->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  Molecule *topmol = mlist->top();
  mlist->make_top(mol); 
  app->scene_resetview();
  mlist->make_top(topmol);

  Py_INCREF(Py_None);
  return Py_None;
}

// is_fixed(molid)
static PyObject *is_fixed(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:trans.is_fixed", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  return PyInt_FromLong(mol->fixed()); 
}

// fix(molid, bool)
static PyObject *fix(PyObject *self, PyObject *args) {
  int molid;
  PyObject *boolobj;
  if (!PyArg_ParseTuple(args, (char *)"iO:trans.fix", &molid, &boolobj))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  app->molecule_fix(molid, PyObject_IsTrue(boolobj));

  Py_INCREF(Py_None);
  return Py_None;
}

// is_shown(molid)
static PyObject *is_shown(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:trans.is_shown", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  return PyInt_FromLong(mol->displayed()); 
}

// show(molid, bool) 
static PyObject *show(PyObject *self, PyObject *args) {
  int molid;
  PyObject *boolobj;
  if (!PyArg_ParseTuple(args, (char *)"iO:trans.show", &molid, &boolobj))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  app->molecule_display(molid, PyObject_IsTrue(boolobj));

  Py_INCREF(Py_None);  
  return Py_None;
}

static PyMethodDef TransMethods[] = {
  {(char *)"rotate", (vmdPyMethod)rotate, METH_VARARGS },
  {(char *)"translate", (vmdPyMethod)translate, METH_VARARGS},
  {(char *)"scale", (vmdPyMethod)scale, METH_VARARGS},
  {(char *)"resetview", (vmdPyMethod)resetview, METH_VARARGS},
  {(char *)"get_center", (vmdPyMethod)get_center, METH_VARARGS},
  {(char *)"get_scale", (vmdPyMethod)get_scale, METH_VARARGS},
  {(char *)"get_trans", (vmdPyMethod)get_trans, METH_VARARGS},
  {(char *)"get_rotation", (vmdPyMethod)get_rotation, METH_VARARGS},
  {(char *)"set_center", (vmdPyMethod)set_center, METH_VARARGS},
  {(char *)"set_scale", (vmdPyMethod)set_scale, METH_VARARGS},
  {(char *)"set_trans", (vmdPyMethod)set_trans, METH_VARARGS},
  {(char *)"set_rotation", (vmdPyMethod)set_rotation, METH_VARARGS},
  {(char *)"is_fixed", (vmdPyMethod)is_fixed, METH_VARARGS},
  {(char *)"fix", (vmdPyMethod)fix, METH_VARARGS},
  {(char *)"is_shown", (vmdPyMethod)is_shown, METH_VARARGS},
  {(char *)"show", (vmdPyMethod)show, METH_VARARGS},
  
  {NULL, NULL}
};



void inittrans() {
  (void) Py_InitModule((char *)"trans", TransMethods);
}


 
