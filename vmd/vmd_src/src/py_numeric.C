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
 *      $RCSfile: py_numeric.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface to the Python Numeric module.
 ***************************************************************************/

#include "py_commands.h"
#ifdef VMDNUMPY

// Promise we're not using any deprecated stuff to squash a compiler warning
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"

/*
 * Note: changes to the timestep method now return an N x 3 array
 * rather than flat 3N arrays.  This may break older plugins like IED
 * if they are not kept up-to-date with VMD.
 */
static const char timestep_doc[] =
"Return a zero-copy reference to atomic coordinates\n\n"
"Args:\n"
"    molid (int): Molecule ID. Defaults to -1 (top molecule)\n"
"    frame (int): Frame to select. Defaults to -1 (current frame)\n"
"Returns:\n"
"    (N x 3 float32 ndarray): Reference to coordinates, where N is the\n"
"        number of atoms in the molecule";
static PyObject *timestep(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "frame", NULL};
  int molid = -1, frame = -1;
  PyArrayObject *result;
  npy_intp dims[2] = {0, 3};
  Timestep *ts;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii:vmdnumpy.timestep",
                                   (char**) kwlist, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();

  if (!valid_molid(molid, app))
    return NULL;

  ts = parse_timestep(app, molid, frame);
  if (!ts) return NULL;

  dims[0] = ts->num;
  result = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT,
                                                      (char *)ts->pos);
  return PyArray_Return(result);
}

static const char velocities_doc[] =
"Return a zero-copy reference to atomic velocites, or None if no velocity\n"
"information is present\n\n"
"Args:\n"
"    molid (int): Molecule ID. Defaults to -1 (top molecule)\n"
"    frame (int): Frame to select. Defaults to -1 (current frame)\n"
"Returns:\n"
"    (N x 3 float32 ndarray): Reference to velocities, where N is the\n"
"        number of atoms in the molecule";
static PyObject *velocities(PyObject *self, PyObject *args,
                            PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "frame", NULL};
  int molid = -1, frame = -1;
  npy_intp dims[2] = {0, 3};
  PyArrayObject *result;
  Timestep *ts;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii:vmdnumpy.velocities",
                                   (char**) kwlist, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();
  if (!valid_molid(molid, app))
    return NULL;

  ts = parse_timestep(app, molid, frame);
  if (!ts)
    return NULL;

  // If no velocities, return None
  if (!ts->vel) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  dims[0] = ts->num;
  result = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT,
                                                      (char *)ts->vel);
  return PyArray_Return(result);
}

static const char atomselect_doc[] =
"Return an array of ints representing flags for on/off atoms in an atom\n"
"selection\n\n"
"Args:\n"
"    selection (str): Atom selection string\n"
"    molid (int): Molecule ID. Defaults to -1 (top molecule)\n"
"    frame (int): Frame to select. Defaults to -1 (current frame)\n"
"Returns:\n"
"    (N, int ndarray): Flags for atoms, where N is the number of atoms\n"
"        in the molecule. The value for an atom will be 1 if it is in the\n"
"        selection, 0 otherwise";
static PyObject *atomselect(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"selection", "molid", "frame", NULL};
  int molid = -1, frame = -1;
  PyArrayObject *result;
  npy_intp dims[1] = {0};
  DrawMolecule *mol;
  AtomSel *atomSel;
  char *sel = 0;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ii:vmdnumpy.atomselect",
                                   (char**) kwlist, &sel, &molid, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (molid == -1)
    molid = app->molecule_top();
  if (!valid_molid(molid, app))
    return NULL;

  // Create a new atom selection
  mol = app->moleculeList->mol_from_id(molid);
  atomSel = new AtomSel(app->atomSelParser, mol->id());
  atomSel->which_frame = frame;

  if (atomSel->change(sel, mol) == AtomSel::NO_PARSE) {
    PyErr_Format(PyExc_ValueError, "cannot parse atom selection text '%s'",
                 sel);
    delete atomSel;
    return NULL;
  }

  dims[0] = mol->nAtoms;

  // As the atom selection object is about to be deleted, we need to copy
  // the atom masks to a new numpy array so they remain valid.
  result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT);
  memcpy(PyArray_DATA(result), atomSel->on, dims[0]*sizeof(NPY_INT));
  delete atomSel;

  return PyArray_Return(result);
}

static PyMethodDef methods[] = {
  {"timestep", (PyCFunction)timestep, METH_VARARGS | METH_KEYWORDS, timestep_doc},
  {"positions", (PyCFunction)timestep, METH_VARARGS | METH_KEYWORDS, timestep_doc},
  {"velocities", (PyCFunction)velocities, METH_VARARGS | METH_KEYWORDS, velocities_doc},
  {"atomselect", (PyCFunction)atomselect, METH_VARARGS | METH_KEYWORDS, atomselect_doc},
  {NULL, NULL}
};

static const char vnumpy_moddoc[] =
"Methods for interacting with atomic positions or velocities as numpy arrays. "
"This can offer significant performance gains over using atomsel types";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef vmdnumpydef = {
  PyModuleDef_HEAD_INIT,
  "vmdnumpy",
  vnumpy_moddoc,
  -1,
  methods,
};
#endif

PyObject* initvmdnumpy() {
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&vmdnumpydef);
#else
  PyObject *module = Py_InitModule3("vmdnumpy", methods, vnumpy_moddoc);
#endif
  _import_array(); // Don't use import_array macro as it expands to return void
  if (PyErr_Occurred()) return NULL;
  return module;
}

#endif // -DVMDNUMPY
