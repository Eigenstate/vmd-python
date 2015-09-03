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
 *      $RCSfile: py_numeric.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.20 $       $Date: 2011/03/05 04:57:37 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface to the Python Numeric module.
 ***************************************************************************/

#include "py_commands.h"
#include "numpy/ndarrayobject.h"

#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"

// make python happier
extern "C" {

/* 
 * Note: changes to the timestep method now return an N x 3 array
 * rather than flat 3N arrays.  This may break older plugins like IED
 * if they are not kept up-to-date with VMD.
 */ 
static char timestep_doc[] = 
  "timestep(molid = -1, frame = -1) -> NumPy (N x 3) float32 array\n"
  "Returns zero-copy reference to atom coordinates.  molid defaults to\n"
  "top molecule; frame defaults to current frame.  Array has shape\n"
  "N x 3 where N is the number of atoms in the molecule.";
static PyObject *timestep(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid = -1;  // default : top
  int frame = -1;  // default : current frame 
  static char *kwlist[] = { (char *)"molid", (char *)"frame", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, (char *)"|ii", kwlist,
        &molid, &frame))
    return NULL;
  Timestep *ts = parse_timestep(get_vmdapp(), molid, frame);
  if (!ts) return NULL;
  npy_intp dims[2] = { ts->num, 3 };
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNewFromData(
      2, dims, NPY_FLOAT, (char *)ts->pos);
  return PyArray_Return(result);
}

static char velocities_doc[] = 
  "velocities(molid = -1, frame = -1) -> NumPy (N x 3) float32 array\n"
  "Returns zero-copy reference to atom velocities.  molid defaults to\n"
  "top molecule; frame defaults to current frame.  Array has shape\n"
  "N x 3 where N is the number of atoms in the molecule.  If timestep\n"
  "holds no velocities, None is returned.";
static PyObject *velocities(PyObject *self, PyObject *args, PyObject *kwds) {
  int molid = -1;  // default : top
  int frame = -1;  // default : current frame 
  static char *kwlist[] = { (char *)"molid", (char *)"frame", NULL };
  if (!PyArg_ParseTupleAndKeywords(args, kwds, (char *)"|ii", kwlist,
        &molid, &frame))
    return NULL;
  Timestep *ts = parse_timestep(get_vmdapp(), molid, frame);
  if (!ts) return NULL;
  if (!ts->vel) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  npy_intp dims[2] = { ts->num, 3 };
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNewFromData(
      2, dims, NPY_FLOAT, (char *)ts->vel);
  return PyArray_Return(result);
}
 
// Return an array of ints representing flags for on/off atoms in an atom
// selection.
// atomselect(molid, frame, selection) -> array
static PyObject *atomselect(PyObject *self, PyObject *args) {
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
  npy_intp dims[1]; dims[0] = mol->nAtoms;
  PyArrayObject *result = (PyArrayObject *)PyArray_SimpleNew(
      1, dims, NPY_INT);
  memcpy(result->data, atomSel->on, mol->nAtoms*sizeof(int));
  delete atomSel;
  return PyArray_Return(result);
}

// end of extern "C" block
}

static PyMethodDef Methods[] = {
  {(char *)"timestep", (PyCFunction)timestep, METH_VARARGS | METH_KEYWORDS, timestep_doc},
  {(char *)"positions", (PyCFunction)timestep, METH_VARARGS | METH_KEYWORDS, timestep_doc},
  {(char *)"velocities", (PyCFunction)velocities, METH_VARARGS | METH_KEYWORDS, velocities_doc},
  {(char *)"atomselect", atomselect, METH_VARARGS, (char *)"Create atom selection flags"},
  {NULL, NULL}
};

void initvmdnumpy() {
  if (_import_array() < 0) {
    PyErr_SetString(PyExc_ValueError, "vmdnumpy module not available.");
    return;
  }
  (void)Py_InitModule((char *)"vmdnumpy", Methods);
}

