#include "py_commands.h"
#include "VMDApp.h"
#include "Measure.h"

static const char bond_doc[] =
"Measures the distance between the atom1 and atom2 over the trajectory, either\n"
"at the given frame or over frames from first to last. Distances between atoms\n"
"in different molecules can be measured by giving specific molids\n\n"
"Args:\n"
"    atom1 (int): Index of first atom\n"
"    atom2 (int): Index of second atom\n"
"    molid (int): Molecule ID of first atom, defaults to top molecule\n"
"    molid2 (int): Molecule ID of second atom, if different. (optional)\n"
"    frame (int): Measure distance in this single frame. Defaults to current\n"
"    first (int): For measuring multiple frames, first frame to measure\n"
"    last (int): For measuring multiple, last frame to measure\n"
"Returns:\n"
"    (list of float): Distance between atoms for given frame(s)";
static PyObject *measure_bond(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"atom1", "atom2", "molid", "molid2", "frame", "first",
                          "last", NULL};
  ResizeArray<float> gValues(1024);
  int first=-1, last=-1, frame=-1;
  PyObject *returnlist = NULL;
  int atmid[2], ret_val, i;
  int molid[2] = {-1, -1};
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|iiiii:measure.bond",
                                   (char**) kwlist, &atmid[0], &atmid[1],
                                   &molid[0], &molid[1], &frame, &first, &last))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // If molid(s) unset, use top molecule
  if (molid[0] == -1)
    molid[0] = app->molecule_top();

  if (molid[1] == -1)
    molid[1] = molid[0];

  // Need either (first, last) or a single frame.
  if (frame != -1 && (first != -1 || last != -1)) {
    PyErr_Warn(PyExc_SyntaxWarning, "frame as well as first or last were "
                                    "specified.\nReturning value for just the "
                                    "frame");
    first = -1;
    last = -1;
  }

  //Calculate the bond length
  ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, frame,
                         first, last, molid[0], MEASURE_BOND);
  if (ret_val < 0) {
    PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
    return NULL;
  }

  // Build the list containing all measured distances
  if (!(returnlist = PyList_New(gValues.num())))
    goto failure;

  for (i = 0; i < gValues.num(); i++) {
    PyList_SET_ITEM(returnlist, i, PyFloat_FromDouble(gValues[i]));

    if (PyErr_Occurred())
      goto failure;
  }

  return returnlist;

failure:
  PyErr_SetString(PyExc_ValueError, "Couldn't build list of distances");
  Py_XDECREF(returnlist);
  return NULL;
}

static const char angle_doc[] =
"Measure angles between given atoms over a trajectory. Can specify multiple\n"
"molids to measure angles between atoms in different molecules.\n\n"
"Args:\n"
"    atom1 (int): Index of first atom\n"
"    atom2 (int): Index of second atom\n"
"    atom3 (int): Index of third atom\n"
"    molid (int): Molecule ID of first atom. Defaults to top molecule\n"
"    molid2 (int): Molecule ID of second atom, if different from first molid\n"
"    molid3 (int): Molecule ID of third atom, if different from first molid\n"
"    frame (int): Single frame to measure angle in. Defaults to current frame\n"
"    first (int): First frame in range to measure in, for multiple frames\n"
"    last (int): Last frame in range to measure in, for multiple frames\n"
"Returns:\n"
"    (list of float): Angle as measured in frame(s)";
static PyObject *measure_angle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"atom1", "atom2", "atom3", "molid", "molid2",
                          "molid3", "frame", "first", "last", NULL};
  ResizeArray<float> gValues(1024);
  int first=-1, last=-1, frame=-1;
  int molid[3] = {-1, -1, -1};
  PyObject *returnlist;
  int atmid[3];
  VMDApp *app;
  int ret_val;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii|iiiiii:measure.angle",
                                   (char**) kwlist, &atmid[0], &atmid[1],
                                   &atmid[2], &molid[0], &molid[1], &molid[2],
                                   &frame, &first, &last))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Default to top molecule
  if (molid[0] == -1)
    molid[0] = app->molecule_top();

  // Other two molids default to the first molid
  if (molid[1] == -1)
    molid[1] = molid[0];

  if (molid[2] == -1)
    molid[2] = molid[0];

  // Warn if passing a single frame and a range
  if (frame != -1 && (first != -1 || last != -1)) {
    PyErr_Warn(PyExc_SyntaxWarning, "frame as well as first or last were "
                                    "specified.\nReturning value for just the "
                                    "frame");
    first = -1;
    last = -1;
  }
    //Calculate the angle
  ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, frame,
                         first, last, molid[0], MEASURE_ANGLE);
  if (ret_val<0) {
    PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
    return NULL;
  }

  //Build the python list.
  returnlist = PyList_New(gValues.num());
  for (int i = 0; i < gValues.num(); i++)
    PyList_SetItem(returnlist, i, Py_BuildValue("f", gValues[i]));

  // Remove reference to list if something went wrong
  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, "Couldn't build list of angles");
    Py_DECREF(returnlist);
    return NULL;
  }

  return returnlist;
}

static const char dihed_doc[] =
"Measures the dihedral angle between atoms over the trajectory. Can specify\n"
"multiple molecule IDs to measure over different loaded molecules.\n\n"
"Args:\n"
"    atom1 (int): First atom in dihedral\n"
"    atom2 (int): Second atom in dihedral\n"
"    atom3 (int): Third atom in dihedral\n"
"    atom4 (int): Fourth atom in dihedral\n"
"    molid (int): Molecule ID to measure in. Defaults to top molecule\n"
"    molid2 (int): Molecule ID for second atom, if different from molid\n"
"    molid3 (int): Molecule ID for third atom, if different from molid\n"
"    molid4 (int): Molecule ID for fourth atom, if different from molid\n"
"    frame (int): Single frame to measure angle in. Defaults to current frame\n"
"    first (int): First frame in range to measure in, for multiple frames\n"
"    last (int): Last frame in range to measure in, for multiple frames\n"
"Returns:\n"
"    (list of float): Dihedral as measured in frame(s)";
static PyObject* measure_dihed(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"atom1", "atom2", "atom3", "atom4", "molid", "molid2",
                          "molid3", "molid4", "frame",  "first",  "last", NULL};
  ResizeArray<float> gValues(1024);
  int first=-1, last=-1, frame=-1;
  int molid[4] = {-1, -1, -1, -1};
  PyObject *returnlist;
  int atmid[4];
  int ret_val;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiii|iiiiiii:measure.dihedral",
                                   (char**) kwlist, &atmid[0], &atmid[1],
                                   &atmid[2], &atmid[3], &molid[0], &molid[1],
                                   &molid[2], &molid[3], &frame, &first, &last))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Molid defaults to top molecule
  if (molid[0] == -1)
    molid[0] = app->molecule_top();

  // Other molecule IDs default to molid
  if (molid[1] == -1)
    molid[1] = molid[0];

  if (molid[2] == -1)
    molid[2] = molid[0];

  if (molid[3] == -1)
    molid[3] = molid[0];

  // Warn if both frame and range of frames are given
  if (frame != -1 && (first != -1 || last != -1)) {
    PyErr_Warn(PyExc_SyntaxWarning, "frame as well as first or last were "
                                    "specified.\nReturning value for just the "
                                    "frame");
    first = -1;
    last = -1;
  }

  //Calculate the dihedral angle
  ret_val = measure_geom(app->moleculeList, molid, atmid, &gValues, frame,
                         first, last, molid[0], MEASURE_DIHED);

  if (ret_val<0) {
    PyErr_SetString(PyExc_RuntimeError, measure_error(ret_val));
    return NULL;
  }

  //Build the python list.
  returnlist = PyList_New(gValues.num());
  for (int i = 0; i < gValues.num(); i++)
    PyList_SetItem(returnlist, i, Py_BuildValue("f", gValues[i]));

  // Remove reference to list if something went wrong
  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, "Couldn't build list of dihedrals");
    Py_DECREF(returnlist);
    return NULL;
  }

  return returnlist;
}

static PyMethodDef methods[] = {
  {"bond", (PyCFunction)measure_bond, METH_VARARGS | METH_KEYWORDS, bond_doc},
  {"angle", (PyCFunction)measure_angle, METH_VARARGS | METH_KEYWORDS, angle_doc},
  {"dihedral", (PyCFunction)measure_dihed, METH_VARARGS | METH_KEYWORDS, dihed_doc},
  {NULL, NULL}
};

static const char measure_moddoc[] =
"Methods to measure bonds, angles, or dihedrals in loaded molecules";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef measuredef = {
  PyModuleDef_HEAD_INIT,
  "measure",
  measure_moddoc,
  -1,
  methods,
};
#endif

PyObject* initmeasure() {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&measuredef);
#else
  PyObject *m = Py_InitModule3("measure", methods, measure_moddoc);
#endif
  return m;
}
