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
 *      $RCSfile: py_trans.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2019/01/17 21:21:03 $
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

static const char rotate_doc[] =
"Rotate the scene about an axis\n\n"
"Args:\n"
"    axis (str): Axis to rotate around, either 'x', 'y', or 'z'\n"
"    angle (float): Angle to rotate by";
static PyObject *py_rotate(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"axis", "angle", NULL};
  VMDApp *app;
  float angle;
  char *axis;
  int len;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s#f:trans.rotate_scene",
                                   (char**) kwlist, &axis, &len, &angle))
    return NULL;

  if (len != 1 || (axis[0] != 'x' && axis[0] != 'y' && axis[0] != 'z')) {
    PyErr_Format(PyExc_ValueError, "axis must be 'x', 'y', or 'z', had '%s'",
                 axis);
    return NULL;
  }

  if (!(app = get_vmdapp()))
    return NULL;

  app->scene_rotate_by(angle, axis[0]);

  Py_INCREF(Py_None); return Py_None;
}

static const char translate_doc[] =
"Translate the scene by a vector\n\n"
"Args:\n"
"    x (float): Amount to move in X direction\n"
"    y (float): Amount to move in Y direction\n"
"    z (float): Amount to move in Z direction";
static PyObject *py_translate(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"x", "y", "z", NULL};
  float x = 0.f, y = 0.f, z = 0.f;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fff:trans.translate_scene",
                                   (char**) kwlist, &x, &y, &z))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  app->scene_translate_by(x,y,z);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char scale_doc[] =
"Set the scale or zoom level for the scene\n\n"
"Args:\n"
"    scale (float): Scale value";
static PyObject *py_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"scale", NULL};
  VMDApp *app;
  float factor;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "f:trans.scale",
                                   (char**) kwlist, &factor))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  app->scene_scale_by(factor);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char center_doc[] =
"Get the coordinates of the displayed center of a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (3-tuple of float) (x,y,z) coordinates of molecule center";
static PyObject *py_get_center(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *newlist = NULL;
  Molecule *mol;
  int i, molid;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:trans.get_center",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  if (!(newlist = PyTuple_New(3)))
    goto failure;

  for (i = 0; i < 3; i++) {
    PyTuple_SET_ITEM(newlist, i, PyFloat_FromDouble(mol->centt[i]));
    if (PyErr_Occurred())
      goto failure;
  }

  return newlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem getting molecule center");
  Py_XDECREF(newlist);
  return NULL;
}

static const char mol_scale_doc[] =
"Get the scaling factor for a molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (float) Scaling factor for molecule";
static PyObject *py_get_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:trans.get_scale",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  return PyFloat_FromDouble(mol->scale);
}

static const char mol_trans_doc[] =
"Get the translation for a molecule relative to scene center\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (3-tuple of float) (x, y, z) translation applied to molecule";
static PyObject *py_get_trans(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *newlist = NULL;
  Molecule *mol;
  int i, molid;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:trans.get_translation",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  if (!(newlist = PyTuple_New(3)))
    goto failure;

  for (i = 0; i < 3; i++) {
    PyTuple_SET_ITEM(newlist, i, PyFloat_FromDouble(mol->globt[i]));
    if (PyErr_Occurred())
      goto failure;
  }

  return newlist;

failure:
  PyErr_Format(PyExc_RuntimeError, "Problem getting molecule '%d' translation",
               molid);
  Py_XDECREF(newlist);
  return NULL;
}

static const char mol_rot_doc[] =
"Gets the rotation of a given molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (16-tuple of float) Rotation matrix for molecule";
static PyObject *py_get_rotation(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *mat = NULL;
  Molecule *mol;
  int i, molid;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:trans.get_rotation",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  if (!(mat = PyTuple_New(16)))
    goto failure;

  for (i = 0; i < 16; i++) {
    PyTuple_SET_ITEM(mat, i, PyFloat_FromDouble(mol->rotm.mat[i]));
    if (PyErr_Occurred())
      goto failure;
  }
  return mat;

failure:
  PyErr_Format(PyExc_RuntimeError, "Problem getting molecule '%d' rotation",
               molid);
  Py_XDECREF(mat);
  return NULL;
}

static const char mcenter_doc[] =
"Set the center of an individual molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to set center in\n"
"    center (list of 3 floats): (x, y, z) coordinates of new center";
static PyObject *py_set_center(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "center", NULL};
  float x, y, z;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i(fff):trans.set_center",
                                   (char**) kwlist, &molid, &x, &y, &z))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  mol->set_cent_trans(x, y, z);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char mscale_doc[] =
"Set the scale of an individual molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to set scale of\n"
"    scale (float): New scale value for molecule";
static PyObject *py_set_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "scale", NULL};
  Molecule *mol;
  VMDApp *app;
  float scale;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "if:trans.set_scale",
                                   (char**) kwlist, &molid, &scale))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }
  mol->set_scale(scale);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char mtrans_doc[] =
"Set the translation of an individual molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to set translation of\n"
"    trans (list of 3 floats): New (x,y,z) translation for molecule";
static PyObject *py_set_trans(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "trans", NULL};
  float x, y, z;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i(fff):trans.set_translation",
                                   (char**) kwlist, &molid, &x, &y, &z))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  mol->set_glob_trans(x, y, z);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char mrot_doc[] =
"Set the rotation of an individual molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to set\n"
"    matrix (list or tuple of 16 floats): Rotation matrix to apply";
static PyObject *py_set_rotation(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "matrix", NULL};
  PyObject *pyseq = NULL;
  PyObject *pylist;
  Molecule *mol;
  int i, molid;
  VMDApp *app;
  float c[16];

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO:trans.set_rotation",
                                   (char**) kwlist, &molid, &pylist))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  // PySequence_Fast does type checking for us and will set an exception
  if (!(pyseq = PySequence_Fast(pylist, "matrix must be a list or tuple")))
    return NULL;

  if (PySequence_Fast_GET_SIZE(pyseq) != 16) {
    PyErr_SetString(PyExc_ValueError, "matrix must have length 16");
    goto failure;
  }

  for (i = 0; i < 16; i++) {
    // Check elements are float first as if they're bools they can get
    // implicitly converted to 1.0
    PyObject *num = PySequence_Fast_GET_ITEM(pyseq, i);
    if (!PyFloat_Check(num)) {
      PyErr_SetString(PyExc_TypeError, "matrix must contain floats");
      goto failure;
    }

    c[i] = PyFloat_AsDouble(num);

    if (PyErr_Occurred())
      goto failure;
  }
  mol->set_rot(c);

  Py_DECREF(pyseq);
  Py_INCREF(Py_None);
  return Py_None;

failure:
  Py_XDECREF(pyseq);
  return NULL;
}

static const char reset_doc[] =
"Centers the view around a given molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to center";
static PyObject *py_resetview(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  Molecule *mol, *topmol;
  MoleculeList *mlist;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:trans.resetview",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  mlist = app->moleculeList;
  if (!(mol = mlist->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  topmol = mlist->top();
  mlist->make_top(mol);
  app->scene_resetview();
  mlist->make_top(topmol);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char fixed_doc[] =
"Query if a molecule is fixed. Fixed molecules do not move when the scene\n"
"or view is changed.\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (bool) True if molecule is fixed";
static PyObject *py_is_fixed(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *result;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:trans.is_fixed",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  result = mol->fixed() ? Py_True : Py_False;
  Py_INCREF(result);
  return result;
}

static const char setfix_doc[] =
"Set the fixed status of a molecule. Fixed molecules do not move when the\n"
"scene or view is changed.\n\n"
"Args:\n"
"    molid (int): Molecule ID to set fixed status for\n"
"    fixed (bool): If molecule should be fixed";
static PyObject *py_fix(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "fixed", NULL};
  int molid, fixed;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&:trans.fix",
                                  (char**) kwlist,  &molid, convert_bool,
                                  &fixed))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }
  app->molecule_fix(molid, fixed);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char show_doc[] =
"Query if a molecule is shown\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (bool) True if molecule is shown";
static PyObject *py_is_shown(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  PyObject *result;
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:trans.is_shown",
                                   (char**) kwlist,  &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  result = mol->displayed() ? Py_True : Py_False;
  Py_INCREF(result);
  return result;
}

static const char setshow_doc[] =
"Set if a molecule is shown\n\n"
"Args:\n"
"    molid (int): Molecule ID to show or hide\n"
"    shown (bool): True if molecule should be shown";
static PyObject *py_show(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "shown", NULL};
  int molid, shown;
  Molecule *mol;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&:trans.show",
                                   (char**) kwlist, &molid, convert_bool,
                                   &shown))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_Format(PyExc_ValueError, "Invalid molecule id '%d'", molid);
    return NULL;
  }

  app->molecule_display(molid, shown);

  Py_INCREF(Py_None);
  return Py_None;
}

// Deprecated methods with old names are retained for backwards compatibility
static PyObject *py_drotate(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyErr_Warn(PyExc_DeprecationWarning, "the 'rotate' method has been "
    "renamed to 'rotate_scene'");
  return py_rotate(self, args, kwargs);
}

static PyObject *py_dtranslate(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyErr_Warn(PyExc_DeprecationWarning, "the 'translate' method has been "
    "renamed to 'translate_scene'");
  return py_translate(self, args, kwargs);
}

static PyObject *py_dscale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyErr_Warn(PyExc_DeprecationWarning, "the 'scale' method has been renamed "
          "to 'scale_scene'");
  return py_scale(self, args, kwargs);
}

static PyObject *py_dstrans(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyErr_Warn(PyExc_DeprecationWarning, "the 'set_trans' method has been "
             "renamed to 'set_translation'");
  return py_set_trans(self, args, kwargs);
}

static PyObject *py_dgtrans(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyErr_Warn(PyExc_DeprecationWarning, "the 'get_trans' method has been "
             "renamed to 'get_translation'");
  return py_get_trans(self, args, kwargs);
}

static PyMethodDef methods[] = {
  {"rotate_scene", (PyCFunction)py_rotate, METH_VARARGS | METH_KEYWORDS, rotate_doc},
  {"translate_scene", (PyCFunction)py_translate, METH_VARARGS | METH_KEYWORDS, translate_doc},
  {"scale_scene", (PyCFunction)py_scale, METH_VARARGS | METH_KEYWORDS, scale_doc},
  {"resetview", (PyCFunction)py_resetview, METH_VARARGS | METH_KEYWORDS, reset_doc},
  {"get_center", (PyCFunction)py_get_center, METH_VARARGS | METH_KEYWORDS, center_doc},
  {"get_scale", (PyCFunction)py_get_scale, METH_VARARGS | METH_KEYWORDS, mol_scale_doc},
  {"get_translation", (PyCFunction)py_get_trans, METH_VARARGS | METH_KEYWORDS, mol_trans_doc},
  {"get_rotation", (PyCFunction)py_get_rotation, METH_VARARGS | METH_KEYWORDS, mol_rot_doc},
  {"set_center", (PyCFunction)py_set_center, METH_VARARGS | METH_KEYWORDS, mcenter_doc},
  {"set_scale", (PyCFunction)py_set_scale, METH_VARARGS | METH_KEYWORDS, mscale_doc},
  {"set_translation", (PyCFunction)py_set_trans, METH_VARARGS | METH_KEYWORDS, mtrans_doc},
  {"set_rotation", (PyCFunction)py_set_rotation, METH_VARARGS | METH_KEYWORDS, mrot_doc},
  {"is_fixed", (PyCFunction)py_is_fixed, METH_VARARGS | METH_KEYWORDS, fixed_doc},
  {"fix", (PyCFunction)py_fix, METH_VARARGS | METH_KEYWORDS, setfix_doc},
  {"is_shown", (PyCFunction)py_is_shown, METH_VARARGS | METH_KEYWORDS, show_doc},
  {"show", (PyCFunction)py_show, METH_VARARGS | METH_KEYWORDS, setshow_doc},

  // The following methods are deprecated as they have been renamed.
  // They're still here for backwards compatiblity, but emit DeprecationWarning
  {"rotate", (PyCFunction)py_drotate, METH_VARARGS | METH_KEYWORDS, rotate_doc},
  {"translate", (PyCFunction)py_dtranslate, METH_VARARGS | METH_KEYWORDS, translate_doc},
  {"scale", (PyCFunction)py_dscale, METH_VARARGS | METH_KEYWORDS, scale_doc},
  {"set_trans", (PyCFunction)py_dstrans, METH_VARARGS | METH_KEYWORDS, mtrans_doc},
  {"get_trans", (PyCFunction)py_dgtrans, METH_VARARGS | METH_KEYWORDS, mol_trans_doc},

  {NULL, NULL}
};

static const char trans_moddoc[] =
"Methods for manipulating the transformations applied to a molecule in the "
"render window, including its position, rotation, center, and scale.";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef transdef = {
  PyModuleDef_HEAD_INIT,
  "trans",
  trans_moddoc,
  -1,
  methods,
};
#endif

PyObject* inittrans(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&transdef);
#else
  PyObject *m = Py_InitModule3("trans", methods, trans_moddoc);
#endif
  return m;
}



