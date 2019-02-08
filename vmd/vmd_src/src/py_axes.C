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
 *      $RCSfile: py_axes.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.13 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python Axes control interface.
 ***************************************************************************/

#include "py_commands.h"
#include "CommandQueue.h"
#include "VMDApp.h"
#include "Axes.h"

// get_location()
static char getloc_doc[] =
"Get current axes location.\n\n"
"Returns:\n"
"    (str) Axes location, one of [OFF, ORIGIN, LOWERLEFT,\n"
"        LOWERRIGHT, UPPERLEFT, UPPERRIGHT]";
static PyObject *get_location(PyObject *self, PyObject *args) {
  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;

  return as_pystring(app->axes->loc_description(app->axes->location()));
}

// set_location(locale)
static char setloc_doc[] =
"Set current axes location.\n\n"
"Args:\n"
"    location (str): New axes location, in [OFF, ORIGIN, LOWERLEFT,\n"
"        LOWERRIGHT, UPPERLEFT, UPPERRIGHT]";
static PyObject *set_location(PyObject *self, PyObject *args, PyObject *kwargs) {

  const char *kwnames[] = {"location", NULL};
  char *location;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:axes.set_location",
                                   (char**) kwnames, &location))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->axes_set_location(location)) {
    PyErr_Format(PyExc_ValueError, "Invalid axes location '%s'", location);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"get_location", (PyCFunction) get_location, METH_NOARGS, getloc_doc},
  {"set_location", (PyCFunction) set_location, METH_VARARGS | METH_KEYWORDS, setloc_doc},
  {NULL, NULL}
};

static const char axes_moddoc[] =
"Methods to get and set the axes location";

#if PY_MAJOR_VERSION >= 3
struct PyModuleDef axesdef = {
  PyModuleDef_HEAD_INIT,
  "axes",
  axes_moddoc,
  -1,
  methods,
};
#endif

PyObject* initaxes(void) {

  PyObject *m;
  VMDApp *app;
#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&axesdef);
#else
  m = Py_InitModule3("axes", methods, axes_moddoc);
#endif

  if (!(app = get_vmdapp()))
    return NULL;

  // Add location string constants to system
  PyModule_AddStringConstant(m, "OFF", app->axes->loc_description(0));
  PyModule_AddStringConstant(m, "ORIGIN", app->axes->loc_description(1));
  PyModule_AddStringConstant(m, "LOWERLEFT", app->axes->loc_description(2));
  PyModule_AddStringConstant(m, "LOWERRIGHT", app->axes->loc_description(3));
  PyModule_AddStringConstant(m, "UPPERLEFT", app->axes->loc_description(4));
  PyModule_AddStringConstant(m, "UPPERRIGHT", app->axes->loc_description(5));

  return m;
}
