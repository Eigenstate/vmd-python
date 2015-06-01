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
 *      $RCSfile: py_axes.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.11 $       $Date: 2010/12/16 04:08:56 $
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
static PyObject *get_location(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":axes.get_location"))
    return NULL;

  VMDApp *app = get_vmdapp(); 
  return PyString_FromString(app->axes->loc_description(app->axes->location()));
}

// set_location(locale)
static PyObject *set_location(PyObject *self, PyObject *args) {
  char *location;
  if (!PyArg_ParseTuple(args, (char *)"s:axes.set_location", &location))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (app->axes_set_location(location)) {
    Py_INCREF(Py_None);
    return Py_None;
  }
  PyErr_SetString(PyExc_ValueError, (char *)"Invalid axes location");
  return NULL;
}
  
static PyMethodDef methods[] = {
  {(char *)"get_location", (vmdPyMethod)get_location, METH_VARARGS},
  {(char *)"set_location", (vmdPyMethod)set_location, METH_VARARGS},
  {NULL, NULL}
};

void initaxes() {
  PyObject *m = Py_InitModule((char *)"axes", methods);
  VMDApp *app = get_vmdapp();
  PyModule_AddStringConstant(m, (char *)"OFF", (char *)app->axes->loc_description(0));
  PyModule_AddStringConstant(m, (char *)"ORIGIN", (char *)app->axes->loc_description(1));
  PyModule_AddStringConstant(m, (char *)"LOWERLEFT", (char *)app->axes->loc_description(2));
  PyModule_AddStringConstant(m, (char *)"LOWERRIGHT", (char *)app->axes->loc_description(3));
  PyModule_AddStringConstant(m, (char *)"UPPERLEFT", (char *)app->axes->loc_description(4));
  PyModule_AddStringConstant(m, (char *)"UPPERRIGHT", (char *)app->axes->loc_description(5));
}
