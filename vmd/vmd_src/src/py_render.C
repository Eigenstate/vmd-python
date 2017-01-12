/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: py_render.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $       $Date: 2016/11/28 03:05:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface to external renderer commands  
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"

// listall(): list the supported render methods

static PyObject *listall(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  PyObject *newlist = PyList_New(0);
  VMDApp *app = get_vmdapp();
  for (int i=0; i<app->filerender_num(); i++)
#if PY_MAJOR_VERSION >= 3
    PyList_Append(newlist, PyUnicode_FromString(app->filerender_name(i)));
#else
    PyList_Append(newlist, PyString_FromString(app->filerender_name(i)));
#endif
  
  return newlist;
}

// render(method, filename)

static PyObject *render(PyObject *self, PyObject *args, PyObject *keywds) {
  char *method, *filename;
  static char *kwlist[] = {
    (char *)"method", (char *)"filename", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"ss", kwlist,
    &method, &filename))
    return NULL;

  VMDApp *app = get_vmdapp();
  
  if (!app->filerender_render(method, filename, NULL)) {
    PyErr_SetString(PyExc_ValueError, "Unable to render to file");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {(char *)"listall", (vmdPyMethod)listall, METH_VARARGS},
  {(char *)"render", (PyCFunction)render, METH_VARARGS | METH_KEYWORDS},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef renderdef = {
    PyModuleDef_HEAD_INIT,
    "render",
    NULL,
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_render(void) {
    PyObject *m = PyModule_Create(&renderdef);
    return m;
}
#else
void initrender() {
  (void)Py_InitModule((char *)"render", methods);
}
#endif
