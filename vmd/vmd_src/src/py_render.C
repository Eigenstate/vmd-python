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
 *      $RCSfile: py_render.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.15 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface to external renderer commands
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"

static const char listall_doc[] =
"List all supported render methods\n\n"
"Returns:\n"
"    (list of str): Supported render methods, suitable for calls to `render()`";
static PyObject *py_listall(PyObject *self, PyObject *args)
{
  PyObject *newlist = NULL;
  VMDApp *app;
  int i;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(newlist = PyList_New(0)))
    goto failure;

  // Need a tmp object here because PyList_Append increments refcount
  for (i=0; i<app->filerender_num(); i++) {
    PyObject *tmp = as_pystring(app->filerender_name(i));
    PyList_Append(newlist, tmp);
    Py_XDECREF(tmp);

    if (PyErr_Occurred())
      goto failure;
  }
  return newlist;

// Don't leak new list reference in case of a failure
failure:
  Py_XDECREF(newlist);
  PyErr_SetString(PyExc_RuntimeError, "Problem listing render methods");
  return NULL;
}

static const char render_doc[] =
"Render the current scene with an external or internal renderer. For some\n"
"rendering engines this entails writing an input file and then invoking an\n"
"external program\n\n"
"Args:\n"
"    method (str): Render method. See `render.listall()` for supported values\n"
"    filename (str): File name to render to. For external rendering engines,\n"
"        filename may be input file to external program";
static PyObject *py_render(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"method", "filename", NULL};
  char *method, *filename;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss:render.render",
                                   (char**) kwlist, &method, &filename))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->filerender_render(method, filename, NULL)) {
    PyErr_Format(PyExc_RuntimeError, "Unable to render to file '%s'", filename);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"listall", (PyCFunction)py_listall, METH_NOARGS, listall_doc},
  {"render", (PyCFunction)py_render, METH_VARARGS | METH_KEYWORDS, render_doc},
  {NULL, NULL}
};

static const char render_moddoc[] =
"Methods to render the current scene in the GUI with an external rendering "
"engine";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef renderdef = {
  PyModuleDef_HEAD_INIT,
  "render",
  render_moddoc,
  -1,
  methods,
};
#endif

PyObject* initrender(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&renderdef);
#else
  PyObject *m = Py_InitModule3("render", methods, render_moddoc);
#endif
  return m;
}
