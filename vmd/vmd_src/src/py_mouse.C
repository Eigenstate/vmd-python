/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"
#include "Mouse.h"
#include "Scene.h" // for DISP_LIGHTS

static const char mode_doc[] =
"Set the mouse behaviour in the graphics window. See help(mouse) for a list\n"
"of values describing available modes\n\n"
"Args:\n"
"    mode (int): Mouse mode to set.\n"
"    lightnum (int): Light being moved, if mode is mouse.LIGHT.";
static PyObject *py_mousemode(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"mode", "lightnum", NULL};
  int lightmode = -1;
  VMDApp *app;
  int mode;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|i:mouse.mode",
                                   (char**) kwlist, &mode, &lightmode))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (mode != Mouse::LIGHT && lightmode != -1) {
    PyErr_SetString(PyExc_ValueError, "Mouse mode must be Mouse::LIGHT if the "
                    "lightnum argument is specified");
    return NULL;
  }

  if (mode == Mouse::LIGHT && (lightmode < 0 || lightmode >= DISP_LIGHTS)) {
    PyErr_Format(PyExc_ValueError, "mouse.LIGHT mode requires a valid light "
                 "number between 0 and %d", DISP_LIGHTS);
    return NULL;
  }

  if (!(app->mouse_set_mode(mode, lightmode))) {
    PyErr_SetString(PyExc_RuntimeError, "Unable to set mouse mode");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"mode", (PyCFunction)py_mousemode, METH_VARARGS | METH_KEYWORDS, mode_doc},
  {NULL, NULL}
};

static const char mouse_moddoc[] =
"Methods for controlling the mouse behavior. Includes available mouse modes "
"and the ability to set the current mouse mode.";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef mousedef = {
  PyModuleDef_HEAD_INIT,
  "mouse",
  mouse_moddoc,
  -1,
  methods,
};
#endif

PyObject* initmouse(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&mousedef);
#else
  PyObject *m = Py_InitModule3("mouse", methods, mouse_moddoc);
#endif
  PyModule_AddIntConstant(m, "ROTATE", Mouse::ROTATION);
  PyModule_AddIntConstant(m, "TRANSLATE", Mouse::TRANSLATION);
  PyModule_AddIntConstant(m, "SCALE", Mouse::SCALING);
  PyModule_AddIntConstant(m, "LIGHT", Mouse::LIGHT);
  PyModule_AddIntConstant(m, "PICK", Mouse::PICK);
  PyModule_AddIntConstant(m, "USERPOINT", Mouse::USERPOINT);
  PyModule_AddIntConstant(m, "QUERY", Mouse::QUERY);
  PyModule_AddIntConstant(m, "CENTER", Mouse::CENTER);
  PyModule_AddIntConstant(m, "LABELATOM", Mouse::LABELATOM);
  PyModule_AddIntConstant(m, "LABELBOND", Mouse::LABELBOND);
  PyModule_AddIntConstant(m, "LABELANGLE", Mouse::LABELANGLE);
  PyModule_AddIntConstant(m, "LABELDIHEDRAL", Mouse::LABELDIHEDRAL);
  PyModule_AddIntConstant(m, "MOVEATOM", Mouse::MOVEATOM);
  PyModule_AddIntConstant(m, "MOVERES", Mouse::MOVERES);
  PyModule_AddIntConstant(m, "MOVEFRAG", Mouse::MOVEFRAG);
  PyModule_AddIntConstant(m, "MOVEMOL", Mouse::MOVEMOL);
  PyModule_AddIntConstant(m, "MOVEREP", Mouse::MOVEREP);
  PyModule_AddIntConstant(m, "FORCEATOM", Mouse::FORCEATOM);
  PyModule_AddIntConstant(m, "FORCERES", Mouse::FORCERES);
  PyModule_AddIntConstant(m, "FORCEFRAG", Mouse::FORCEFRAG);
  PyModule_AddIntConstant(m, "ADDBOND", Mouse::ADDBOND);

  return m;
}

