/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"
#include "Mouse.h"

static PyObject *mousemode(PyObject *self, PyObject *args) {
  int mode, submode = -1;
  if (!PyArg_ParseTuple(args, (char *)"i|i", &mode, &submode)) 
    return NULL;

  VMDApp *app = get_vmdapp();
  app->mouse_set_mode(mode, submode);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {(char *)"mode", (vmdPyMethod)mousemode, METH_VARARGS,
    (char *)"mode(mode, submode) -- set mouse behavior in graphics window"},
  {NULL, NULL}
};

void initmouse() {
  PyObject *m = Py_InitModule((char *)"mouse", methods);
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

}

