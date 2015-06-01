/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "py_commands.h"
#include "VMDTkinterMenu.h"
#include "VMDApp.h"

static PyObject *show(PyObject *self, PyObject *args) {
  char *name;
  int onoff;
  if (!PyArg_ParseTuple(args, (char *)"s|i", &name, &onoff))
    return NULL;
  VMDApp *app = get_vmdapp();
  if (PyTuple_GET_SIZE(args) > 1) {
    app->menu_show(name, onoff);
  }
  return Py_BuildValue("i", app->menu_status(name));
}

static PyObject *location(PyObject *self, PyObject *args) {
  char *name;
  PyObject *loc;
  if (!PyArg_ParseTuple(args, (char *)"s|O", &name, &loc))
    return NULL;
  VMDApp *app = get_vmdapp();
  int x, y;
  if (PyTuple_GET_SIZE(args) > 1) {
    // parse loc argument
    if (!PyArg_ParseTuple(loc, (char *)"ii", &x, &y))
      return NULL;
    app->menu_move(name, x, y);
  }
  app->menu_location(name, x, y);
  return Py_BuildValue("(ii)", x, y);
}

static PyObject *addmenu(PyObject *self, PyObject *args) {
  char *name;
  PyObject *root;
  if (!PyArg_ParseTuple(args, (char *)"sO", &name, &root))
    return NULL;
  VMDApp *app = get_vmdapp();
  VMDMenu *menu = new VMDTkinterMenu(name, root, app);
  if (!app->add_menu(menu)) {
    delete menu;
    PyErr_SetString(PyExc_ValueError, (char *)"Could not add menu");
    return NULL;
  }
  app->menu_add_extension(name,name);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *registermenu(PyObject *self, PyObject *args) {
  char *name, *menupath=NULL;
  PyObject *func;
  if (!PyArg_ParseTuple(args, (char *)"sO|s", &name, &func, &menupath))
    return NULL;
  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_ValueError, (char *)"func argument must be callable");
    return NULL;
  }
  if (!menupath) 
    menupath = name;
  VMDApp *app = get_vmdapp();
  VMDTkinterMenu *menu = new VMDTkinterMenu(name, NULL, app);
  if (!app->add_menu(menu)) {
    delete menu;
    PyErr_SetString(PyExc_ValueError, (char *)"Could not add menu");
    return NULL;
  }
  menu->register_windowproc(func);
  app->menu_add_extension(name,menupath);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {(char *)"add", (vmdPyMethod)addmenu, METH_VARARGS,
    (char *)"add(name, root) -- add to VMD extension menu"},
  {(char *)"register", (vmdPyMethod)registermenu, METH_VARARGS,
    (char *)"register(name, func, menupath) -- func returns Tk() instance"},
  {(char *)"show", (vmdPyMethod)show, METH_VARARGS,
    (char *)"show(name, onoff) -- show/hide registered window"},
  {(char *)"location", (vmdPyMethod)location, METH_VARARGS, 
    (char *)"location(name, (x, y)) -- set menu location"},
  {NULL, NULL}
};

void initvmdmenu() {
  (void) Py_InitModule((char *)"vmdmenu", methods);
}

