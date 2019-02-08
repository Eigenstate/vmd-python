/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "py_commands.h"
#include "VMDTkinterMenu.h"
#include "VMDApp.h"

static const char show_doc[] =
"Show or hide a registered window, or queries window visibility\n\n"
"Args:\n"
"    name (str): Window name\n"
"    visible (bool): Whether or not window is visible, or None to query"
"Returns:\n"
"    (bool): Updated visibility status of menu";
static PyObject *py_menushow(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "visible", NULL};
  PyObject *shownobj = Py_None;
  PyObject *result;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|O!:menu.show",
                                   (char**) kwlist, &name,
                                   &PyBool_Type, &shownobj))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Check menu name actually exists
  if (app->menu_id(name) == -1) {
    PyErr_Format(PyExc_ValueError, "No such menu '%s' to show/hide", name);
    return NULL;
  }

  // If visible is not None, set visibility of window if (shownobj != Py_None)
    app->menu_show(name, PyObject_IsTrue(shownobj));

  // Return visibility status of window
  result = app->menu_status(name) ? Py_True : Py_False;
  Py_INCREF(result);
  return result;
}

static const char loc_doc[] =
"Sets or queries menu location\n\n"
"Args:\n"
"    name (str): Menu name\n"
"    location (tuple of ints): (x,y) position to set menu location, or None\n"
"        to query current menu location\n"
"Returns:\n"
"    (tuple of ints): (x, y) updated menu location";
static PyObject* py_location(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "location", NULL};
  int x = -1, y = -1;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|(ii):menu.location",
                                   (char**) kwlist, &name, &x, &y))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Check menu name actually exists
  if (app->menu_id(name) == -1) {
    PyErr_Format(PyExc_ValueError, "No such menu '%s' to set location", name);
    return NULL;
  }

  // If a new position is specified, move the menu
  if (x != -1 && y != -1)
    app->menu_move(name, x, y);

  // Return current menu location
  app->menu_location(name, x, y);
  return Py_BuildValue("(ii)", x, y);
}

static const char add_doc[] =
"Add a new entry to a VMD menu\n\n"
"Args:\n"
"    name (str): Name of new menu to add\n"
"    root (TKinter menu): Root menu to add this menu under\n";
static PyObject* py_addmenu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "root", NULL};
  PyObject *root;
  VMDMenu *menu;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO:menu.add", (char**) kwlist,
                                   &name, &root))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Create the menu
  menu = new VMDTkinterMenu(name, root, app);

  // Try to add the menu to the root menu
  if (!app->add_menu(menu)) {
    delete menu;
    PyErr_Format(PyExc_ValueError, "Could not add menu '%s'", name);
    return NULL;
  }
  app->menu_add_extension(name,name);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char register_doc[] =
"Add an item to a menu and register a function to be called when it is "
"selected.\n\nArgs:\n"
"    name (str): Name of menu to add\n"
"    function (callable): Function to call when menu selected\n"
"    menupath (str): Path to add menu to, if it will be a submenu.\n"
"        Defaults to adding to the root menu";
static PyObject* py_registermenu(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"name", "function", "menupath", NULL};
  char *menupath = NULL;
  VMDTkinterMenu *menu;
  PyObject *func;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|s:menu.register",
                                   (char**) kwlist, &name, &func, &menupath))
    return NULL;

  // Check that the function is actually a callable
  if (!PyCallable_Check(func)) {
    PyErr_Format(PyExc_ValueError, "function argument for menu '%s' must "
                 "be callable", name);
    return NULL;
  }

  // Default for menupath is just being its own menu
  if (!menupath)
    menupath = name;

  // Make a new menu and try to add it to the app
  if (!(app = get_vmdapp()))
    return NULL;

  menu = new VMDTkinterMenu(name, NULL, app);
  if (!app->add_menu(menu)) {
    delete menu;
    PyErr_Format(PyExc_ValueError, "Could not add menu '%s'", name);
    return NULL;
  }

  // Register the callback for the menu
  menu->register_windowproc(func);
  app->menu_add_extension(name,menupath);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"add", (PyCFunction)py_addmenu, METH_VARARGS | METH_KEYWORDS, add_doc},
  {"register", (PyCFunction)py_registermenu, METH_VARARGS | METH_KEYWORDS, register_doc},
  {"show", (PyCFunction)py_menushow, METH_VARARGS | METH_KEYWORDS, show_doc},
  {"location", (PyCFunction)py_location, METH_VARARGS | METH_KEYWORDS, loc_doc},
  {NULL, NULL}
};

static const char menu_moddoc[] =
"Methods to manipulate GUI menus";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef menudef = {
  PyModuleDef_HEAD_INIT,
  "vmdmenu",
  menu_moddoc,
  -1,
  methods,
};
#endif

PyObject* initvmdmenu(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&menudef);
#else
  PyObject *m = Py_InitModule3("vmdmenu", methods, menu_moddoc);
#endif
  return m;
}

