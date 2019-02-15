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
 *      $RCSfile: py_atomsel.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $      $Date: 2018/03/20 05:04:30 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   New Python atom selection interface
 ***************************************************************************/

#include "py_commands.h"
#include "AtomSel.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "SymbolTable.h"
#include "Measure.h"
#include "SpatialSearch.h"

static const char getmacro_doc[] =
"Gets the atom selection string corresponding to a macro\n\n"
"Args:\n"
"    name (str): Macro name\n"
"Returns:\n"
"    (str): Atom selection string that macro expands to";
static PyObject *py_getmacro(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", NULL};
  const char *s;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:selection.get_macro",
                                   (char**) kwlist, &name))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(s = app->atomSelParser->get_custom_singleword(name))) {
    PyErr_Format(PyExc_ValueError, "No such macro '%s'", name);
    return NULL;
  }

  return as_pystring(s);
}

static const char allmacro_doc[] =
"Get all defined macros\n\n"
"Returns:\n"
"    (list of str): All defined macro names";
static PyObject *py_allmacros(PyObject *self, PyObject *args)
{
  PyObject *result = NULL;
  SymbolTable *table;
  const char *s;
  VMDApp *app;
  int i;

  if (!(app = get_vmdapp()))
    return NULL;

  table = app->atomSelParser;

  if (!(result = PyList_New(0)))
    goto failure;

  for (i = 0; i < table->num_custom_singleword(); i++) {
    s = table->custom_singleword_name(i);

    // Handle PyList_Append not stealing a reference
    if (s && strlen(s)) {
      PyObject *tmp = as_pystring(s);
      PyList_Append(result, tmp);
      Py_XDECREF(tmp);
    }

    if (PyErr_Occurred())
      goto failure;
  }

  return result;

failure:
  Py_XDECREF(result);
  PyErr_SetString(PyExc_RuntimeError, "Problem listing macro names");
  return NULL;
}

static const char addmacro_doc[] =
"Create a new atom selection macro. A macro is a word or words that expand\n"
"to be a much larger selection string, for example 'noh' is a built-in macro\n"
"that expands to 'not hydrogen'\n\n"
"Args:\n"
"    name (str): Macro name\n"
"    selection (str): Atom selection that macro will expand to";
static PyObject *py_addmacro(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "selection", NULL};
  char *name, *selection;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss:selection.add_macro",
                                   (char**) kwlist, &name, &selection))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->atomSelParser->add_custom_singleword(name, selection)) {
    PyErr_Format(PyExc_ValueError, "Unable to create macro '%s'", name);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char delmacro_doc[] =
"Delete an atom selection macro, by name\n\n"
"Args:\n"
"    name (str): Macro name to delete";
static PyObject *py_delmacro(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", NULL};
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:selection.del_macro",
                                   (char**) kwlist, &name))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->atomSelParser->remove_custom_singleword(name)) {
    PyErr_Format(PyExc_ValueError, "Unable to remove macro '%s'", name);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

#define SYMBOL_TABLE_FUNC(funcname, elemtype) \
static PyObject *funcname(PyObject *self) { \
  VMDApp *app = get_vmdapp(); \
  if (!app) return NULL; \
  PyObject *result = PyList_New(0); \
  if (!result) return NULL; \
  SymbolTable *table = app->atomSelParser; \
  int i, n = table->fctns.num(); \
  for (i=0; i<n; i++) \
    if (table->fctns.data(i)->is_a == elemtype && \
        strlen(table->fctns.name(i))) { \
      PyObject *tmp = as_pystring(table->fctns.name(i)); \
      PyList_Append(result, tmp); \
      Py_XDECREF(tmp); \
  } \
  return result; \
}

SYMBOL_TABLE_FUNC(keywords, SymbolTableElement::KEYWORD)
SYMBOL_TABLE_FUNC(booleans, SymbolTableElement::SINGLEWORD)
SYMBOL_TABLE_FUNC(functions, SymbolTableElement::FUNCTION)
SYMBOL_TABLE_FUNC(stringfunctions, SymbolTableElement::STRINGFCTN)

#undef SYMBOL_TABLE_FUNC

/* List of functions exported by this module */
static PyMethodDef selection_methods[] = {
  {"add_macro", (PyCFunction)py_addmacro, METH_VARARGS | METH_KEYWORDS, addmacro_doc},
  {"get_macro", (PyCFunction)py_getmacro, METH_VARARGS | METH_KEYWORDS, getmacro_doc},
  {"all_macros", (PyCFunction)py_allmacros, METH_NOARGS, allmacro_doc},
  {"del_macro", (PyCFunction)py_delmacro, METH_VARARGS | METH_KEYWORDS, delmacro_doc},
  {"keywords", (PyCFunction)keywords, METH_NOARGS,
    "keywords() -> List of available atom selection keywords."},
  {"booleans", (PyCFunction)booleans, METH_NOARGS,
    "booleans() -> List of available atom selection boolean tokens."},
  {"functions", (PyCFunction)functions, METH_NOARGS,
    "functions() -> List of available atom selection functions."},
  {"stringfunctions", (PyCFunction)stringfunctions, METH_NOARGS,
    "stringfunctions() -> List of available atom selection string functions."},
  { NULL, NULL }
};

static const char module_doc[] =
"Methods for creating, modifying, or deleting macros for atom selections.";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef selectiondef = {
    PyModuleDef_HEAD_INIT,
    "selection",
    module_doc,
    -1,
    selection_methods,
};
#endif

PyObject* initselection() {

#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&selectiondef);
#else
  PyObject *m = Py_InitModule3("selection", selection_methods, module_doc);
#endif

  Py_INCREF((PyObject *)&Atomsel_Type);
  if (PyModule_AddObject(m, "selection", (PyObject *)&Atomsel_Type))
    return NULL;
  return m;
}

