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
 *      $RCSfile: py_material.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface for managing materials.
 ***************************************************************************/

#include "py_commands.h"
#include "CmdMaterial.h"
#include "utilities.h"
#include "MaterialList.h"
#include "CommandQueue.h"
#include "VMDApp.h"
#include <stdlib.h>
#include <ctype.h>

static const char listall_doc[] =
"Lists all currently defined materials\n\n"
"Returns:\n"
"    (list of str): Available material names";
static PyObject* py_listall(PyObject *self, PyObject *args)
{
  PyObject *newlist = NULL;
  MaterialList *mlist;
  VMDApp *app;
  int i;

  if (!(app = get_vmdapp()))
    return NULL;

  mlist = app->materialList;
  if (!(newlist = PyList_New(0)))
    goto failure;

  // Need to avoid double counting references since PyList_Append does
  // not steal a reference and increments the count itself
  for (i = 0; i < mlist->num(); i++) {
    PyObject *tmp = as_pystring(mlist->material_name(i));
    PyList_Append(newlist, tmp);
    Py_XDECREF(tmp);

    if (PyErr_Occurred())
      goto failure;
  }

  return newlist;

failure:
  PyErr_SetString(PyExc_ValueError, "Cannot get materials");
  Py_XDECREF(newlist);
  return NULL;
}

static const char settings_doc[] =
"Get settings that comprise the definition of a given material\n\n"
"Args:\n"
"    name (str): Material name to query\n"
"Returns:\n"
"    (dict str->str): Material properties and values";
static PyObject* py_settings(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", NULL};
  MaterialList *mlist;
  PyObject *dict;
  VMDApp *app;
  char *name;
  int ind;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:material.settings",
                                   (char**) kwlist, &name))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  mlist = app->materialList;
  ind = mlist->material_index(name);
  if (ind < 0) {
    PyErr_Format(PyExc_ValueError, "No such material '%s'", name);
    return NULL;
  }

  dict = PyDict_New();
  PyDict_SetItemString(dict, "ambient",
    PyFloat_FromDouble(mlist->get_ambient(ind)));
  PyDict_SetItemString(dict, "specular",
    PyFloat_FromDouble(mlist->get_specular(ind)));
  PyDict_SetItemString(dict, "diffuse",
    PyFloat_FromDouble(mlist->get_diffuse(ind)));
  PyDict_SetItemString(dict, "shininess",
    PyFloat_FromDouble(mlist->get_shininess(ind)));
  PyDict_SetItemString(dict, "mirror",
    PyFloat_FromDouble(mlist->get_mirror(ind)));
  PyDict_SetItemString(dict, "opacity",
    PyFloat_FromDouble(mlist->get_opacity(ind)));
  PyDict_SetItemString(dict, "outline",
    PyFloat_FromDouble(mlist->get_outline(ind)));
  PyDict_SetItemString(dict, "outlinewidth",
    PyFloat_FromDouble(mlist->get_outlinewidth(ind)));

  // Transmode is a boolean, so handle reference counting
  PyDict_SetItemString(dict, "transmode",
    mlist->get_transmode(ind) ? Py_True : Py_False);
  Py_INCREF(PyDict_GetItemString(dict, "transmode"));

  if (PyErr_Occurred()) {
    PyErr_Format(PyExc_ValueError, "Error getting settings for material '%s'",
                 name);
    Py_DECREF(dict);
    return NULL;
  }

  return dict;
}

static const char default_doc[] =
"Restore the default settings of a material\n\n"
"Args:\n"
"    name (str): Material name to restore";
static PyObject* py_set_default(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"name", NULL};
  VMDApp *app;
  char *name;
  int ind;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:material.default",
                                   (char**) kwlist, &name))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Get material index from name
  ind = app->materialList->material_index(name);
  if (ind < 0) {
    PyErr_Format(PyExc_ValueError, "No such material '%s'", name);
    return NULL;
  }

  if (!app->material_restore_default(ind)) {
    PyErr_Format(PyExc_ValueError, "Material '%s' has no default settings",
                 name);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char add_doc[] =
"Create a new material with the given name. Optionally, copy material settings\n"
"from an existing material. If no name is given, make up a new name\n\n"
"Args:\n"
"    name (str): Name of new material. If None, will be made up if copy is None\n"
"    copy (str): Name of material to copy settings from. If None,\n"
"        material settings will be initialized to Opaque material";
static PyObject* py_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "copy", NULL};
  char *name = NULL, *copy = NULL;
  MaterialList *mlist;
  char *result;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|zz:material.add",
                                   (char**) kwlist, &name, &copy))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  mlist = app->materialList;

  // Check that material name does not yet exist
  if (name && mlist->material_index(name) >= 0) {
    PyErr_Format(PyExc_ValueError, "Material '%s' already exists", name);
    return NULL;
  }

  // Check that copy material does exist
  if (copy && mlist->material_index(copy) < 0) {
    PyErr_Format(PyExc_ValueError, "Material to copy '%s' doesn't exist", copy);
    return NULL;
  }

  if (!(result = (char*) app->material_add(name, copy))) {
    PyErr_Format(PyExc_ValueError, "Unable to add material '%s'", name);
    return NULL;
  }

  return as_pystring(result);
}

static const char del_doc[] =
"Deletes a material with given name\n\n"
"Args:\n"
"    name (str): Material to delete";
static PyObject* py_matdelete(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", NULL};
  char *name = NULL;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "z:material.delete",
                                   (char**) kwlist, &name))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->material_delete(name)) {
    PyErr_Format(PyExc_ValueError, "Unable to delete material '%s'", name);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char rename_doc[] =
"Change the name of an existing material\n\n"
"Args:\n"
"    name (str): Name of material to change\n"
"    newname (str): New name for material";
static PyObject* py_rename(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "newname", NULL};
  char *oldname, *newname;
  MaterialList *mlist;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss:material.rename",
                                   (char**) kwlist, &oldname, &newname))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  mlist = app->materialList;

  // Check current material is defined
  if (mlist->material_index(oldname) < 0) {
    PyErr_Format(PyExc_ValueError, "Material '%s' to rename doesn't exist",
                 oldname);
    return NULL;
  }

  // Check new name is not yet defined
  if (mlist->material_index(newname) >= 0) {
    PyErr_Format(PyExc_ValueError, "New name '%s' is already a material",
                 newname);
    return NULL;
  }

  if (!app->material_rename(oldname, newname)) {
    PyErr_Format(PyExc_ValueError, "Could not change material name '%s'->'%s'",
                 oldname, newname);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char change_doc[] =
"Change properties of a material. All property arguments are optional and are\n"
"used to set the new value of each property.\n\n"
"Args:\n"
"    name (str): Material name to change\n"
"    ambient (float): Ambient light of material\n"
"    diffuse (float): Diffuse light scattering of material\n"
"    specular (float): Amount of specular highlights from material\n"
"    shininess (float): Shininess of material\n"
"    mirror (float): Reflectivity of material\n"
"    opacity (float): Opacity of material\n"
"    outline (float): Amount of outline on material\n"
"    outline_width (float): Width of outline on material\n"
"    transmode (bool): If angle-modulated transparency should be used\n";
static PyObject* py_change(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"name", "ambient", "diffuse", "specular", "shininess",
                          "mirror", "opacity", "outline", "outlinewidth",
                          "transmode", NULL};
  float ambient, specular, diffuse, shininess, mirror, opacity;
  float outline, outlinewidth;
  MaterialList *mlist;
  int transmode;
  VMDApp *app;
  char *name;


  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ffffffffO&:material.change",
                                   (char**) kwlist, &name, &ambient, &diffuse,
                                   &specular, &shininess, &mirror, &opacity,
                                   &outline, &outlinewidth, convert_bool,
                                   &transmode))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  // Check material exists
  mlist = app->materialList;
  if (mlist->material_index(name) < 0) {
    PyErr_Format(PyExc_ValueError, "Material '%s' to change does not exist",
                 name);
    return NULL;
  }

  if (PyDict_GetItemString(kwargs, "ambient"))
    app->material_change(name, MAT_AMBIENT, ambient);
  if (PyDict_GetItemString(kwargs, "specular"))
    app->material_change(name, MAT_SPECULAR, specular);
  if (PyDict_GetItemString(kwargs, "diffuse"))
    app->material_change(name, MAT_DIFFUSE, diffuse);
  if (PyDict_GetItemString(kwargs, "shininess"))
    app->material_change(name, MAT_SHININESS, shininess);
  if (PyDict_GetItemString(kwargs, "mirror"))
    app->material_change(name, MAT_MIRROR, mirror);
  if (PyDict_GetItemString(kwargs, "opacity"))
    app->material_change(name, MAT_OPACITY, opacity);
  if (PyDict_GetItemString(kwargs, "outline"))
    app->material_change(name, MAT_OUTLINE, outline);
  if (PyDict_GetItemString(kwargs, "outlinewidth"))
    app->material_change(name, MAT_OUTLINEWIDTH, outlinewidth);
  if (PyDict_GetItemString(kwargs, "transmode"))
    app->material_change(name, MAT_TRANSMODE, (float) transmode);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"listall", (PyCFunction)py_listall, METH_NOARGS, listall_doc},
  {"settings", (PyCFunction)py_settings, METH_VARARGS | METH_KEYWORDS, settings_doc},
  {"add", (PyCFunction)py_add, METH_VARARGS | METH_KEYWORDS, add_doc},
  {"delete", (PyCFunction)py_matdelete, METH_VARARGS | METH_KEYWORDS, del_doc},
  {"rename", (PyCFunction)py_rename, METH_VARARGS | METH_KEYWORDS, rename_doc},
  {"change", (PyCFunction)py_change, METH_VARARGS | METH_KEYWORDS, change_doc },
  {"default", (PyCFunction)py_set_default, METH_VARARGS | METH_KEYWORDS, default_doc },
  {NULL, NULL}
};

static const char material_moddoc[] =
"Methods to control the materials used to render molecules or graphics objects";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef materialdef = {
  PyModuleDef_HEAD_INIT,
  "material",
  material_moddoc,
  -1,
  methods,
};
#endif

PyObject* initmaterial(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&materialdef);
#else
  PyObject *m = Py_InitModule3("material", methods, material_moddoc);
#endif
  return m;
}

