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
 *      $RCSfile: py_color.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.23 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python color control interface.
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"

static const char categories_doc[] =
"Get available color categories\n\n"
"Returns:\n"
"    (list of str): Available color categories";
static PyObject* py_categories(PyObject *self, PyObject *args)
{
  PyObject *newlist = NULL;
  VMDApp *app;
  int i, num;

  if (!(app = get_vmdapp()))
    return NULL;

  num = app->num_color_categories();

  if (!(newlist = PyList_New(num)))
    goto failure;

  for (i = 0; i < num; i++) {
    PyList_SET_ITEM(newlist, i, as_pystring(app->color_category(i)));
    if (PyErr_Occurred())
      goto failure;
  }
  return newlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem listing color categories");
  Py_XDECREF(newlist);
  return NULL;
}

static const char get_colormap_doc[] =
"Get name/color pairs in a given colormap category\n\n"
"Args:\n"
"    name (str): Colormap to query\n\n"
"Returns:\n"
"   (dict str->str): Name/color pairs in colormap";
static PyObject* py_get_colormap(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwnames[] = {"name", NULL};
  PyObject *newdict = NULL;
  int i, num_names;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:color.get_colormap",
                                   (char**) kwnames, &name))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  num_names = app->num_color_category_items(name);
  if (!num_names) {
    PyErr_Format(PyExc_ValueError, "Colormap '%s' does not exist", name);
    return NULL;
  }

  if (!(newdict = PyDict_New()))
    goto failure;

  // Populate colormap dictionary
  for (i = 0; i < num_names; i++) {
    const char *key = app->color_category_item(name, i);
    const char *value = app->color_mapping(name, key);

    // SetItemString does correctly steal a reference.
    if (PyDict_SetItemString(newdict, key, as_pystring(value)))
      goto failure;
  }
  return newdict;

failure:
  PyErr_Format(PyExc_RuntimeError, "Problem getting colormap '%s'", name);
  Py_XDECREF(newdict);
  return NULL;
}

static const char set_colormap_doc[] =
"Update name/color pairs in given color category\n\n"
"Args:\n"
"    name (str): Name of colormap to update\n"
"    pairs (dict str->str): Colors to update and new values. Keys must come\n"
"        from the keys listed by `get_colormap` for that color category. Not\n"
"        all keys need to be listed. Values must be legal color names";
static PyObject* py_set_colormap(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {

  const char *kwnames[] = {"name", "pairs", NULL};
  PyObject *newdict, *keys, *vals;
  PyObject *result = NULL;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO!:color.set_colormap",
                                   (char**) kwnames, &name, &PyDict_Type,
                                   &newdict))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  keys = PyDict_Keys(newdict);
  vals = PyDict_Values(newdict);

  for (int i=0; i<PyList_Size(keys); i++) {
    char *keyname = as_charptr(PyList_GetItem(keys, i));
    char *valname = as_charptr(PyList_GetItem(vals, i));

    if (!keyname || !valname || PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "set_colormap dictionary invalid");
      goto cleanup;
    }

    if (!app->color_change_name(name, keyname, valname)) {
      PyErr_SetString(PyExc_ValueError,
                      "Invalid color category or item specified");
      goto cleanup;
    }
  }
  result = Py_None;

  // Getting the keys and values from the dictionary makes a new reference.
  // We tell Python we're done with it so as to not leak memory.
  // This needs to happen even if there was a problem setting color, which is
  // why we don't return NULL from the error checking statements above.
cleanup:
  Py_DECREF(keys);
  Py_DECREF(vals);

  Py_XINCREF(result);
  return result;
}

static const char get_colors_doc[] =
"Get all legal color names and corresponding RGB values.\n\n"
"Returns:\n"
"    (dict str->3 tuple of float): Color name and RGB value. RGB values\n"
"        should be in the range 0 to 1.";
static PyObject* py_get_colors(PyObject *self, PyObject *args) {

  PyObject *newdict = NULL, *newtuple = NULL;
  const char *name;
  float col[3];
  VMDApp *app;
  int i, j;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(newdict = PyDict_New()))
    goto failure;

  for (i = 0; i < app->num_regular_colors(); i++) {

    name = app->color_name(i);
    if (!app->color_value(name, col, col+1, col+2))
      goto failure;

    if (!(newtuple = PyTuple_New(3)))
      goto failure;

    for (j = 0; j < 3; j++) {
      PyTuple_SET_ITEM(newtuple, j, PyFloat_FromDouble(col[j]));
      if (PyErr_Occurred())
        goto failure;
    }

    PyDict_SetItemString(newdict, name, newtuple);

    if (PyErr_Occurred())
      goto failure;
  }
  return newdict;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem getting color names");
  Py_XDECREF(newdict);
  Py_XDECREF(newtuple);
  return NULL;
}

static const char get_colorlist_doc[] =
"Get list of all defined colors by RGB value\n\n"
"Returns:\n"
"    (list of 3-tuple): Currently defined RGB values";
static PyObject* py_get_colorlist(PyObject *self, PyObject *args)
{

  PyObject *newlist = NULL, *newtuple = NULL;
  int i, j, listlen;
  const char *name;
  float col[3];
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  listlen = app->num_regular_colors();
  if (!(newlist = PyList_New(listlen)))
    goto failure;

  for (i = 0; i < listlen; i++) {

    name = app->color_name(i);
    if (!app->color_value(name, col, col+1, col+2))
      goto failure;

    if (!(newtuple = PyTuple_New(3)))
      goto failure;

    for (j = 0; j < 3; j++) {
      PyTuple_SET_ITEM(newtuple, j, PyFloat_FromDouble(col[j]));
      if (PyErr_Occurred())
        goto failure;
    }

    PyList_SET_ITEM(newlist, i, newtuple);
    if (PyErr_Occurred())
      goto failure;

  }
  return newlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem getting color list");
  Py_XDECREF(newlist);
  Py_XDECREF(newtuple);
  return NULL;
}

static const char set_colors_doc[] =
"Change the RGB values for named colors.\n\n"
"Args:\n"
"    colors (dict str->3-tuple of floats): Name and RGB values for colors \n"
"        to update. RGB values should be in the range 0 to 1.";
static PyObject* py_set_colors(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *newdict, *newtuple, *keys, *vals;
  const char *kwnames[] = {"colors", NULL};
  PyObject *retval = NULL;
  char *keyname;
  float rgb[3];
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!:color.set_colors",
                                   (char**) kwnames, &PyDict_Type, &newdict))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  keys = PyDict_Keys(newdict);
  vals = PyDict_Values(newdict);

  for (int i=0; i<PyList_Size(keys); i++) {
    // Get color name from input dictionary
    keyname = as_charptr(PyList_GetItem(keys, i));
    if (PyErr_Occurred())
        goto cleanup;

    // Check this color name actually exists
    if (app->color_index(keyname) < 0) {
      PyErr_Format(PyExc_ValueError, "Unknown color '%s'", keyname);
      goto cleanup;
    }

    // Unpack value tuples into 3 floats
    newtuple = PyList_GetItem(vals, i);
    if (!PyTuple_Check(newtuple) || PyTuple_Size(newtuple) != 3) {
      PyErr_SetString(PyExc_ValueError,
                      "color definition must be 3-tuple of floats");
      goto cleanup;
    }

    for (int j=0; j<3; j++) {
      rgb[j] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(newtuple, j));

      if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "color definition must be floats");
        goto cleanup;
      }
    }

    // Finally actually change the color
    app->color_change_rgb(keyname, rgb[0], rgb[1], rgb[2]);
  }
  retval = Py_None;

  // Getting the keys and values from the dictionary makes a new reference.
  // We tell Python we're done with it so as to not leak memory.
  // This needs to happen even if there was a problem setting color, which is
  // why we don't return NULL from the error checking statements above.
cleanup:
  Py_DECREF(keys);
  Py_DECREF(vals);

  Py_XINCREF(retval);
  return retval;
}

static const char set_colorid_doc[] =
"Set RGB value of a color at a given index\n\n"
"Args:\n"
"    id (int): Color ID to change\n"
"    rgb (3-tuple of floats): New RGB value for color. Values should be in\n"
"        the range 0 to 1.";
static PyObject* py_set_colorid(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwnames[] = {"id", "rgb", NULL};
  float rgb[3];
  int colorid;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i(fff):color.set_colorid",
                                   (char**) kwnames, &colorid, &rgb[0], &rgb[1],
                                   &rgb[2]))
  	return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (colorid >= app->num_regular_colors() || colorid < 0) {
  	PyErr_Format(PyExc_ValueError, "color index '%d' out of range", colorid);
  	return NULL;
  }

  app->color_change_rgb(app->color_name(colorid), rgb[0], rgb[1], rgb[2]);
  Py_INCREF(Py_None);
  return Py_None;
}

static const char scale_method_doc[] =
"Get the current colorscale method\n\n"
"Returns:\n"
"    (str) Current method name";
static PyObject* py_scale_method(PyObject *self, PyObject *args)
{

  VMDApp *app;
  const char *method;

  if (!(app = get_vmdapp()))
    return NULL;

  method = app->colorscale_method_name(app->colorscale_method_current());

  return as_pystring(method);
}

static const char scale_methods_doc[] =
"Get list of available colorscale methods\n\n"
"Returns:\n"
"    (list of str) Available colorscale methods";
static PyObject* py_scale_methods(PyObject *self, PyObject *args)
{

  PyObject *newlist = NULL;
  VMDApp *app;
  int i, num;

  if (!(app = get_vmdapp()))
    return NULL;

  num = app->num_colorscale_methods();
  if (!(newlist = PyList_New(num)))
    goto failure;

  for (i = 0; i < num; i++) {
    PyList_SET_ITEM(newlist, i, as_pystring(app->colorscale_method_name(i)));
    if (PyErr_Occurred())
      goto failure;
  }

  return newlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem listing colorscales");
  Py_XDECREF(newlist);
  return NULL;
}

static const char scale_midpoint_doc[] =
"Get current colorscale midpoint value\n\n"
"Returns:\n"
"    (float) Current midpoint";
static PyObject* py_scale_midpoint(PyObject *self, PyObject *args)
{
  float mid, min, max;
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  app->colorscale_info(&mid, &min, &max);
  return PyFloat_FromDouble(mid);
}

static const char scale_min_doc[] =
"Get current colorscale minimum value\n\n"
"Returns:\n"
"    (float): Current minimum";
static PyObject* py_scale_min(PyObject *self, PyObject *args)
{
  float mid, min, max;
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  app->colorscale_info(&mid, &min, &max);
  return PyFloat_FromDouble(min);
}

static const char scale_max_doc[] =
"Get current colorscale max.\n\n"
"Returns:\n"
"    (float) Current maximum value";
static PyObject* py_scale_max(PyObject *self, PyObject *args)
{
  float mid, min, max;
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  app->colorscale_info(&mid, &min, &max);
  return PyFloat_FromDouble(max);
}

static const char set_scale_doc[] =
"Set colorscale parameters. One or more parameters may be set with each "
"function invocation.\n\n"
"Args:\n"
"    method (str): Coloring method. Valid values are in scale_methods()\n"
"    midpoint (float): Midpoint of color scale\n"
"    min (float): Minimum value of color scale\n"
"    max (float): Maximum value of color scale\n";
static PyObject* py_set_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{

  const char *kwnames[] = {"method", "midpoint", "min", "max", NULL};
  float midpoint = -1, min = -1, max = -1;
  char *method = NULL;
  VMDApp *app;

  // Set midpoint, min and max to the current values
  if (!(app = get_vmdapp()))
    return NULL;
  app->colorscale_info(&midpoint, &min, &max);

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|zfff:color.set_scale",
                                   (char**) kwnames, &method, &midpoint, &min,
                                   &max))
    return NULL;

  if (method) {
    int ind = app->colorscale_method_index(method);
    if (ind < 0) {
      PyErr_SetString(PyExc_ValueError, "Invalid color scale method");
      return NULL;
    }
    app->colorscale_setmethod(ind);
  }
  app->colorscale_setvalues(midpoint, min, max);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ColorMethods[] = {
  {"categories", (PyCFunction)py_categories, METH_NOARGS, categories_doc},
  {"get_colormap", (PyCFunction)py_get_colormap, METH_VARARGS | METH_KEYWORDS, get_colormap_doc},
  {"set_colormap", (PyCFunction)py_set_colormap, METH_VARARGS | METH_KEYWORDS, set_colormap_doc},
  {"get_colors", (PyCFunction)py_get_colors, METH_NOARGS, get_colors_doc},
  {"get_colorlist", (PyCFunction)py_get_colorlist, METH_NOARGS, get_colorlist_doc},
  {"set_colors", (PyCFunction)py_set_colors, METH_VARARGS | METH_KEYWORDS, set_colors_doc},
  {"set_colorid", (PyCFunction)py_set_colorid, METH_VARARGS | METH_KEYWORDS, set_colorid_doc},
  {"scale_method", (PyCFunction)py_scale_method, METH_NOARGS, scale_method_doc},
  {"scale_methods", (PyCFunction)py_scale_methods, METH_NOARGS, scale_methods_doc},
  {"scale_midpoint", (PyCFunction)py_scale_midpoint, METH_NOARGS, scale_midpoint_doc},
  {"scale_min", (PyCFunction)py_scale_min, METH_NOARGS, scale_min_doc},
  {"scale_max", (PyCFunction)py_scale_max, METH_NOARGS, scale_max_doc},
  {"set_scale", (PyCFunction)py_set_scale, METH_VARARGS | METH_KEYWORDS, set_scale_doc},
  {NULL, NULL}
};


static const char color_moddoc[] =
"Contains methods for working with colors, including changing color "
"definitions, color maps, or edit the color scale. All RGB and color scale "
"values should be in the range 0 to 1";

#if PY_MAJOR_VERSION >= 3
struct PyModuleDef colordef = {
  PyModuleDef_HEAD_INIT,
  "color",
  color_moddoc,
  -1,
  ColorMethods,
};
#endif

PyObject* initcolor() {

  PyObject *module;
#if PY_MAJOR_VERSION >= 3
  module = PyModule_Create(&colordef);
#else
  module = Py_InitModule3("color", ColorMethods, color_moddoc);
#endif

  return module;
}


