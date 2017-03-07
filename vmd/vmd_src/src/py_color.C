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
 *      $RCSfile: py_color.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.20 $       $Date: 2016/11/28 03:05:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python color control interface.
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"

static char categories_doc[] =
  "categories() -> list\n"
  "Return list of available color categories";
static PyObject *categories(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  VMDApp *app = get_vmdapp();
  int num = app->num_color_categories();
  PyObject *newlist = PyList_New(num);
  for (int i=0; i<num; i++) {
    PyList_SET_ITEM(newlist, i,
#if PY_MAJOR_VERSION >= 3
      PyUnicode_FromString(app->color_category(i))
#else
      PyString_FromString(app->color_category(i))
#endif
    );
  }
  return newlist;
}

static char get_colormap_doc[] =
  "get_colormap(name) -> dictionary\n"
  "Return dictionary of name/color pairs in the given category";
static PyObject *get_colormap(PyObject *self, PyObject *args) {
  char *name;
  if (!PyArg_ParseTuple(args, (char *)"s", &name))
    return NULL;

  VMDApp *app = get_vmdapp();
  int num_names = app->num_color_category_items(name);
  PyObject *newdict = PyDict_New();
  for (int i=0; i<num_names; i++) {
    const char *key = app->color_category_item(name, i);
    const char *value = app->color_mapping(name, key);
#if PY_MAJOR_VERSION >= 3
    PyDict_SetItemString(newdict, (char *)key, PyUnicode_FromString(value));
#else
    PyDict_SetItemString(newdict, (char *)key, PyString_FromString(value));
#endif
  }
  return newdict;
}

static char set_colormap_doc[] =
  "set_colormap(name, dict) -> None\n"
  "Update name/color pairs in given color category.";
static PyObject *set_colormap(PyObject *self, PyObject *args) {
  char *name;
  PyObject *newdict;

  if (!PyArg_ParseTuple(args, (char *)"sO!", &name, &PyDict_Type, &newdict))
    return NULL;

  VMDApp *app = get_vmdapp();
  PyObject *keys = PyDict_Keys(newdict);
  PyObject *vals = PyDict_Values(newdict);
  int error = 0;
  for (int i=0; i<PyList_Size(keys); i++) {
#if PY_MAJOR_VERSION >= 3
    char *keyname = PyUnicode_AsUTF8(PyList_GET_ITEM(keys, i));
#else
    char *keyname = PyString_AsString(PyList_GET_ITEM(keys, i));
#endif
    if (PyErr_Occurred()) {
      error = 1;
      break;
    }
#if PY_MAJOR_VERSION >= 3
    char *valname = PyUnicode_AsUTF8(PyList_GET_ITEM(vals, i));
#else
    char *valname = PyString_AsString(PyList_GET_ITEM(vals, i));
#endif
    if (PyErr_Occurred()) {
      error = 1;
      break;
    }
    if (!app->color_changename(name, keyname, valname)) {
      PyErr_SetString(PyExc_ValueError,
        (char *)"Invalid color category or item specified");
      return NULL;
    }
  }
  Py_DECREF(keys);
  Py_DECREF(vals);
  if (error)
    return NULL;

  Py_INCREF(Py_None);
  return Py_None;
}

static char get_colors_doc[] =
  "get_colors() -> dictionary\n"
  "Returns dictionary of name/rgb pairs.  rgb is a 3-tuple.";
static PyObject *get_colors(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  VMDApp *app = get_vmdapp();
  PyObject *newdict = PyDict_New();
  for (int i=0; i<app->num_regular_colors(); i++) {
    float col[3];
    const char *name = app->color_name(i);
    if (!app->color_value(name, col, col+1, col+2)) {
      PyErr_SetString(PyExc_ValueError, (char *)
        "Unable to get color definition");
      return NULL;
    }
    PyObject *newtuple = PyTuple_New(3);
    for (int j=0; j<3; j++)
      PyTuple_SET_ITEM(newtuple, j, PyFloat_FromDouble(col[j]));
    PyDict_SetItemString(newdict, (char *)name, newtuple);
  }
  return newdict;
}

static char get_colorlist_doc[] =
  "get_colorlist() -> list\n"
  "Returns list of rgb values.  rgb is a 3-tuple.";
static PyObject *get_colorlist(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  VMDApp *app = get_vmdapp();
  int listlen = app->num_regular_colors();
  PyObject *newlist = PyList_New(listlen);
  for (int i=0; i<listlen; i++) {
    float col[3];
    const char *name = app->color_name(i);
    if (!app->color_value(name, col, col+1, col+2)) {
      PyErr_SetString(PyExc_ValueError, (char *)
        "Unable to get color definition");
      return NULL;
    }
    PyObject *newtuple = PyTuple_New(3);
    for (int j=0; j<3; j++)
      PyTuple_SET_ITEM(newtuple, j, PyFloat_FromDouble(col[j]));
    PyList_SET_ITEM(newlist, i, newtuple);
  }
  return newlist;
}

static char set_colors_doc[] =
    "set_colors(dict) -> None\n"
    "Set colors using given dicionary of name/rgb pairs";
static PyObject *set_colors(PyObject *self, PyObject *args) {
  PyObject *newdict;
  if (!PyArg_ParseTuple(args, (char *)"O!", &PyDict_Type, &newdict))
    return NULL;

  VMDApp *app = get_vmdapp();
  PyObject *keys = PyDict_Keys(newdict);
  PyObject *vals = PyDict_Values(newdict);
  int error = 0;
  for (int i=0; i<PyList_Size(keys); i++) {
#if PY_MAJOR_VERSION >= 3
    char *keyname = PyUnicode_AsUTF8(PyList_GET_ITEM(keys, i));
#else
    char *keyname = PyString_AsString(PyList_GET_ITEM(keys, i));
#endif
    if (PyErr_Occurred()) {
      error = 1;
      break;
    }
    if (app->color_index(keyname) < 0) {
      PyErr_SetString(PyExc_ValueError, (char *)"Unknown color");
      error = 1;
      break;
    }
    PyObject *newtuple = PyList_GET_ITEM(vals, i);
    if (!PyTuple_Check(newtuple) || PyTuple_Size(newtuple) != 3) {
      PyErr_SetString(PyExc_ValueError, (char *)"color definition must be 3-tuple of floats");
      error = 1;
      break;
    }
    float rgb[3];
    for (int j=0; j<3; j++)
      rgb[j] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(newtuple, j));
    if (PyErr_Occurred()) {
      error = 1;
      break;
    }
    app->color_changevalue(keyname, rgb[0], rgb[1], rgb[2]);
  }
  Py_DECREF(keys);
  Py_DECREF(vals);
  if (error)
    return NULL;

  Py_INCREF(Py_None);
  return Py_None;
}
static char set_colorid_doc[] =
    "set_colors(id, (r, g, b)) -> None\n"
    "Set colors using given color index and rgb values";
static PyObject *set_colorid(PyObject *self, PyObject *args) {
	int colorid;
	float rgb[3];
	if (!PyArg_ParseTuple(args, (char *)"i(fff)", &colorid, &rgb[0], &rgb[1], &rgb[2])) {
		return NULL;
	}
	VMDApp *app = get_vmdapp();
	if (colorid >= app->num_regular_colors() || colorid < 0) {
		PyErr_SetString(PyExc_ValueError, (char *) "color index out of range");
		return NULL;
	}
	const char *name = app->color_name(colorid);
	app->color_changevalue(name, rgb[0], rgb[1], rgb[2]);
	//We declare that we are returning a pyobject, so lets actually return one!
	Py_INCREF(Py_None);
	return Py_None;
}
static char scale_method_doc[] =
    "scale_method() -> string\n"
    "Return current colorscale method name";
static PyObject *scale_method(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  VMDApp *app = get_vmdapp();
  const char *method =
    app->colorscale_method_name(app->colorscale_method_current());
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(method);
#else
  return PyString_FromString(method);
#endif
}

static char scale_methods_doc[] =
    "scale_methods() -> list\n"
    "Return list of available colorscale methods";
static PyObject *scale_methods(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  VMDApp *app = get_vmdapp();
  int num = app->num_colorscale_methods();
  PyObject *newlist = PyList_New(num);
  for (int i=0; i<num; i++) {
    PyList_SET_ITEM(newlist, i,
#if PY_MAJOR_VERSION >= 3
      PyUnicode_FromString(app->colorscale_method_name(i))
#else
      PyString_FromString(app->colorscale_method_name(i))
#endif
    );
  }
  return newlist;
}

static char scale_midpoint_doc[] =
    "scale_midpoint() -> float\n"
    "Return current colorscale midpoint value";
static PyObject *scale_midpoint(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  float mid, min, max;
  get_vmdapp()->colorscale_info(&mid, &min, &max);
  return PyFloat_FromDouble(mid);
}

static char scale_min_doc[] =
    "scale_min() -> float\n"
    "Return current colorscale midpoint min";
static PyObject *scale_min(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  float mid, min, max;
  get_vmdapp()->colorscale_info(&mid, &min, &max);
  return PyFloat_FromDouble(min);
}

static char scale_max_doc[] =
    "scale_max() -> float\n"
    "Return current colorscale midpoint max";
static PyObject *scale_max(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  float mid, min, max;
  get_vmdapp()->colorscale_info(&mid, &min, &max);
  return PyFloat_FromDouble(max);
}

static char set_scale_doc[] =
    "set_scale(method, midpoint, min, max) -> None\n"
    "Set colorscale parameters";
static PyObject *set_scale(PyObject *self, PyObject *args, PyObject *keywds) {
  static char *kwlist[] = {
    (char *)"method", (char *)"midpoint", (char *)"min", (char *)"max", NULL
  };

  char *method = NULL;
  float midpoint = -1, min = -1, max = -1;
  VMDApp *app = get_vmdapp();
  app->colorscale_info(&midpoint, &min, &max);

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"|sfff", kwlist,
                                   &method, &midpoint, &min, &max))
    return NULL;

  if (method) {
    int ind = app->colorscale_method_index(method);
    if (ind < 0) {
      PyErr_SetString(PyExc_ValueError, (char *)"Invalid color scale method");
      return NULL;
    }
    app->colorscale_setmethod(ind);
  }
  app->colorscale_setvalues(midpoint, min, max);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ColorMethods[] = {
  {(char *)"categories", (vmdPyMethod)categories, METH_VARARGS, categories_doc},
  {(char *)"get_colormap", (vmdPyMethod)get_colormap, METH_VARARGS, get_colormap_doc},
  {(char *)"set_colormap", (vmdPyMethod)set_colormap, METH_VARARGS, set_colormap_doc},
  {(char *)"get_colors", (vmdPyMethod)get_colors, METH_VARARGS, get_colors_doc},
  {(char *)"get_colorlist", (vmdPyMethod)get_colorlist, METH_VARARGS, get_colorlist_doc},
  {(char *)"set_colors", (vmdPyMethod)set_colors, METH_VARARGS, set_colors_doc},
  {(char *)"set_colorid", (vmdPyMethod)set_colorid, METH_VARARGS, set_colorid_doc},
  {(char *)"scale_method", (vmdPyMethod)scale_method, METH_VARARGS, scale_method_doc},
  {(char *)"scale_methods", (vmdPyMethod)scale_methods, METH_VARARGS, scale_methods_doc},
  {(char *)"scale_midpoint", (vmdPyMethod)scale_midpoint, METH_VARARGS, scale_midpoint_doc},
  {(char *)"scale_min", (vmdPyMethod)scale_min, METH_VARARGS, scale_min_doc},
  {(char *)"scale_max", (vmdPyMethod)scale_max, METH_VARARGS, scale_max_doc},
  {(char *)"set_scale", (PyCFunction)set_scale, METH_VARARGS | METH_KEYWORDS, set_scale_doc},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef colordef = {
    PyModuleDef_HEAD_INIT,
    "color",
    NULL,
    -1,
    ColorMethods,
};
#endif

PyObject* initcolor() {
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&colordef);
#else
  PyObject *module = Py_InitModule((char *)"color", ColorMethods);
#endif
  return module;
}


