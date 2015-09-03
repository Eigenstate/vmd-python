/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: py_molrep.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.23 $       $Date: 2010/12/16 04:08:57 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface for managing graphical representations
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"

// num(molid)
static PyObject *molrep_num(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i:molrep.num", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  return PyInt_FromLong(app->num_molreps(molid));
}

// addrep(molid, style=None, color=None, selection=None, material=None)
static PyObject *addrep(PyObject *self, PyObject *args, PyObject *keywds) {
  int molid;
  char *style=NULL, *color=NULL, *selection=NULL, *material=NULL;
  static char *kwlist[] = {
    (char *)"molid", (char *)"style", (char *)"color", (char *)"selection", 
    (char *)"material", NULL
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywds, 
        (char *)"i|ssss:molrep.addrep", kwlist,
        &molid, &style, &color, &selection, &material))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (style && !app->molecule_set_style(style)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid style");
    return NULL;
  }
  if (color && !app->molecule_set_color(color)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid color");
    return NULL;
  }
  if (selection && !app->molecule_set_selection(selection)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid selection");
    return NULL;
  }
  if (material && !app->molecule_set_material(material)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid material");
    return NULL;
  }
  app->molecule_addrep(molid);
  
  Py_INCREF(Py_None);
  return Py_None;
}

// delrep(molid, rep)
static PyObject *delrep(PyObject *self, PyObject *args) {
  int molid, rep;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.delrep", &molid, &rep))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  app->molrep_delete(molid, rep);   
  
  Py_INCREF(Py_None);
  return Py_None;
}

// get_style(molid, rep)
static PyObject *get_style(PyObject *self, PyObject *args) {
  int molid, rep;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_style", &molid, &rep))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  return PyString_FromString((char *)app->molrep_get_style(molid, rep));
}

// get_selection(molid, rep)
static PyObject *get_selection(PyObject *self, PyObject *args) {
  int molid, rep;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_selection", &molid, &rep))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  return PyString_FromString((char *)app->molrep_get_selection(molid, rep));
}

// get_color(molid, rep)
static PyObject *get_color(PyObject *self, PyObject *args) {
  int molid, rep;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_color", &molid, &rep))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  return PyString_FromString((char *)app->molrep_get_color(molid, rep));
}

// get_material(molid, rep)
static PyObject *get_material(PyObject *self, PyObject *args) {
  int molid, rep;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_material", &molid, &rep))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  return PyString_FromString((char *)app->molrep_get_material(molid, rep));
}

// get_scaleminmax(molid, rep)
static PyObject *get_scaleminmax(PyObject *self, PyObject *args) {
  int molid, rep;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_scaleminmax", &molid, &rep))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  float min, max;
  if (!app->molrep_get_scaleminmax(molid, rep, &min, &max)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to get color scale range for this rep");
    return NULL;
  }
  return Py_BuildValue((char *)"[f,f]", min, max);
}

// set_scaleminmax(molid, rep, min, max)
static PyObject *set_scaleminmax(PyObject *self, PyObject *args) {
  int molid, rep;
  float min, max;
  if (!PyArg_ParseTuple(args, (char *)"iiff:molrep.set_scaleminmax", &molid, 
        &rep, &min, &max))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  if (!app->molrep_set_scaleminmax(molid, rep, min, max)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to get color scale range for this rep");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// reset_scaleminmax(molid, rep)
static PyObject *reset_scaleminmax(PyObject *self, PyObject *args) {
  int molid, rep;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.auto_scaleminmax", &molid, &rep))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 
  if (!app->molrep_reset_scaleminmax(molid, rep)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to reset color scale range for this rep");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// modrep(molid, rep, style, sel, color, material, scaleminmax)
// All but molid and rep are optional
static PyObject *modrep(PyObject *self, PyObject *args, PyObject *keywds) {

  char *style=NULL, *sel=NULL, *color=NULL, *material=NULL;
  PyObject *scaleminmax = NULL;
  int molid, rep;
 
  static char *kwlist[] = {
    (char *)"molid", (char *)"rep", (char *)"style", (char *)"sel", 
    (char *)"color", (char *)"material", (char *)"scaleminmax", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"ii|ssssO!:molrep.modrep", kwlist,
    &molid, &rep, &style, &sel, &color, &material, &PyList_Type, &scaleminmax))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (!app->molecule_valid_id(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  } 
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid rep number");
    return NULL;
  } 

  int rc = 1;
  if (style) 
    rc &= app->molrep_set_style(molid, rep, style);
  if (sel) 
    rc &= app->molrep_set_selection(molid, rep, sel);
  if (color) 
    rc &= app->molrep_set_color(molid, rep, color);
  if (material) 
    rc &= app->molrep_set_material(molid, rep, material);
  if (scaleminmax) {
    if (PyList_Size(scaleminmax) == 2) {
      float min = (float)PyFloat_AsDouble(PyList_GET_ITEM(scaleminmax, 0));
      float max = (float)PyFloat_AsDouble(PyList_GET_ITEM(scaleminmax, 1));
      if (PyErr_Occurred()) return NULL;
      rc &= app->molrep_set_scaleminmax(molid, rep, min, max);
    } else {
      PyErr_SetString(PyExc_ValueError, (char *)"scaleminmax must have two items");
      return NULL;
    }
  }
  return PyInt_FromLong(rc);
}

// get_repname(molid, repid)
static PyObject *get_repname(PyObject *self, PyObject *args) {
  int molid, repid;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_repname", &molid, &repid))
    return NULL;
  VMDApp *app = get_vmdapp();
  const char *name = app->molrep_get_name(molid, repid);
  if (!name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  return PyString_FromString((char *)name);
}
  
// repindex(molid, name)
static PyObject *repindex(PyObject *self, PyObject *args) {
  int molid;
  char *name;
  if (!PyArg_ParseTuple(args, (char *)"is:molrep.repindex", &molid, &name))
    return NULL;
  VMDApp *app = get_vmdapp();
  int repid = app->molrep_get_by_name(molid, name);
  return PyInt_FromLong(repid);
}

// get_autoupdate(molid, repid)
static PyObject *get_autoupdate(PyObject *self, PyObject *args) {
  int molid, repid;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_autoupdate", &molid, &repid))
    return NULL;
  VMDApp *app = get_vmdapp();
  const char *name = app->molrep_get_name(molid, repid);
  if (!name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  return PyInt_FromLong(app->molrep_get_selupdate(molid, repid));
}

// set_autoupdate(molid, repid, int onoff)
static PyObject *set_autoupdate(PyObject *self, PyObject *args) {
  int molid, repid, onoff;
  if (!PyArg_ParseTuple(args, (char *)"iii:molrep.set_autoupdate", &molid, &repid, &onoff))
    return NULL;
  VMDApp *app = get_vmdapp();
  if (!app->molrep_set_selupdate(molid, repid, onoff)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// get_colorupdate(molid, repid)
static PyObject *get_colorupdate(PyObject *self, PyObject *args) {
  int molid, repid;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_colorupdate", &molid, &repid))
    return NULL;
  VMDApp *app = get_vmdapp();
  const char *name = app->molrep_get_name(molid, repid);
  if (!name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  return PyInt_FromLong(app->molrep_get_colorupdate(molid, repid));
}

// set_colorupdate(molid, repid, int onoff)
static PyObject *set_colorupdate(PyObject *self, PyObject *args) {
  int molid, repid, onoff;
  if (!PyArg_ParseTuple(args, (char *)"iii:molrep.set_colorupdate", &molid, &repid, &onoff))
    return NULL;
  VMDApp *app = get_vmdapp();
  if (!app->molrep_set_colorupdate(molid, repid, onoff)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// get_smoothing(molid, repid)
static PyObject *get_smoothing(PyObject *self, PyObject *args) {
  int molid, repid;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_smoothing", &molid, &repid))
    return NULL;
  VMDApp *app = get_vmdapp();
  const char *name = app->molrep_get_name(molid, repid);
  if (!name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  return PyInt_FromLong(app->molrep_get_smoothing(molid, repid));
}

// set_smoothing(molid, repid, int n)
static PyObject *set_smoothing(PyObject *self, PyObject *args) {
  int molid, repid, n;
  if (!PyArg_ParseTuple(args, (char *)"iii:molrep.set_smoothing", &molid, &repid, &n))
    return NULL;
  if (n < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Smoothing window must be 0 or higher");
    return NULL;
  }
  VMDApp *app = get_vmdapp();
  if (!app->molrep_set_smoothing(molid, repid, n)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

// get_visible(molid, repid)
static PyObject *get_visible(PyObject *self, PyObject *args) {
  int molid, repid;
  if (!PyArg_ParseTuple(args, (char *)"ii:molrep.get_visible", &molid, &repid))
    return NULL;
  VMDApp *app = get_vmdapp();
  const char *name = app->molrep_get_name(molid, repid);
  if (!name) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  return PyInt_FromLong(app->molrep_is_shown(molid, repid));
}

// set_visible(molid, repid, int n)
static PyObject *set_visible(PyObject *self, PyObject *args) {
  int molid, repid, n;
  if (!PyArg_ParseTuple(args, (char *)"iii:molrep.set_visible", &molid, &repid, &n))
    return NULL;
  if (n < 0) {
    PyErr_SetString(PyExc_ValueError, (char *)"Smoothing window must be 0 or higher");
    return NULL;
  }
  VMDApp *app = get_vmdapp();
  if (!app->molrep_show(molid, repid, n)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molid or repid");
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {(char *)"num", (vmdPyMethod)molrep_num, METH_VARARGS},
  {(char *)"addrep", (PyCFunction)addrep, METH_VARARGS | METH_KEYWORDS},
  {(char *)"delrep", (vmdPyMethod)delrep, METH_VARARGS},
  {(char *)"get_style", (vmdPyMethod)get_style, METH_VARARGS},
  {(char *)"get_color", (vmdPyMethod)get_color, METH_VARARGS},
  {(char *)"get_selection", (vmdPyMethod)get_selection, METH_VARARGS},
  {(char *)"get_material", (vmdPyMethod)get_material, METH_VARARGS},
  {(char *)"modrep", (PyCFunction)modrep, METH_VARARGS | METH_KEYWORDS},
  {(char *)"get_repname", (vmdPyMethod)get_repname, METH_VARARGS},
  {(char *)"repindex", (vmdPyMethod)repindex, METH_VARARGS},
  {(char *)"get_autoupdate", (vmdPyMethod)get_autoupdate, METH_VARARGS},
  {(char *)"set_autoupdate", (vmdPyMethod)set_autoupdate, METH_VARARGS},
  {(char *)"get_scaleminmax", (vmdPyMethod)get_scaleminmax, METH_VARARGS},
  {(char *)"set_scaleminmax", (vmdPyMethod)set_scaleminmax, METH_VARARGS},
  {(char *)"reset_scaleminmax", (vmdPyMethod)reset_scaleminmax, METH_VARARGS},
  {(char *)"get_colorupdate", (vmdPyMethod)get_colorupdate, METH_VARARGS},
  {(char *)"set_colorupdate", (vmdPyMethod)set_colorupdate, METH_VARARGS},
  {(char *)"get_smoothing", (vmdPyMethod)get_smoothing, METH_VARARGS},
  {(char *)"set_smoothing", (vmdPyMethod)set_smoothing, METH_VARARGS},
  {(char *)"get_visible", (vmdPyMethod)get_visible, METH_VARARGS},
  {(char *)"set_visible", (vmdPyMethod)set_visible, METH_VARARGS},
  {NULL, NULL}
};

void initmolrep() {
  (void) Py_InitModule((char *)"molrep", methods);
}

