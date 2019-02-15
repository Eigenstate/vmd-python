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
 *      $RCSfile: py_molrep.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.25 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface for managing graphical representations
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"

// Helper function to check if a representation is valid
static int valid_rep(int rep, int molid, VMDApp *app)
{
  if (rep < 0 || rep >= app->num_molreps(molid)) {
    PyErr_Format(PyExc_ValueError, "Invalid rep number '%d", rep);
    return 0;
  }
  return 1;
}

static const char num_doc[] =
"Get number of representations present for molecule\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n"
"Returns:\n"
"    (int): Number of representation";
static PyObject* py_molrep_num(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", NULL};
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:molrep.num",
                                   (char**) kwlist, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  return as_pyint(app->num_molreps(molid));
}

static const char addrep_doc[] =
"Add a representation to the molecule. If optional arguments are not\n"
"specified, they default to whatever the previously added representation has.\n"
"Args:\n"
"    molid (int): Molecule ID to add represenation to\n"
"    style (str): Representation style (like 'NewCartoon'), optional\n"
"    color (str): Coloring method (like 'ColorID 1' or 'Type'), optional\n"
"    selection (str): Atoms to apply representation to, optional\n"
"    material (str): Material for represenation (like 'Opaque')\n"
"Returns:\n"
"    (int): Index of added representation";
static PyObject *py_addrep(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "style", "color", "selection", "material",
                          NULL};
  char *style = NULL, *color = NULL, *selection = NULL, *material = NULL;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|zzzz:molrep.addrep",
                                   (char**) kwlist, &molid, &style, &color,
                                   &selection, &material))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  if (style && !app->molecule_set_style(style)) {
    PyErr_Format(PyExc_ValueError, "Invalid style '%s'", style);
    return NULL;
  }

  if (color && !app->molecule_set_color(color)) {
    PyErr_Format(PyExc_ValueError, "Invalid color '%s'", color);
    return NULL;
  }

  if (selection && !app->molecule_set_selection(selection)) {
    PyErr_Format(PyExc_ValueError, "Invalid selection '%s'", selection);
    return NULL;
  }

  if (material && !app->molecule_set_material(material)) {
    PyErr_Format(PyExc_ValueError, "Invalid material '%s'", material);
    return NULL;
  }

  if (!(app->molecule_addrep(molid))) {
    PyErr_SetString(PyExc_RuntimeError, "Could not add representation");
    return NULL;
  }

  return as_pyint(app->num_molreps(molid) - 1);
}

static const char delrep_doc[] =
"Delete a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID to delete representation from\n"
"    rep (int): Representation index to delete\n";
static PyObject *py_delrep(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, rep;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.delrep",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  if (!valid_rep(rep, molid, app))
    return NULL;

  app->molrep_delete(molid, rep);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char style_doc[] =
"Get the style associated with a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (str): Representation style, like 'NewCartoon'";
static PyObject *py_get_style(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, rep;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_style",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  return as_pystring(app->molrep_get_style(molid, rep));
}

static const char select_doc[] =
"Get the atom selection associated with a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (str): Atom selection";
static PyObject *py_get_selection(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  VMDApp *app;
  int molid, rep;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_selection",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  return as_pystring(app->molrep_get_selection(molid, rep));
}

static const char color_doc[] =
"Get the coloring scheme associated with a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (str): Coloring scheme, like 'Type' or 'ColorID 5'";
static PyObject *py_get_color(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, rep;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_color",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  return as_pystring(app->molrep_get_color(molid, rep));
}

static const char material_doc[] =
"Get the material associated with a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (str): Material used, like 'Opaque'";
static PyObject *py_get_material(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, rep;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_material",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  return as_pystring(app->molrep_get_material(molid, rep));
}

static const char get_scale_doc[] =
"Get the minimum and maximum color scale values for a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (2-tuple of float): (min, max) color scale values";
static PyObject *py_get_scaleminmax(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, rep;
  float min, max;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_scaleminmax",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  if (!app->molrep_get_scaleminmax(molid, rep, &min, &max)) {
    PyErr_Format(PyExc_ValueError, "Unable to get color scale range for "
                 "molid '%d' representation '%d'", molid, rep);
    return NULL;
  }
  return Py_BuildValue("(f,f)", min, max);
}

static const char set_scale_doc[] =
"Set the minimum and maximum color scale values for a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to modify\n"
"    scale_min (float): Minimum scale value\n"
"    scale_max (float): Maximum scale value";
static PyObject *py_set_scaleminmax(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", "scale_min", "scale_max", NULL};
  int molid, rep;
  float min, max;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiff:molrep.set_scaleminmax",
                                   (char**) kwlist, &molid, &rep, &min, &max))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  if (!app->molrep_set_scaleminmax(molid, rep, min, max)) {
    PyErr_Format(PyExc_RuntimeError, "Unable to set color scale range for "
                 "molid '%d' representation '%d'", molid, rep);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char reset_doc[] =
"Automatically set the color scale minimum and maximum values to span the\n"
"input data\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to modify";
static PyObject *py_reset_scaleminmax(PyObject *self, PyObject *args,
                                      PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, rep;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.reset_scaleminmax",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  if (!app->molrep_reset_scaleminmax(molid, rep)) {
    PyErr_Format(PyExc_ValueError, "Unable to reset color scale range for "
                 "molid '%d' representation '%d'", molid, rep);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char modrep_doc[] =
"Modify properties of a representation. Any number of optional arguments may\n"
"be specified.\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to modify\n"
"    style (str): Representation style (like 'NewCartoon'), optional\n"
"    color (str): Coloring method (like 'ColorID 1' or 'Type'), optional\n"
"    selection (str): Atoms to apply representation to, optional\n"
"    material (str): Material for represenation (like 'Opaque')\n"
"    scaleminmax (2-tuple or list of float): (min, max) values for color scale\n"
"Returns:\n"
"    (bool): If modification(s) were successful";
static PyObject *py_modrep(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", "style", "selection", "color",
                          "material", "scaleminmax", NULL};
  char *style = NULL, *sel = NULL, *color = NULL, *material = NULL;
  float min = -1.f, max = -1.f;
  PyObject *ret;
  int molid, rep;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|zzzz(ff):molrep.modrep",
                                   (char**) kwlist, &molid, &rep, &style, &sel,
                                   &color, &material, &min, &max))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  int rc = 1;
  if (style)
    rc &= app->molrep_set_style(molid, rep, style);

  if (sel)
    rc &= app->molrep_set_selection(molid, rep, sel);

  if (color)
    rc &= app->molrep_set_color(molid, rep, color);

  if (material)
    rc &= app->molrep_set_material(molid, rep, material);

  if (min != -1 && max != -1) {
    rc &= app->molrep_set_scaleminmax(molid, rep, min, max);
  }

  ret = rc ? Py_True : Py_False;
  Py_INCREF(ret);
  return ret;
}

static const char repname_doc[] =
"Get the name of a representation\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (str): Representation name";
static PyObject *py_get_repname(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, rep;
  const char *name;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_repname",
                                   (char**) kwlist, &molid, &rep))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(rep, molid, app))
    return NULL;

  name = app->molrep_get_name(molid, rep);
  if (!name) {
    PyErr_Format(PyExc_ValueError, "Could not get name for molid '%d' rep '%d'",
                 molid, rep);
    return NULL;
  }

  return as_pystring(name);
}

static const char repindex_doc[] =
"Get the index of a representation from its name\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    name (str): Representation name\n"
"Returns:\n"
"    (int): Representation index, or None if no such representation";
static PyObject* py_repindex(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "name", NULL};
  int molid, repid;
  VMDApp *app;
  char *name;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "is:molrep.repindex",
                                   (char**) kwlist, &molid, &name))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app))
    return NULL;

  repid = app->molrep_get_by_name(molid, name);

  // Return None if representation doesn't exist
  if (repid == -1) {
    Py_INCREF(Py_None);
    return Py_None;
  }

  return as_pyint(repid);
}

static const char autoupdate_doc[] =
"Get if representation automatically updates its atom selection\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (bool): Status of autoupdate";
static PyObject* py_get_autoupdate(PyObject *self, PyObject *args,
                                   PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, repid;
  PyObject *result;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_autoupdate",
                                   (char**) kwlist, &molid, &repid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  result = app->molrep_get_selupdate(molid, repid) ? Py_True : Py_False;

  Py_INCREF(result);
  return result;
}

static const char set_autoupdate_doc[] =
"Set if the representation should automatically update its atom selection\n"
"when the frame is changed. Useful for selections like 'within'\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"    autoupdate (bool): Whether or not autoupdate is on";
static PyObject *py_set_autoupdate(PyObject *self, PyObject *args,
                                   PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", "autoupdate", NULL};
  int molid, repid, onoff;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiO&:molrep.set_autoupdate",
                                   (char**) kwlist, &molid, &repid,
                                   convert_bool, &onoff))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  if (!app->molrep_set_selupdate(molid, repid, onoff)) {
    PyErr_Format(PyExc_ValueError, "Cannot set selection update molid '%d'"
                "  rep '%d'", molid, repid);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char cupdate_doc[] =
"Query if the representations color automatically updates\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (bool): If colorupdate is set";
static PyObject* py_get_colorupdate(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, repid;
  PyObject *retval;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_colorupdate",
                                   (char**) kwlist, &molid, &repid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  retval = app->molrep_get_colorupdate(molid, repid) ? Py_True : Py_False;
  Py_INCREF(retval);

  return retval;
}

static const char set_cupdate_doc[] =
"Sets if the representation's color should automatically update when the\n"
"frame is changed. Useful for distance based coloring, etc.\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"    autoupdate (bool): If color should automatically update";
static PyObject* py_set_colorupdate(PyObject *self, PyObject *args,
                                    PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", "autoupdate", NULL};
  int molid, repid, onoff;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiO&:molrep.set_colorupdate",
                                   (char**) kwlist, &molid, &repid,
                                   convert_bool, &onoff))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  if (!app->molrep_set_colorupdate(molid, repid, onoff)) {
    PyErr_Format(PyExc_ValueError, "Cannot set color update molid '%d' rep '%d'",
                 molid, repid);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char get_smooth_doc[] =
"Get the number of frames over which a representation is smoothed\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (int): Number of frames representation is smoothed over";
static PyObject* py_get_smoothing(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, repid;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_smoothing",
                                   (char**) kwlist, &molid, &repid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  return as_pyint(app->molrep_get_smoothing(molid, repid));
}

static const char set_smooth_doc[] =
"Sets the number of frames over which a representation is smoothed\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"    smoothing (int): Smoothing window";
static PyObject* py_set_smoothing(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", "smoothing", NULL};
  int molid, repid, n;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iii:molrep.set_smoothing",
                                   (char**) kwlist, &molid, &repid, &n))
    return NULL;

  if (n < 0) {
    PyErr_Format(PyExc_ValueError, "Smoothing window must be 0 or higher."
                 " Got %d", n);
    return NULL;
  }

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  app->molrep_set_smoothing(molid, repid, n);
  Py_INCREF(Py_None);
  return Py_None;
}

static const char visible_doc[] =
"Query if a representation is visible\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"Returns:\n"
"    (bool): If representation is visible";
static PyObject* py_get_visible(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", NULL};
  int molid, repid;
  PyObject *retval;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii:molrep.get_visible",
                                   (char**) kwlist, &molid, &repid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  retval = app->molrep_is_shown(molid, repid) ? Py_True : Py_False;
  Py_INCREF(retval);
  return retval;
}

static const char set_visible_doc[] =
"Set if a representation is visible\n\n"
"Args:\n"
"    molid (int): Molecule ID with representation\n"
"    rep (int): Representation index to query\n"
"    visible (bool): If representation should be displayed";
static PyObject* py_set_visible(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"molid", "rep", "visible", NULL};
  int molid, repid, n;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiO&:molrep.set_visible",
                                   (char**) kwlist, &molid, &repid,
                                   convert_bool, &n))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!valid_molid(molid, app) || !valid_rep(repid, molid, app))
    return NULL;

  app->molrep_show(molid, repid, n);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"num", (PyCFunction)py_molrep_num, METH_VARARGS | METH_KEYWORDS, num_doc},
  {"addrep", (PyCFunction)py_addrep, METH_VARARGS | METH_KEYWORDS, addrep_doc},
  {"delrep", (PyCFunction)py_delrep, METH_VARARGS | METH_KEYWORDS, delrep_doc},
  {"get_style", (PyCFunction)py_get_style, METH_VARARGS | METH_KEYWORDS, style_doc},
  {"get_color", (PyCFunction)py_get_color, METH_VARARGS | METH_KEYWORDS, color_doc},
  {"get_selection", (PyCFunction)py_get_selection, METH_VARARGS | METH_KEYWORDS, select_doc},
  {"get_material", (PyCFunction)py_get_material, METH_VARARGS | METH_KEYWORDS, material_doc},
  {"modrep", (PyCFunction)py_modrep, METH_VARARGS | METH_KEYWORDS, modrep_doc},
  {"get_repname", (PyCFunction)py_get_repname, METH_VARARGS | METH_KEYWORDS, repname_doc},
  {"repindex", (PyCFunction)py_repindex, METH_VARARGS | METH_KEYWORDS, repindex_doc},
  {"get_autoupdate", (PyCFunction)py_get_autoupdate, METH_VARARGS | METH_KEYWORDS, autoupdate_doc},
  {"set_autoupdate", (PyCFunction)py_set_autoupdate, METH_VARARGS | METH_KEYWORDS, set_autoupdate_doc},
  {"get_scaleminmax", (PyCFunction)py_get_scaleminmax, METH_VARARGS | METH_KEYWORDS, get_scale_doc},
  {"set_scaleminmax", (PyCFunction)py_set_scaleminmax, METH_VARARGS | METH_KEYWORDS, set_scale_doc},
  {"reset_scaleminmax", (PyCFunction)py_reset_scaleminmax, METH_VARARGS | METH_KEYWORDS, reset_doc},
  {"get_colorupdate", (PyCFunction)py_get_colorupdate, METH_VARARGS | METH_KEYWORDS, cupdate_doc},
  {"set_colorupdate", (PyCFunction)py_set_colorupdate, METH_VARARGS | METH_KEYWORDS, set_cupdate_doc},
  {"get_smoothing", (PyCFunction)py_get_smoothing, METH_VARARGS | METH_KEYWORDS, get_smooth_doc},
  {"set_smoothing", (PyCFunction)py_set_smoothing, METH_VARARGS | METH_KEYWORDS, set_smooth_doc},
  {"get_visible", (PyCFunction)py_get_visible, METH_VARARGS | METH_KEYWORDS, visible_doc},
  {"set_visible", (PyCFunction)py_set_visible, METH_VARARGS | METH_KEYWORDS, set_visible_doc},
  {NULL, NULL}
};

static const char rep_moddoc[] =
"Methods for controlling graphical representations associated with a molecule";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef molrepdef = {
  PyModuleDef_HEAD_INIT,
  "molrep",
  rep_moddoc,
  -1,
  methods,
};
#endif

PyObject* initmolrep(void) {
#if PY_MAJOR_VERSION >= 3
  PyObject *module = PyModule_Create(&molrepdef);
#else
  PyObject *module =  Py_InitModule3("molrep", methods, rep_moddoc);
#endif
  return module;
}

