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
 *      $RCSfile: py_display.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.34 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python OpenGL display control interface.
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"
#include "DisplayDevice.h"

static const char update_doc[] =
"Force a render window update, without updating FLTK menus";
static PyObject* py_update(PyObject *self, PyObject *args)
{
  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;

  app->display_update();
  Py_INCREF(Py_None);
  return Py_None;
}

static const char update_ui_doc[] =
"Update the render window and all user interfaces";
static PyObject* py_update_ui(PyObject *self, PyObject *args)
{
  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;

  app->display_update_ui();

  Py_INCREF(Py_None);
  return Py_None;
}

static const char update_on_doc[] =
"Tell VMD to regularly update display and GUI menus";
static PyObject* py_update_on(PyObject *self, PyObject *args)
{
  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;

  app->display_update_on(1);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char update_off_doc[] =
"Stop updating the display. Updates will only occur when `update()` is called";
static PyObject* py_update_off(PyObject *self, PyObject *args)
{
  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;

  app->display_update_on(0);

  Py_INCREF(Py_None);
  return Py_None;
}

static const char set_doc[] =
"Sets display properties. One or more properties may be set at a time.\n\n"
"Args:\n"
"    eyesep (float): Eye separation\n"
"    focallength (float): Focal length\n"
"    height (float): Screen height relative to the camera\n"
"    distance (float): Screen distance relative to the camera\n"
"    nearclip (float): Near clipping plane distance\n"
"    farclip (float): Far clipping plane distance\n"
"    antialias (bool): If antialiasing is on\n"
"    depthcueue (bool): If depth cueuing is used\n"
"    culling (bool): If backface culling is used. Can reduce performance\n"
"    stereo (bool): If stereo mode is on\n"
"    projection (str): Projection mode, in [Perspective, Orthographic]\n"
"    size (list of 2 ints): Display window size, in px\n"
"    ambientocclusion (bool): If ambient occlusion is used\n"
"    aoambient (float): Amount of ambient light\n"
"    aodirect (float): Amount of direct light\n"
"    shadows (bool): If shadows should be rendered\n"
"    dof (bool): If depth of field effects should be rendered\n"
"    dof_fnumber (float): F-number for depth of field effects\n"
"    dof_focaldist (float): Focal distance for depth of field effects";
static PyObject* py_set(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"eyesep", "focallength", "height", "distance",
                          "nearclip", "farclip", "antialias", "depthcue",
                          "culling", "stereo", "projection", "size",
                          "ambientocclusion", "aoambient", "aodirect",
                          "shadows", "dof", "dof_fnumber", "dof_focaldist",
                          NULL};

  float eyesep, focallength, height, distance, nearclip, farclip;
  float aoambient, aodirect, dof_fnumber, dof_focaldist;
  int antialias, depthcue, culling, ao, shadows, dof;
  char *stereo, *projection;
  int num_keys = 19;
  PyObject *size;
  VMDApp *app;
  int i, w, h;


  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
                                   "|ffffffO&O&O&ssOO&ffO&O&ff:display.set",
                                   (char**) kwlist, &eyesep, &focallength,
                                   &height, &distance, &nearclip, &farclip,
                                   convert_bool, &antialias, convert_bool,
                                   &depthcue, convert_bool, &culling, &stereo,
                                   &projection, &size, convert_bool, &ao,
                                   &aoambient, &aodirect, convert_bool,
                                   &shadows, convert_bool, &dof, &dof_fnumber,
                                   &dof_focaldist))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;


  /*
   * If both nearclip and farclip will be set, the setting can fail even if
   * both new values define a valid range, as the setting is performed on one
   * clip plane at a time and it is compared to the *current* value of the
   * other. Here, we set both clip planes to be in valid locations beforehand
   * so this failure won't happen.
   */
  if (PyDict_GetItemString(kwargs, "nearclip")
   && PyDict_GetItemString(kwargs, "farclip")) {
    if (nearclip >= farclip)
      goto cliperror;

    if (nearclip >= app->display->far_clip())
      app->display_set_farclip(nearclip + 1.0, 0);

    if (farclip <= app->display->near_clip())
      app->display_set_nearclip(farclip - 1.0, 0);
  }

  // Use the kwargs dictionary directly to get the commands
  for (i = 0; i < num_keys; i++) {

    if (! PyDict_GetItemString(kwargs, kwlist[i]))
      continue;

    switch (i) {
      case 0: app->display_set_eyesep(eyesep); break;
      case 1: app->display_set_focallen(focallength); break;
      case 2: app->display_set_screen_height(height); break;
      case 3: app->display_set_screen_distance(distance); break;
      case 4:
        if (nearclip >= app->display->far_clip())
          goto cliperror;
        app->display_set_nearclip(nearclip, 0);
        break;
      case 5:
        if (farclip <= app->display->near_clip())
          goto cliperror;
        app->display_set_farclip(farclip, 0);
        break;
      case 6: app->display_set_aa(antialias); break;
      case 7: app->display_set_depthcue(depthcue); break;
      case 8: app->display_set_culling(culling); break;
      case 9: app->display_set_stereo(stereo); break;
      case 10:
        if (!app->display_set_projection(projection)) {
          PyErr_SetString(PyExc_ValueError, "Invalid projection");
          goto failure;
        }
        break;
      case 11:
        if (!PySequence_Check(size) || PySequence_Size(size) != 2
            || is_pystring(size)) {
          PyErr_SetString(PyExc_ValueError,
                          "size argument must be a two-element list or tuple");
          goto failure;
        }
        w = as_int(PySequence_GetItem(size, 0));
        h = as_int(PySequence_GetItem(size, 1));
        if (PyErr_Occurred())
            goto failure;

        app->display_set_size(w, h);

        break;
      case 12: app->display_set_ao(ao); break;
      case 13: app->display_set_ao_ambient(aoambient); break;
      case 14: app->display_set_ao_direct(aodirect); break;
      case 15: app->display_set_shadows(shadows); break;
      case 16: app->display_set_dof(dof); break;
      case 17: app->display_set_dof_fnumber(dof_fnumber); break;
      case 18: app->display_set_dof_focal_dist(dof_focaldist); break;
      default: ;
    } // end switch
  }   // end loop over keys

  Py_INCREF(Py_None);
  return Py_None;

cliperror:
PyErr_SetString(PyExc_ValueError, "Invalid clip plane settings. Near clip "
                "plane cannot be larger than far clip plane");
failure:
    return NULL;
}

static char get_doc[] =
"Query display properties\n\n"
"Args:\n"
"    query (str): Property to query. See keywords for `display.set()` for a\n"
"        comprehensive list of properties.\n"
"Returns:\n"
"    (either float, bool, list of 2 ints, or str): Value of queried parameter\n"
"        with datatype depending on the parameter type. See `display.set()`\n"
"        for a list of all parameters and types.";
static PyObject* py_get(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"query", NULL};
  PyObject *result = NULL;
  DisplayDevice *disp;
  VMDApp *app;
  char *key;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:display.get",
                                  (char**) kwlist,  &key))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  disp = app->display;

  if (!strcmp(key, "eyesep")) {
    result = PyFloat_FromDouble(disp->eyesep());

  } else if (!strcmp(key, "focallength")) {
    result = PyFloat_FromDouble(disp->eye_dist());

  } else if (!strcmp(key, "height")) {
    result = PyFloat_FromDouble(disp->screen_height());

  } else if (!strcmp(key, "distance")) {
    result = PyFloat_FromDouble(disp->distance_to_screen());

  } else if (!strcmp(key, "nearclip")) {
    result = PyFloat_FromDouble(disp->near_clip());

  } else if (!strcmp(key, "farclip")) {
    result = PyFloat_FromDouble(disp->far_clip());

  } else if (!strcmp(key, "antialias")) {
    result = disp->aa_enabled() ? Py_True : Py_False;
    Py_INCREF(result);

  } else if (!strcmp(key, "depthcue")) {
    result = disp->cueing_enabled() ? Py_True : Py_False;
    Py_INCREF(result);

  } else if (!strcmp(key, "culling")) {
    result = disp->culling_enabled() ? Py_True : Py_False;
    Py_INCREF(result);

  } else if (!strcmp(key, "stereo")) {
    result = as_pystring(disp->stereo_name(disp->stereo_mode()));

  } else if (!strcmp(key, "projection")) {
    result = as_pystring(disp->get_projection());

  } else if (!strcmp(key, "size")) {
    int w, h;
    app->display_get_size(&w, &h);
    result = Py_BuildValue("[i,i]", w, h);

  } else if (!strcmp(key, "ambientocclusion")) {
    result = disp->ao_enabled() ? Py_True : Py_False;
    Py_INCREF(result);

  } else if (!strcmp(key, "aoambient")) {
    result = PyFloat_FromDouble(disp->get_ao_ambient());

  } else if (!strcmp(key, "aodirect")) {
    result = PyFloat_FromDouble(disp->get_ao_direct());

  } else if (!strcmp(key, "shadows")) {
    result = disp->shadows_enabled() ? Py_True : Py_False;
    Py_INCREF(result);

  } else if (!strcmp(key, "dof")) {
    result = disp->dof_enabled() ? Py_True : Py_False;
    Py_INCREF(result);

  } else if (!strcmp(key, "dof_fnumber")) {
    result = PyFloat_FromDouble(disp->get_dof_fnumber());

  } else if (!strcmp(key, "dof_focaldist")) {
    result = PyFloat_FromDouble(disp->get_dof_focal_dist());

  } else {
    PyErr_Format(PyExc_ValueError, "Invalid query '%s'", key);
    goto failure;
  }

  if (PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError, "Problem getting display attribute");
    goto failure;
  }

  return result;

failure:
  Py_XDECREF(result);
  return NULL;
}

static const char stereomodes_doc[] =
"Get available stereo modes\n\n"
"Returns:\n"
"    (list of str): Available modes";
static PyObject* py_stereomodes(PyObject *self, PyObject *args)
{
  PyObject *newlist = NULL;
  DisplayDevice *disp;
  VMDApp *app;
  int j, num;

  if (!(app = get_vmdapp()))
    return NULL;

  disp = app->display;
  num = disp->num_stereo_modes();

  newlist = PyList_New(num);
  if (!newlist || PyErr_Occurred())
    goto failure;

  for (j = 0; j < num; j++) {
    PyList_SET_ITEM(newlist, j, as_pystring(disp->stereo_name(j)));
    if (PyErr_Occurred())
      goto failure;
  }

  return newlist;

failure:
  PyErr_SetString(PyExc_RuntimeError, "Problem listing stero modes");
  Py_XDECREF(newlist);
  return NULL;
}

static PyMethodDef DisplayMethods[] = {
  {"update", (PyCFunction)py_update, METH_NOARGS, update_doc},
  {"update_ui", (PyCFunction)py_update_ui, METH_NOARGS, update_ui_doc},
  {"update_on", (PyCFunction)py_update_on, METH_NOARGS, update_on_doc},
  {"update_off", (PyCFunction)py_update_off, METH_NOARGS, update_off_doc},
  {"set", (PyCFunction)py_set, METH_VARARGS | METH_KEYWORDS, set_doc},
  {"get", (PyCFunction)py_get, METH_VARARGS | METH_KEYWORDS, get_doc},
  {"stereomodes", (PyCFunction)py_stereomodes, METH_NOARGS, stereomodes_doc},
  {NULL, NULL}
};

static const char disp_moddoc[] =
"Contains methods to set various parameters in the graphical display, as "
"well as controlling how the UI and render window are updated";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef displaydef = {
  PyModuleDef_HEAD_INIT,
  "display",
  disp_moddoc,
  -1,
  DisplayMethods,
};
#endif

PyObject* initdisplay() {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&displaydef);
#else
  PyObject *m = Py_InitModule3("display", DisplayMethods, disp_moddoc);
#endif
  // XXX elminate these hard-coded string names
  PyModule_AddStringConstant(m, (char *)"PROJ_PERSP", (char *)"Perspective");
  PyModule_AddStringConstant(m, (char *)"PROJ_ORTHO", (char *)"Orthographic");
  return m;
}

