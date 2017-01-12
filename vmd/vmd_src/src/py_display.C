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
 *      $RCSfile: py_display.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.33 $       $Date: 2016/11/28 03:05:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python OpenGL display control interface.
 ***************************************************************************/

#include "py_commands.h"
#include "VMDApp.h"
#include "DisplayDevice.h"

// update()
// force a screen update, but not a GUI or TUI update
static PyObject *update(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":display.update"))
    return NULL;

  VMDApp *app = get_vmdapp();
  app->display_update();

  Py_INCREF(Py_None);
  return Py_None;
}
    
// update_ui()
// update the screen as well as all user interfaces
static PyObject *update_ui(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":display.update_ui"))
    return NULL;

  VMDApp *app = get_vmdapp();
  app->display_update_ui();

  Py_INCREF(Py_None);
  return Py_None;
}
 
// update_on: Tell VMD to regularly update the screen
static PyObject *update_on(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":display.update_on"))
    return NULL;

  VMDApp *app = get_vmdapp();
  app->display_update_on(1);

  Py_INCREF(Py_None);
  return Py_None;
}

// update_off: Tell VMD not to update the screen
static PyObject *update_off(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":display.update_off"))
    return NULL;

  VMDApp *app = get_vmdapp();
  app->display_update_on(0);

  Py_INCREF(Py_None);
  return Py_None;
}

static char *kwlist[] = {
  (char *)"eyesep", (char *)"focallength", (char *)"height", 
  (char *)"distance", (char *)"nearclip", (char *)"farclip",
  (char *)"antialias", (char *)"depthcue", (char *)"culling", 
  (char *)"stereo", (char *)"projection", (char *)"size", 
  (char *)"ambientocclusion", (char *)"aoambient", (char *)"aodirect", (char *) "shadows",
  (char *)"dof", (char *)"dof_fnumber", (char *)"dof_focaldist",
  NULL };
static const int num_keys = 19;

// set(keywords)
static PyObject *set(PyObject *self, PyObject *args, PyObject *keywds) {

  float eyesep, focallength, height, distance, nearclip, farclip, aoambient, aodirect, dof_fnumber, dof_focaldist;
  PyObject *antialias, *depthcue, *culling, *dof, *ao, *shadows;
  char *stereo, *projection;
  PyObject *size;

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"|ffffffOOOssOOffOOff:display.set", kwlist,
    &eyesep, &focallength, &height, &distance, &nearclip, &farclip,
    &antialias, &depthcue, &culling, &stereo,
    &projection, &size, &ao, &aoambient, &aodirect, &shadows, &dof, &dof_fnumber, &dof_focaldist))
    return NULL;

  // Figure out which keys were set, and queue the corresponding command
  VMDApp *app = get_vmdapp();
  int w, h;
  for (int i=0; i<num_keys; i++) {
    if (PyDict_GetItemString(keywds, kwlist[i]) == NULL) 
      continue;
    switch (i) {
      case 0: app->display_set_eyesep(eyesep); break;
      case 1: app->display_set_focallen(focallength); break;
      case 2: app->display_set_screen_height(height); break;
      case 3: app->display_set_screen_distance(distance); break;
      case 4: app->display_set_nearclip(nearclip, 0); break;
      case 5: app->display_set_farclip(farclip, 0); break;
      case 6: app->display_set_aa(PyObject_IsTrue(antialias)); break;
      case 7: app->display_set_depthcue(PyObject_IsTrue(depthcue)); break;
      case 8: app->display_set_culling(PyObject_IsTrue(culling)); break;
      case 9: app->display_set_stereo(stereo); break;
      case 10: 
        if (!app->display_set_projection(projection)) {
          PyErr_SetString(PyExc_ValueError, "Invalid projection");
          return NULL;
        }
        break;
      case 11:
        if (!PyList_Check(size) || PyList_Size(size) != 2) {
          PyErr_SetString(PyExc_ValueError, "size argument must be a two-element list");
          return NULL;
        }
#if PY_MAJOR_VERSION >= 3
        w = PyLong_AsLong(PyList_GET_ITEM(size, 0));
        h = PyLong_AsLong(PyList_GET_ITEM(size, 1));
#else
        w = PyInt_AsLong(PyList_GET_ITEM(size, 0));
        h = PyInt_AsLong(PyList_GET_ITEM(size, 1));
#endif
        if (PyErr_Occurred()) return NULL;
        app->display_set_size(w, h);
        break;
      case 12: app->display_set_ao(PyObject_IsTrue(ao)); break;
      case 13: app->display_set_ao_ambient(aoambient); break;
      case 14: app->display_set_ao_direct(aodirect); break;
      case 15: app->display_set_shadows(PyObject_IsTrue(shadows)); break;
      case 16: app->display_set_dof(PyObject_IsTrue(dof)); break;
      case 17: app->display_set_dof_fnumber(dof_fnumber); break;
      case 18: app->display_set_dof_focal_dist(dof_focaldist); break;
      default: ;
    } // end switch
  }   // end loop over keys

  Py_INCREF(Py_None);
  return Py_None;
}
     
static char *get_kwlist[] = {
  (char *)"eyesep", (char *)"focallength", (char *)"height", 
  (char *)"distance", (char *)"nearclip", (char *)"farclip",
  (char *)"antialias", (char *)"depthcue", (char *)"culling", 
  (char *)"stereo", (char *)"projection", (char *)"size", 
  (char *)"ambientocclusion", (char *)"aoambient", (char *)"aodirect", (char *) "shadows",
  (char *)"dof", (char *)"dof_fnumber", (char *)"dof_focaldist"
};

static const int num_get_keys = 19;

// get(key)
static PyObject *get(PyObject *self, PyObject *args) {

  char *key;
  if (!PyArg_ParseTuple(args, (char *)"s:display.get", &key))
    return NULL;
   
  int i=0;
  for (; i<num_get_keys; i++) 
    if (!strcmp(key, get_kwlist[i]))
      break;
  if (i == num_get_keys) {
    PyErr_SetString(PyExc_ValueError, "Invalid attribute");
    return NULL;
  }
  VMDApp *app = get_vmdapp();
  DisplayDevice *disp = app->display;
  int w, h;
  switch (i) {
    case 0: return PyFloat_FromDouble(disp->eyesep()); 
    case 1: return PyFloat_FromDouble(disp->eye_dist()); 
    case 2: return PyFloat_FromDouble(disp->screen_height()); 
    case 3: return PyFloat_FromDouble(disp->distance_to_screen()); 
    case 4: return PyFloat_FromDouble(disp->near_clip());
    case 5: return PyFloat_FromDouble(disp->far_clip());
#if PY_MAJOR_VERSION >= 3
    case 6: return PyLong_FromLong(disp->aa_enabled() ? 1 : 0); 
    case 7: return PyLong_FromLong(disp->cueing_enabled() ? 1 : 0); 
    case 8: return PyLong_FromLong(disp->culling_enabled() ? 1 : 0);
    case 9: return PyUnicode_FromString( 
              disp->stereo_name(disp->stereo_mode())); 
    case 10: return PyUnicode_FromString(
              disp->get_projection());
    case 12: return PyLong_FromLong(disp->ao_enabled() ? 1 : 0); 
    case 15: return PyLong_FromLong(disp->shadows_enabled() ? 1 : 0);
    case 16: return PyLong_FromLong(disp->dof_enabled() ? 1 : 0);
#else
    case 6: return PyInt_FromLong(disp->aa_enabled() ? 1 : 0); 
    case 7: return PyInt_FromLong(disp->cueing_enabled() ? 1 : 0); 
    case 8: return PyInt_FromLong(disp->culling_enabled() ? 1 : 0);
    case 9: return PyString_FromString( 
              disp->stereo_name(disp->stereo_mode())); 
    case 10: return PyString_FromString(
              disp->get_projection());
    case 12: return PyInt_FromLong(disp->ao_enabled() ? 1 : 0); 
    case 15: return PyInt_FromLong(disp->shadows_enabled() ? 1 : 0);
    case 16: return PyInt_FromLong(disp->dof_enabled() ? 1 : 0);
#endif
    case 11: 
             app->display_get_size(&w, &h);
             return Py_BuildValue((char *)"[i,i]", w, h);
    case 13: return PyFloat_FromDouble(disp->get_ao_ambient());
    case 14: return PyFloat_FromDouble(disp->get_ao_direct());
    case 17: return PyFloat_FromDouble(disp->get_dof_fnumber());
    case 18: return PyFloat_FromDouble(disp->get_dof_focal_dist());
    default: ;
  }
  PyErr_SetString(PyExc_RuntimeError, "Internal error in get()");
  return NULL;
}

static PyObject *stereomodes(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)":display.stereomodes"))
    return NULL;

  DisplayDevice *disp = get_vmdapp()->display;  
  int num = disp->num_stereo_modes();
  PyObject *newlist = PyList_New(num);
  for (int j=0; j<num; j++)
#if PY_MAJOR_VERSION >= 3
    PyList_SET_ITEM(newlist, j, PyBytes_FromString(disp->stereo_name(j)));
#else
    PyList_SET_ITEM(newlist, j, PyString_FromString(disp->stereo_name(j)));
#endif
  return newlist;
}
 
static PyMethodDef DisplayMethods[] = {
  {(char *)"update", (vmdPyMethod)update, METH_VARARGS },
  {(char *)"update_ui", (vmdPyMethod)update_ui, METH_VARARGS },
  {(char *)"update_on", (vmdPyMethod)update_on, METH_VARARGS },
  {(char *)"update_off", (vmdPyMethod)update_off, METH_VARARGS },
  {(char *)"set", (PyCFunction)set, METH_VARARGS | METH_KEYWORDS},
  {(char *)"get", (vmdPyMethod)get, METH_VARARGS},
  {(char *)"stereomodes", (vmdPyMethod)stereomodes, METH_VARARGS},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef displaydef = {
  PyModuleDef_HEAD_INIT,
  "display",
  NULL,
  -1,
  DisplayMethods,
  NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_display(void) {
   PyObject *m = PyModule_Create(&displaydef); 
#else
void initdisplay() {
  PyObject *m = Py_InitModule((char *)"display", DisplayMethods);
#endif
  // XXX elminate these hard-coded string names
  PyModule_AddStringConstant(m, (char *)"PROJ_PERSP", (char *)"Perspective");
  PyModule_AddStringConstant(m, (char *)"PROJ_ORTHO", (char *)"Orthographic");
#if PY_MAJOR_VERSION >= 3
  return m;
#endif
}
 
