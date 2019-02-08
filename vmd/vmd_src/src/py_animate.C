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
 *      $RCSfile: py_animate.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.19 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python interface to animation functions.
 ***************************************************************************/

#include "py_commands.h"
#include "CommandQueue.h"
#include "VMDApp.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "Animation.h"

// once()
static const char once_doc[] = "Animate once through all frames.";
static PyObject *py_once(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_style(Animation::ANIM_ONCE);

  Py_INCREF(Py_None);
  return Py_None;
}

// rock()
static const char rock_doc[] = "Animate back and forth between first and last frames.";
static PyObject *py_rock(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;

  app->animation_set_style(Animation::ANIM_ROCK);

  Py_INCREF(Py_None);
  return Py_None;
}

// loop()
static const char loop_doc[] = "Animate in a continuous loop.";
static PyObject *py_loop(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_style(Animation::ANIM_LOOP);

  Py_INCREF(Py_None);
  return Py_None;
}

// style() : return current style name as a string
static const char style_doc[] =
"Returns current animation style\n\n"
"Returns:\n"
"    (str) style, in ['Rock', 'Once', 'Loop']";
static PyObject *py_style(PyObject *self, PyObject *args) {
  int stylenum;
  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;

  stylenum = app->anim->anim_style();
  return as_pystring((char*) animationStyleName[stylenum]);
}

// goto(frame)
static const char goto_doc[] =
"Display a givenframe on the next display update\n\n"
"Args:\n"
"    frame (int): Frame index to display";
static PyObject *py_anim_goto(PyObject *self, PyObject *args, PyObject *kwargs) {
  const char *kwnames[] = {"frame", NULL};
  VMDApp *app;
  int frame;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:animate.goto",
                                   (char**) kwnames, &frame))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_frame(frame);

  Py_INCREF(Py_None);
  return Py_None;
}

// reverse()
static const char reverse_doc[] = "Start animating frames in reverse order.";
static PyObject *py_reverse(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_dir(Animation::ANIM_REVERSE);

  Py_INCREF(Py_None);
  return Py_None;
}

// forward()
static const char forward_doc[] = "Start animating frames in forward order.";
static PyObject *py_forward(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_dir(Animation::ANIM_FORWARD);

  Py_INCREF(Py_None);
  return Py_None;
}

// prev()
static const char prev_doc[] = "Animate to the previous frame and stop.";
static PyObject *py_prev(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_dir(Animation::ANIM_REVERSE1);

  Py_INCREF(Py_None);
  return Py_None;
}

// next()
static const char next_doc[] = "Animate to the next frame and stop.";
static PyObject *py_next(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_dir(Animation::ANIM_FORWARD1);

  Py_INCREF(Py_None);
  return Py_None;
}

// pause()
static const char pause_doc[] = "Pause the animation.";
static PyObject *py_pause(PyObject *self, PyObject *args) {

  VMDApp *app;
  if (!(app = get_vmdapp()))
    return NULL;
  app->animation_set_dir(Animation::ANIM_PAUSE);

  Py_INCREF(Py_None);
  return Py_None;
}

// speed(value)
static const char speed_doc[] =
"Set or get animation speed\n\n"
"Args:\n"
"    value (float): New value for speed, between 0 and 1, or None to query\n"
"Returns:\n"
"    (float) Current value for speed";
static PyObject *py_speed(PyObject *self, PyObject *args, PyObject *kwargs) {

  const char *kwnames[] = {"value", NULL};
  float value = -1.0f;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|f:animate.speed",
                                   (char**) kwnames, &value))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (value != -1.0f) {
    if (value < 0.0f || value > 1.0f) {
        PyErr_SetString(PyExc_ValueError, "speed must be between 0 and 1");
        return NULL;
    }
    app->animation_set_speed(value);
  }

  // always return current value
  return PyFloat_FromDouble(app->anim->speed());
}

// skip(value)
static const char skip_doc[] =
"Set or get stride for animation frames. A skip value of 1 shows every frame,\n"
"a value of 2 shows every other frame, etc.\n\n"
"Args:\n"
"    value (int): New value for stride, or None to query\n\n"
"Returns:\n"
"   (int) Current value for stride";
static PyObject *py_skip(PyObject *self, PyObject *args, PyObject *kwargs) {

  const char *kwnames[] = {"value", NULL};
  int skip = 0;
  VMDApp *app;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i:animate.stride",
                                   (char**) kwnames, &skip))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (skip) {
    if (skip < 1) {
        PyErr_SetString(PyExc_ValueError, "skip must be 1 or greater");
        return NULL;
    }
    app->animation_set_stride(skip);
  }

  return as_pyint(app->anim->skip());
}

// is_active(molid)
static const char is_active_doc[] =
"Returns whether a given molecule is active (updated during animation)\n\n"
"Args:\n"
"    molid (int): Molecule ID to query\n\n"
"Returns:\n"
"    (bool) If molecule is active";
static PyObject *py_is_active(PyObject *self, PyObject *args, PyObject *kwargs) {

  const char *kwnames[] = {"molid", NULL};
  Molecule *mol;
  VMDApp *app;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i:animate.is_active",
                                   (char**) kwnames, &molid))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }

  return mol->active ? Py_True : Py_False;
}

// activate(molid, bool)
static const char activate_doc[] =
"Set the active status of a molecule. Active molecules update their coordinate"
" frames during animation, inactive ones do not.\n\n"
"Args:\n"
"    molid (int): Molecule ID to change\n"
"    active (bool): New active status of molecule.";
static PyObject *py_activate(PyObject *self, PyObject *args, PyObject *kwargs) {

  const char *kwnames[] = {"molid", "active", NULL};
  Molecule *mol;
  VMDApp *app;
  int status;
  int molid;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO&:animate.activate",
                                   (char**) kwnames, &molid, convert_bool,
                                   &status))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!(mol = app->moleculeList->mol_from_id(molid))) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  app->molecule_activate(molid, status);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef methods[] = {
  {"once", (PyCFunction) py_once, METH_NOARGS, once_doc},
  {"rock", (PyCFunction) py_rock, METH_NOARGS, rock_doc },
  {"loop", (PyCFunction) py_loop, METH_NOARGS, loop_doc },
  {"style", (PyCFunction) py_style, METH_NOARGS, style_doc },
  {"goto", (PyCFunction) py_anim_goto, METH_VARARGS | METH_KEYWORDS, goto_doc },
  {"reverse", (PyCFunction) py_reverse, METH_NOARGS, reverse_doc },
  {"forward", (PyCFunction) py_forward, METH_NOARGS, forward_doc },
  {"prev", (PyCFunction) py_prev, METH_NOARGS, prev_doc },
  {"next", (PyCFunction) py_next, METH_NOARGS, next_doc },
  {"pause", (PyCFunction) py_pause, METH_NOARGS, pause_doc },
  {"speed", (PyCFunction) py_speed, METH_VARARGS | METH_KEYWORDS, speed_doc },
  {"skip", (PyCFunction) py_skip, METH_VARARGS | METH_KEYWORDS, skip_doc },
  {"is_active", (PyCFunction) py_is_active, METH_VARARGS | METH_KEYWORDS, is_active_doc },
  {"activate", (PyCFunction) py_activate, METH_VARARGS | METH_KEYWORDS, activate_doc },
  {NULL, NULL}
};

static const char animate_moddoc[] =
"Methods for controlling molecules with multiple frames loaded";

#if PY_MAJOR_VERSION >= 3
  struct PyModuleDef animatedef = {
    PyModuleDef_HEAD_INIT,
    "animate",
    animate_moddoc,
    -1,
    methods,
  };
#endif

PyObject* initanimate() {
  PyObject *m;

#if PY_MAJOR_VERSION >= 3
  m =  PyModule_Create(&animatedef);
#else
  m =  Py_InitModule3("animate", methods, animate_moddoc);
#endif
  return m;
}

