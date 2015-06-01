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
 *      $RCSfile: py_animate.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2010/12/16 04:08:56 $
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
static char once_doc[] = "once() -> None\nAnimate once through all frames.";
static PyObject *once(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_style(Animation::ANIM_ONCE);
 
  Py_INCREF(Py_None);
  return Py_None;
}
  
// rock()
static char rock_doc[] = "rock() -> None\nAnimate back and forth between first and last frames.";
static PyObject *rock(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_style(Animation::ANIM_ROCK);
 
  Py_INCREF(Py_None);
  return Py_None;
}
  
// loop()
static char loop_doc[] = "loop() -> None\nAnimate in a continuous loop.";
static PyObject *loop(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_style(Animation::ANIM_LOOP);
 
  Py_INCREF(Py_None);
  return Py_None;
}

// style() : return current style name as a string
static char style_doc[] = "style() -> string\nReturns current animation style ('rock', 'once', or 'loop').";
static PyObject *style(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;

  int stylenum = get_vmdapp()->anim->anim_style();
  return PyString_FromString(animationStyleName[stylenum]);
}
 
// goto(frame)
static char goto_doc[] = "goto(frame) -> None\nGo to frame on the next display update.";
static PyObject *anim_goto(PyObject *self, PyObject *args) {
  int frame;
  if (!PyArg_ParseTuple(args, (char *)"i", &frame))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_frame(frame);
 
  Py_INCREF(Py_None);
  return Py_None;
}
 
// reverse()
static char reverse_doc[] = "reverse() -> None\nStart animating in reverse.";
static PyObject *reverse(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_dir(Animation::ANIM_REVERSE);
 
  Py_INCREF(Py_None);
  return Py_None;
}
 
// forward()
static char forward_doc[] = "forward() -> None\nStart animating forward.";
static PyObject *forward(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_dir(Animation::ANIM_FORWARD);
 
  Py_INCREF(Py_None);
  return Py_None;
}
 
// prev()
static char prev_doc[] = "prev() -> None\nAnimate to the previous frame and stop.";
static PyObject *prev(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_dir(Animation::ANIM_REVERSE1);
 
  Py_INCREF(Py_None);
  return Py_None;
}
 
// next()
static char next_doc[] = "next() -> None\nAnimate to the next frame and stop.";
static PyObject *next(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_dir(Animation::ANIM_FORWARD1);
 
  Py_INCREF(Py_None);
  return Py_None;
}
 
// pause()
static char pause_doc[] = "pause() -> None\nPause the animation.";
static PyObject *pause(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  app->animation_set_dir(Animation::ANIM_PAUSE);
 
  Py_INCREF(Py_None);
  return Py_None;
}
 
// speed(value)
static char speed_doc[] = "speed(value) -> current value\nSet animation speed; pass -1 to get current speed; returns new speed.";
static PyObject *speed(PyObject *self, PyObject *args) {
  float value = -1.0f;
  if (!PyArg_ParseTuple(args, (char *)"|f",&value))
    return NULL;
 
  VMDApp *app = get_vmdapp();
  if (value > 0) { 
    app->animation_set_speed(value);
  } 
  
  // always return current value
  return PyFloat_FromDouble(app->anim->speed());
}

// skip(value)
static char skip_doc[] = "skip(value) -> new value\nSet stride for animation frames; pass -1 to get current value only;\nreturns new value.";
static PyObject *skip(PyObject *self, PyObject *args) {
  int value = 0;
  if (!PyArg_ParseTuple(args, (char *)"|i",&value))
    return NULL;
  
  VMDApp *app = get_vmdapp();
  if (value > 0) { 
    app->animation_set_stride(value);
  }
  return PyInt_FromLong(app->anim->skip()); 
}

// is_active(molid)
static char is_active_doc[] = "is_active(molid) -> boolean\nIs given molecule active (animateable)?";
static PyObject *is_active(PyObject *self, PyObject *args) {
  int molid;
  if (!PyArg_ParseTuple(args, (char *)"i", &molid))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  return PyInt_FromLong(mol->active);
}

// activate(molid, bool)
static char activate_doc[] = "activate(molid, bool) -> None\nActivate/inactivate the given molecule.";
static PyObject *activate(PyObject *self, PyObject *args) {
  int molid;
  PyObject *boolobj;
  if (!PyArg_ParseTuple(args, (char *)"iO", &molid, &boolobj))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->mol_from_id(molid);
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"Invalid molecule id");
    return NULL;
  }
  app->molecule_activate(molid, PyObject_IsTrue(boolobj));

  Py_INCREF(Py_None);
  return Py_None;
}
 
static PyMethodDef methods[] = {
  {(char *)"once", (vmdPyMethod)once, METH_VARARGS, once_doc},
  {(char *)"rock", (vmdPyMethod)rock, METH_VARARGS, rock_doc },
  {(char *)"loop", (vmdPyMethod)loop, METH_VARARGS, loop_doc },
  {(char *)"style", (vmdPyMethod)style, METH_VARARGS, style_doc },
  {(char *)"goto", (vmdPyMethod)anim_goto, METH_VARARGS, goto_doc },
  {(char *)"reverse", (vmdPyMethod)reverse, METH_VARARGS, reverse_doc },
  {(char *)"forward", (vmdPyMethod)forward, METH_VARARGS, forward_doc },
  {(char *)"prev", (vmdPyMethod)prev, METH_VARARGS, prev_doc },
  {(char *)"next", (vmdPyMethod)next, METH_VARARGS, next_doc },
  {(char *)"pause", (vmdPyMethod)pause, METH_VARARGS, pause_doc },
  {(char *)"speed", (vmdPyMethod)speed, METH_VARARGS, speed_doc },
  {(char *)"skip", (vmdPyMethod)skip, METH_VARARGS, skip_doc },
  {(char *)"is_active", (vmdPyMethod)is_active, METH_VARARGS, is_active_doc },
  {(char *)"activate", (vmdPyMethod)activate, METH_VARARGS, activate_doc },
  {NULL, NULL}
};

void initanimate() {
  (void) Py_InitModule((char *)"animate", methods);
}

 


