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
 *      $RCSfile: py_imd.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2011/02/04 17:49:39 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Python IMD interface.
 ***************************************************************************/

#include "py_commands.h"

#ifdef VMDIMD

#include "CmdIMD.h"
#include "CommandQueue.h"
#include "VMDApp.h"
#include "MoleculeList.h"
#include "IMDMgr.h"

// connect(host, port)
static PyObject *imdconnect(PyObject *self, PyObject *args, PyObject *keywds) {
  
  char *host;
  int port;

  static char *kwlist[] = {
    (char *)"host", (char *)"port", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"si", kwlist, &host, &port))
    return NULL;

  VMDApp *app = get_vmdapp();
  Molecule *mol = app->moleculeList->top();
  if (!mol) {
    PyErr_SetString(PyExc_ValueError, (char *)"No molecule loaded");
    return NULL;
  }
  if (app->imdMgr->connected()) {
    PyErr_SetString(PyExc_ValueError, (char *)"Can't create new IMD connection: already connected.");
    return NULL;
  }
  if (!app->imd_connect(mol->id(), host, port)) {
    PyErr_SetString(PyExc_ValueError, (char *)"Unable to connect to IMD server");
    return NULL;
  } 
  Py_INCREF(Py_None);
  return Py_None;
}

// pause()
static PyObject *pause(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  VMDApp *app = get_vmdapp();
  app->imdMgr->togglepause();
  app->commandQueue->runcommand(new CmdIMDSim(CmdIMDSim::PAUSE_TOGGLE)); 
  Py_INCREF(Py_None);
  return Py_None;
}

// detach()
static PyObject *detach(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  VMDApp *app = get_vmdapp();
  app->imdMgr->detach();
  app->commandQueue->runcommand(new CmdIMDSim(CmdIMDSim::DETACH));
  Py_INCREF(Py_None);
  return Py_None;
}

// kill()
static PyObject *kill(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)""))
    return NULL;
  VMDApp *app = get_vmdapp();
  app->imdMgr->kill();
  app->commandQueue->runcommand(new CmdIMDSim(CmdIMDSim::KILL)); 
  
  Py_INCREF(Py_None);
  return Py_None;
}

// transfer(rate) : rate is optional; returns current (new) value
static PyObject *transfer(PyObject *self, PyObject *args, PyObject *keywds) {

  int rate = -1;
  static char *kwlist[] = {
    (char *)"rate", NULL
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"|i", kwlist, &rate))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (rate > 0) {
    app->imdMgr->set_trans_rate(rate);
    app->commandQueue->runcommand(
      new CmdIMDRate(CmdIMDRate::TRANSFER, rate));
  }
  return PyInt_FromLong(app->imdMgr->get_trans_rate());
}

// keep(rate): rate is optional, return current (new) value
static PyObject *keep(PyObject *self, PyObject *args, PyObject *keywds) {
  
  int rate = -1;
  static char *kwlist[] = {
    (char *)"rate", NULL
  };
  if (!PyArg_ParseTupleAndKeywords(args, keywds, (char *)"|i", kwlist, &rate))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (rate >= 0) {
    app->imdMgr->set_keep_rate(rate);
    app->commandQueue->runcommand(
      new CmdIMDRate(CmdIMDRate::KEEP, rate));
  }
  return PyInt_FromLong(app->imdMgr->get_keep_rate());
}


// copyunitcell(True/False)
static PyObject *copyunitcell(PyObject *self, PyObject *args) {

  PyObject *boolobj; 
  if (!PyArg_ParseTuple(args, (char *)"O:imd.copyunitcell", &boolobj))
    return NULL;

  VMDApp *app = get_vmdapp();
  if (PyObject_IsTrue(boolobj)) {
    app->imdMgr->set_copyunitcell(1);
    app->commandQueue->runcommand(new CmdIMDCopyUnitCell(CmdIMDCopyUnitCell::COPYCELL_ON));
  } else {
    app->imdMgr->set_copyunitcell(0);
    app->commandQueue->runcommand(new CmdIMDCopyUnitCell(CmdIMDCopyUnitCell::COPYCELL_OFF));
  }
  return Py_None;
}

static PyObject *imdconnected(PyObject *self, PyObject *args) {
  if (!PyArg_ParseTuple(args, (char *)"")) return NULL;
  VMDApp *app = get_vmdapp();
  return Py_BuildValue( "O", 
      app->imdMgr->connected() ? Py_True : Py_False );
}

static PyMethodDef methods[] = {
  {(char *)"connected", (vmdPyMethod)imdconnected, METH_VARARGS,
    (char *)"connected() -- True/False" },
  {(char *)"connect", (PyCFunction)imdconnect, METH_VARARGS | METH_KEYWORDS,
    (char *)"connect(host, port) -- establish IMD connection simulation on host:port"},
  {(char *)"pause", (vmdPyMethod)pause, METH_VARARGS, 
    (char *)"pause() -- pause a running IMD simulation"},
  {(char *)"detach", (vmdPyMethod)detach, METH_VARARGS, 
    (char *)"detach() -- detach from a running IMD simulation"},
  {(char *)"kill", (vmdPyMethod)kill, METH_VARARGS,
    (char *)"kill() -- halt a running IMD simulation (also detaches)"},
  {(char *)"transfer", (PyCFunction)transfer,METH_VARARGS | METH_KEYWORDS,
    (char *)"transfer(rate = -1) -- set/get how often timesteps are sent"},
  {(char *)"keep", (PyCFunction)keep, METH_VARARGS | METH_KEYWORDS,
    (char *)"keep(rate = -1) -- set/get how often timesteps are saved "},
  {(char *)"copyunitcell", (PyCFunction)copyunitcell, METH_VARARGS,
    (char *)"copyunitcell(True/False) -- copy unitcell information from previous frame"},
  {NULL, NULL}
};

#else
// need to create an imd module even if IMD isn't available so that the
// python startup doesn't fail.
static PyMethodDef methods[] = {
  {NULL, NULL}
};
#endif

void initimd() {
  (void) Py_InitModule((char *)"imd", methods);
}

