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
 *      $RCSfile: py_imd.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.19 $       $Date: 2019/01/17 21:21:03 $
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

static const char connect_doc[] =
"Connect to an IMD server\n\n"
"Args:\n"
"    host (str): Server hostname\n"
"    port (int): Port running IMD server";
static PyObject* py_imdconnect(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"host", "port", NULL};
  Molecule *mol;
  VMDApp *app;
  char *host;
  int port;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "si:imd.connect",
                                   (char**) kwlist, &host, &port))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;
  mol = app->moleculeList->top();

  if (!mol) {
    PyErr_SetString(PyExc_ValueError, "No molecule loaded");
    return NULL;
  }

  if (app->imdMgr->connected()) {
    PyErr_SetString(PyExc_ValueError, "Can't create new IMD connection: "
                    "already connected.");
    return NULL;
  }

  if (!app->imd_connect(mol->id(), host, port)) {
    PyErr_Format(PyExc_RuntimeError, "Unable to connect to IMD server '%s:%d'",
                 host, port);
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

static const char pause_doc[] =
"Pause a running IMD simulation";
static PyObject* py_imdpause(PyObject *self, PyObject *args)
{

  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->imdMgr->connected()) {
    PyErr_SetString(PyExc_ValueError, "Not connected to an IMD server");
    return NULL;
  }

  app->imdMgr->togglepause();
  app->commandQueue->runcommand(new CmdIMDSim(CmdIMDSim::PAUSE_TOGGLE));

  Py_INCREF(Py_None);
  return Py_None;
}

static const char detach_doc[] =
"Detach from a running IMD simulation";
static PyObject *py_imddetach(PyObject *self, PyObject *args)
{
  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->imdMgr->connected()) {
    PyErr_SetString(PyExc_ValueError, "Not connected to an IMD server");
    return NULL;
  }

  app->imdMgr->detach();
  app->commandQueue->runcommand(new CmdIMDSim(CmdIMDSim::DETACH));

  Py_INCREF(Py_None);
  return Py_None;
}

static const char kill_doc[] =
"Halt a running IMD simulation. Also detaches.";
static PyObject* py_imdkill(PyObject *self, PyObject *args)
{

  VMDApp *app;

  if (!(app = get_vmdapp()))
    return NULL;

  if (!app->imdMgr->connected()) {
    PyErr_SetString(PyExc_ValueError, "Not connected to an IMD server");
    return NULL;
  }

  app->imdMgr->kill();
  app->commandQueue->runcommand(new CmdIMDSim(CmdIMDSim::KILL));

  Py_INCREF(Py_None);
  return Py_None;
}

static const char transfer_doc[] =
"Get and/or set how often timesteps are sent to the IMD server\n\n"
"Args:\n"
"    rate (int): New transfer rate, or None to query. Rate must be greater\n"
"        than 0\n"
"Returns:\n"
"    (int): Updated transfer rate";
static PyObject* py_imdtransfer(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
  const char *kwlist[] = {"rate", NULL};
  PyObject *rateobj = NULL;
  VMDApp *app;
  int rate;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:imd.transfer",
                                   (char**) kwlist, &rateobj))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (rateobj) {
    rate = as_int(rateobj);

    if (rate <= 0) {
      PyErr_SetString(PyExc_ValueError, "transfer rate must be > 0");
      return NULL;
    }

    app->imdMgr->set_trans_rate(rate);
    app->commandQueue->runcommand(new CmdIMDRate(CmdIMDRate::TRANSFER, rate));
  }

  return as_pyint(app->imdMgr->get_trans_rate());
}

static const char keep_doc[] =
"Get and/or set how often timesteps are saved.\n\n"
"Args:\n"
"    rate (int): Save frequency, or None to query. Rate must be greater than\n"
"        or equal to 0\n"
"Returns:\n"
"    (int): Updated save frequency";
static PyObject* py_imdkeep(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwlist[] = {"rate", NULL};
  PyObject *rateobj = NULL;
  VMDApp *app;
  int rate;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:imd.keep",
                                   (char**) kwlist, &rateobj))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  if (rateobj) {
    rate = as_int(rateobj);

    if (rate < 0) {
      PyErr_SetString(PyExc_ValueError, "keep value must be >= 0");
      return NULL;
    }

    app->imdMgr->set_keep_rate(rate);
    app->commandQueue->runcommand(new CmdIMDRate(CmdIMDRate::KEEP, rate));
  }

  return as_pyint(app->imdMgr->get_keep_rate());
}

static const char copy_doc[] =
"Set if unit cell information should be copied from previous frame\n\n"
"Args:\n"
"    copy (bool): If cell information should be copied";
static PyObject* py_copyunitcell(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
  const char *kwlist[] = {"copy", NULL};
  CmdIMDCopyUnitCell::CmdIMDCopyUnitCellCommand c;
  VMDApp *app;
  int copy;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&:imd.copyunitcell",
                                   (char**) kwlist, convert_bool, &copy))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  app->imdMgr->set_copyunitcell(copy);
  c = copy ? CmdIMDCopyUnitCell::COPYCELL_ON : CmdIMDCopyUnitCell::COPYCELL_OFF;
  app->commandQueue->runcommand(new CmdIMDCopyUnitCell(c));

  Py_INCREF(Py_None);
  return Py_None;
}

static const char connected_doc[] =
"Query if an IMD connection exists\n\n"
"Returns:\n"
"    (bool): True if a connection exists, False otherwise";
static PyObject* py_imdconnected(PyObject *self, PyObject *args)
{
  VMDApp *app;
  PyObject *result;

  if (!(app = get_vmdapp()))
    return NULL;

  result = app->imdMgr->connected() ? Py_True : Py_False;
  Py_INCREF(result);
  return result;
}

static PyMethodDef methods[] = {
  {"connected", (PyCFunction)py_imdconnected, METH_NOARGS, connected_doc},
  {"connect", (PyCFunction)py_imdconnect, METH_VARARGS | METH_KEYWORDS, connect_doc},
  {"pause", (PyCFunction)py_imdpause, METH_NOARGS, pause_doc},
  {"detach", (PyCFunction)py_imddetach, METH_NOARGS, detach_doc},
  {"kill", (PyCFunction)py_imdkill, METH_NOARGS, kill_doc},
  {"transfer", (PyCFunction)py_imdtransfer, METH_VARARGS | METH_KEYWORDS, transfer_doc},
  {"keep", (PyCFunction)py_imdkeep, METH_VARARGS | METH_KEYWORDS, keep_doc},
  {"copyunitcell", (PyCFunction)py_copyunitcell, METH_VARARGS | METH_KEYWORDS, copy_doc},
  {NULL, NULL}
};

static const char imd_moddoc[] =
"Methods for controlling interactive molecular dynamics simulations";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef imddef = {
  PyModuleDef_HEAD_INIT,
  "imd",
  imd_moddoc,
  -1,
  methods,
};
#endif

PyObject* initimd() {
#if PY_MAJOR_VERSION >= 3
  PyObject *m = PyModule_Create(&imddef);
#else
  PyObject *m =  Py_InitModule3("imd", methods, imd_moddoc);
#endif
  return m;
}

#endif

