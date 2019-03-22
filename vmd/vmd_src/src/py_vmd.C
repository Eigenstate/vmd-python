
#include "py_commands.h"
#include "vmd.h"
#include "DisplayDevice.h"
#include "UIText.h"

#ifdef VMDTCL
#include <tcl.h>
#endif

static PyObject *py_vmdupdate(PyObject *self, PyObject *args) {
  VMDApp *app;
  if (!(app = get_vmdapp())) {
    fprintf(stderr, "no app!!\n");
    Py_INCREF(Py_None);
    return Py_None;
  }
  return as_pyint(app->VMDupdate(1));
}

static const char exit_doc[] =
"Exits VMD\n\n"
"Args:\n"
"    message (str): Message to print\n"
"    code (int): Code to return. Defaults to 0 (normal exit)\n"
"    delay (int): Number of seconds to delay before exiting. Defaults to 0\n";
static PyObject *py_vmdexit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwnames[] = {"message", "code", "delay", NULL};
  int code = 0, pauseseconds = 0;
  VMDApp *app;
  char *msg;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs,"s|ii:vmd.VMDupdate",
                                   (char**) kwnames, &msg, &code,
                                   &pauseseconds))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  app->VMDexit(msg, code, pauseseconds);
  // don't call VMDshutdown, because calling Tcl_Finalize() crashes Tkinter
  // Maybe add a callback to the Python atexit module?
  Py_INCREF(Py_None);
  return Py_None;
}

#ifdef VMDTCL
static const char evaltcl_doc[] =
"Evaluates a command in the TCL interpreter. This can be useful for accessing "
"functionality that is not yet implemented via the Python interface.\n\n"
"Args:\n"
"    command (str): TCL command to evaluate\n"
"Returns:\n"
"    (str) Output from the TCL interpreter";
static PyObject *py_evaltcl(PyObject *self, PyObject *args, PyObject *kwargs)
{
  const char *kwnames[] = {"command", NULL};
  const char *cmd, *result;
  Tcl_Interp *interp;
  VMDApp *app;
  int rc;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s:vmd.evaltcl",
                                   (char**) kwnames, &cmd))
    return NULL;

  if (!(app = get_vmdapp()))
    return NULL;

  interp = app->uiText->get_tcl_interp();
  rc = Tcl_Eval(interp, cmd);
  result = Tcl_GetStringResult(interp);

  if (rc != TCL_OK) {
    PyErr_SetString(PyExc_ValueError, result);
    return NULL;
  }
  return as_pystring(result);
}
#endif

static PyMethodDef VMDAppMethods [] = {
    {"VMDupdate", (PyCFunction) py_vmdupdate, METH_VARARGS },
    {"VMDexit", (PyCFunction) py_vmdexit, METH_VARARGS | METH_KEYWORDS, exit_doc },
#ifdef VMDTCL
    {"VMDevaltcl", (PyCFunction) py_evaltcl, METH_VARARGS | METH_KEYWORDS, evaltcl_doc },
    {"evaltcl", (PyCFunction) py_evaltcl, METH_VARARGS | METH_KEYWORDS, evaltcl_doc},
#endif
    { NULL, NULL }
};

static VMDApp *the_app = NULL;
static PyThreadState *event_tstate = NULL;

#if defined(VMD_SHARED)
static int vmd_input_hook() {
  if (the_app) {
    the_app->VMDupdate(1);
  }
  PyEval_RestoreThread(event_tstate);
  PyRun_SimpleString(
      "try:\n"
      "\timport Tkinter\n"
      "\twhile Tkinter.tkinter.dooneevent(Tkinter.tkinter.DONT_WAIT):\n"
      "\t\tpass\n"
      "except:\n"
      "\tpass\n");
  PyEval_SaveThread();
  return 0;
}
#endif

static const char vmd_moddoc[] =
"The main VMD python module. Provides access to the `evaltcl` method for "
"accessing features not yet available in the Python interface. When VMD "
"is built as a standalone Python module, the vmd module serves as an access "
"point for all other modules, such as `vmd.molecule` for example";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef vmddef = {
  PyModuleDef_HEAD_INIT,
  "vmd",
  vmd_moddoc,
  -1,
  VMDAppMethods,
};

#define INITERROR return NULL
extern "C" PyObject* PyInit_vmd() {
#else
#define INITERROR return
extern "C" void initvmd() {
#endif
  // Assume that VMD should not initialize or use MPI
  // It is conceivable we would want to be able to load the VMD
  // Python module into a MPI-based Python run, and enable the
  // MPI features of VMD, but we'll have to determine the best way
  // to detect this and it will need to be tested since we may have
  // to handle this case differently than the normal MPI case where
  // VMD explicitly does MPI initialization and shutdown itself.
  int mpienabled = 0;

  // If there's already a VMDapp in get_vmdapp, then we must be running
  // inside a standalone VMD instead of being loaded as a python extension.
  // Don't throw an error - just load the methods for interoperability
  // in case vmd.so is in the PYTHONPATH of the standalone application.
  if (get_vmdapp()) {
#if PY_MAJOR_VERSION >= 3
  return PyModule_Create(&vmddef);
#else
  (void)Py_InitModule3("vmd", VMDAppMethods, vmd_moddoc);
  return;
#endif
  }

  int argc = 1;
  char *argv[] = {(char*) Py_GetProgramFullPath()};
  char **argp = argv;
  if (!VMDinitialize(&argc, (char ***) &argp, mpienabled)) {
    INITERROR;
  }

  // XXX this is a hack, and it would be better to tie this into
  //     VMDApp more directly at some later point, but the regular
  //     VMD startup code is similarly lame, so we'll use it for now.
  const char *disp = getenv("VMDDISPLAYDEVICE");
  if (!disp) disp = "text";

  int loc[2] = { 50, 50 };
  int size[2] = { 400, 400 };
  VMDgetDisplayFrame(loc, size);

  VMDApp *app = new VMDApp(1, argv, mpienabled);
  app->VMDinit(1, argv, disp, loc, size);

  // read application defaults
  VMDreadInit(app);

  // don't read .vmdrc or other user-defined startup files if running
  // as a python module because it's too easy for that to cause unintended
  // behavior

  set_vmdapp(app);

  // set my local static
  the_app = app;

#if PY_MAJOR_VERSION >= 3
  PyObject *vmdmodule = PyModule_Create(&vmddef);
#else
  PyObject *vmdmodule = Py_InitModule3("vmd", VMDAppMethods, vmd_moddoc);
#endif
  if (!vmdmodule) {
    msgErr << "Failed to import vmd module" << sendmsg;
    PyErr_Print();
    INITERROR;
  }

  int i = 0;
  while (py_initializers[i].name) {
    const char *name = py_initializers[i].name;
    PyObject *module = (*(py_initializers[i].initfunc))();
    if (!module) {
      msgErr << "Failed to initialize builtin module " << name << sendmsg;
      PyErr_Print();
      continue;
    }

    int retval = PyModule_AddObject(vmdmodule, CAST_HACK name, module);
    if (retval || PyErr_Occurred()) {
      msgErr << "Failed to import builtin module " << name << sendmsg;
      msgErr << "Aborting module import" << sendmsg;
      INITERROR;
    }
    i++;
  }

  event_tstate = PyThreadState_Get();
#if defined(VMD_SHARED)
  PyOS_InputHook = vmd_input_hook;
#endif

#if PY_MAJOR_VERSION >= 3
  return vmdmodule;
#endif
}

