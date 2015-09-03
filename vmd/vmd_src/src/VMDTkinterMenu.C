
#include "VMDTkinterMenu.h"

// do the equivalent of
// root.wm_protocol('WM_DELETE_WINDOW', root.wm_withdraw)

static void preserve_window(PyObject *root) {
  PyObject *cb = PyObject_GetAttrString(root, (char *)"wm_withdraw");
  if (cb) {
    Py_XDECREF(PyObject_CallMethod(root, (char *)"wm_protocol", (char *)"sO",
          "WM_DELETE_WINDOW", cb));
    Py_DECREF(cb);
  }
}

VMDTkinterMenu::VMDTkinterMenu(const char *menuname, PyObject *theroot,
    VMDApp *vmdapp)
: VMDMenu(menuname, vmdapp), root(theroot), func(NULL) { 
  if (root) {
    Py_INCREF(root);
    preserve_window(root);
  }
}

VMDTkinterMenu::~VMDTkinterMenu() {
  Py_XDECREF(root);
  Py_XDECREF(func);
}

void VMDTkinterMenu::register_windowproc(PyObject *newfunc) {
  Py_XDECREF(func);
  func = newfunc;
  Py_INCREF(func);
}

void VMDTkinterMenu::do_on() {
  if (!root) {
    if (func) {
      if (!(root = PyObject_CallObject(func, NULL))) {
        // func failed; don't bother calling it again.
        Py_DECREF(func);
        func = NULL;
        return;
      }
    }
    if (root) preserve_window(root);
  }
  // root.wm_deiconify()
  if (root)
    Py_XDECREF(PyObject_CallMethod(root, (char *)"wm_deiconify", NULL));
}

void VMDTkinterMenu::do_off() {
  // root.wm_withdraw()
  if (root)
    Py_XDECREF(PyObject_CallMethod(root, (char *)"wm_withdraw", NULL));
}

void VMDTkinterMenu::move(int x, int y) {
  // root.wm_geometry(+x+y)
  if (root) {
    char buf[100];
    sprintf(buf, "+%d+%d", x, y);
    Py_XDECREF(PyObject_CallMethod(root, (char *)"wm_geometry", "s", buf));
  }
}

void VMDTkinterMenu::where(int &x, int &y) {
  // root.wm_geometry()
  if (root) {
    PyObject *result = PyObject_CallMethod(root, (char *)"wm_geometry", NULL);
    if (result) {
      char *str = PyString_AsString(result);
      int w, h;
      if (sscanf(str, "%dx%d+%d+%d", &w, &h, &x, &y) != 4) {
        fprintf(stderr, "couldn't parse output of geometry: %s\n", str);
      }
      Py_XDECREF(result);
    }
  } else {
    // give default values
    x=y=0;
  }
}

