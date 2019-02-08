/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef VMDTKINTERMENU_H
#define VMDTKINTERMENU_H

#if defined(__APPLE__)
// use the Apple-provided Python framework
#include "Python/Python.h"
#else
#include "Python.h"
#endif

#include "VMDMenu.h"

/// VMDMenu subclass to manage Tkinter extension menus added by the user
class VMDTkinterMenu: public VMDMenu {
private:
  // handle to the Tk() instance
  PyObject *root;
  // window creation function
  PyObject *func;

protected:
  /// virtual routines to handle toolkit-specific actions
  virtual void do_on();
  virtual void do_off();

public:
  /// class constructor and destructor
  VMDTkinterMenu(const char *menuname, PyObject *root, VMDApp *);
  virtual ~VMDTkinterMenu();

  /// pass a callable object which returns a root window handle.
  void register_windowproc(PyObject *func);

  /// Move the menu to a new place on the screen
  virtual void move(int, int);

  /// return the current location of the form
  virtual void where(int &, int &);
};

#endif

