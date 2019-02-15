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
 *      $RCSfile: VMDTkMenu.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.13 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Class to manage Tk extension menus added by the user.
 ***************************************************************************/

#ifndef VMDTKMENU_H
#define VMDTKMENU_H

#include "VMDMenu.h"
#include <tcl.h>

/// VMDMenu subclass to manage Tk extension menus added by the user
class VMDTkMenu: public VMDMenu {
protected:
  Tcl_Interp *interp;   ///< Tcl interpreter handle
  char *path;           ///< Tk window path, i.e. .zoomseq

  /// Name of proc which returns a window handle; can be used to defer
  /// window creation until the first time the menu is turned on.
  char *windowProc;

  /// registers newly-created window with the window manager
  void create_window();

  /// virtual routines to handle toolkit-specific actions
  virtual void do_on();
  virtual void do_off();

public:
  /// class constructor and destructor
  VMDTkMenu(const char *menuname, const char *windowpath, 
            VMDApp *, Tcl_Interp *);
  virtual ~VMDTkMenu();

  /// register a window creation proc.  Return success.
  int register_proc(const char *newproc);

  /// Move the menu to a new place on the screen
  virtual void move(int, int);

  /// return the current location of the form
  virtual void where(int &, int &);
};

#endif

