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
 *      $RCSfile: VMDTkMenu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.20 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Class to manage Tk extension menus registered by the user.
 ***************************************************************************/

#include "VMDTkMenu.h"
#include "utilities.h"
#include "JString.h"
#include "Inform.h"

VMDTkMenu::VMDTkMenu(const char *menuname, const char *pathname, 
                     VMDApp *vmdapp, Tcl_Interp *tclinterp)
: VMDMenu(menuname, vmdapp) {

  interp = tclinterp;
  windowProc = NULL;
  path = NULL;
  if (pathname) {
    path = stringdup(pathname); // allocated via new []
    create_window();
  }
}

void VMDTkMenu::create_window() {
  // register creation of this window.  Gives the callback function a 
  // chance to configure window manager options for this window so it 
  // behaves well.
  JString buf("vmd_tkmenu_cb ");
  buf += path;
  buf += " create ";
  buf += get_name(); 
  if (Tcl_Eval(interp, (char *)(const char *)buf) != TCL_OK) {
    msgErr << "Error creating Tk extension window " << get_name() << sendmsg;
    msgErr << Tcl_GetStringResult(interp) << sendmsg;
  }
}

int VMDTkMenu::register_proc(const char *proc) {
  if (!proc) return FALSE;
  delete [] windowProc;
  windowProc = stringdup(proc);
  return TRUE;
}

VMDTkMenu::~VMDTkMenu() {
  if (path) {
    JString buf("vmd_tkmenu_cb ");
    buf += path;
    buf += " remove";
    Tcl_Eval(interp, (char *)(const char *)buf); 
    delete [] path; // must be freed with delete []
  }

  delete [] windowProc;
}

void VMDTkMenu::do_on() {
  if (!path) {
    // try to get the window handle from the windowProc
    if (!windowProc) {
      return;
    }
    if (Tcl_Eval(interp, windowProc) != TCL_OK) {
      msgErr << "Creation of window for '" << get_name() << "' failed (" 
             << Tcl_GetStringResult(interp) << ")." << sendmsg;
      delete [] windowProc;
      windowProc = NULL;
      return;
    }
    path = stringdup(Tcl_GetStringResult(interp));
    create_window();
  }
  JString buf("vmd_tkmenu_cb ");
  buf += path;
  buf += " on";
  Tcl_Eval(interp, (char *)(const char *)buf);
}

void VMDTkMenu::do_off() {
  if (!path) return;
  JString buf("vmd_tkmenu_cb ");
  buf += path;
  buf += " off";
  Tcl_Eval(interp, (char *)(const char *)buf);
}

void VMDTkMenu::move(int x, int y) {
  if (!path) return;
  char numbuf[20];
  sprintf(numbuf, "%d %d", x, y);
  JString buf("vmd_tkmenu_cb ");
  buf += path;
  buf += " move ";
  buf += numbuf;
  Tcl_Eval(interp, (char *)(const char *)buf);
}

void VMDTkMenu::where(int &x, int &y) {
  if (!path) { x=y=0; return; }
  JString buf("vmd_tkmenu_cb ");
  buf += path;
  buf += " loc";
  Tcl_Eval(interp, (char *)(const char *)buf); 
  Tcl_Obj *result = Tcl_GetObjResult(interp);
  int objc;
  Tcl_Obj **objv;
  if (Tcl_ListObjGetElements(interp, result, &objc, &objv) != TCL_OK ||
      objc != 2 || 
      Tcl_GetIntFromObj(interp, objv[0], &x) ||
      Tcl_GetIntFromObj(interp, objv[1], &y)) {
    x = y = 0;
  }
}

