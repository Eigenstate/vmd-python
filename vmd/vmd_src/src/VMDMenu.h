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
 *      $RCSfile: VMDMenu.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  VMD menu base class.
 ***************************************************************************/
#ifndef VMDMENU_H
#define VMDMENU_H

#include "UIObject.h"
#include "utilities.h"

/// base class for all GUI menus in VMD
class VMDMenu: public UIObject {
private:
  char *name;   ///< name of the manu

protected:
  // virtual routines to handle toolkit-specific actions
  virtual void freeze() {}   ///< freeze the menu (suspend updates/events)
  virtual void unfreeze() {} ///< thaw the menu (allow updates)

public:
  VMDMenu(const char *, VMDApp *);
  virtual ~VMDMenu();

  const char *get_name() { return name; } ///< return menu name

  /// Move the menu to a new place on the screen
  virtual void move(int, int) = 0;

  /// return the current location of the form
  virtual void where(int &, int &) = 0;
    
  /// This will make the "molno"-th molecule be the selected one
  /// in the VMDMenu. The VMDMenu will return TRUE if it has processed this
  /// event, and FALSE if it ignores it. The first molecule starts at 0.
  virtual int selectmol(int molno) {return FALSE;};
};

#endif

