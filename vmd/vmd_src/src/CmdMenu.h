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
 *      $RCSfile: CmdMenu.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.33 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *     The menu commands are defined here.  These tell the app to turn a
 * menu on or off; to move a menu; or print the cooordinates of a menu
 * to infoMsg.  A "menu" is a UIObject that has a window.
 *
 ***************************************************************************/
#ifndef CMDMENU_H
#define CMDMENU_H

#include "Command.h"
#include "utilities.h"

/// turn a menu on/off
class CmdMenuShow : public Command {
public:
  int turnOn;		// if T, turn on; if F, turn off
  char *menuname;       // name of menu to switch

protected:
  virtual void create_text(void);

public:
  CmdMenuShow(const char *name, int turnon);
  ~CmdMenuShow(void);
};


/// Add an item to the VMD extension menu
class CmdMenuExtensionAdd : public Command {
public:
  char *name;
  char *menupath;
  CmdMenuExtensionAdd(const char *aName, const char *aPath) 
  : Command(Command::MENU_TK_ADD) {
    name = stringdup(aName);
    menupath = stringdup(aPath);
  }
  ~CmdMenuExtensionAdd() {
    delete [] name;
    delete [] menupath;
  }
};

/// Remove an item from the VMD extension menu
class CmdMenuExtensionRemove : public Command {
public:
  char *name;
  CmdMenuExtensionRemove(const char *aName) 
  : Command(Command::MENU_TK_REMOVE) {
    name = stringdup(aName);
  }
  ~CmdMenuExtensionRemove() {
    delete [] name;
  }
};

#endif


