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
 *	$RCSfile: CmdMenu.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.34 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *     The menu commands are defined here.  These tell the app to turn a
 * menu on or off; to move a menu; or print the cooordinates of a menu
 * to infoMsg.  A "menu" is a UIObject that has a window.
 *
 ***************************************************************************/

#include "CmdMenu.h"
#include "utilities.h" // for stringdup

//////////// turn a menu on/off

void CmdMenuShow::create_text(void) {
  *cmdText << "menu " << menuname << (turnOn ? " on" : " off");
  *cmdText << ends;
}

CmdMenuShow::CmdMenuShow(const char *name, int turnon )
  : Command(Command::MENU_SHOW)  {
  turnOn = turnon;
  menuname = stringdup(name);
}

CmdMenuShow::~CmdMenuShow(void) {
  delete [] menuname;
}

