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
 *	$RCSfile: UIObject.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.43 $	$Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * User Interface Object base class.  All user interface modules are derived
 * from this; it provides methods for registering with the command processor
 * and 'signing up' for which commands it is interested in, as well as
 * generating commands and events.
 *
 ***************************************************************************/

#include "UIObject.h"
#include "CommandQueue.h"
#include "Command.h"
#include "Inform.h"
#include "utilities.h"
#include "VMDApp.h"

// class constructor: list to register with and name
UIObject::UIObject(VMDApp *vmdapp) {
  app = vmdapp; 
  cmdQueue = app->commandQueue;

  // register this object
  cmdQueue->register_UI(this);
  maxCmds = Command::TOTAL;
  
  // init the command flag array  
  doCmd = new char[maxCmds];
  for(int i=0; i < maxCmds; doCmd[i++] = FALSE);

  make_callbacks = FALSE;
  is_on = FALSE;
}


UIObject::~UIObject(void) {
  cmdQueue->unregister_UI(this); 
  delete [] doCmd;
}

// note that we are/are not interested in a command
void UIObject::command_wanted(int cmd) {
  if(cmd >= 0 && cmd < maxCmds)
    doCmd[cmd] = TRUE;
}

void UIObject::runcommand(Command *cmd) {
  cmdQueue->runcommand(cmd);
}

