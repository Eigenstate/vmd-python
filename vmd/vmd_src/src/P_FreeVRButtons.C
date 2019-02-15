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
 *	$RCSfile: P_FreeVRButtons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.20 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 ***************************************************************************/

#include <freevr.h>
#include "P_FreeVRButtons.h"
#include "VMDApp.h"
#include "TextEvent.h"
#include "CommandQueue.h"
#include "Inform.h"

int FreeVRButtons::do_start(const SensorConfig *) {
  // XXX Somehow check that a FreeVR environment exists.  If it doesn't,
  // return false.
  if ( 0 ) {
    return 0;  // no FreeVR, cannot run FreeVR buttons.
  }
  return 1;    // FreeVR is active.
}

void FreeVRButtons::update() {
  // the mapping of buttons is different in FreeVR than CAVElib
  stat[0]=vrGet2switchValue(1);
  stat[1]=vrGet2switchValue(2);
  stat[2]=vrGet2switchValue(3);
  stat[3]=vrGet2switchValue(0);

  // XXX Bill wants to send button events to user-defined scripts...
  // 4-6 are in one of the default configs as user buttons.
  int i; 
  for (i=4; i<11; i++) {
#if 1
    if (vrGet2switchDelta(i)) {  // report state only when it changes 
#else
    if (vrGet2switchValue(i)) {  // report current state on every query
#endif
//      msgInfo << "Generated AUX event for FreeVR button[" << i << "]" << sendmsg;    
      Command *cmd = new UserKeyEvent(DisplayDevice::WIN_KBD, ((char) '0'+i), (int) DisplayDevice::AUX);
      app->commandQueue->runcommand(cmd);
    }
  }
}

