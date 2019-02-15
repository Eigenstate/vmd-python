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
 *	$RCSfile: SpaceballButtons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.5 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************/

#include "VMDApp.h"
#include "ResizeArray.h"
#include "P_Buttons.h"
#include "SpaceballButtons.h"
#include "Inform.h"

SpaceballButtons::SpaceballButtons(VMDApp *vmdapp) {
  app = vmdapp;
  numButtons = 31;
}

int SpaceballButtons::do_start(const SensorConfig *) {
  msgInfo << "Opening VMD console Spaceball device (buttons)." << sendmsg;
  return 1;
}

void SpaceballButtons::update() {
  float tx, ty, tz, rx, ry, rz;
  tx=ty=tz=rx=ry=rz=0.0f;
  int buttons;
  buttons=0;

  // query VMDApp spaceball for events
  if (app != NULL) {
    app->spaceball_get_tracker_status(tx, ty, tz, rx, ry, rz, buttons);
  }

  stat[0] = 0;
  for(int i = 0; i < numButtons; i++) {
    stat[i] = (buttons >> i) & 1;
    stat[0] |= (buttons >> i) & 1;
  }
}

