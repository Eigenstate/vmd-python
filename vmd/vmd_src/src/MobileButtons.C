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
 *	$RCSfile: MobileButtons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.3 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************/
#include "VMDApp.h"
#include "ResizeArray.h"
#include "P_Buttons.h"
#include "MobileButtons.h"
#include "Inform.h"

MobileButtons::MobileButtons(VMDApp *vmdapp) {
  app = vmdapp;
  numButtons = 31;
}

int MobileButtons::do_start(const SensorConfig *) {
  msgInfo << "Opening VMD console Mobile device (buttons)." << sendmsg;
  return 1;
}

void MobileButtons::update() {
  float tx, ty, tz, rx, ry, rz;
  tx=ty=tz=rx=ry=rz=0.0f;
  int buttons;
  buttons=0;

  // query VMDApp for events
  if (app != NULL) {
    app->mobile_get_tracker_status(tx, ty, tz, rx, ry, rz, buttons);
  }

  stat[0] = 0;
  for(int i = 0; i < numButtons; i++) {
    stat[i] = (buttons >> i) & 1;
    stat[0] |= (buttons >> i) & 1;
  }
}

