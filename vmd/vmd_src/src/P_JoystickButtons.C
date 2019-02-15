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
 *	$RCSfile: P_JoystickButtons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.18 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#ifdef WINGMAN

#include <stdio.h>
#include "Inform.h"
#include "ResizeArray.h"

extern "C" {
#include "wingforce.h"
}
#include "P_Buttons.h"
#include "P_JoystickButtons.h"

JoystickButtons::JoystickButtons() {
  joy = NULL;
}

int JoystickButtons::do_start(const SensorConfig *config) {
  if (joy) return 0;
  const char *name = config->getname();
  char joyname[100];
  if(name[0]!='/') snprintf(joyname,100,"/%s",name);
  else snprintf(joyname,100,"%s",name);
  fprintf(stderr,"Opening joystick: %s\n",joyname);
  joy = joy_open(joyname);
  return (joy != NULL);
}

JoystickButtons::~JoystickButtons() {
  joy_close(joy);
}

void JoystickButtons::update() {
  int x, y, t, hat, buttons;
  int i;

  if(joy==NULL) return;

  joy_getstatus(joy, &x, &y, &t, &hat, &buttons);
  
  for(i=0;i<MAX_BUTTONS;i++) {
    stat[i] = buttons%2;
    buttons >>= 1;
  }
}

#endif //WINGMAN
