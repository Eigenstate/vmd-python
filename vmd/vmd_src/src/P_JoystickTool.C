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
 *	$RCSfile: P_JoystickTool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.40 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#include "P_JoystickTool.h"

JoystickTool::JoystickTool(int id, VMDApp *vmdapp, Displayable *disp) 
: Tool(id, vmdapp, disp) {
  constrained=0;
}

void JoystickTool::do_event() {
  int i;

  if(!constrained && isgrabbing()) {
    if(!target(TARGET_GRAB, pos, 0)) {
      //      constrained=-1; return; 
    }
    constrained=1;
    for(i=0;i<dimension();i++) {
      pos[i]=Tool::position()[i];
      constraint[i]=pos[i];
    }
    setconstraint(100,constraint);
    sendforce();
  }
  else if(constrained && !isgrabbing()) {
    let_go();
    constrained=0;
    forceoff();
  }

  if(constrained == 1) {
    for(i=0;i<dimension();i++)
      pos[i] += 0.1f * (Tool::position()[i]-constraint[i]);
  }
  else {
    for(i=0;i<dimension();i++)
      pos[i] = Tool::position()[i];
  }
}

const float *JoystickTool::position() const {
  if(constrained) return pos;
  else return Tool::position();
}

