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
 *	$RCSfile: P_GrabTool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.34 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#include "P_GrabTool.h"
#include "utilities.h"
#include "Matrix4.h"

GrabTool::GrabTool(int id, VMDApp *vmdapp, Displayable *disp) 
: Tool(id, vmdapp, disp) {
  targetting=0;
}

void GrabTool::do_event() {
  float mypos[3], kick[3]={.5,0,0}; /// XXX user configurable
  float zero[9]={0,0,0, 0,0,0, 0,0,0};
  Matrix4 o;

  if (!position()) 
    return;

  if (!wasgrabbing && isgrabbing()) {
    if (target(TARGET_GRAB, mypos, 0)) {
      o = *orientation();
      vec_copy(mypos, position());
      o.multpoint3d(kick, kick);

      setforcefield(position(), kick, zero);
      sendforce();
      
      targetting = 1;
    } else { 
      targetting=0;
    }
  } else if (!isgrabbing() && wasgrabbing && targetting) {
    o = *orientation();
    vec_copy(mypos, position());
    o.multpoint3d(kick, kick);
    vec_negate(kick, kick);

    setforcefield(position(), kick, zero);
    sendforce();
  } else {
    forceoff();
  }
  wasgrabbing = isgrabbing();
}

