/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: P_Tracker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.22 $	$Date: 2010/12/16 04:08:32 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#include <string.h>
#include "Matrix4.h"
#include "P_Tracker.h"

VMDTracker::VMDTracker() {
  dim = 3;
  pos[0]=pos[1]=pos[2]=0;
  orient = new Matrix4();
  orient->identity();
}

VMDTracker::~VMDTracker() {
  delete orient;
}

int VMDTracker::start(const SensorConfig *config) {
  set_scale(config->getscale());
  set_offset(config->getoffset());
  set_right_rot(config->getright_rot());
  set_left_rot(config->getleft_rot());
  return do_start(config);
}

  
