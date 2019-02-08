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
 *	$RCSfile: P_CaveTracker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.19 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * -dispdev cave
 *
 ***************************************************************************/

#include <cave.macros.h>
#include "CaveRoutines.h"
#include "Matrix4.h"
#include "P_Tracker.h"
#include "P_CaveTracker.h"

void CaveTracker::update() {
  CAVEGetSensorPosition(CAVESENSOR(caveTrackerNum),CAVE_NAV_FRAME, pos);
  
  /* "classical" Euler angles */
  float angles[3];
  CAVEGetSensorOrientation(CAVESENSOR(caveTrackerNum), CAVE_NAV_FRAME, angles);
  orient->identity();
  orient->rot(angles[1],'y');
  orient->rot(angles[0],'x');
  orient->rot(angles[2],'z');
  orient->rot(90,'y'); // to face forward (-z)
}

int CaveTracker::do_start(const SensorConfig *config) {
  // Must check that we are actually running in a CAVE here
  if (!vmd_cave_is_initialized() || CAVEController == NULL) {
     return 0;
  }

  if (!config->require_cave_name()) return 0;
  if (!config->have_one_sensor()) return 0;
  int num = (*config->getsensors())[0];
  caveTrackerNum = num+1;
  return 1;
}
 
  
