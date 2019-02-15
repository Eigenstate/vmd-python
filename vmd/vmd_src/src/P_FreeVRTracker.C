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
 *	$RCSfile: P_FreeVRTracker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.18 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 * -dispdev freevr
 ***************************************************************************/

#include "Matrix4.h"
#include "P_Tracker.h"
#include "P_FreeVRTracker.h"
#include <freevr.h>

void FreeVRTracker::update() {
  // FreeVR Wand is sensor 1
  #define WAND_SENSOR     1

  vrPoint wand_location;
  vrPointGetRWFrom6sensor(&wand_location, WAND_SENSOR);
  pos[0] = wand_location.v[0];
  pos[1] = wand_location.v[1];
  pos[2] = wand_location.v[2];

  /* get Euler angles for wand orientation */
  vrEuler wand_orientation;
  vrEulerGetRWFrom6sensor(&wand_orientation, WAND_SENSOR);

  orient->identity();
  orient->rot(wand_orientation.r[0],'x');
  orient->rot(wand_orientation.r[1],'y');
  orient->rot(wand_orientation.r[2],'z');
  orient->rot(90,'y'); // to face forward (-z)
}

int FreeVRTracker::do_start(const SensorConfig *config) {
  // Must check that we are actually running in FreeVR here; if not, return 0.
  if (!config->require_freevr_name()) return 0;
  if (!config->have_one_sensor()) return 0;
  return 1;
}
