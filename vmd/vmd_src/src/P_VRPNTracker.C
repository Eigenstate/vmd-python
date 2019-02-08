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
 *	$RCSfile: P_VRPNTracker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.29 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#if defined(VMDVRPN)

#include "P_VRPNTracker.h"

#include "quat.h"
#include "Matrix4.h"
#include "utilities.h"

/* Newer revs of VRPN require a calling convention macro */
#if !defined(VRPN_CALLBACK)
#define VRPN_CALLBACK
#endif


static void VRPN_CALLBACK handle_vrpn_tracker(void *userdata, const vrpn_TRACKERCB t) {
  VRPNTrackerUserData *data = (VRPNTrackerUserData *)userdata;

  data->pos[0] = (float) t.pos[0];
  data->pos[1] = (float) t.pos[1];
  data->pos[2] = (float) t.pos[2];

  double q[4]; /* we need to swap X and Z to make it work right */
  q[0] = -t.quat[Q_Z];
  q[1] =  t.quat[Q_Y];
  q[2] =  t.quat[Q_X];
  q[3] =  t.quat[Q_W];

  double rotation[16];
  Matrix4 *orient = data->orient;
  int i;
  q_to_ogl_matrix(rotation, q);
  for(i=0;i<4;i++) orient->mat[4*0+i]=(float)rotation[i];
  for(i=0;i<4;i++) orient->mat[4*1+i]=(float)rotation[i+4];
  for(i=0;i<4;i++) orient->mat[4*2+i]=(float)rotation[i+8];
  for(i=0;i<4;i++) orient->mat[4*3+i]=(float)rotation[i+12];
}

VRPNTracker::VRPNTracker() {
  tkr = NULL;

  // set userdata to point to parent class data 
  userdata.pos = pos;
  userdata.orient = orient;
}

int VRPNTracker::do_start(const SensorConfig *config) {
  if (tkr) return 0;
  if (!config->have_one_sensor()) return 0;

  char myUSL[100];
  config->make_vrpn_address(myUSL);
  int mysensor = (*config->getsensors())[0];

  // create new tracker connection
  tkr = new vrpn_Tracker_Remote(myUSL);

  // register callback for position and orientation updates
  tkr->register_change_handler(&userdata, handle_vrpn_tracker, 
                               mysensor);
  return 1;
}

VRPNTracker::~VRPNTracker(void) {
  delete tkr;
}

void VRPNTracker::update() {

  if(!alive()) { // insert hack here to fix VRPN's connection bug someday
    moveto(0,0,0);
    orient->identity();
    return;
  }
  tkr->mainloop(); // clear out old data
}

#endif


