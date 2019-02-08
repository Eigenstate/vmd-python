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
 *	$RCSfile: P_VRPNTracker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.26 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * A Tracker that gets its info over the net through VRPN.  There are
 * multiple Trackers using the same connection to a computer, so they
 * need to cooperate in creative ways to get sensor updates.
 *
 ***************************************************************************/
#ifndef P_VRPNTRACKER_H
#define P_VRPNTRACKER_H

#if defined(VMDVRPN)

#include "P_Tracker.h"
#include "vrpn_Tracker.h"

/// VRPN tracker position and orientation data
struct VRPNTrackerUserData {
  float *pos;
  Matrix4 *orient;
};


/// VMDTracker subclass the manage VRPN tracker devices
class VRPNTracker : public VMDTracker {
private:
  /// representations of the remote device
  vrpn_Tracker_Remote *tkr;

  /// we pass the address of this struct as user data when we register our
  /// VRPN callback function
  VRPNTrackerUserData userdata;

protected:
  virtual int do_start(const SensorConfig *);

public:
  VRPNTracker();
  ~VRPNTracker();
  const char *device_name() const { return "vrpntracker"; }
  virtual VMDTracker *clone() { return new VRPNTracker; }

  inline virtual int alive() {
    if(tkr) if(tkr->connectionPtr())
      return tkr->connectionPtr()->doing_okay();
    return 0;
  }

  virtual void update();
};

#endif

#endif
