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
 *	$RCSfile: P_VRPNFeedback.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.28 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * This is a Feedback object that sends forces over the net through VRPN.
 *
 ***************************************************************************/
#ifndef P_VRPNFEEDBACK_H
#define P_VRPNFEEDBACK_H 

#if defined(VMDVRPN)

#include "P_Feedback.h"
#include "vrpn_ForceDevice.h"  // for definitions of vrpn_float32

/// Feedback subclass for sending haptic forces to VRPN over the net
class VRPNFeedback : public Feedback {
 private:
  vrpn_ForceDevice_Remote *fdv;
  vrpn_float32 F[3], jac[3][3]; // a force starting at the origin

 protected:
  virtual int do_start(const SensorConfig *);

 public:
  VRPNFeedback();
  ~VRPNFeedback();
 
  virtual const char *device_name() const { return "vrpnfeedback"; }
  virtual Feedback *clone() { return new VRPNFeedback; }

  virtual void update();
  inline virtual int alive() { // VRPN doesn't let us *really* check this :(
    if(fdv) if(fdv->connectionAvailable()) return 1;
    return 0;
  }
  virtual void addconstraint(float k, const float *location);
  virtual void addplaneconstraint(float k, const float *point,
				  const float *normal);
  virtual void addforcefield(const float *origin, const float *force,
			     const float *jacobian);
  inline virtual void zeroforce();
  inline virtual void forceoff();
  virtual void sendforce(const float *initial_pos);
};

#endif

#endif
