/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the      
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: P_FreeVRTracker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.17 $	$Date: 2016/11/28 03:05:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 * Representation of the Tracker in FreeVR
 ***************************************************************************/

#include "P_Tracker.h"

class VMDApp;

/// VMDTracker subclass that interfaces to the FreeVR wand
class FreeVRTracker : public VMDTracker {
 private:
  VMDApp *app;

 public:
  inline FreeVRTracker(VMDApp *vmdapp) : VMDTracker() { app=vmdapp; };
  virtual VMDTracker *clone() { return new FreeVRTracker(app); }
  const char *device_name() const { return "freevrtracker"; }
  virtual void update();
  inline virtual int alive() { return 1; }

 protected:
  virtual int do_start(const SensorConfig *);
};

