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
 *	$RCSfile: MobileTracker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.3 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A Tracker that gets its info from a WiFi smartphone, tablet, etc.
 *
 ***************************************************************************/
#ifndef MOBILETRACKER_H
#define MOBILETRACKER_H

#include "P_Tracker.h"
class VMDApp;

/// VMDTracker subclass that gets its info from the Mobile driver
class MobileTracker : public VMDTracker {
private:
  VMDApp *app;
  float transInc, rotInc, scaleInc;

protected:
  virtual int do_start(const SensorConfig *);
 
public:
  MobileTracker(VMDApp *);
  ~MobileTracker();
  virtual const char *device_name() const { return "mobiletracker"; } 
  virtual VMDTracker *clone() { return new MobileTracker(app); }

  virtual void update();
  inline virtual int alive() { return 1; }
};

#endif
