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
 *	$RCSfile: MobileTracker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.1 $	$Date: 2011/06/22 19:02:47 $
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
