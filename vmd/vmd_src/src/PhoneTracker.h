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
 *	$RCSfile: PhoneTracker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.5 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A Tracker that gets its info from a WiFi smartphone
 *
 ***************************************************************************/
#ifndef PHONETRACKER_H
#define PHONETACKER_H

#include "P_Tracker.h"
class VMDApp;

/// VMDTracker subclass that gets its info from the Phone driver
class PhoneTracker : public VMDTracker {
private:
  VMDApp *app;
  float transInc, rotInc, scaleInc;

protected:
  virtual int do_start(const SensorConfig *);
 
public:
  PhoneTracker(VMDApp *);
  ~PhoneTracker();
  virtual const char *device_name() const { return "phonetracker"; } 
  virtual VMDTracker *clone() { return new PhoneTracker(app); }

  virtual void update();
  inline virtual int alive() { return 1; }
};

#endif
