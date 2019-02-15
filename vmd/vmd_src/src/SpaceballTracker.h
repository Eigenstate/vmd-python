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
 *	$RCSfile: SpaceballTracker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.17 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * A Tracker that gets its info from the Spaceball driver.
 *
 ***************************************************************************/
#ifndef SPACEBALLTRACKER_H
#define SPACEBALLTRACKER_H

#include "P_Tracker.h"
#if defined(VMDLIBSBALL)
#include "sball.h"
#endif
class VMDApp;

/// VMDTracker subclass that gets its info from the Spaceball driver
class SpaceballTracker : public VMDTracker {
private:
  VMDApp *app;
  int uselocal;                      ///< use the main VMD spaceball
#if defined(VMDLIBSBALL)
  SBallHandle sball;                 ///< handle from spaceball I/O library
#endif
  float transInc, rotInc, scaleInc;

protected:
  virtual int do_start(const SensorConfig *);
 
public:
  SpaceballTracker(VMDApp *);
  ~SpaceballTracker();
  virtual const char *device_name() const { return "sballtracker"; } 
  virtual VMDTracker *clone() { return new SpaceballTracker(app); }

  virtual void update();
  inline virtual int alive() { return 1; }
};

#endif
