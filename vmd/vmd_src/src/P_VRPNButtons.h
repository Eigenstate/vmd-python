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
 *	$RCSfile: P_VRPNButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.27 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * This is a Buttons that gets its info over the net from VRPN.
 *
 ***************************************************************************/

#ifndef P_VRPN_BUTTONS_H
#define P_VRPN_BUTTONS_H

#if defined(VMDVRPN)

#include "P_Buttons.h"
#include "vrpn_Button.h"
#include "ResizeArray.h"

/// Buttons subclass that gets its info over the net from VRPN
class VRPNButtons : public Buttons {
private:
  vrpn_Button_Remote *btn; 

protected:
  virtual int do_start(const SensorConfig *);

public:
  VRPNButtons();
  ~VRPNButtons();
  
  virtual const char *device_name() const { return "vrpnbuttons"; }
  virtual Buttons *clone() { return new VRPNButtons; }

  virtual void update();

  inline virtual int alive() {
    if(btn) if(btn->connectionPtr())
      return btn->connectionPtr()->doing_okay();
    return 0;
  }
};

#endif

#endif
