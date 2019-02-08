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
 *	$RCSfile: P_VRPNButtons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.28 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#if defined(VMDVRPN)

#include "P_VRPNButtons.h"
#include "vrpn_Tracker.h"
#include "utilities.h"

/* Newer revs of VRPN require a calling convention macro */
#if !defined(VRPN_CALLBACK)
#define VRPN_CALLBACK 
#endif

/*** VRPN Event Handler Function ***/
static void VRPN_CALLBACK handle_vrpn_button(void *userdata, const vrpn_BUTTONCB b) {
  int *stat = (int *)userdata;
  stat[b.button] = b.state;
}

VRPNButtons::VRPNButtons() {
  btn = NULL;
}

int VRPNButtons::do_start(const SensorConfig *config) {
  if (btn) return 0;
  char myUSL[100];
  config->make_vrpn_address(myUSL);
  btn = new vrpn_Button_Remote(myUSL);
  btn->register_change_handler(stat, handle_vrpn_button);
  return 1;
}

void VRPNButtons::update() {
  btn->mainloop();
}

VRPNButtons::~VRPNButtons(void) {
  if(btn) {
    //    btn->unregister_change_handler(&mycon->buttondata,handle_vrpn_button);
    //    delete btn; // VRPN has broken destructors
  }
}

#endif

