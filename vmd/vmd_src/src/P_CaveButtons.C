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
 *	$RCSfile: P_CaveButtons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.20 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#include <cave.macros.h>
#include "CaveRoutines.h"
#include "ResizeArray.h"
#include "P_Buttons.h"
#include "P_CaveButtons.h"

CaveButtons::CaveButtons() {
  numButtons = 0; // no buttons until we know the CAVE is running
}

int CaveButtons::do_start(const SensorConfig *) {
  if (!vmd_cave_is_initialized() || CAVEController == NULL) {
    return 0;     // return false; cannot run without CAVE environment
  }
  numButtons = CAVEController->num_buttons;
  return 1;
}

void CaveButtons::update() {
  for(int i = 0; i < numButtons; i++) {
    stat[i] = (CAVEController->button[i]);
  }
}

