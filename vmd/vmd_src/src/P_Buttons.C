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
 *	$RCSfile: P_Buttons.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.20 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * A Buttons is a representation for a set of n boolean inputs.  This
 * fairly abstract class should be subclassed to make Buttons objects
 * that actually know how to get their buttons.  This is somewhat
 * parallel to the Tracker object, compare them!
 *
 ***************************************************************************/

#include "P_Buttons.h"

int Buttons::start(const SensorConfig *config) {
  const ResizeArray<int> *theused = config->getsensors();
  int i;
  for(i=0; i<MAX_BUTTONS; i++)
    stat[i]=0;

  used.clear();
  for(i=0; i<theused->num(); i++) {
    used.append((*theused)[i]);
  }

  return do_start(config);
}
