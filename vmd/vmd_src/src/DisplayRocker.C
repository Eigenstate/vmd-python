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
 *	$RCSfile: DisplayRocker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.9 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************/

#include "DisplayRocker.h"

DisplayRocker::DisplayRocker(Displayable *d)
: Displayable(d) {
  Rocking = rockOnce = FALSE;
  rockSteps = currRockStep = 0;
  rockAmount = 0.0;
  rockAxis = 'y';
}

///////////////////////////  public routines 

// 'rock' the scene, by moving it through a given arc (possibly just a
// continuous circle), with a specified number of steps.
// There are two ways:
//   a) doOnce == TRUE: the rocking is done once, from a --> b
//   b) doOnce == FALSE: the rocking is done until told to stop,
//      from (a+b)/2 --> b --> a --> b .....
// Note that if steps < 0, the rocking is continuous, that is no ending point
// is specified so the rotation is continually in one direction.  In this
// case doOnce means nothing, and is automatically used as if it were FALSE.
void DisplayRocker::start_rocking(float a, char ax, int steps, int doOnce) {
  // set rocking parameters
  Rocking = TRUE;
  rockSteps = steps;  // if < 0, continuous
  rockAmount = a;
  rockAxis = ((ax >= 'x' && ax <= 'z') ? ax : 'y');

  // when currRockStep == rockSteps, flip rockAmount or stop
  rockOnce = (doOnce && steps >= 0);
  currRockStep = (rockOnce ? 0 : (int)( ((float)steps)/2.0 ));
}

void DisplayRocker::prepare() {
  if (Rocking) {
    parent->add_rot(rockAmount, rockAxis);
    if (rockSteps >= 0 && ++currRockStep >= rockSteps) {
      currRockStep = 0;
      rockAmount *= -1.0;		// reverse direction of rocking
      if (rockOnce)
        stop_rocking();			// rocked once; now quit
    }
  }
}

