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
 *	$RCSfile: Animation.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.55 $	$Date: 2010/12/16 04:08:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Animation class, which stores a list of pointers to Timestep objects
 * that contain 3D coordinates and other data that varies with time.
 *
 ***************************************************************************/

#include "Animation.h"
#include "VMDApp.h"
#include "CmdAnimate.h"
#include <stdio.h>

// strings describing animation styles
const char *animationStyleName[Animation::ANIM_TOTAL_STYLES] = {
  "Once", "Loop", "Rock" 
};
const char *animationDirName[Animation::ANIM_TOTAL_DIRS] = {
  "forward", "next", "reverse", "prev", "pause"
};


// constructor
Animation::Animation(VMDApp *app_) : UIObject(app_), mlist(*app->moleculeList) {
  lastTime = time_of_day();       // get the current time for determining speed
  speed(1.0);                     // set initial speed to maximum
  skip(1);                        // don't skip any frames yet
  anim_dir(ANIM_PAUSE);           // not yet animating
  anim_style(ANIM_LOOP);          // loop back to beginning when end reached
}


void Animation::goto_frame(int fr) {
  for (int i=0; i<mlist.num(); i++) {
    Molecule *m = mlist.molecule(i);
    if (m->active) {
      // XXX backward compatibility lameness
      int theframe = fr;
      if (fr == -1) 
        theframe = 0;
      else if (fr == -2) 
        theframe = m->numframes();
      m->override_current_frame(theframe);
      m->change_ts();
    }
  }
}


// update the animation list based on current mode; return if curr frame change
int Animation::check_event() {
  // if we're paused, do nothing
  if (animDir == ANIM_PAUSE) 
    return 0;

  // other animation modes depend on the delay.  Check if a sufficient
  // delay has elapsed.
  double curTime = time_of_day();
  double dt = curTime - lastTime;
  if (dt <= (SPEED_FACTOR - Speed)) 
    return 0;

  // time to update frames.  Cache the current time before proceeding.
  lastTime = curTime;
  int curframe = frame();
  int n = num();

  // nothing to do if there are fewer than two frames
  if (n < 2) 
    return 0;

  // skip the current frame ahead the proper amount
  for (int i=0; i < frameSkip; i++) {
    if (animDir == ANIM_REVERSE || animDir == ANIM_REVERSE1) {
      if (curframe <= 0) {
        if (animStyle == ANIM_LOOP) {
          curframe = n-1;
        } else if (animStyle == ANIM_ROCK) {
          animDir = ANIM_FORWARD;
        } else if (animStyle == ANIM_ONCE) {
          animDir = ANIM_PAUSE;
        }
      } else {
        --curframe;
      }
    } else if (animDir == ANIM_FORWARD || animDir == ANIM_FORWARD1) {
      if (curframe >= n-1) {
        if (animStyle == ANIM_LOOP) {
          curframe = 0;
        } else if (animStyle == ANIM_ROCK) {
          animDir = ANIM_REVERSE;
        } else if (animStyle == ANIM_ONCE) {
          animDir = ANIM_PAUSE;
        }
      } else {
        ++curframe;
      }
    }
  }

  goto_frame(curframe);
  // we generated an event, so let other UIs know about it.
  runcommand(new CmdAnimNewFrame);

  // these two modes stop after one action
  if (animDir == ANIM_FORWARD1 || animDir == ANIM_REVERSE1)
    animDir = ANIM_PAUSE;
  
  return 1;
}


void Animation::skip(int newsk) { 
  frameSkip = ( newsk >= 1 ? newsk : 1);
}


float Animation::speed(float newsp)  {
  if (newsp < 0.0)
    Speed = 0.0;
  else if (newsp > 1.0)
    Speed = SPEED_FACTOR;
  else
    Speed = newsp*SPEED_FACTOR;

  return Speed;
}


void Animation::anim_style(AnimStyle as) { 
  animStyle = as; 
}







