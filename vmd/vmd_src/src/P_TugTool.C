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
 *	$RCSfile: P_TugTool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.54 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#include "P_TugTool.h"
#include "utilities.h"
#include "Displayable.h"

TugTool::TugTool(int id, VMDApp *vmdapp, Displayable *disp) 
: Tool(id, vmdapp, disp) {
  int i;
  for(i=0;i<3;i++) offset[i]=0;
  tugging=0;
  springscale = 1.0;
}

void TugTool::do_event() {

  if(!tugging) { // we haven't started tugging, update UIVR
    tool_location_update();
  }

  if (istugging()) {  // Tugging is enabled...
    if (!tugging || !is_targeted()) { // but we're not tugging now
      if(!target(TARGET_TUG, tugged_pos, 0)) {
        // Didn't pick anything, so return
        tugging = 0;
        return;
      }
      tugging = 1;
      // We're starting the force field, so set the offset
      vec_sub(offset, Tool::position(), tugged_pos);
      start_tug();
    }
    target(TARGET_TUG, tugged_pos, 1);
    // Apply the force field...
    float offset_tugged_pos[3]; // offset+tugged_pos
    vec_add(offset_tugged_pos,offset,tugged_pos);
    set_tug_constraint(offset_tugged_pos);
    
    // and send the proper force to UIVR for display and possible
    // export
    float diff[3];
    vec_sub(diff, Tool::position(), offset_tugged_pos);
    // diff now is in my units, but we should do it in mol units
    vec_scale(diff,dtool->scale/getTargetScale(),diff);
    // scale by the force scaling spring constant
    vec_scale(diff,forcescale,diff); 
    do_tug(diff);
  }
  else if (tugging) { // Tugging has been disabled, so turn it off.
    let_go();
    tugging = 0;
    forceoff();
    offset[0]=offset[1]=offset[2]=0;
  }
}

void TugTool::set_tug_constraint(float *cpos) {
  setconstraint(50, cpos);
  sendforce();
}

void TugTool::do_tug(float *force) {
  tug(force);
}

