/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/
// Tool subclass implementing a function similar to TugTool 
// except that the force is only applied in the direction
// the tool is oriented.

#include "P_PinchTool.h"
#include "Matrix4.h"
#include "utilities.h"

PinchTool::PinchTool(int id, VMDApp *vmdapp, Displayable *disp) 
: Tool(id, vmdapp, disp) {
  int i;
  for(i=0;i<3;i++) offset[i]=0;
  tugging=0;
  springscale = 1.0;
}

void PinchTool::do_event() {
  float zero[3] = {0,0,0};

  if (istugging()) {  // Tugging is enabled...
    if (!tugging) {   // but we're not tugging yet
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

    // set a plane constraint corresponding to the force the TugTool would
    // apply, but dotted into the orientation of the controller.
    const Matrix4 *o = orientation();
    float zaxis[] = {1,0,0};
    float orientaxis[3];
    o->multpoint3d(zaxis, orientaxis);
    setplaneconstraint(50, offset_tugged_pos, orientaxis);  
    sendforce();
 
    // and send the proper force to UIVR for display and possible export
    float diff[3];
    vec_sub(diff, Tool::position(), offset_tugged_pos);
    vec_scale(diff,40000 * forcescale*springscale,diff); 
    float res[3];
    vec_scale(res, dot_prod(diff, orientaxis), orientaxis);
    tug(res);
  }
  else if (tugging) { // Tugging has been disabled, so turn it off.
    tug(zero);
    let_go();
    tugging = 0;
    forceoff();
    offset[0]=offset[1]=offset[2]=0;
  }
}
