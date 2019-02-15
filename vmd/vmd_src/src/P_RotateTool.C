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
 *	$RCSfile: P_RotateTool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.46 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#ifdef VMDVRPN

#include <math.h>

#include "Matrix4.h"
#include "quat.h"
#include "utilities.h"
#include "Displayable.h"
#include "Command.h"
#include "UIObject.h"
#include "JString.h"

#include "P_Buttons.h"
#include "P_Feedback.h"
#include "P_Tracker.h"
#include "P_Tool.h"
#include "P_RotateTool.h"
#include "P_UIVR.h"

int RotateTool::isgrabbing() {
  if(grab_toggle && constrained != -1) return 1;
  else return 0;
}

RotateTool::RotateTool(int id, VMDApp *vmdapp, Displayable *disp) 
: Tool(id,vmdapp, disp) {
  grab_toggle = 0;
  button_was_down = 0;
  constrained = 0;
}

void RotateTool::do_event() {
  int i,j;
  float newpos[3], point[3], normal[3], cpoint[3];
  q_vec_type qpoint, qnormal, last_qpoint;
  q_type rot;
  q_matrix_type qmat;
  Matrix4 mat;

  if(!wasgrabbing && isgrabbing()) {
    target(TARGET_GRAB, newpos, 0);
  }
  wasgrabbing = isgrabbing();

  if (orientation() == NULL) {
    return;
  }

  if(button_was_down && !Tool::isgrabbing()) {
    grab_toggle = !grab_toggle;
    button_was_down = 0;
  }
  else if(!button_was_down && Tool::isgrabbing()) {
    button_was_down = 1;
  }

  if(!constrained && (Tool::isgrabbing() || grab_toggle)) {
    q_make(qoffset,1,0,0,0);

    // get the center of rotation
    if(!target(TARGET_GRAB, rotatecenter, 0)) {
      constrained = -1;
      return;
    }

    // compute the radius of rotation
    rotateradius = distance(Tool::position(), rotatecenter);

    // record the current position, direction and normal to the sphere
    for(i=0;i<dimension();i++) {
      old_pos[i] = Tool::position()[i];
      old_normal[i] = old_pos[i]/rotateradius;
      qpoint[i] = (double)orientation()->mat[4*0+i];
      qnormal[i] = -(double)old_normal[i];
    }

    // set up the starting position
    for(i=0;i<4;i++) for(j=0;j<4;j++) 
     start.mat[4*i+j] = orientation()->mat[4*i+j];

    // and rotate it to be normal, just for show
    q_from_two_vecs(rot,qpoint,qnormal);
    q_to_row_matrix(qmat,rot);
    for(i=0;i<4;i++) for(j=0;j<4;j++) mat.mat[4*i+j]=(float) qmat[i][j];
    mat.multmatrix(start);
    start=mat;

    // set this flag last so it doesn't affect orientation computations
    constrained=1;
  }
  else if(constrained && !(Tool::isgrabbing() || grab_toggle)){
    let_go();
    constrained=0;
    forceoff();
  }

  if(constrained == 1) {
    float dist;
    // set up the position we are reporting
    dist = distance(Tool::position(),rotatecenter) / rotateradius;
    for(i=0;i<dimension();i++) {
      constrainedpos[i] = rotatecenter[i] +
	(Tool::position()[i] - rotatecenter[i])/dist;
    }

    // constrain the position to a plane
    vec_sub(normal,Tool::position(),rotatecenter);
    vec_normalize(normal);
    vec_scale(cpoint,rotateradius,normal);
    vec_add(point,rotatecenter,cpoint);
    setplaneconstraint(100,point,normal);
    
    // now add detentes (!)
    for(i=0;i<3;i++) {
      dist = (constrainedpos[i] - rotatecenter[i])/rotateradius;
      if(dist < 0.1 && dist > -0.1) {
	float normal[3]={0,0,0};
	normal[i]=1;
	addplaneconstraint(100,rotatecenter,normal);      
      }
    }

    sendforce();

    // compute the difference between the old normal and the new;
    // rotate the offset matrix by this amount.
    for(i=0;i<dimension();i++) {
      qpoint[i] = normal[i];
      last_qpoint[i] = old_normal[i];
      old_normal[i] = normal[i];
    }
    q_from_two_vecs(rot,last_qpoint,qpoint);
    q_mult(qoffset,rot,qoffset);
    q_to_row_matrix(qmat,qoffset);
    for(i=0;i<4;i++) {
      for(j=0;j<4;j++) {
	mat.mat[4*i+j]=(float) qmat[i][j];
      }
    }
    offset = mat;
  }
}

const float *RotateTool::position() const {
  const float *newpos = Tool::position();
  if(constrained != 1) return newpos;
  return constrainedpos;
}

const Matrix4 *RotateTool::orientation() {
  if(constrained != 1) {
    return Tool::orientation();
  }
  product = offset;
  product.multmatrix(start);
  return &product;
}

#endif
