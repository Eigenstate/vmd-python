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
 *	$RCSfile: P_RotateTool.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.35 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * This tool, meant for use with a haptic device, is useful for
 * accurate rotations.  Pressing the button grabs the molecule and
 * fixes you to the surface of a sphere.  Moving around its surface
 * rotates as if you are grabbing a big trackball.  I want to add 3d
 * detentes!
 *
 ***************************************************************************/
#ifdef VMDVRPN
#include "P_Tool.h"

/// Tool subclass implementing a rotational orientation control with 
/// haptic feedback which constrains the position of the pointer to the
/// surface of a sphere while it is being manipulated
class RotateTool : public Tool {
public:
  RotateTool(int id, VMDApp *, Displayable *);
  virtual void do_event();
  virtual const float *position() const;
  virtual int isgrabbing();
  virtual const Matrix4 *orientation();
  
  const char *type_name() const { return "rotate"; }
private:
  int grab_toggle;
  int button_was_down;

  int constrained;
  float rotatecenter[3];
  float rotateradius;
  float constrainedpos[3];
  float old_pos[3];
  float old_normal[3];
  q_type qoffset;
  Matrix4 offset;
  Matrix4 start;
  Matrix4 product;
};

#endif
