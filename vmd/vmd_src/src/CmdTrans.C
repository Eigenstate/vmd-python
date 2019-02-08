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
 *	$RCSfile: CmdTrans.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.43 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Command objects for transforming the current scene.
 *
 ***************************************************************************/

#include <math.h>
#include "CmdTrans.h"

///////////////// apply a generic transformation matrix to the scene
void CmdRotMat::create_text(void) {
  *cmdText << "rotmat " << (byOrTo == CmdRotMat::BY ? "by " : "to ");
  for (int i=0; i<3; i++) *cmdText << " " 
      << rotMat.mat[4*i] << " " << rotMat.mat[4*i+1] <<" " 
      << rotMat.mat[4*i+2] << " ";
  *cmdText << ends;
}

CmdRotMat::CmdRotMat(const Matrix4& m, int by_or_to)
: Command(Command::ROTMAT) {
   byOrTo = by_or_to;
   rotMat = m;
}


///////////////// rotate the current scene

void CmdRotate::create_text(void) {
  *cmdText << "rotate " << axis;
  *cmdText << ( byOrTo == CmdRotate::BY ? " by " : " to ");
  *cmdText << deg;
  if(steps > 0)
    *cmdText << " " << (deg / ((float)steps));
  *cmdText << ends;
}

// first constructor: a single rotation, no smooth transition
CmdRotate::CmdRotate(float a, char ax, int by_or_to)
  : Command(Command::ROTATE) {

  steps = (-1);

  // make sure the axis specified is a legal one ...
  if(ax >= 'x' && ax <= 'z') {
    byOrTo = by_or_to;
    axis = ax;
    deg = a;
  } else {
    // if not legal, just do no rotation.
    byOrTo = CmdRotate::BY;
    axis = 'y';
    deg = 0.0;
  }
}

// second constructor: a smooth rotation in given increments ...
// only useful for "by" rotations.  If "to" is given to this constructor,
// a single-step rotation is done.
CmdRotate::CmdRotate(float a, char ax, int by_or_to, float inc)
  : Command(Command::ROTATE) {

  // make sure the axis specified is a legal one ...
  if(ax >= 'x' && ax <= 'z' && inc != 0) {
    byOrTo = by_or_to;
    axis =  ax;
  
    // determine by how much to rotate, and number of steps to use.  If we
    // are doing 'to' rotation, just do it in one big step.
    if(byOrTo == CmdRotate::TO) {
      steps = (-1);
      deg = a;
    } else {
      steps = (int)(fabs(a / inc) + 0.5);

      // make sure there is at least one step
      if(steps < 1) {
        steps = (-1);
	deg = a;
      } else {
        deg = (float) (a < 0.0 ? - fabs(inc) : fabs(inc));
      }
    }

  } else {
    // if not legal, just do no rotation.
    byOrTo = CmdRotate::BY;
    axis = 'y';
    deg = 0.0;
    steps = (-1);
  }
}


///////////////// translate the current scene
void CmdTranslate::create_text(void) {
  *cmdText << "translate ";
  *cmdText << (byOrTo == CmdTranslate::BY ? "by " : "to ");
  *cmdText << x << " " << y << " " << z << ends;
}

CmdTranslate::CmdTranslate(float nx,float ny, float nz, int by_or_to)
  : Command(Command::TRANSLATE) {
  x = nx;  y = ny;  z = nz;
  byOrTo = by_or_to;
}


///////////////// scale the current scene
void CmdScale::create_text(void) {
  *cmdText << "scale ";
  *cmdText << ( byOrTo == CmdScale::BY ? "by " : "to ");
  *cmdText << s;
  *cmdText << ends;
}

CmdScale::CmdScale(float ns, int by_or_to)
  : Command(Command::SCALE) {
  s = ns;
  byOrTo = by_or_to;
}


///////////////// rock the current scene
void CmdRockOn::create_text(void) {
  *cmdText << "rock " << axis << " by " << deg;
  if(steps >= 0)
    *cmdText << " " << steps;
  *cmdText << ends;
}

CmdRockOn::CmdRockOn(float a, char ax, int nsteps)
  : Command(Command::ROCKON) {
  deg = a;
  axis = ((ax >= 'x' && ax <= 'z') ? ax : 'y');
  steps = nsteps;
}


///////////////// stop rocking the current scene
void CmdRockOff::create_text(void) {
  *cmdText << "rock off" << ends;
}

CmdRockOff::CmdRockOff() 
: Command(Command::ROCKOFF) { }
