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
 *	$RCSfile: Axes.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.39 $	$Date: 2011/02/17 21:58:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * A Displayable object which consists of a set of axes, which may be
 * drawn anywhere on the screen, at any size.
 *
 ***************************************************************************/
#ifndef AXES_H
#define AXES_H

#include "Displayable.h"
#include "DispCmds.h"


/// A Displayable object which consisting of a set of axes
class Axes : public Displayable {
public:
  /// locations for the axes
  enum AxesPos { NO_AXES = 0, AXES_ORIGIN, AXES_LOWERLEFT, 
  	AXES_LOWERRIGHT, AXES_UPPERLEFT, AXES_UPPERRIGHT, AXESPOS_TOTAL };

private:
  DisplayDevice *disp;

  /// lines defining the axes
  float origin[3], xLine[3], yLine[3], zLine[3];
  float xLineCap[3], yLineCap[3], zLineCap[3];
  float xText[3], yText[3], zText[3];

  int usecolors[5];        ///< colors of the 3 arrows, sphere, and text
  int axesPos;             ///< current axes position
  float Aspect;            ///< most recent aspect ratio of the display
  int colorCat;            ///< color category index, if < 0, use defaults
  int movedAxes;           ///< whether axes were moved by mouse/pointer      
  int need_create_cmdlist; ///< flag indicating we need to regen draw cmds

  // display command objects used to render the axes
  DispCmdBeginRepGeomGroup cmdBeginRepGeomGroup;
  DispCmdSphereRes sphres;
  DispCmdSphereType sphtype;
  DispCmdColorIndex xcol;
  DispCmdCylinder xcyl;
  DispCmdCone xcap;
  DispCmdSphere sph;
  DispCmdText txt;
  DispCmdPickPoint pickPoint;
  DispCmdComment cmdCommentX;

  /// regenerate the command list
  void create_cmdlist(void);

protected:
  virtual void do_color_changed(int);

public:
  /// constructor: the display device to take aspect ratio from
  Axes(DisplayDevice *, Displayable *);
  virtual ~Axes(void);

  int location(int);                           ///< set axes display mode
  int location(void) { return axesPos; }       ///< return axes display mode
  char *loc_description(int);                  ///< return location descripton
  int locations(void){ return AXESPOS_TOTAL; } ///< return number of locations

  //
  // public virtual routines
  //
  
  /// prepare for drawing ... do any updates needed right before draw.
  virtual void prepare();

  /// called when a pick moves:
  ///	args = display to use, obj picked, button, mode, tag, dim, pos
  /// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
  /// For 3D version: x,y,z are transformed position of pointer
  /// For the Axes, when they are selected and the pointer moves, we wish
  /// to move the axes as well.
  virtual void pick_move(PickMode *, DisplayDevice *, int, int, const float *);
};

#endif

