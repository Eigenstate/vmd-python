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
 *	$RCSfile: Animation.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.42 $	$Date: 2010/12/16 04:08:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Animation class, which stores a list of pointers to Timestep objects
 * that contain 3D coordinates and other data that varies with time.
 *
 ***************************************************************************/
#ifndef ANIMATION_H
#define ANIMATION_H

#include "MoleculeList.h"
#include "UIObject.h"

#define SPEED_FACTOR	0.5f // max fraction of a sec between redraws

/*!
 * Animation (1) provides an interface for defining how frames in a collection
 * of molecules get animated over time; and (2) notifies those molecules at
 * the appropriate time.
 *
 */

class Animation : public UIObject {
public:
  // enums for Animation options
  enum AnimDir 	{ ANIM_FORWARD, ANIM_FORWARD1, ANIM_REVERSE, ANIM_REVERSE1, 
                  ANIM_PAUSE, ANIM_TOTAL_DIRS };
  enum AnimStyle  { ANIM_ONCE, ANIM_LOOP, ANIM_ROCK, ANIM_TOTAL_STYLES };

private:
  /// the collection of molecules managed by this Animation instance.
  /// The animation range is controlled by the top molecule in the mlist.
  MoleculeList &mlist;
 
  /// last time the image was drawn, for use with determining speed
  double lastTime;

  int frameSkip;       ///< frames to skip to next position when animating
  float Speed;         ///< animation speed, from 0.0 (slowest) to 1.0 (fastest)
  AnimDir animDir;     ///< current animation direction
  AnimStyle animStyle; ///< what to do when you get to the end of the loop

public:
  /// constructor: take VMDApp pointer for UIObject baseclass
  Animation( VMDApp * );
  
  /// total number of frames currently stored
  /// FIXME: should be const, but MoleculeList lacks const accessors
  int num() { 
      Molecule *m = mlist.top();
      if (m) return m->numframes();
      return 0;
  }

  /// return the current frame number (frames 0...(frames -1); -1 => no frames)
  /// FIXME: should be const, but MoleculeList lacks const accessors
  int frame() { 
      Molecule *m = mlist.top();
      if (m) return m->frame();
      return -1;
  }

  /// move each molecule to the specified frame, clamped by the range of
  /// the respective molecules.  Any ongoing animation is paused.
  void goto_frame(int fr);

  /// UIObject interface
  virtual int check_event();

  void skip(int newsk);                            ///< set # frames to skip
  int skip() const { return frameSkip; }           ///< get # frames to skip
  void anim_dir(AnimDir ad) { animDir = ad; }      ///< set animation direction
  AnimDir anim_dir() const { return animDir; }     ///< get animation direction
  void anim_style(AnimStyle as);                   ///< set animation style
  AnimStyle anim_style() const { return animStyle; } ///< get animation style


  /// animation speed methods:
  ///	newsp should be from 0 (min speed) to 1 (max speed)
  float speed(float newsp);
  float speed() const { return (float) (Speed / SPEED_FACTOR); }
};

/// static storage for strings describing animation styles
extern const char *animationStyleName[Animation::ANIM_TOTAL_STYLES];

/// static storage for strings describing animation directions
extern const char *animationDirName[Animation::ANIM_TOTAL_DIRS];

#endif

