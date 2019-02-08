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
 *	$RCSfile: P_TugTool.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.42 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************/

/// A tool for interacting with MD simulations
/** This is a tool for use with running MD simulations.  It allows the
    user to grab atoms and pull them around, by sending forces to
    UIVR. */

#ifndef P_TUGTOOL_H
#define P_TUGTOOL_H

#include "P_Tool.h"
class TugTool : public Tool {
 public:
  TugTool(int id, VMDApp *, Displayable *);
  virtual void do_event();
  virtual int isgrabbing() { return 0; } // tug instead of grabbing!

  virtual void setspringscale(float sc) {
    springscale=sc;
    Tool::setspringscale(sc);
  }
 
  const char *type_name() const { return "tug"; }
protected:
  virtual void start_tug() {}

private:
  virtual int istugging() { return Tool::isgrabbing(); }

  // applies a force (returns the actual force applied in the argument)
  virtual void do_tug(float *force);

  virtual void set_tug_constraint(float *pos);

  int tugging;
  float tugged_pos[3];
  float offset[3];    // An offset so that initial force is always zero
  float springscale;
};

#endif
