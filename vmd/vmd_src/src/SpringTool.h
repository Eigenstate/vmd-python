/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/// A tool for connecting atoms with springs.
/** A SpringTool is just like a TugTool, except that when you release
   the button, a GeometrySpring is added between the atom you were
   pulling and the atom you were near */

#ifndef P_SPRINGTOOL_H
#define P_SPRINGTOOL_H

#include "P_Tool.h"
class SpringTool : public Tool {
 public:
  SpringTool(int id, VMDApp *, Displayable *);
  virtual void do_event();
  virtual int isgrabbing() { return 0; } // tug instead of grabbing!
  
  virtual void setspringscale(float sc) {
    springscale=sc;
    Tool::setspringscale(sc);
  }
 
  const char *type_name() const { return "spring"; }

protected:
  virtual void start_tug() {}

 private:
  virtual int istugging() { return Tool::isgrabbing(); }
  virtual void do_tug(float *force);
  virtual void set_tug_constraint(float *pos);

  int tugging;
  float tugged_pos[3];
  float offset[3];    // An offset so that initial force is always zero
  float springscale;
};

#endif
