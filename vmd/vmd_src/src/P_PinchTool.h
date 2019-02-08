/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/
#ifndef P_PINCHTOOL_H
#define P_PINCHTOOL_H

#include "P_Tool.h"

/// Tool subclass implementing a function similar to TugTool
/// except that the force is only applied in the direction
/// the tool is oriented.
class PinchTool : public Tool {
 public:
  PinchTool(int id, VMDApp *, Displayable *);
  virtual void do_event();
  virtual int isgrabbing() { return 0; } // tug instead of grabbing!

  virtual void setspringscale(float sc) {
    springscale=sc;
    Tool::setspringscale(sc);
  }
 
  const char *type_name() const { return "pinch"; }

protected:
  virtual void start_tug() {}

 private:
  virtual int istugging() { return Tool::isgrabbing(); }

  int tugging;
  float tugged_pos[3];
  float offset[3];    // An offset so that initial force is always zero
  float springscale;
};

#endif
