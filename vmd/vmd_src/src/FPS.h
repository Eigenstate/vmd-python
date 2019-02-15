/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/
#ifndef FPS_H 
#define FPS_H 

#include "Displayable.h"
class DisplayDevice;

/// Displayable subclass to display the current display update rate onscreen
class FPS : public Displayable {
private:
  DisplayDevice *disp; ///< display device
  double last_update;  ///< time of last redraw
  int loop_count;      ///< number of times through display loop since redraw
  int colorCat;        ///< color category to use, If < 0, use default color
  int usecolor;        ///< color index to use

public:
  /// constructor: the display device to take aspect ratio from
  FPS(DisplayDevice *, Displayable *);
  virtual void prepare();

protected:
  virtual void do_color_changed(int);

};

#endif

