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
 *      $RCSfile: CaveDisplayDevice.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.37 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * a CAVE specific display device for VMD
 ***************************************************************************/
#ifndef CAVEDISPLAYDEVICE_H
#define CAVEDISPLAYDEVICE_H

#include "OpenGLRenderer.h"

/// DisplayDevice subclass that runs in the CAVE
class CaveDisplayDevice : public OpenGLRenderer {
private:
  int doneGLInit;             ///< have we initialized the graphics yet?
  void cave_gl_init_fn(void); ///< setup graphics state on CAVE displays
  
public:
  CaveDisplayDevice(void);                     ///< constructor
  virtual ~CaveDisplayDevice(void);            ///< destructor
  virtual void set_stereo_mode(int = 0);       ///< ignore stereo mode changes
  virtual void render(const VMDDisplayList *); ///< CAVE renderer, init check
  virtual void normal(void);                   ///< prevent view mode changes
  virtual void update(int do_update = TRUE);   ///< prevent buffer swaps 
};
#endif

