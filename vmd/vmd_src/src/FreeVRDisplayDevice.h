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
 *      $RCSfile: FreeVRDisplayDevice.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.20 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * a FreeVR specific display device for VMD
 ***************************************************************************/
#ifndef FREEVRDISPLAYDEVICE_H
#define FREEVRDISPLAYDEVICE_H

#include "OpenGLRenderer.h"

/// OpenGLRenderer subclass for FreeVR displays
class FreeVRDisplayDevice : public OpenGLRenderer {
private:
  int doneGLInit;                ///< have we initialized the graphics yet?
  void freevr_gl_init_fn(void);  ///< setup graphics state on FreeVR displays
  
public:
  FreeVRDisplayDevice(void);                   ///< constructor
  virtual ~FreeVRDisplayDevice(void);          ///< destructor
  virtual void set_stereo_mode(int = 0);       ///< ignore stereo mode changes
  virtual void render(const VMDDisplayList *); ///< FreeVR renderer + init chk
  virtual void normal(void);                   ///< prevent view mode changes
  virtual void update(int do_update = TRUE);   ///< prevent buffer swaps 
};
#endif

