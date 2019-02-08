/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr VRJuggler patches contributed by Martijn Kragtwijk: m.kragtwijk@rug.nl
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VRJugglerDisplayDevice.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific button device for VMD
 ***************************************************************************/
#ifndef VRJUGGLERDISPLAYDEVICE_H
#define VRJUGGLERDISPLAYDEVICE_H

#include "OpenGLRenderer.h"

/// DisplayDevice subclass that runs in the VRJuggler
class VRJugglerDisplayDevice : public OpenGLRenderer {
private:
  int doneGLInit;             ///< have we initialized the graphics yet?
  void vrjuggler_gl_init_fn(void); ///< setup graphics state on VRJuggler displays
  
public:
  VRJugglerDisplayDevice(void);                     ///< constructor
  virtual ~VRJugglerDisplayDevice(void);            ///< destructor
  virtual void set_stereo_mode(int = 0);       ///< ignore stereo mode changes
  virtual void render(const VMDDisplayList *); ///< VRJuggler renderer, init check
  virtual void normal(void);                   ///< prevent view mode changes
  virtual void update(int do_update = TRUE);   ///< prevent buffer swaps 

  //virtual int supports_gui() { return FALSE; }
  virtual int supports_gui() { return TRUE; }
};
#endif  //  VRJUGGLERDISPLAYDEVICE_H

