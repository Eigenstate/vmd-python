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
 *      $RCSfile: VRJugglerButtons.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific button device for VMD
 ***************************************************************************/
#include "P_Buttons.h"
class VRJugglerScene;

/// Buttons subclass that gets its info from the VRJuggler wand.
class VRJugglerButtons : public Buttons {
public:
  VRJugglerButtons(VRJugglerScene* scene);
  virtual const char *device_name() const { return "vrjugglerbuttons"; }
  virtual Buttons *clone() { return new VRJugglerButtons(mScene); }
  virtual void update();
  inline virtual int alive() { return 1; }

protected:
  /// Check that we are running in a VRJuggler environment.
  virtual int do_start(const SensorConfig *);
  VRJugglerScene* mScene;
};

