/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr VRJuggler patches contributed by Martijn Kragtwijk: m.kragtwijk@rug.nl
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VRJugglerButtons.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $       $Date: 2010/12/16 04:08:47 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific button device for VMD
 ***************************************************************************/
#include "VRJugglerButtons.h"
#include "VRJugglerScene.h"

VRJugglerButtons::VRJugglerButtons(VRJugglerScene* scene) :Buttons(), mScene(scene) {
};

int VRJugglerButtons::do_start(const SensorConfig *) {
  // XXX Somehow check that a VRJuggler environment exists.  If it doesn't,
  // return false.
  return 1; // VRJuggler is active.
}

void VRJugglerButtons::update() {
  stat[0]=mScene->getWandButton(0);
  stat[1]=mScene->getWandButton(1);
}

