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
 *      $RCSfile: VRJugglerTracker.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific tracker device for VMD
 ***************************************************************************/
#include "P_Tracker.h"
#include "VRJugglerTracker.h"
#include "VRJugglerScene.h"

VRJugglerTracker::VRJugglerTracker(VRJugglerScene* scene)
  : VMDTracker(),
    mScene(scene)
{

}; // VRJuggler needs no initialization

void VRJugglerTracker::update() {

  #define WAND_SENSOR     1

  
  //float units =5;// mScene->application->getDrawScaleFactor();
  float x,y,z; 
  mScene->getWandXYZ(x,y,z); 

     // Get the wand matrix
//  vrPointGetRWFrom6sensor(&wand_location, WAND_SENSOR);
  pos[0] = x;//0.0;//wand_location.v[0];
  pos[1] = y;//0.0;//wand_location.v[1];
  pos[2] = z;//0.0;//wand_location.v[2];

  /* "classical" Euler angles */
  float azi, elev, roll;

  // XXX hack to get us by for now until VRJuggler can do this, or 
  // something like this.
  azi=0.0;  
  elev=0.0;
  roll=0.0;
  // CAVE version
  // CAVEGetWandOrientation(azi, elev, roll);

  Matrix4 rot;

  //  mScene->getWandRot(azi, elev, roll); // get the wand rot as euler angles (degrees)
  mScene->getWandRotMat(rot); // get the wand rot matrix

  /*  orient->identity();
  orient->rot(azi,'y');
  orient->rot(elev,'x');
  orient->rot(roll,'z');
  orient->rot(90,'y'); // to face forward (-z)
*/
  orient->loadmatrix(rot);
  orient->rot(90,'y'); // to face forward (-z)

  // change the tool such that it doesn't the wanda isn't at the tip but at the other end
  float displacementIn[3] = {0.5, 0, 0}; // why in x direction? 
  //float displacementIn[3] = {0.0, 0, 0}; // why in x direction? 
                                         //because the tool is originally pointing towards positive x?
  float displacementOut[3] = {0, 0, 0};
  
  orient->multpoint3d(displacementIn, displacementOut); // multiply displacement by rotation
                               // and ad it to position
  pos[0]+=displacementOut[0];
  pos[1]+=displacementOut[1];
  pos[2]+=displacementOut[2];
}

int VRJugglerTracker::do_start(const SensorConfig *config) {
  // Must check that we are actually running in VRJuggler here; if not, 
  // return 0.

//  if (!config->require_freevr_name()) return 0;
//  if (!config->have_one_sensor()) return 0;
  return 1;
}
