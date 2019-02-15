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
 *	$RCSfile: SpaceballTracker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.18 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/


#include <stdlib.h> // for getenv(), abs() etc.
#include <string.h>
#include "VMDApp.h"
#include "SpaceballTracker.h"
#include "Spaceball.h"
#include "Matrix4.h"
#include "Inform.h"
#include "utilities.h"

SpaceballTracker::SpaceballTracker(VMDApp *vmdapp) {
  app = vmdapp; // copy VMDApp pointer for use in accessing local spaceball

  uselocal=0;   // zero out the local Spaceball flag

#if defined(VMDLIBSBALL)
  sball=NULL;   // zero it out to begin with
#endif
}

int SpaceballTracker::do_start(const SensorConfig *config) {
#if defined(VMDLIBSBALL)
  if (sball) return FALSE;
#endif
  if (!config->require_local()) return 0;
  if (!config->have_one_sensor()) return 0;

  char *myUSL = stringdup(config->getname());

  if (!strupcmp(myUSL, "VMDLOCAL")) {
    msgInfo << "Opening VMD console Spaceball device (tracker)." << sendmsg;
    uselocal=1; // Use the main VMD spaceball for input

    // set the default translation and rotation increments
    // these really need to be made user modifiable at runtime
    transInc = 1.0f;
      rotInc = 1.0f;
    scaleInc = 1.0f;
  } else {
#if defined(VMDLIBSBALL)
    msgInfo << "Opening Spaceball tracker (direct I/O) on port: " << myUSL << sendmsg;
    sball = sball_open(myUSL);
    if (sball == NULL) 
      msgErr << "Failed to open Spaceball serial port, tracker disabled" 
             << sendmsg; 

    // set the default translation and rotation increments
    // these really need to be made user modifiable at runtime
    transInc = 1.0f / 6000.0f;
      rotInc = 1.0f /   50.0f;
    scaleInc = 1.0f / 6000.0f;
#else
    msgErr << "Cannot open Spaceball with direct I/O, not compiled with "
              "LIBSBALL option" << sendmsg;
#endif
  }


  // reset the position
  moveto(0,0,0);
  orient->identity();

  delete [] myUSL;

  return TRUE;
}

SpaceballTracker::~SpaceballTracker(void) {
#if defined(VMDLIBSBALL)
  if (sball != NULL)
    sball_close(sball);
#endif
}

void SpaceballTracker::update() {
  Matrix4 temp;

  if(!alive()) {
    moveto(0,0,0);
    orient->identity();
    return;
  }

  if (uselocal) {
    float tx, ty, tz, rx, ry, rz;
    tx=ty=tz=rx=ry=rz=0.0f;
    int buttons;
    buttons=0;

    // query VMDApp spaceball for events
    if (app != NULL) {
      app->spaceball_get_tracker_status(tx, ty, tz, rx, ry, rz, buttons);
    }

    // Z-axis rotation/trans have to be negated in order to convert to the
    // VMD coordinate system
    temp.identity();
    temp.rot(rx, 'x');
    temp.rot(ry, 'y');
    temp.rot(rz, 'z');
    temp.multmatrix(*orient);
    orient->loadmatrix(temp);
    pos[0] += tx;
    pos[1] += ty;
    pos[2] += tz;
  } else {
    int tx, ty, tz, rx, ry, rz, buttons;
    tx=ty=tz=rx=ry=rz=buttons=0;
#if defined(VMDLIBSBALL)
    if (sball != NULL ) {
      if (!sball_getstatus(sball, &tx, &ty, &tz, &rx, &ry, &rz, &buttons))
        return;
    }
#endif
    // Z-axis rotation/trans have to be negated in order to convert to the
    // VMD coordinate system
    temp.identity();
    temp.rot( ((float)rx)*rotInc, 'x' );
    temp.rot( ((float)ry)*rotInc, 'y' );
    temp.rot(-((float)rz)*rotInc, 'z' );
    temp.multmatrix(*orient);
    orient->loadmatrix(temp);
    pos[0] += tx * transInc;
    pos[1] += ty * transInc;
    pos[2] +=-tz * transInc;
  }
}

