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
 *	$RCSfile: MobileTracker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.4 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Listen for UDP packets from WiFi mobile input devices such 
 *   as smartphones, tablets, etc.
 *
 ***************************************************************************/
#include <stdlib.h> // for getenv(), abs() etc.
#include <string.h>
#include <math.h>
#include "VMDApp.h"
#include "MobileTracker.h"
#include "Matrix4.h"
#include "Inform.h"
#include "utilities.h"

MobileTracker::MobileTracker(VMDApp *vmdapp) {
  app = vmdapp; // copy VMDApp pointer for use in accessing local spaceball
}

int MobileTracker::do_start(const SensorConfig *config) {
  if (!config->require_local()) return 0;
  if (!config->have_one_sensor()) return 0;

  char *myUSL = stringdup(config->getname());

printf("Mobile USL: '%s'\n", myUSL);
  
  // set the default translation and rotation increments
  // these really need to be made user modifiable at runtime
  transInc = 1.0f;
    rotInc = 0.01f;
  scaleInc = 1.0f;

  // reset the position
  moveto(0,0,0);
  orient->identity();

  delete [] myUSL;

  return TRUE;
}

MobileTracker::~MobileTracker(void) {
}

void MobileTracker::update() {
  Matrix4 temp;

  if(!alive()) {
    moveto(0,0,0);
    orient->identity();
    return;
  }

  if (app != NULL ) {
    float tx, ty, tz, rx, ry, rz;
    tx=ty=tz=rx=ry=rz=0.0f;
    int buttons=0;

printf("polling mobile status socket..\n");
    app->mobile_get_tracker_status(tx, ty, tz, rx, ry, rz, buttons);

    temp.identity();
    temp.rot( ((float)rx)*rotInc, 'x' );
    temp.rot( ((float)ry)*rotInc, 'y' );
    temp.rot( ((float)rz)*rotInc, 'z' );
    temp.multmatrix(*orient);
    orient->loadmatrix(temp);
    pos[0] += tx * transInc;
    pos[1] += ty * transInc;
    pos[2] +=-tz * transInc;
  }
}

