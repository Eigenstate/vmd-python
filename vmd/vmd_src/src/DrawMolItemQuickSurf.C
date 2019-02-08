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
 *	$RCSfile: DrawMolItemQuickSurf.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.25 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A continuation of rendering types from DrawMolItem
 *
 *   This file only contains code for fast gaussian surface representations 
 ***************************************************************************/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "DrawMolItem.h"
#include "BaseMolecule.h" // need volume data definitions
#include "Inform.h"
#include "Isosurface.h"
#include "MoleculeList.h"
#include "Scene.h"
#include "VolumetricData.h"
#include "VMDApp.h"
#include "utilities.h"
#include "WKFUtils.h" // timers
#include "Measure.h"

#define MYSGN(a) (((a) > 0) ? 1 : -1)

void DrawMolItem::draw_quicksurf(float *pos, int quality, float radscale, float isovalue, float gridspacing) {
  const int *colidx = NULL;
  const float *cmap = NULL;

  if (atomSel->selected < 1)
    return;

  if (!mol->numframes() || gridspacing <= 0.0f)
    return; 

#if 0
  // XXX this needs to be fixed so that things like the
  //     draw multiple frames feature will work correctly 
  int frame = mol->frame(); // draw currently active frame
  const Timestep *ts = mol->get_frame(frame);
#endif
  const float *radii = mol->radius();

  int usecolor = draw_volume_get_colorid();

  // Use the active per-atom color map to generate a volumetric 
  // texture during the calculation of the density map
  if ((atomColor->method() != AtomColor::COLORID) &&
      (atomColor->method() != AtomColor::MOLECULE)) {
    colidx = atomColor->color;
    cmap = scene->color_value(0); // get start of color map
  } else {
    cmap = scene->color_value(usecolor); // active color 
  }

  // Set color, material, and add the rep comment token
  sprintf(commentBuffer, "MoleculeID: %d ReprID: %d Beginning QuickSurf",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);
  append(DMATERIALON); // enable lighting and shading
  cmdColorIndex.putdata(usecolor, cmdList);

  // Extract the surface from a density map computed from atom positions/radii
  mol->app->qsurf->calc_surf(atomSel, mol, pos, radii, quality, radscale, 
                             gridspacing, isovalue, colidx, cmap, cmdList);
}

