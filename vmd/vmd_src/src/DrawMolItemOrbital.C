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
 *	$RCSfile: DrawMolItemOrbital.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.51 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A continuation of rendering types from DrawMolItem
 *
 *   This file only contains representations for visualizing QM orbital data
 ***************************************************************************/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "DrawMolItem.h"
#include "DrawMolecule.h"
#include "BaseMolecule.h" // need volume data definitions
#include "Inform.h"
#include "Isosurface.h"
#include "MoleculeList.h"
#include "Scene.h"
#include "Orbital.h"
#include "QMData.h"
#include "VolumetricData.h"
#include "VMDApp.h"
#include "utilities.h"
#include "WKFUtils.h" // timers

#define MYSGN(a) (((a) > 0) ? 1 : -1)

void DrawMolItem::draw_orbital(int density, int wavefnctype, int wavefncspin, 
                               int wavefncexcitation, int orbid, 
                               float isovalue, 
                               int drawbox, int style, 
                               float gridspacing, int stepsize, int thickness) {
  if (!mol->numframes() || gridspacing <= 0.0f)
    return; 

  // only recalculate the orbital grid if necessary
  int regenorbital=1;
  int useorbgridfromrep = -1;  // XXX repID with an existing grid we can reuse 

  if (density != orbgridisdensity ||
      wavefnctype != waveftype ||
      wavefncspin != wavefspin ||
      wavefncexcitation != wavefexcitation ||
      orbid != gridorbid ||
      gridspacing != orbgridspacing ||
      orbvol == NULL || 
      needRegenerate & MOL_REGEN ||
      needRegenerate & SEL_REGEN) {

#if defined(VMDENABLEORBITALGRIDBACKDOOR)
    //
    // XXX hack to look for an existing orbital grid we can reuse as-is...
    //
    // This search loop allows the molecular orbital representations 
    // within the same molecule to reuse any existing
    // rep's molecular orbital grid if the orbital ID and various 
    // grid-specific parameters are all compatible.  This optimization
    // short-circuits the need for a rep to compute its own grid if
    // any other rep already has what it needs.  For large QM/MM scenes,
    // this optimization can be worth as much as a 2X speedup when
    // orbital computation dominates animation performance.
    int repcnt = mol->repList.num();
    int r;
    for (r=0; r<repcnt && useorbgridfromrep < 0; r++) {
      DrawMolItem *dmi = mol->repList[r];
      if (dmi->repNumber != repNumber) {
        AtomRep *ar = dmi->atomRep;
        if (ar->method() == AtomRep::ORBITAL) {
          if (orbid == dmi->gridorbid &&
              wavefnctype == dmi->waveftype &&
              wavefncspin == dmi->wavefspin &&
              wavefncexcitation == dmi->wavefexcitation &&
              gridspacing == dmi->orbgridspacing &&
              density == dmi->orbgridisdensity &&
              dmi->orbvol != NULL) {
//            printf("Rep[%d]: orbid %d,  rep[%d]: %d\n", repNumber, orbid, r, dmi->gridorbid);
            useorbgridfromrep=r;
            delete orbvol;
            orbvol = dmi->orbvol; // XXX watch out when borrowing orbvol!
          }
        }
      }
    }

    if (useorbgridfromrep >= 0)
      regenorbital=0;
    else  
#endif 
      regenorbital=1;
  }

  double motime=0, voltime=0, gradtime=0;
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer);

  if (regenorbital) {
    // XXX this needs to be fixed so that things like the
    //     draw multiple frames feature will work correctly for Orbitals
    int frame = mol->frame(); // draw currently active frame
    const Timestep *ts = mol->get_frame(frame);

    if (!ts->qm_timestep || !mol->qm_data || 
        !mol->qm_data->num_basis || orbid < 1) {
      wkf_timer_destroy(timer);
      return;
    }

    // Find the  timestep independent wavefunction ID tag
    // by comparing type, spin, and excitation with the
    // signatures of existing wavefunctions.
    int waveid = mol->qm_data->find_wavef_id_from_gui_specs(
                  wavefnctype, wavefncspin, wavefncexcitation);

    // Translate the wavefunction ID into the index the
    // wavefunction has in this timestep
    int iwave = ts->qm_timestep->get_wavef_index(waveid);

    if (iwave<0 || 
        !ts->qm_timestep->get_wavecoeffs(iwave) ||
        !ts->qm_timestep->get_num_orbitals(iwave) ||
        orbid > ts->qm_timestep->get_num_orbitals(iwave)) {
      wkf_timer_destroy(timer);
      return;
    }

    // Get the orbital index for this timestep from the orbital ID.
    int orbindex = ts->qm_timestep->get_orbital_index_from_id(iwave, orbid);

    // Build an Orbital object and prepare to calculate a grid
    Orbital *orbital = mol->qm_data->create_orbital(iwave, orbindex, 
                                                    ts->pos, ts->qm_timestep);

    // Set the bounding box of the atom coordinates as the grid dimensions
    orbital->set_grid_to_bbox(ts->pos, 3.0, gridspacing);

    // XXX needs more testing, can get stuck for certain orbitals
#if 0
    // XXX for GPU, we need to only optimize to a stepsize of 4 or more, as
    //     otherwise doing this actually slows us down rather than speeding up
    //     orbital.find_optimal_grid(0.01, 4, 8);
    // 
    // optimize: minstep 2, maxstep 8, threshold 0.01
    orbital->find_optimal_grid(0.01, 2, 8);
#endif

    // Calculate the molecular orbital
    orbital->calculate_mo(mol, density);

    motime = wkf_timer_timenow(timer);

    // query orbital grid origin, dimensions, and axes 
    const int *numvoxels = orbital->get_numvoxels();
    const float *origin = orbital->get_origin();

    float xaxis[3], yaxis[3], zaxis[3];
    orbital->get_grid_axes(xaxis, yaxis, zaxis);

    // build a VolumetricData object for rendering
    char dataname[64];
    sprintf(dataname, "molecular orbital %i", orbid);

    // update attributes of cached orbital grid
    orbgridisdensity = density;
    waveftype = wavefnctype;
    wavefspin = wavefncspin;
    wavefexcitation = wavefncexcitation;
    gridorbid = orbid;
    orbgridspacing = gridspacing;
    delete orbvol;
    orbvol = new VolumetricData(dataname, origin, 
                                xaxis, yaxis, zaxis,
                                numvoxels[0], numvoxels[1], numvoxels[2],
                                orbital->get_grid_data());
    delete orbital;

    voltime = wkf_timer_timenow(timer);

    orbvol->compute_volume_gradient(); // calc gradients: smooth vertex normals

    gradtime = wkf_timer_timenow(timer);
  } // regen the orbital grid...


  // draw the newly created VolumetricData object 
  sprintf(commentBuffer, "MoleculeID: %d ReprID: %d Beginning Orbital",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  if (drawbox > 0) {
    // don't texture the box if color by volume is active
    if (atomColor->method() == AtomColor::VOLUME) {
      append(DVOLTEXOFF);
    }
    // wireframe only?  or solid?
    if (style > 0 || drawbox == 2) {
      draw_volume_box_lines(orbvol);
    } else {
      draw_volume_box_solid(orbvol);
    }
    if (atomColor->method() == AtomColor::VOLUME) {
      append(DVOLTEXON);
    }
  }

  if ((drawbox == 2) || (drawbox == 0)) {
    switch (style) {
      case 3:
        // shaded points isosurface looping over X-axis, 1 point per voxel
        draw_volume_isosurface_lit_points(orbvol, isovalue, stepsize, thickness);
        break;

      case 2:
        // points isosurface looping over X-axis, max of 1 point per voxel
        draw_volume_isosurface_points(orbvol, isovalue, stepsize, thickness);
        break;

      case 1:
        // lines implementation, max of 18 line per voxel (3-per triangle)
        draw_volume_isosurface_lines(orbvol, isovalue, stepsize, thickness);
        break;

      case 0:
      default:
        // trimesh polygonalized surface, max of 6 triangles per voxel
        draw_volume_isosurface_trimesh(orbvol, isovalue, stepsize);
        break;
    }
  }

  // XXX if we reused the orbital grid from another rep, we have to 
  //     null out orbvol so we don't try and free another reps memory later...
  if (useorbgridfromrep >= 0) {
    orbvol = NULL; // XXX watch out, un-copy the pointer to the borrowed grid
  }

  if (regenorbital) {
    double surftime = wkf_timer_timenow(timer);
    if (surftime > 5) {
      char strmsg[1024];
      sprintf(strmsg, "Total MO rep time: %.3f [MO: %.3f vol: %.3f grad: %.3f surf: %.2f]",
              surftime, motime, voltime - motime, gradtime - motime, surftime - gradtime);

      msgInfo << strmsg << sendmsg;
    }
  }

  wkf_timer_destroy(timer);
}

