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
 *	$RCSfile: DrawForce.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.57 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Another Child Displayable component for a remote molecule; this displays
 * and stores the information about the interactive forces being applied to
 * the molecule.  If no forces are being used, this draws nothing.
 *
 * The force information is retrieved from the Atom list in the parent
 * molecule.  No forces are stored here.
 *
 * This name is now a misnomer as accelerations are changed, _not_ forces.
 * This eliminates the problem of having hydrogens acceleterating 12 time
 * faster than carbons, etc.
 *
 * And now I'm changing it back.  We should draw the actual force...
 *
 ***************************************************************************/

#include "DrawForce.h"
#include "DrawMolecule.h"
#include "DisplayDevice.h"
#include "Scene.h"
#include "Atom.h"
#include "Timestep.h"
#include "Inform.h"
#include "utilities.h"
#include "Mouse.h"
#include "VMDApp.h"

// Force arrow scaling constant
// XXX ideally this should be a user-defined scaling coefficient
//     settable in the text commands and GUI interfaces

// Since we now have MUCH faster machines for running simulations,
// we should expect IMD users to be able to apply weaker pulling 
// forces to achieve the same end results now.  Given this, we
// can now use a much larger scaling factor (1.0f) on the force arrows 
// than we did in the past (0.3333f).
#define DRAW_FORCE_SCALE 1.0f

////////////////////////////  constructor  

DrawForce::DrawForce(DrawMolecule *mr)
	: Displayable(mr) {

  // save data
  mol = mr;

  // initialize variables
  needRegenerate = TRUE;
  colorCat = (-1);
}


///////////////////////////  protected virtual routines

void DrawForce::do_color_changed(int ccat) {
  // right now this does nothing, since we always redraw the list.  But
  // the general thing it would do is set the flag that a redraw is needed,
  // so looking ahead I'll do this now.
  if(ccat == colorCat) {
    needRegenerate = TRUE;
  }
}

//////////////////////////////// private routines 

// regenerate the command list
void DrawForce::create_cmdlist(void) {
  // do we need to recreate everything?
  if(needRegenerate) {
    // regenerate both data block and display commands
    needRegenerate = FALSE;
    reset_disp_list();

    // only put in commands if there is a current frame
    Timestep *ts = mol->current();
    if (ts) {
      const float *tsforce = ts->force;

      // if we have a force array, draw arrows for all non-zero forces
      if (tsforce != NULL) {
        append(DMATERIALON);

        // for each atom, if it has a nonzero user force, then display it
        long maxidx = mol->nAtoms * 3L;
        for (int idx=0; idx<maxidx; idx+=3) {
          // check for nonzero forces
          if (tsforce[idx]>0.0f || tsforce[idx+1]>0.0f || tsforce[idx+2]>0.0f ||
              tsforce[idx]<0.0f || tsforce[idx+1]<0.0f || tsforce[idx+2]<0.0f) {
            // get position of atom, and the position of the force vector
            float fval[3], p2[3], p3[3];
            float *p1 = ts->pos + idx;
            for (int k=0; k<3; k++) {
              fval[k] = tsforce[idx + k] * DRAW_FORCE_SCALE;
              p2[k] = p1[k] + fval[k];
              p3[k] = p1[k] + 0.8f * fval[k];
            }

            // find length of force
            float p2norm = norm(fval);

            // set arrow color
            int sc = (int)p2norm;
            if (sc >= MAPCLRS)
              sc = MAPCLRS - 1;
            cmdColorIndex.putdata(MAPCOLOR(sc), cmdList);

            // compute the cone radii
            float rada = 0.2f * p2norm;
            if (rada > 0.3f)
              rada = 0.3f;
            float radb = 1.5f * rada;

            // draw the arrow with two cones
            cmdCone.putdata(p3, p1, rada, 0, 9, cmdList);
            cmdCone.putdata(p3, p2, radb, 0, 9, cmdList);
          }
        }
      }
    }
  }
}


//////////////////////////////// public routines 
// prepare for drawing ... do any updates needed right before draw.
void DrawForce::prepare() {

  if (parent->needUpdate()) {
    needRegenerate = TRUE;
  }
  
  create_cmdlist();
}

