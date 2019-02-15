/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#include "SpringTool.h"
#include "Matrix4.h"
#include "utilities.h"
#include "Molecule.h"
#include "VMDApp.h"
#include "PickList.h"
#include "MoleculeList.h"

SpringTool::SpringTool(int id, VMDApp *vmdapp, Displayable *disp) 
: Tool(id, vmdapp, disp) {
  offset[0]=offset[1]=offset[2]=0;
  tugging=0;
  springscale = 1.0;
}


static int get_nearby_atom(VMDApp *app, const float *pos,
             int *molret, int *atomret) {
  int mol = -1;
  int atom = -1;
  int tag; 
  
  Pickable *p;
  if (app->display) {
    // pick a nearby atom
    p = app->pickList->pick_check(3, pos, tag, NULL, 0.2f);

    if (p) {
      Molecule *m = app->moleculeList->check_pickable(p);
      if (m) {
        mol = m->id();
        atom = tag;
      }
    }
  }

  if (atom == -1 || mol == -1) return 0;
  *molret = mol;
  *atomret = atom;
  return 1;
}


void SpringTool::do_event() {
  // always update!
  tool_location_update();

  if (istugging()) {  // Tugging is enabled...
    if (!tugging || !is_targeted()) { // but we're not tugging now
      if (!target(TARGET_TUG, tugged_pos, 0)) {
        // Didn't pick anything, so return
        tugging = 0;
        return;
      }

      tugging = 1;
      // We're starting the force field, so set the offset
      vec_sub(offset, Tool::position(), tugged_pos);
      start_tug();
    }
    target(TARGET_TUG, tugged_pos, 1);
    // Apply the force field...
    float offset_tugged_pos[3]; // offset+tugged_pos
    vec_add(offset_tugged_pos,offset,tugged_pos);
    set_tug_constraint(offset_tugged_pos);
    
    float diff[3];
    vec_sub(diff, Tool::position(), offset_tugged_pos);
    // diff now is in my units, but we should do it in mol units
    vec_scale(diff, dtool->scale/getTargetScale(), diff);
    // scale by the force scaling spring constant
    vec_scale(diff, forcescale, diff); 
    do_tug(diff);
  } else if (tugging) { // Tugging has been disabled
    int mol1=-1, mol2=-1;
    int atom1=-1, atom2=-1;
    int ret1 = get_nearby_atom(app, position(), &mol1, &atom1);
    int ret2 = get_targeted_atom(&mol2, &atom2);

    if (ret1 && ret2) { // got two atoms successfully
      if (mol1 == mol2) { // on the same molecule
	if (atom1 != atom2) { // two *different* atoms
          int atombuf[2];
          int molbuf[2];
          atombuf[0] = atom1; atombuf[1] = atom2;
          molbuf[0] = mol1; molbuf[1] = mol2;
          app->label_add("Springs", 2, molbuf, atombuf, NULL, 0.0f, 1);
        }	
      }
    }

    // now let go
    let_go();
    tugging = 0;
    forceoff();
    offset[0]=offset[1]=offset[2]=0;
  } 
}


void SpringTool::set_tug_constraint(float *newpos) {
  setconstraint(50, newpos);
  sendforce();
}


void SpringTool::do_tug(float *force) {
  tug(force);
}

