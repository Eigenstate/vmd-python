/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: PickModeCenter.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.32 $	$Date: 2016/11/28 03:05:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Pick on an atom to change the molecule's centering/global matricies
 ***************************************************************************/

#include <math.h>
#include <string.h>
#include "PickModeCenter.h"
#include "Pickable.h"
#include "VMDApp.h"
#include "MoleculeList.h"

PickModeCenter::PickModeCenter(VMDApp *vmdapp) : app(vmdapp) {}

void PickModeCenter::pick_molecule_start(DrawMolecule *, DisplayDevice *,
               int, int tag, const int * cell, int dim, const float *pos) {
  pAtom = tag;
  memcpy(pPos, pos, dim*sizeof(float));
  memcpy(pCell, cell, 3L*sizeof(int));
  needName = TRUE; 
}

void PickModeCenter::pick_molecule_move(DrawMolecule *, DisplayDevice *,
                                        int, int dim, const float *pos) {
  if(needName) {
    float mvdist = 0.0;
    for(int i=0; i < dim; i++)
      mvdist += (float) fabs(pPos[i] - pos[i]);
    if(mvdist > 0.02 )
      needName = FALSE;
  }
}

void PickModeCenter::pick_molecule_end(DrawMolecule *mol, DisplayDevice *) {
  if (!needName) return;

  const Timestep *ts = mol->current();
  const float *coord = ts->pos + pAtom * 3L;
  float tcoord[3]; // transformed coordinate
  Matrix4 mat;
  ts->get_transform_from_cell(pCell, mat);
  mat.multpoint3d(coord, tcoord);

  // and apply the result to all the active molecules
  // XXX There should be a high-level command for this.
  DrawMolecule *m;
  for (int m_id = 0; m_id < app->moleculeList->num(); m_id++) {
    m = app->moleculeList->molecule(m_id);
    if (m->active) {
      m->change_center(tcoord[0], tcoord[1], tcoord[2]);
    }
  }
}

