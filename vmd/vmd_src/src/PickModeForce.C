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
 *      $RCSfile: PickModeForce.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Pick mode for applying forces to atoms
 ***************************************************************************/

#include "PickModeForce.h"
#include "Atom.h"
#include "Molecule.h"
#include "DisplayDevice.h"
#include "Mouse.h"

void PickModeForce::get_force(const float *atomPos, const float *pos,
                              DrawMolecule *m, int dim, DisplayDevice *d, 
                              float *newforce) {

  float atomWorldPos[3], mouseWorldPos[3];
  const float *newpos;
  if(dim == 2) {

    // convert atom position to world coordinates
    m->tm.multpoint3d(atomPos, atomWorldPos);

    // find the 3D position (in world coords) of the mouse
    d->find_3D_from_2D(atomWorldPos, pos, mouseWorldPos);

    // indicate this new position as the mouse position
    newpos = mouseWorldPos;
  } else {
    // for 3D pointer, just use the pointer position as-is, but
    newpos = pos;
  }

  // now convert back from world coords to object coords
  Matrix4 tminv(m->tm);
  tminv.inverse();
  tminv.multpoint3d(newpos, newforce);
}

// This #define is copied from DrawForce

#define FORCE_SCALE (10.0f / 14.0f)

static const float zeroforce[] = {0.0f, 0.0f, 0.0f};

void PickModeForceAtom::pick_molecule_start(DrawMolecule *m, DisplayDevice *,
                             int btn, int tag, const int *cell, int /* dim */, 
                             const float * /* pos */ ) {

  // if the middle or right button is down, clear the force
  if ((btn == Mouse::B_MIDDLE || btn == Mouse::B_RIGHT) && tag >= 0) {
    // XXX downcast to Molecule
    ((Molecule *)m)->addPersistentForce(tag, zeroforce);
    mytag = -1;
  } else if (cell[0] || cell[1] || cell[2]) {
    // don't apply force to periodic images
    mytag = -1;
  } else {
    mytag = tag;
  }
}

void PickModeForceAtom::pick_molecule_move(DrawMolecule *m, DisplayDevice *d,
                                           int tag, int dim, 
                                           const float *pos) {
 
  if (mytag < 0) return;
  float *atomPos = m->current()->pos + 3*tag;
  float newforce[3];
  get_force(atomPos, pos, m, dim, d, newforce);
  newforce[0] = FORCE_SCALE * (newforce[0] - atomPos[0]); 
  newforce[1] = FORCE_SCALE * (newforce[1] - atomPos[1]); 
  newforce[2] = FORCE_SCALE * (newforce[2] - atomPos[2]); 
  // XXX downcast to Molecule
  ((Molecule *)m)->addPersistentForce(tag, newforce);
}

void PickModeForceResidue::pick_molecule_start(DrawMolecule *m, DisplayDevice *,
                             int btn, int tag, const int *, int /* dim */, 
                             const float * /* pos */ ) {

  // if the middle or right button is down, clear the force
  if ((btn == Mouse::B_MIDDLE || btn == Mouse::B_RIGHT) && tag >= 0) {
    Residue *res = m->atom_residue(tag);
    if (res) {
      int natm = (res->atoms).num();
      for (int i=0; i < natm; i++)
        ((Molecule *)m)->addPersistentForce(res->atoms[i], zeroforce);
    }
    mytag = -1;
  } else {
    mytag = tag;
  }
}

void PickModeForceResidue::pick_molecule_move(DrawMolecule *m, DisplayDevice *d,
                                           int tag, int dim, 
                                           const float *pos) {
 
  if (mytag < 0) return;
  float *atomPos = m->current()->pos + 3*tag;
  float newforce[3];
  get_force(atomPos, pos, m, dim, d, newforce);
  newforce[0] = FORCE_SCALE * (newforce[0] - atomPos[0]); 
  newforce[1] = FORCE_SCALE * (newforce[1] - atomPos[1]); 
  newforce[2] = FORCE_SCALE * (newforce[2] - atomPos[2]); 
  Residue *res = m->atom_residue(tag);
  if (res) {
    int natm = (res->atoms).num();
    for (int i=0; i < natm; i++)
      ((Molecule *)m)->addPersistentForce(res->atoms[i], newforce);
  }
}

void PickModeForceFragment::pick_molecule_start(DrawMolecule *m, 
                                         DisplayDevice *, int btn, 
                                         int tag, const int *, int /* dim */, 
                                         const float * /* pos */ ) {

  // if the middle or right button is down, clear the force
  if ((btn == Mouse::B_MIDDLE || btn == Mouse::B_RIGHT) && tag >= 0) {
    Fragment *frag = m->atom_fragment(tag);
    if (frag) {
      int nres = frag->num();
      for (int r=0; r < nres; r++) {
        Residue *res = m->residue((*frag)[r]);
        int natm = (res->atoms).num();
        for (int i=0; i < natm; i++)
          ((Molecule *)m)->addPersistentForce(res->atoms[i], zeroforce);
      }  // loop over residues
    }    // if (frag)
    mytag = -1;
  } else {     
    mytag = tag;
  }
}

void PickModeForceFragment::pick_molecule_move(DrawMolecule *m, 
                                           DisplayDevice *d,
                                           int tag, int dim, 
                                           const float *pos) {
 
  if (mytag < 0) return; 
  float *atomPos = m->current()->pos + 3*tag;
  float newforce[3];
  get_force(atomPos, pos, m, dim, d, newforce);
  newforce[0] = FORCE_SCALE * (newforce[0] - atomPos[0]); 
  newforce[1] = FORCE_SCALE * (newforce[1] - atomPos[1]); 
  newforce[2] = FORCE_SCALE * (newforce[2] - atomPos[2]); 
  Fragment *frag = m->atom_fragment(tag);
  if (frag) {
    int nres = frag->num();
    for (int r=0; r < nres; r++) {
      Residue *res = m->residue((*frag)[r]);
      int natm = (res->atoms).num();
      for (int i=0; i < natm; i++)
        ((Molecule *)m)->addPersistentForce(res->atoms[i], newforce);
    }  // loop over residues
  }    // if (frag)
}
