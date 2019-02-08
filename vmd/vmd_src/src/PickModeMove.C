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
 *      $RCSfile: PickModeMove.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.24 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Pick mode for moving atoms (no forces applied)
 ***************************************************************************/

#include <math.h>
#include <string.h>
#include "PickModeMove.h"
#include "DrawMolecule.h"
#include "utilities.h"
#include "VMDQuat.h"
#include "Mouse.h"
#include "DisplayDevice.h"

///////////////////// public virtual routines  ////////////////////////

void PickModeMove::get_pointer_pos(DrawMolecule *m, DisplayDevice *d, 
                                   int moveAtom, const int *cell,
                                   int dim, const float *pos,
                                   float *retpos) {

  Timestep *ts = m->current();
  if(!(moveAtom < 0 || ts == NULL)) {
    // get the current pointer position in 3 space
    float atomPos[3];
    vec_copy(atomPos, ts->pos + moveAtom * 3L); 
    const float *newpos;
    float mouseObjPos[3];
    if(dim == 2) {
      float atomWorldPos[3];

      // Apply unit cell transform
      Matrix4 mat;
      ts->get_transform_from_cell(cell, mat);
      mat.multpoint3d(atomPos, atomPos);

      // convert the atom position to world coordinates
      (m->tm).multpoint3d(atomPos, atomWorldPos);

      // find the 3D position (in world coords) of the mouse
      d->find_3D_from_2D(atomWorldPos, pos, mouseObjPos);

      // indicate the mouse world pos is the position to convert back to
      // molecule object coordinates
      newpos = mouseObjPos;

    } else {

      // for 3D pointer, just use the pointer position as-is
      newpos = pos;
    }
    vec_copy(retpos, newpos);
  } 
}

void PickModeMove::pick_molecule_start(DrawMolecule *m, DisplayDevice *d, 
                                       int theBtn, int tag, const int *cell,
                                       int dim, const float *pos) {

  // convert the pointer position to world coordinates; store in lastPos
  memcpy(lastCell, cell, 3L*sizeof(int));
  get_pointer_pos(m, d, tag, cell, dim, pos, lastPos);
  btn = theBtn;
}

///////////////////////////////////
// compute the transformation matrix for a rotation with the mouse
// newpos should be in world coordinates
Quat PickModeMove::calc_rot_quat(int dim, int b, DisplayDevice *d,
                                 const float *mat, const float *newpos) {
  Quat quatx, quaty, quatz;                               
  Quat transquat;
  if (dim == 2) {                           
    if ((b == Mouse::B_MIDDLE || b == Mouse::B_RIGHT) || 
        d->shift_state() & DisplayDevice::ALT) {    
      quatz.rotate('z',(newpos[0] - lastPos[0])*150); 
    } else {                                
      quaty.rotate('y',(newpos[0] - lastPos[0])*150);  
      quatx.rotate('x',(-newpos[1] + lastPos[1])*150);
    }                                       
  } else {                                 
    quatx.rotate('x', (newpos[0] - lastPos[0])*150);  
    quaty.rotate('y', (newpos[1] - lastPos[1])*150);  
    quatz.rotate('z', (newpos[2] - lastPos[2])*150);  
  }                                         
  Quat rotq;                               
  rotq.fromMatrix(mat);     
  transquat = rotq;                  
  transquat.mult(quatz);            
  transquat.mult(quaty);           
  transquat.mult(quatx);         
  rotq.invert();               
  transquat.mult(rotq);      
  return transquat;
}  

void PickModeMove::pick_molecule_move(DrawMolecule *m, DisplayDevice *d,
			int tag, int dim, const float *pos) {
  
  float newpos[3], atomPos[3];

  // Convert the pointer position into world coordinates
  get_pointer_pos(m,d,tag,lastCell,dim,pos,newpos);

  // Copy the current atom coordinates into a buffer
  Timestep *ts = m->current();
  memcpy(atomPos, ts->pos + 3L*tag, 3L*sizeof(float));

  // if the shift key is pressed, do rotations
  if (d->shift_state() & DisplayDevice::SHIFT) {
    Quat transquat = calc_rot_quat(dim,btn,d,(m->rotm).mat, newpos);
    rotate(m, tag, atomPos, transquat);
  } 
  else { // do translations
    // convert the pointer position to object coordinates, subtract the 
    // coordinates of the picked atom, and add the result to all atoms in
    // the residue.

    // Apply unit cell transform
    Matrix4 mat;
    ts->get_transform_from_cell(lastCell, mat);
    mat.multpoint3d(atomPos, atomPos);

    Matrix4 tminv(m->tm);
    tminv.inverse();

    float moveAmount[3];
    tminv.multpoint3d(newpos, moveAmount);
    vec_sub(moveAmount, moveAmount, atomPos);
 
    translate(m, tag, moveAmount);
  }
 
  vec_copy(lastPos, newpos);

  // tell the the molecule to update now - don't wait for the pick to end
  // XXX some reps (e.g., volumetric reps) don't care about changed coordinates
  m->force_recalc(DrawMolItem::MOL_REGEN | DrawMolItem::SEL_REGEN);
}

void PickModeMoveAtom::translate(DrawMolecule *m, int tag, const float *p) {
  float *pos = m->current()->pos+3L*tag;
  vec_add(pos, pos, p);
}

void PickModeMoveResidue::rotate(DrawMolecule *m, int tag, const float *p,
                                 const Quat &q) {

  Residue *res = m->atom_residue(tag);
  if (res) {
    int natm = (res->atoms).num();
    for (int n=0; n<natm; n++) {
      float *ap = m->current()->pos + 3L * (res->atoms)[n];
      // Rotate about the picked atom
      vec_sub(ap, ap, p);
      q.multpoint3(ap,ap);
      vec_add(ap, ap, p);
    }
  }
}

void PickModeMoveResidue::translate(DrawMolecule *m, int tag, const float *p) {

  Residue *res = m->atom_residue(tag);
  if(res) {
    int natm = (res->atoms).num();
    for(int n = 0; n < natm; n++) {
      float *ap = m->current()->pos + 3L * (res->atoms)[n];
      vec_add(ap, ap, p);
    }
  }
}

void PickModeMoveFragment::rotate(DrawMolecule *m, int tag, const float *p, 
                                  const Quat &q) {

  Fragment *frag = m->atom_fragment(tag);
  if (frag) {
    int nres = frag->num();
    for (int r=0; r < nres; r++) {
      Residue *res = m->residue((*frag)[r]);
      int natm = (res->atoms).num();
      for (int n=0; n < natm; n++) {
        float *ap = m->current()->pos + 3L * (res->atoms)[n];
        vec_sub(ap,ap,p);
        q.multpoint3(ap,ap); 
        vec_add(ap,ap,p);
      }
    }
  }
}

void PickModeMoveFragment::translate(DrawMolecule *m, int tag, const float *p){
  Fragment *frag = m->atom_fragment(tag);
  if (frag) {
    int nres = frag->num();
    for (int r=0; r < nres; r++) {
      Residue *res = m->residue((*frag)[r]);
      int natm = (res->atoms).num();
      for (int n=0; n < natm; n++) {
        float *ap = m->current()->pos + 3L * (res->atoms)[n];
        vec_add(ap,ap,p);
      }
    }
  }
}

void PickModeMoveMolecule::rotate(DrawMolecule *m, int, const float *p, 
                                  const Quat &q) {
  int natm = m->nAtoms;
  for (int i=0; i < natm; i++) {
    float *ap = m->current()->pos + 3L*i;
    vec_sub(ap,ap,p);
    q.multpoint3(ap,ap); 
    vec_add(ap,ap,p);
  }
}

void PickModeMoveMolecule::translate(DrawMolecule *m, int, const float *p){ 

  int natm = m->nAtoms;
  for (int i=0; i < natm; i++) {
    float *ap = m->current()->pos + 3L*i;
    vec_add(ap,ap,p);
  }
}

void PickModeMoveRep::rotate(DrawMolecule *m, int tag, const float *p, 
                                  const Quat &q) {
  if (!m->components()) 
    return;
  int rep = m->highlighted_rep();
  if (rep < 0 || rep >= m->components())
    return;

  const AtomSel *sel = m->component(rep)->atomSel;
  const int n = sel->num_atoms;
  const int *on = sel->on;
  if (!on[tag]) 
    return;
  float *ap = m->current()->pos;
  for (int i=0; i<n; i++) {
    if (on[i]) {
      float *pos = ap + 3L*i;
      vec_sub(pos, pos, p);
      q.multpoint3(pos, pos);
      vec_add(pos, pos, p);
    }
  }
} 
   
void PickModeMoveRep::translate(DrawMolecule *m, int tag, const float *p) {

  if (!m->components()) 
    return;
  int rep = m->highlighted_rep();
  if (rep < 0 || rep >= m->components())
    return;

  const AtomSel *sel = m->component(rep)->atomSel;
  const int n = sel->num_atoms;
  const int *on = sel->on;
  if (!on[tag]) 
    return;
  float *ap = m->current()->pos;
  for (int i=0; i<n; i++) {
    if (on[i]) {
      vec_add(ap+3L*i, ap+3L*i, p);
    }
  }
}

void PickModeMove::pick_molecule_end(DrawMolecule *, DisplayDevice *) {
  // pick_move takes care of everything
}

