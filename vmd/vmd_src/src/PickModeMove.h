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
 *      $RCSfile: PickModeMove.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $       $Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Pick mode for moving atoms (no forces applied)
 ***************************************************************************/

#ifndef PICKMODEMOVE_H
#define PICKMODEMOVE_H

#include "PickMode.h"
class DrawMolecule;
class Quat;

/// PickMode subclass for moving atoms (no forces applied)
class PickModeMove : public PickMode {
private:
  float lastPos[3]; ///< world position of the pointer for the previous call
  int lastCell[3];  ///< unit cell of the atom we're dragging
  int btn;          ///< which button
  
  /// convert the current pointer position to world coordinates
  void get_pointer_pos(DrawMolecule *, DisplayDevice *, int atm, 
                       const int *cell,int dim,
                       const float *pos, float *newpos); 

  /// compute a rotation transformation, based on the difference between
  /// lastPos and the current pointer position
  Quat calc_rot_quat(int dim, int btn, DisplayDevice *, const float *mat, 
                     const float *pos); 

protected:
  PickModeMove() {} 

  /// subclasses figure out which atoms are to be translated or rotated.
  virtual void translate(DrawMolecule *, int tag, const float *) = 0;
  virtual void rotate(DrawMolecule *, int tag, const float *, const Quat &) = 0;

public:
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *,
                             int btn, int tag, const int *cell, 
                             int dim, const float *pos);
  virtual void pick_molecule_move(DrawMolecule *, DisplayDevice *,
                             int tag, int dim, const float *pos);
  virtual void pick_molecule_end(DrawMolecule *, DisplayDevice *);
};


/// PickMode subclass for moving atoms (no forces applied)
class PickModeMoveAtom : public PickModeMove {
protected:
  virtual void translate(DrawMolecule *, int tag, const float *);
  virtual void rotate(DrawMolecule *, int, const float *, const Quat &) {}
  
public:
  PickModeMoveAtom() {}
};


/// PickMode subclass for moving residues (no forces applied)
class PickModeMoveResidue : public PickModeMove {
protected:
  virtual void translate(DrawMolecule *, int tag, const float *);
  virtual void rotate(DrawMolecule *, int tag, const float *, const Quat &);

public:
  PickModeMoveResidue() {}
};


/// PickMode subclass for moving fragments (no forces applied)
class PickModeMoveFragment : public PickModeMove {
protected:
  virtual void translate(DrawMolecule *, int tag, const float *);
  virtual void rotate(DrawMolecule *, int tag, const float *, const Quat &);

public:
  PickModeMoveFragment() {}
};


/// PickMode subclass for moving whole molecules (no forces applied)
class PickModeMoveMolecule : public PickModeMove {
protected:
  virtual void translate(DrawMolecule *, int tag, const float *);
  virtual void rotate(DrawMolecule *, int tag, const float *, const Quat &);

public:
  PickModeMoveMolecule() {}
};


/// PickMode subclass for moving a selected rep's atoms (no forces applied)
class PickModeMoveRep : public PickModeMove {
protected:
  virtual void translate(DrawMolecule *, int tag, const float *);
  virtual void rotate(DrawMolecule *, int tag, const float *, const Quat &);
  
public:
  PickModeMoveRep() {}
};

#endif

