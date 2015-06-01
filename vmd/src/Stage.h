/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: Stage.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.38 $	$Date: 2011/02/17 23:17:19 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A Stage is a displayable object that acts as floor for other objects.
 * It is intended to be a point of reference.
 *
 ***************************************************************************/
#ifndef STAGE_H
#define STAGE_H

#include "Displayable.h"
#include "DispCmds.h"

/// Displayable subclass implementing a checkered "stage"
class Stage : public Displayable {
public:
  /// enumerated locations for the stage
  enum StagePos { NO_STAGE = 0, STAGE_ORIGIN, STAGE_LOWER, STAGE_UPPER,
  	STAGE_LEFT, STAGE_RIGHT, STAGE_BEHIND, STAGEPOS_TOTAL };

private:
  // corners defining the stage
  float corner1[3], corner2[3], corner3[3], corner4[3];
  int usecolors[2];           ///< color indices for even/odd panels
  int Panels;                 ///< number of panels each side is divided into
  float Size;                 ///< size of stage walls
  float inc;                  ///< width of panel
  int stagePos;               ///< current stage position
  int need_update;            ///< do we need an update
  DispCmdColorIndex cmdColor; ///< display command to set panel color
  DispCmdSquare cmdSquare;    ///< display command to draw panels
  DispCmdPickPoint pickPoint; ///< pick points to allow Stage to be moved
  int colorCat;               ///< color category to use, if < 0, use defaults
  int movedStage;             ///< whether stage was moved by mouse/pointer

  /// regenerate the command listo
  void create_cmdlist(void);

protected:
  virtual void do_color_changed(int);

public:
  /// constructor: the parent displayable 
  Stage(Displayable *);

  /// set stage display mode; return success
  int location(int);

  /// return stage display mode
  int location(void) { return stagePos; }

  /// return descripton of location
  char *loc_description(int);

  /// return total number of locations
  int locations(void) { return STAGEPOS_TOTAL; }

  /// get number of panels (must be >= 1)
  int panels(void) { return Panels; }

  /// set number of panels (must be >= 1)
  int panels(int newp) {
    if (newp == Panels) return TRUE;
    if (newp >= 1 && newp <= 30) {
      Panels = newp;
      inc = (Size * 2.0f) / (float) Panels;
      need_update = TRUE;
      return TRUE; // success
    }
    return FALSE; // failed
  }


  /// return the size of the walls (half of side length) 
  float size(void) { return Size; }

  /// set the size of the walls (half of side length)
  float size(float newsize) {
    Size = newsize;
    inc = (Size * 2.0f) / (float) Panels;
    need_update = TRUE;
    return TRUE; // success
  }

  //
  // public virtual routines
  //
  
  /// prepare for drawing ... do any updates needed right before draw.
  virtual void prepare();

  /// called when a pick moves:
  ///   args = display to use, obj picked, button, mode, tag, dim, pos
  /// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
  /// For 3D version: x,y,z are transformed position of pointer
  /// For the stage, when they are selected and the pointer moves, we wish
  /// to move the stage as well.
  virtual void pick_move(PickMode *, DisplayDevice *, int, int, const float *);

};

#endif

