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
 *	$RCSfile: Stage.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.51 $	$Date: 2011/02/17 23:17:19 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * A Displayable3D object which consists of a set of Stage, which may be
 * drawn anywhere on the screen, at any size.
 *
 ***************************************************************************/

#include "Stage.h"
#include "DisplayDevice.h"
#include "Scene.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"


// string descriptions of stage locations
static char *stageloc[Stage::STAGEPOS_TOTAL] = {
  (char *) "Off", (char *) "Origin", (char *) "Bottom", 
  (char *) "Top", (char *) "Left", (char *) "Right", (char *) "Behind" };


// default colors
#define STAGEEVENCOL	REGGREY
#define STAGEODDCOL	REGSILVER

// number of square tiles to draw in each direction
#define STAGE_PANELS    8

Stage::Stage(Displayable *disp) : Displayable(disp) {
  // Displayable characteristics
  rot_off();
  scale_off();
  glob_trans_off();
  cent_trans_off();
  
  // put stage in lower part of image by default
  movedStage = FALSE;           // stage has not been moved yet
  Size = 1.0f;                  // initial size of wall, half of total length
  Panels = 0;
  panels(STAGE_PANELS);		// really sets the panels, and panel size
  stagePos = STAGEPOS_TOTAL;    // (inits value so purify doesn't complain)
  location(NO_STAGE);	        // position the stage

  colorCat = scene->add_color_category("Stage");
  scene->add_color_item(colorCat, "Even", STAGEEVENCOL);
  scene->add_color_item(colorCat, "Odd", STAGEODDCOL);

  do_color_changed(colorCat);
}


//////////////////////////  protected virtual routines  ////////////////////

void Stage::do_color_changed(int clr) {
  if (clr == colorCat) {
    usecolors[0] = scene->category_item_value(colorCat, 
        scene->category_item_index(colorCat, "Even"));
    usecolors[1] = scene->category_item_value(colorCat, 
        scene->category_item_index(colorCat, "Odd"));
    // color changed for us, recreate command list
    need_update = TRUE;
  }
}

//////////////////////////  public virtual routines  ////////////////////

// create the drawing command list
void Stage::create_cmdlist(void) {
  int i, j, k, odd;
  float c[4][3];
  float mycorner1[3] = { -1.0, 0.0, -1.0 };
  mycorner1[0] *= Size;
  mycorner1[1] *= Size;
  mycorner1[2] *= Size;

  memset(c, 0, sizeof(c));
  c[0][1] = c[1][1] = c[2][1] = c[3][1] = 0.0;

  // do reset first
  reset_disp_list();

  // turn on material characteristics
  append(DMATERIALON);

  // draw odd/even squares separately
  for (k=0; k < 2; k++) {
    odd = (k == 1);
    
    // set color in checkerboard
    if (odd)
      cmdColor.putdata(usecolors[0],cmdList);
    else
      cmdColor.putdata(usecolors[1],cmdList);
    
    // start in lower-left corner
    c[0][0] = c[3][0] = mycorner1[0];
    c[1][0] = c[2][0] = mycorner1[0] + inc;

    for (i=0; i < Panels; i++) {
      c[0][2] = c[1][2] = mycorner1[2];
      c[2][2] = c[3][2] = mycorner1[2] + inc;
      for (j=0; j < Panels; j++) {
        if (!odd) {
          cmdSquare.putdata(c[2], c[1], c[0], cmdList);
          pickPoint.putdata(c[2], 0, cmdList);
        }

	odd = !odd;
        c[0][2] += inc;  c[1][2] += inc;  c[2][2] += inc;  c[3][2] += inc;
      }
      c[0][0] += inc;  c[1][0] += inc;  c[2][0] += inc;  c[3][0] += inc;
      if (Panels % 2 == 0)
        odd = !odd;
    }
  }
}


// set stage display mode; return success
int Stage::location(int ap) {
  movedStage = FALSE;

  if (ap != stagePos) {
    stagePos = ap;
    if (ap == NO_STAGE) {
      off();
    } else {
      on();
      need_update = TRUE;
    }
  }

  return TRUE;
}


// return descripton of location
char *Stage::loc_description(int ap) {
  return stageloc[ap];
}


// routine to prepare the displayable object; must set the origin properly
void Stage::prepare() {
  float rot_amount = 0.0, strans[3];
  char rot_axis = 'z';
  
  strans[0] = strans[1] = strans[2] = 0.0;

  // move the stage to its proper position
  if (need_update && stagePos != NO_STAGE) {
    switch (stagePos) {
      case STAGE_ORIGIN:
        // no offset/rotate, draw as-is
        break;

      case STAGE_LOWER:
        strans[1] = -1.0;
        break;

      case STAGE_UPPER:
        strans[1] = 1.0;
        break;

      case STAGE_LEFT:
        strans[0] = -1.0;
        rot_amount = -90.0;
        break;

      case STAGE_RIGHT:
        strans[0] = 1.0;
        rot_amount = 90.0;
        break;

      case STAGE_BEHIND:
        strans[2] = -1.0;
        rot_axis = 'x';
        rot_amount = 90.0;
        break;

      case NO_STAGE:
        return; // exit without drawing

      default:
        msgErr << "Stage: Illegal stage location " << stagePos << " specified."
               << sendmsg;
        stagePos = STAGE_ORIGIN;
        return; // exit without drawing
    }

    // update the current transformation
    need_update = FALSE;
    
    // (re)create the command list
    create_cmdlist();

    if (stagePos == NO_STAGE || movedStage)
      return; // don't update/modify Stage position

    // reset tranformation
    glob_trans_on();
    rot_on();
    reset_transformation();
    set_glob_trans(strans[0], strans[1], strans[2]);
    add_rot(rot_amount, rot_axis);
    rot_off();
    glob_trans_off();
  }
}


//
// When the Stage is picked and moved with a pointer, this is used to move
// the Stage to a new position.  The initial pointer position is remembered,
// and subsequent motions add a global translation to the Stage.
// This is done for any pick mode, and any button.  This is only done if the
// item selected is actually the specific Stage object.
//

// called when a pick moves:
//   args = display to use, obj picked, button, mode, tag, dim, pos
// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
// For 3D version: x,y,z are transformed position of pointer
// For the Stage, when it is selected and the pointer moves, we wish
// to move the Stage as well.
void Stage::pick_move(PickMode *, DisplayDevice *d,
                      int, int dim, const float *pos) {
  float moveStageOrigPos[3], newStageOrigPos[3];
  const float *newpos;

  // calculate amount to translate stage
  if (dim == 2) {
    float origin[3] = { 0.0, 0.0, 0.0 };
    tm.multpoint3d(origin, moveStageOrigPos);
    d->find_3D_from_2D(moveStageOrigPos, pos, newStageOrigPos);
    newpos = newStageOrigPos;
  } else {
    newpos = pos;
  }

  // apply transformation
  glob_trans_on();
  set_glob_trans(newpos[0], newpos[1], newpos[2]);
  glob_trans_off();

  movedStage = TRUE;
}


