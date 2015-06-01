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
 *	$RCSfile: Axes.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.70 $	$Date: 2011/02/17 21:58:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * A Displayable object which consists of a set of axes, which may be
 * drawn anywhere on the screen, at any size.
 *
 ***************************************************************************/

#include <string.h>
#include <stdio.h>
#include "Axes.h"
#include "DisplayDevice.h"
#include "DispCmds.h"
#include "utilities.h"
#include "Scene.h"

// radius for the cylinders that make up the axes
#define AXESRAD         0.12f
#define AXESCAPRAD      0.25f
#define AXESRODLEN      0.85f
#define AXESTXTLEN      1.35f
#define AXESRES         32
#define AXESSPHRES      6

// default colors to use
#define AXESXCOL        REGRED
#define AXESYCOL        REGGREEN
#define AXESZCOL        REGBLUE
#define AXESOCOL        REGCYAN
#define AXESTCOL        REGWHITE


// string descriptions of axes locations
static char *axesloc[Axes::AXESPOS_TOTAL] = {
  (char *) "Off",        (char *) "Origin",    (char *) "LowerLeft", 
  (char *) "LowerRight", (char *) "UpperLeft", (char *) "UpperRight"};


//////////////////////////  constructor
Axes::Axes(DisplayDevice *d, Displayable *par) : Displayable(par), disp(d) {
  colorCat = (-1); // indicate we don't yet have a color object to use
  need_create_cmdlist = TRUE;
  movedAxes = FALSE;

  // initialize coordinates for axes lines
  origin[0] = yLine[0] = zLine[0] = 0.0;  
  xLineCap[1] = xLineCap[2] = 0.0;
  xLine[0] = AXESRODLEN; xLineCap[0] = 1.0;

  origin[1] = xLine[1] = zLine[1] = 0.0;  
  yLineCap[0] = yLineCap[2] = 0.0;
  yLine[1] = AXESRODLEN; yLineCap[1] = 1.0;

  origin[2] = xLine[2] = yLine[2] = 0.0;
  zLineCap[0] = zLineCap[1] = 0.0;
  zLine[2] = AXESRODLEN; zLineCap[2] = 1.0;

  xText[0] = AXESTXTLEN * xLine[0]; xText[1] = xLine[1]; xText[2] = xLine[2];
  yText[1] = AXESTXTLEN * yLine[1]; yText[0] = yLine[0]; yText[2] = yLine[2];
  zText[2] = AXESTXTLEN * zLine[2]; zText[0] = zLine[0]; zText[1] = zLine[1];
  
  // Displayable characteristics
  rot_on();
  glob_trans_off();
  cent_trans_off();
  
  // set scaling factor to a small amount
  scale_on();
  set_scale(0.25);
  scale_off();
  
  // put axes in lower left corner by default
  axesPos = AXES_LOWERLEFT;
  Aspect = (-1.0);

  colorCat = scene->add_color_category("Axes");
  
  // add components, and their default colors
  scene->add_color_item(colorCat, "X", AXESXCOL);
  scene->add_color_item(colorCat, "Y", AXESYCOL);
  scene->add_color_item(colorCat, "Z", AXESZCOL);
  scene->add_color_item(colorCat, "Origin", AXESOCOL);
  scene->add_color_item(colorCat, "Labels", AXESTCOL);

  do_color_changed(colorCat);
}


//////////////////////////  destructor
Axes::~Axes(void) {
  // do nothing
}


//////////////////////////  protected virtual routines  ////////////////////

void Axes::do_color_changed(int clr) {
  if (clr == colorCat) {
    usecolors[0] = scene->category_item_value(colorCat, "X");
    usecolors[1] = scene->category_item_value(colorCat, "Y");
    usecolors[2] = scene->category_item_value(colorCat, "Z");
    usecolors[3] = scene->category_item_value(colorCat, "Origin");
    usecolors[4] = scene->category_item_value(colorCat, "Labels");

    // color changed for us, recreate command list
    need_create_cmdlist = TRUE;
  }
}

//////////////////////////  public virtual routines  ////////////////////

// create the drawing command list
void Axes::create_cmdlist(void) {
  char commentBuffer[128];

  reset_disp_list(); // regenerate both data block and display commands
  sprintf(commentBuffer, "VMD: Starting axes output.");
  cmdCommentX.putdata(commentBuffer, cmdList);

  cmdBeginRepGeomGroup.putdata("vmd_axes", cmdList);

#if defined(USELINEAXES)
  DispCmdLine cmdline;
  float arrow1[3], arrow2[3];

  // turn on material characteristics
  append(DMATERIALOFF);

  // Draw axes as lines of width 2, faster for wimpy machines...
  DispCmdLineWidth cmdwidth;
  cmdwidth.putdata(2, cmdList);

  // x-axis
  xcol.putdata(usecolors[0], cmdList);
  cmdline.putdata(origin, xLineCap, cmdList);
  arrow1[0] = xLine[0];
  arrow1[1] = xLine[1] - AXESCAPRAD/sqrt(2.0f);
  arrow1[2] = xLine[2] + AXESCAPRAD/sqrt(2.0f);
  arrow2[0] = xLine[0];
  arrow2[1] = xLine[1] + AXESCAPRAD/sqrt(2.0f);
  arrow2[2] = xLine[2] - AXESCAPRAD/sqrt(2.0f);
  cmdline.putdata(xLineCap, arrow1, cmdList);
  cmdline.putdata(xLineCap, arrow2, cmdList);

  // y-axis
  xcol.putdata(usecolors[1], cmdList);
  cmdline.putdata(origin, yLineCap, cmdList);
  arrow1[0] = yLine[0] + AXESCAPRAD/sqrt(2.0f);
  arrow1[1] = yLine[1];
  arrow1[2] = yLine[2] - AXESCAPRAD/sqrt(2.0f);
  arrow2[0] = yLine[0] - AXESCAPRAD/sqrt(2.0f);
  arrow2[1] = yLine[1];
  arrow2[2] = yLine[2] + AXESCAPRAD/sqrt(2.0f);
  cmdline.putdata(yLineCap, arrow1, cmdList);
  cmdline.putdata(yLineCap, arrow2, cmdList);

  // z-axis
  xcol.putdata(usecolors[2], cmdList);
  cmdline.putdata(origin, zLineCap, cmdList);
  arrow1[0] = zLine[0] - AXESCAPRAD/sqrt(2.0f);
  arrow1[1] = zLine[1] + AXESCAPRAD/sqrt(2.0f);
  arrow1[2] = zLine[2];
  arrow2[0] = zLine[0] + AXESCAPRAD/sqrt(2.0f);
  arrow2[1] = zLine[1] - AXESCAPRAD/sqrt(2.0f);
  arrow2[2] = zLine[2];
  cmdline.putdata(zLineCap, arrow1, cmdList);
  cmdline.putdata(zLineCap, arrow2, cmdList);
#else
  // Draw axes as solid cylinders/cones, for faster machines...
  // turn on material characteristics
  append(DMATERIALON);

  // set sphere type and resolution
  sphres.putdata(AXESSPHRES,  cmdList);
  sphtype.putdata(SOLIDSPHERE, cmdList);

  // put in commands to draw lines
  // x-axis
  xcol.putdata(usecolors[0], cmdList);
  xcyl.putdata(origin, xLine, AXESRAD, AXESRES, 0, cmdList);
  xcap.putdata(xLine, xLineCap, AXESCAPRAD, 0, AXESRES, cmdList);

  // y-axis
  xcol.putdata(usecolors[1], cmdList);
  xcyl.putdata(origin, yLine, AXESRAD, AXESRES, 0, cmdList);
  xcap.putdata(yLine, yLineCap, AXESCAPRAD, 0, AXESRES, cmdList);

  // z-axis
  xcol.putdata(usecolors[2], cmdList);
  xcyl.putdata(origin, zLine, AXESRAD, AXESRES, 0, cmdList);
  xcap.putdata(zLine, zLineCap, AXESCAPRAD, 0, AXESRES, cmdList);

  // put in commands to draw sphere at origin
  xcol.putdata(usecolors[3], cmdList);
  sph.putdata(origin, AXESRAD, cmdList);

  // turn off material characteristics
  append(DMATERIALOFF);
#endif

  // put in commands to label the axes
  xcol.putdata(usecolors[4], cmdList);
  txt.putdata(xText, "x", 1.0f, cmdList);
  txt.putdata(yText, "y", 1.0f, cmdList);
  txt.putdata(zText, "z", 1.0f, cmdList);

  // put in commands to draw pickpoints at axes endpoints
  pickPoint.putdata(origin,   0, cmdList);
  pickPoint.putdata(xLine,    1, cmdList);
  pickPoint.putdata(xLineCap, 2, cmdList);
  pickPoint.putdata(yLine,    3, cmdList);
  pickPoint.putdata(yLineCap, 4, cmdList);
  pickPoint.putdata(zLine,    5, cmdList);
  pickPoint.putdata(zLineCap, 6, cmdList);

  // done drawing axes
  sprintf(commentBuffer, "VMD: Done with axes.");
  cmdCommentX.putdata(commentBuffer, cmdList);

  need_create_cmdlist = FALSE;
}


// set axes display mode; return success
int Axes::location(int ap) {
  axesPos = ap;
  movedAxes = FALSE;
  if (ap == NO_AXES) {
    off();
  } else {
    on();
    Aspect = -1.0;
  }

  return TRUE;
}


// return descripton of location
char *Axes::loc_description(int ap) {
  return axesloc[ap];
}


//////////////////  public virtual Displayable routines  ////////////////////

// routine to prepare the displayable object; must set the origin properly
void Axes::prepare() {
  float asp, xpos, ypos;
  float poscale = 0.95f;

  // recreate command list if needed
  if (need_create_cmdlist)
    create_cmdlist();

  if (axesPos == NO_AXES || movedAxes) {
    return; // don't update/modify the Axes position
  }
  
  if ((asp = disp->aspect()) != Aspect) {
    // move the axes to their proper position
    switch (axesPos) {
      case AXES_LOWERLEFT:
        xpos = -poscale * asp;
        ypos = -poscale;
        break;
      case AXES_LOWERRIGHT:
        xpos = poscale * asp;
        ypos = -poscale;
        break;
      case AXES_UPPERLEFT:
        xpos = -poscale * asp;
        ypos = poscale;
        break;
      case AXES_UPPERRIGHT:
        xpos = poscale * asp;
        ypos = poscale;
        break;
      default:
        xpos = ypos = 0.0;
    }

    // update the current transformation
    Aspect = asp;
    glob_trans_on();
    set_glob_trans(xpos, ypos, 0.0);
    glob_trans_off();
  }
}


//////////////////  public virtual Pickable routines  ////////////////////


//
// When the Axes are picked and moved with a pointer, this is used to move
// the Axes to a new position.  The initial pointer position is remembered,
// and subsequent motions add a global translation to the Axes.
// This is done for any pick mode, and any button.  This is only done if the
// item selected is actually the specific Axes object.
//

// called when a pick moves:
//   args = display to use, obj picked, button, mode, tag, dim, pos
// For 2D version: x & y are 0 ... 1, represent 'relative, scaled' coords.
// For 3D version: x,y,z are transformed position of pointer
// For the Axes, when they are selected and the pointer moves, we wish
// to move the axes as well.
void Axes::pick_move(PickMode *, DisplayDevice *d, 
                     int, int dim, const float *pos) {
  float moveAxesOrigPos[3], newAxesOrigPos[3];
  const float *newpos;

  // calculate amount to translate axes
  if (dim == 2) {
    tm.multpoint3d(origin, moveAxesOrigPos);
    d->find_3D_from_2D(moveAxesOrigPos, pos, newAxesOrigPos);
    newpos = newAxesOrigPos;
  } else {
    newpos = pos;
  }
  
  // apply transformation
  glob_trans_on();
  set_glob_trans(newpos[0], newpos[1], newpos[2]);
  glob_trans_off();
  
  movedAxes = TRUE;
}

