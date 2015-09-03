/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#include <stdio.h>
#include "FPS.h"
#include "DisplayDevice.h"
#include "Displayable.h"
#include "DispCmds.h"
#include "Scene.h"

FPS::FPS(DisplayDevice *d, Displayable *par) 
: Displayable(par), disp(d) {
  last_update = time_of_day(); // setup time of day with initial value
  loop_count = 0;

  // disable transformations on this displayable
  rot_off();
  glob_trans_off();
  cent_trans_off();
  scale_off();

  // set the text color category, index, etc. 
  colorCat = scene->add_color_category("Display");
  if (colorCat < 0)
    colorCat = scene->category_index("Display");

  usecolor = scene->add_color_item(colorCat, "FPS", REGWHITE);
  do_color_changed(colorCat);
}

void FPS::prepare() {
  if (!displayed()) return;
  ++loop_count;
  double curtime = time_of_day();
  // force a redraw....
  need_matrix_recalc();
  // but don't redraw indicator more than twice per second
  if (curtime - last_update < 0.5) return;
  double rate = loop_count / (curtime - last_update);
  last_update = curtime;
  loop_count = 0;

  reset_disp_list();
  append(DMATERIALOFF);

  float asp = disp->aspect();
  float poscale = 1.2f;
  float pos[3];
  pos[0] = asp * poscale;
  pos[1] = poscale;
  pos[2] = 0;
  
  DispCmdColorIndex cmdColor;
  cmdColor.putdata(usecolor, cmdList);

  DispCmdText cmdText;
  char buf[20];
  sprintf(buf, "%5.2f", rate);
  cmdText.putdata(pos, buf, 1.0f, cmdList);
}

void FPS::do_color_changed(int clr) {
  if (clr == colorCat) {
    usecolor = scene->category_item_value(colorCat, "FPS");
  }
}

