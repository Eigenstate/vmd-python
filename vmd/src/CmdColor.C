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
 *      $RCSfile: CmdColor.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.44 $      $Date: 2010/12/16 04:08:07 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Command objects for affecting molecules.
 *
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "CmdColor.h"
#include "utilities.h"

// the following defines commands for the Cmdtypes:
// COLOR_NAME, COLOR_CHANGE, COLOR_SCALE_METHOD, COLOR_SCALE_MIDPOINT,
// COLOR_SCALE_MIN, COLOR_SCALE_MAX

void CmdColorName::create_text(void) { 
  *cmdText << "color " << cCatStr << " " << cNameStr;
  *cmdText << " " << cColStr;
  *cmdText << ends;
}

// constructor: category name, item name, new color
CmdColorName::CmdColorName(const char *cCat, const char *cName, const char *cCol)
: Command(Command::COLOR_NAME) {
  cCatStr = stringdup(cCat);
  cNameStr = stringdup(cName);
  cColStr = stringdup(cCol);
}
CmdColorName::~CmdColorName() {
  delete [] cCatStr;  
  delete [] cNameStr;  
  delete [] cColStr;  
}

///////////////  change the rgb settings for a specified color

void CmdColorChange::create_text() { 
  *cmdText << "color change rgb " << color << " ";
  *cmdText << newR << " " << newG << " " << newB;
  *cmdText << ends;
}

CmdColorChange::CmdColorChange(const char *cCol, float r, float g, float b) 
: Command(Command::COLOR_CHANGE) {
  
  newR = r;
  newG = g;
  newB = b;
  color = stringdup(cCol);
}
CmdColorChange::~CmdColorChange() {
  delete [] color;
}

void CmdColorScaleMethod::create_text() {
  *cmdText << "color scale method " << method << ends;
}

// constructor: new method
CmdColorScaleMethod::CmdColorScaleMethod(const char *nm)
	: Command(Command::COLOR_SCALE_METHOD) {
  method = stringdup(nm);
}
CmdColorScaleMethod::~CmdColorScaleMethod() {
  delete [] method;
}

void CmdColorScaleSettings::create_text() {
  *cmdText << "color scale midpoint " << mid << "\n";
  *cmdText << "color scale min " << min << "\n";
  *cmdText << "color scale max " << max << "\n";
  *cmdText << ends;
}

CmdColorScaleSettings::CmdColorScaleSettings(float newmid, float newmin, 
                                             float newmax)
: Command(Command::COLOR_SCALE_SETTINGS), mid(newmid), min(newmin), max(newmax)
{}


CmdColorItem::CmdColorItem(const char *cat, const char *nm, const char *def)
: Command(Command::COLOR_ADD_ITEM) {
  category = stringdup(cat);
  name     = stringdup(nm);
  defcolor = stringdup(def);
}
CmdColorItem::~CmdColorItem() {
  delete [] category;
  delete [] name;
  delete [] defcolor;
}
void CmdColorItem::create_text() {
  *cmdText << "color add item {" << category << "} {" << name << "} {"
    << defcolor << "}" << ends;
}

CmdColorScaleColors::CmdColorScaleColors(const char *nm, 
    const float themin[3], const float themid[3], const float themax[3])
: Command(COLOR_SCALE_COLORS) {
  method = nm;
  memcpy(colors[0], themin, 3*sizeof(float));
  memcpy(colors[1], themid, 3*sizeof(float));
  memcpy(colors[2], themax, 3*sizeof(float));
}

void CmdColorScaleColors::create_text() {
  *cmdText << "color scale colors " << method;
  for (int i=0; i<3; i++)
    *cmdText << "{" << colors[i][0] << " " << colors[i][1] << " " 
      << colors[i][2] << "}" << " ";
  *cmdText << ends;
}

