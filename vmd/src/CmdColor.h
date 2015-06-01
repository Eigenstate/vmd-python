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
 *      $RCSfile: CmdColor.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.31 $      $Date: 2010/12/16 04:08:07 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Command objects for affecting molecules.
 *
 ***************************************************************************/
#ifndef CMDCOLOR_H
#define CMDCOLOR_H

#include "Command.h"

class VMDApp;

// the following defines commands for the Cmdtypes:
// COLOR_NAME, COLOR_CHANGE, COLOR_SCALE_METHOD, COLOR_SCALE_MIDPOINT,
// COLOR_SCALE_MIN, COLOR_SCALE_MAX

///  change the color index for a specifed name in a specified category
class CmdColorName : public Command {
protected:
  virtual void create_text(void);

public:
  char *cCatStr, *cNameStr, *cColStr;

  /// constructor: category name, item name, new color
  CmdColorName(const char *, const char *, const char *);
  ~CmdColorName();
};


/// change the rgb settings for a specified color
class CmdColorChange : public Command {
public:
  char *color; 
  float newR, newG, newB;

protected:
  virtual void create_text();

public:
  // constructor: color name, R, G, B
  CmdColorChange(const char *, float, float, float);
  ~CmdColorChange();
};


/// change the method used to calculate the color scale
class CmdColorScaleMethod : public Command {
protected:
  virtual void create_text();

public:
  char *method;
  CmdColorScaleMethod(const char *);
  ~CmdColorScaleMethod();
};


/// Change the settings for the color scale
class CmdColorScaleSettings : public Command {
protected:
  virtual void create_text(void);

public:
  float mid, min, max;
  CmdColorScaleSettings(float newmid, float newmin, float newmax);
};

class CmdColorScaleColors : public Command {
protected:
  virtual void create_text();
public:
  const char *method;
  float colors[3][3];
  CmdColorScaleColors(const char *, const float *, const float *, const float *);
};

class CmdColorItem : public Command {
protected:
  virtual void create_text();
public:
  char *category, *name, *defcolor;
  CmdColorItem(const char *cat, const char *nm, const char *def);
  ~CmdColorItem();
};
#endif

