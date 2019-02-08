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
 *      $RCSfile: CmdLabel.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.41 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Command objects used to create, list, delete, or graph labels for measuring
 * geometries.
 *
 ***************************************************************************/
#ifndef CMDLABEL_H
#define CMDLABEL_H

#include "Command.h"

// The following uses the Cmdtypes LABEL_ADD, LABEL_ADDSPRING,
// LABEL_SHOW, LABEL_LIST, and LABEL_DELETE from the
// Command class

/// add a new label 
class CmdLabelAdd : public Command {
private:
  char geomcatStr[16];
  char *geomitems[4];
  int num_geomitems;
  
protected:
  virtual void create_text(void);

public:
  /// constructor: category name, # items, item molid's, item atomid's. 
  CmdLabelAdd(const char *, int, int *molid, int *atomid);
  ~CmdLabelAdd(void);
};


/// add a new spring
class CmdLabelAddspring : public Command {
private:
  int molid;
  int atom1;
  int atom2;
  float k;

protected:
  virtual void create_text(void);

public:
  /// constructor: category name, # items, item molid's, item atomid's. 
  CmdLabelAddspring(int themol, int theatom1, int theatom2,
		    float thek);
};


/// toggle a geometry category on/off
class CmdLabelShow : public Command {
private:
  char geomcatStr[16];
  int item;		///< which item to toggle (if < 0 toggle entire cat.)
  int show;		///< if T, turn on; if F, turn off

protected:
  virtual void create_text(void);

public:
  CmdLabelShow(const char *category, int n, int onoff);
};


/// delete the Nth label in a category
class CmdLabelDelete : public Command {
private:
  char geomcatStr[16];
  int item;		///< which item to delete (if < 0 delete all in cat.)

protected:
  virtual void create_text(void);

public:
  CmdLabelDelete(const char *, int); ///< constructor: category name, item
};

class CmdLabelTextSize : public Command {
protected:
  virtual void create_text();
public:
  const float size;
  CmdLabelTextSize(float newsize)
  : Command(LABEL_TEXTSIZE), size(newsize) {}
};

class CmdLabelTextThickness : public Command {
protected:
  virtual void create_text();
public:
  const float thickness;
  CmdLabelTextThickness(float newthickness)
  : Command(LABEL_TEXTTHICKNESS), thickness(newthickness) {}
};

class CmdLabelTextOffset : public Command {
protected:
  virtual void create_text();
public:
  char *nm;
  int n;
  const float m_x, m_y;
  CmdLabelTextOffset(const char *name, int ind, float x, float y);
  ~CmdLabelTextOffset();
};

class CmdLabelTextFormat: public Command {
protected:
  virtual void create_text();
public:
  char *nm;
  int n;
  char *format;
  CmdLabelTextFormat(const char *name, int ind, const char *format);
  ~CmdLabelTextFormat();
};
#endif

