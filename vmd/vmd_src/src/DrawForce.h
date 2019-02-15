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
 *	$RCSfile: DrawForce.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.26 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Another Child Displayable component for a remote molecule; this displays
 * and stores the information about the interactive forces being applied to
 * the molecule.  If no forces are being used, this draws nothing.
 *
 * The force information is retrieved from the Atom list in the parent
 * molecule.  No forces are stored here.
 *
 ***************************************************************************/
#ifndef DRAWFORCE_H
#define DRAWFORCE_H

#include "Displayable.h"
#include "DispCmds.h"

class DrawMolecule;

/// A Displayable subclass for drawing forces applied by IMD
class DrawForce : public Displayable {
private:
  DrawMolecule *mol;               ///< parent molecule
  DispCmdColorIndex cmdColorIndex; ///< color index to use when drawing
  DispCmdCone cmdCone;             ///< cone geometry used to draw force arrow
  void create_cmdlist(void);       ///< regenerate the command list
  int needRegenerate;              ///< flag controlling redraws of the list
  int colorCat;                    ///< color category we use for our colors

protected:
  virtual void do_color_changed(int);

public:
  DrawForce(DrawMolecule *);       ///< constructor: parent molecule
  virtual void prepare();          ///< prepare for drawing, do needed updates
};

#endif

