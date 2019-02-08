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
 *	$RCSfile: CmdRender.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.29 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Render a scene (so far, there is only one, the global one) as some
 * sort of rendered output; postscipt, rayshade, POVray, raster3D, etc.
 *
 ***************************************************************************/
#ifndef CMDRENDER_H
#define CMDRENDER_H

#include "Command.h"

/// render the global scene
class CmdRender : public Command {
public:
  char *filename;
  char *method;  ///< what kind of output?  "postscript", "rayshade", etc.
  char *extcmd;  ///< command to run if the rendering is successful

protected:
  virtual void create_text(void);

public:
  /// constructor: filename, method, external cmd
  CmdRender(const char *, const char *, const char *);
  virtual ~CmdRender(void);
};

/// set the render execution command 
class CmdRenderOption : public Command {
public:
  char *method;
  char *option;

public:
  /// constructor takes method and option.  
  CmdRenderOption(const char *, const char *);
  ~CmdRenderOption();
};

#endif


