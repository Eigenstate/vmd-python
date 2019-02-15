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
 *	$RCSfile: CmdRender.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.39 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Render a scene (so far, there is only one, the global one) as some
 * sort of rendered output; postscipt, rayshade, POVray, raster3D, etc.
 *
 ***************************************************************************/

#include <stdlib.h>

#include "CmdRender.h"
#include "utilities.h"


///////////////////////// render the global scene
CmdRender::CmdRender(const char *newfilename, const char *newmethod, 
                     const char *newcmd)
	: Command(Command::RENDER) {
  filename = stringdup(newfilename);
  method = stringdup(newmethod);
  extcmd = (newcmd ? stringdup(newcmd) : (char *) NULL);
}

CmdRender::~CmdRender(void) {
  delete [] filename;
  delete [] method;
  if(extcmd)  delete [] extcmd;
}

void CmdRender::create_text(void) {
  *cmdText << "render " << method << " " << filename;
  if(extcmd)
    *cmdText << " " << extcmd;
  *cmdText << ends;
}

///// CmdRenderOption
CmdRenderOption::CmdRenderOption(const char *met, const char *opt)
: Command(Command::RENDER_OPTION) {
  method = stringdup(met);
  option = stringdup(opt);
}

CmdRenderOption::~CmdRenderOption() {
  delete [] method;
  delete [] option;
}

