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
 *	$RCSfile: P_PrintTool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.13 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Very simple tool that prints tracker position and orientation data
 *
 ***************************************************************************/

#include "P_PrintTool.h"
#include "utilities.h"
#include "Matrix4.h"
#include "Inform.h"

PrintTool::PrintTool(int id, VMDApp *vmdapp, Displayable *disp) 
: Tool(id, vmdapp, disp) {
  targetting=0;
}

void PrintTool::do_event() {
  float p[3];
  Matrix4 o;

  if(!position()) return;
  vec_copy(p, position());
  o = *orientation();

  msgInfo << "Tool[" << id() << "] pos: " 
          << p[0] << ", " << p[1] << ", " << p[2] << sendmsg;  

  msgInfo << "Tool[" << id() << "] orientation: " 
          << o.mat[ 0] << o.mat[ 1] << o.mat[ 2] << o.mat[ 3] 
          << o.mat[ 4] << o.mat[ 5] << o.mat[ 6] << o.mat[ 7] 
          << o.mat[ 8] << o.mat[ 9] << o.mat[10] << o.mat[11] 
          << o.mat[12] << o.mat[13] << o.mat[14] << o.mat[15] 
          << sendmsg;  
}

