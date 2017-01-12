############################################################################
#cr                                                                       
#cr            (C) Copyright 1995-2007 The Board of Trustees of the            
#cr                        University of Illinois                         
#cr                         All Rights Reserved                           
#cr                                                                       
############################################################################

############################################################################
# RCS INFORMATION:
#
# 	$RCSfile: draw.tcl,v $
# 	$Author: johns $	$Locker:  $		$State: Exp $
#	$Revision: 1.7 $	$Date: 2007/01/12 20:11:31 $
#
############################################################################
# DESCRIPTION:
#   easy interface to the drawing (graphics) routines
#
############################################################################

# Add graphics primitives to the top molecule.  If no molecule is loaded,
# create a blank one first.  If a proc of the form vmd_draw_$symbol is found,
# use that instead of whatever might be defined by the graphics command.

proc draw {symbol args} {
  if { [molinfo num] < 1 } {
    mol load graphics graphics
  }
  set mol [molinfo top]
  if { [llength [info commands vmd_draw_$symbol]] } { 
    return [eval vmd_draw_$symbol $mol $args]
  }
  return [eval graphics $mol $symbol $args]
}
