############################################################################
#cr
#cr            (C) Copyright 1995-2009 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: mdff_sim.tcl,v $
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.3 $       $Date: 2019/01/10 16:06:22 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>
#
# mdff sim -- creates a simulated map from an atomic structure
#

package require mdff_tmp
package provide mdff_sim 0.2

namespace eval ::MDFF::Sim:: {

  variable defaultTargetResolution 10.0
}

proc ::MDFF::Sim::mdff_sim_usage { } {

  variable defaultTargetResolution

  puts "Usage: mdff sim <atomselection> -o <output map> ?options?"
  puts "Options:"
  puts "  -res <target resolution in Angstroms> (default: $defaultTargetResolution)"
  puts "  -spacing <grid spacing in Angstroms> (default based on res)"
  puts "  -allframes (average over all frames)"

}

proc ::MDFF::Sim::mdff_sim { args } {

  variable defaultTargetResolution

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_sim_usage
    error ""
  }

  set sel [lindex $args 0]
  if { [$sel num] == 0 } {
    error "mdff_sim: empty atomselection."
  }

  # should we use all frames?
  set pos [lsearch -exact $args {-allframes}]
  if { $pos != -1 } {
    set allFrames 1
    set args [lreplace $args $pos $pos]
  } else {
    set allFrames 0
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -o     { set arg(o)     $val }
      -res   { set arg(res)   $val }
      -spacing { set arg(spacing) $val }
    }
  }

  if { [info exists arg(spacing)] } {
    set gridspacing $arg(spacing)
  }

  if { [info exists arg(res)] } {
    set targetResolution $arg(res)
  } else {
    set targetResolution $defaultTargetResolution
  }
  
  if { [info exists arg(o)] } {
    set dxout $arg(o)
  } else {
    error "Missing output dx map."
  }

  if { [info exists gridspacing] } {
    voltool sim $sel -res $targetResolution -spacing $gridspacing -o $dxout 
  } else {
    voltool sim $sel -res $targetResolution -o $dxout 
  }
  return

}


