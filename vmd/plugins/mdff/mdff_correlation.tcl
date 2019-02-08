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
#       $RCSfile: mdff_correlation.tcl,v $
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.3 $       $Date: 2019/01/10 16:06:22 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>
#
# mdff ccc
#

package require mdff_tmp
package provide mdff_correlation 0.2

namespace eval ::MDFF::Correlation:: {
}

proc ::MDFF::Correlation::mdff_ccc_usage { } {

 
  puts "Usage: mdff ccc <atom selection> -i <input map> -res <map resolution in Angstroms> ?options?"
  puts "Options:"
  puts "  -spacing <grid spacing in Angstroms> (default based on res)"
  puts "  -threshold <x> (ignores voxels with values below x threshold.)"
  puts "  -allframes (average over all frames)"
  
}

proc ::MDFF::Correlation::mdff_ccc { args } {


  set nargs [llength [lindex $args 0]]
  if {$nargs == 0} {
    mdff_ccc_usage
    error ""
  }

  set sel [lindex $args 0]
  if { [$sel num] == 0 } {
    error "Empty atom selection."
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
      -i { set arg(i) $val }
      -res { set arg(res) $val }
      -spacing { set arg(spacing) $val }
      -threshold { set arg(threshold) $val }
    }
  }

  if { [info exists arg(i)] } {
    set inputMap $arg(i)
  } else {
    error "Missing input map."
  }

  if { [info exists arg(res)] } {
    set res $arg(res)
  } else {
    error "Missing input map resolution."
  }

  if { [info exists arg(spacing)] } {
    set spacing $arg(spacing)
  }

  if { [info exists arg(threshold)] } {
    set threshold $arg(threshold)
    set use_threshold 1
  } else {
    set use_threshold 0
  }

  if $use_threshold {
    if { [info exists spacing] } {
        set cc [voltool cc $sel -i $inputMap -res $res -thresholddensity $threshold -spacing $spacing]
      } else {
        set cc [voltool cc $sel -i $inputMap -res $res -thresholddensity $threshold]
      }
  } else {
    if { [info exists spacing] } {
        set cc [voltool cc $sel -i $inputMap -res $res -spacing $spacing]
      } else {
        set cc [voltool cc $sel -i $inputMap -res $res]
      }
  }

  return $cc

}

