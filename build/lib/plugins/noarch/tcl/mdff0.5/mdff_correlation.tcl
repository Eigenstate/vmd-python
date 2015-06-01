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
#       $Author: ltrabuco $        $Locker:  $             $State: Exp $
#       $Revision: 1.1 $       $Date: 2009/08/06 20:07:33 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>
#
# mdff ccc
#

package require volutil
package require mdff_tmp
package provide mdff_correlation 0.2

namespace eval ::MDFF::Correlation:: {
  variable defaultGridspacing 1.0
}

proc ::MDFF::Correlation::mdff_ccc_usage { } {

  variable defaultGridspacing
 
  puts "Usage: mdff ccc <atom selection> -i <input map> -res <map resolution in Angstroms> ?options?"
  puts "Options:"
  puts "  -spacing <grid spacing in Angstroms> (default: $defaultGridspacing)"
  puts "  -threshold <x sigmas>"
  puts "  -allframes (average over all frames)"
  
}

proc ::MDFF::Correlation::mdff_ccc { args } {

  variable defaultGridspacing

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
  } else {
    set spacing $defaultGridspacing
  }

  if { [info exists arg(threshold)] } {
    set threshold $arg(threshold)
    set use_threshold 1
  } else {
    set use_threshold 0
  }

  # Get temporary filenames
  set tmpDir [::MDFF::Tmp::tmpdir]
  set tmpDX [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_corr -suffix .dx -tmpdir $tmpDir]]
  set tmpDX2 [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_corr -suffix .dx -tmpdir $tmpDir]]
  set tmpLog [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_corr -suffix .log -tmpdir $tmpDir]]

  # Create simulated map
  if $allFrames {
    ::MDFF::Sim::mdff_sim $sel -o $tmpDX -res $res -spacing $spacing -allframes
  } else {
    ::MDFF::Sim::mdff_sim $sel -o $tmpDX -res $res -spacing $spacing
  }

  if $use_threshold {
    # Set voxels above the given threshold to NAN
    ::VolUtil::volutil -threshold $threshold $tmpDX -o $tmpDX2
    # Calculate correlation
    ::VolUtil::volutil -tee $tmpLog -quiet -safe -corr $inputMap $tmpDX2
  } else {
    # Calculate correlation
    ::VolUtil::volutil -tee $tmpLog -quiet -corr $inputMap $tmpDX
  }

  file delete $tmpDX
  file delete $tmpDX2

  # parse the output to get the correlation coefficient
  set file [open $tmpLog r]
  gets $file line
  while {$line != ""} {
    if { [regexp {^Correlation coefficient = (.*)} $line fullmatch cc] } {
      break
    }
    gets $file line
  }
  close $file

  file delete $tmpLog

  return $cc

}

