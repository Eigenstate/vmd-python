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
#       $RCSfile: mdff_map.tcl,v $
#       $Author: ltrabuco $        $Locker:  $             $State: Exp $
#       $Revision: 1.1 $       $Date: 2009/08/06 20:07:33 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>
#
# mdff griddx -- creates a map for docking
# mdff delete -- deletes volume corresponding to atomic structure
# mdff hist -- calculates a density histogram
# 

# TODO
# - 'mdff delete' should use the input map gridspacing by default, but
#    that information has to be provided by volutil first

package require volutil
package require multiplot
package require mdff_tmp
package provide mdff_map 0.2

namespace eval ::MDFF::Map:: {

  variable defaultGridspacing 1.0
  variable defaultTargetResolution 10.0
  variable defaultNBins 10
  variable defaultThreshold 0
  variable defaultSmoothKernel 3.0


}

proc ::MDFF::Map::mdff_griddx_usage { } {
  variable defaultThreshold

  puts "Usage: mdff griddx -i <input map> -o <output dx map> ?options?"
  puts "Options:"
  puts "  -threshold <value> (default: $defaultThreshold)"
}

proc ::MDFF::Map::mdff_griddx { args } {

  variable defaultThreshold

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_griddx_usage
    error ""
  }

  foreach {name val} $args {
    switch -- $name {
      -i { set arg(i) $val }
      -o { set arg(o) $val }
      -threshold { set arg(threshold) $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(i)] } {
    set inMap $arg(i)
  } else {
    error "Missing input map."
  }

  if { [info exists arg(o)] } {
    set outDX $arg(o)
  } else {
    error "Missing output dx map."
  }

  if { [info exists arg(threshold)] } {
    set threshold $arg(threshold)
  } else {
    set threshold $defaultThreshold
  }

  # Get temporary filename
  set tmpDir [::MDFF::Tmp::tmpdir]
  set tmpDX [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_griddx -suffix .dx -tmpdir $tmpDir]]

  # TODO: change volutil so that we can do everything in once step
  #       even when the threshold is not zero; i.e., the option
  #       -dockgrid should take the threshold as an extra argument

  if { $threshold != 0 } {
    ::VolUtil::volutil -clamp $threshold: -o $tmpDX $inMap
    ::VolUtil::volutil -smult -1 -o $tmpDX $tmpDX
    ::VolUtil::volutil -range 0:1 -o $outDX $tmpDX
    file delete $tmpDX
  } else {
    ::VolUtil::volutil -dockgrid 1 -o $outDX $inMap
  }

  return

}

proc ::MDFF::Map::mdff_delete_usage { } {

  variable defaultTargetResolution
  variable defaultGridspacing

  puts "Usage: mdff delete <atom selection> -i <input map> -o <output dx map> ?options?"
  puts "Options:"
  puts "  -res <target resolution in Angstroms> (default: $defaultTargetResolution)"
  puts "  -spacing <grid spacing for \"mask\" map in Angstroms> (default: $defaultGridspacing)"
  puts "  -allframes -- average over all frames"

  return

}

proc ::MDFF::Map::mdff_delete { args } {

  variable defaultGridspacing
  variable defaultTargetResolution

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_delete_usage
    error ""
  }

  set sel [lindex $args 0]
  if { [$sel num] == 0 } {
    error "mdff_delete: empty atomselection."
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
      -o { set arg(o) $val }
      -res { set arg(res) $val }
      -spacing { set arg(spacing) $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(i)] } {
    set inMap $arg(i)
  } else {
    error "Missing input map."
  }

  if { [info exists arg(o)] } {
    set outMap $arg(o)
  } else {
    error "Missing output dx map."
  }

  if { [info exists arg(res)] } {
    set res $arg(res)
  } else {
    set res $defaultTargetResolution
  }

  if { [info exists arg(spacing)] } {
    set spacing $arg(spacing)
  } else {
    set spacing $defaultGridspacing
  }

  # Get temporary filenames
  set tmpDir [::MDFF::Tmp::tmpdir]
  set tmpDX [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_delete -suffix .dx -tmpdir $tmpDir]]
  set tmpDX2 [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_delete -suffix .dx -tmpdir $tmpDir]]

  # create simulated map tmpDX
  if $allFrames {
    ::MDFF::Sim::mdff_sim $sel -o $tmpDX -res $res -spacing $spacing -allframes
  } else {
    ::MDFF::Sim::mdff_sim $sel -o $tmpDX -res $res -spacing $spacing
  }

  # create a "mask" based on simulated map
  ::VolUtil::volutil -smult -1 $tmpDX -o $tmpDX2
  ::VolUtil::volutil -range 0:1 $tmpDX2 -o $tmpDX
  ::VolUtil::volutil -mult -union $inMap $tmpDX -o $outMap

  file delete $tmpDX $tmpDX2

}

proc ::MDFF::Map::mdff_histogram_usage { } {

  variable defaultNBins

  puts "Usage: mdff histogram -i <input map> ?options?"
  puts "Options:"
  puts "  -nbins <number of bins> (default: $defaultNBins)"

  return

}

proc ::MDFF::Map::mdff_histogram { args } {

  variable defaultNBins

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_histogram_usage
    error ""
  }

  foreach {name val} $args {
    switch -- $name {
      -i { set arg(i) $val }
      -nbins { set arg(nbins) $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(i)] } {
    set inMap $arg(i)
  } else {
    error "Missing input map."
  }

  if { [info exists arg(nbins)] } {
    set nbins $arg(nbins)
  } else {
    set nbins $defaultNBins
  }

  # Get temporary filename
  set tmpDir [::MDFF::Tmp::tmpdir]
  set tmpLog [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_hist -suffix .log -tmpdir $tmpDir]]

  ::VolUtil::volutil -tee $tmpLog -quiet -hist -histnbins $nbins $inMap

  # parse the output and plot the histogram...
  set file [open $tmpLog r]
  gets $file line
  while {$line != ""} {
    if { [regexp {^Density histogram with min = (.*), max = (.*), nbins = (\d+)} $line fullmatch min max nbins] } {
      gets $file histogram
      break
    }
    gets $file line
  }
  close $file
  file delete $tmpLog

  # calculate x axis
  set xlist [list]
  set delta [expr {($max - $min) / $nbins}]
  for {set i 0} {$i < $nbins} {incr i} {
    lappend xlist [expr {$min + (0.5 * $delta) + $i * $delta}]
  }

  set plot [multiplot -x $xlist -y $histogram -title "Density histogram" -xlabel "Density" -ylabel "Number of voxels" -lines -plot]

}


proc ::MDFF::Map::mdff_edge_usage { } {
  variable defaultSmoothKernel

  puts "Usage: mdff edge -i <input map> -o <output dx map> ?options?"
  puts "Options:"
  puts "  -kernel <Gaussian kernel> (default: $defaultSmoothKernel)"
}

proc ::MDFF::Map::mdff_edge { args } {

  variable defaultSmoothKernel 3.0

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_edge_usage
    error ""
  }

  foreach {name val} $args {
    switch -- $name {
      -i { set arg(i) $val }
      -o { set arg(o) $val }
      -kernel { set arg(kernel) $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if { [info exists arg(i)] } {
    set inMap $arg(i)
  } else {
    error "Missing input map."

  }

  if { [info exists arg(o)] } {
    set outDX $arg(o)
  } else {
    error "Missing output dx map."
  }

  if { [info exists arg(kernel)] } {
    set kernel $arg(kernel)
  } else {
    set kernel $defaultSmoothKernel 
  }

  # Get temporary filenames
  set tmpDir [::MDFF::Tmp::tmpdir]
  set tmpDX [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_edge -suffix .dx -tmpdir $tmpDir]]
  set tmpDX2 [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_edge -suffix .dx -tmpdir $tmpDir]]
  set tmpDX3 [file join $tmpDir \
    [::MDFF::Tmp::tmpfilename -prefix mdff_edge -suffix .dx -tmpdir $tmpDir]]

  # TODO: change volutil so that we can do everything in one step

  # this function is doing B = A + gauus(A) * (1 - binmask(A))
  # yielding the original information intact, and adding smooth 
  # edges 
  ::VolUtil::volutil -invmask -o $tmpDX $inMap
  ::VolUtil::volutil -smooth $defaultSmoothKernel -o $tmpDX2 $inMap
  ::VolUtil::volutil -mult $tmpDX $tmpDX2 -o $tmpDX3
  ::VolUtil::volutil -add $inMap $tmpDX3 -o $outDX

  file delete $tmpDX $tmpDX2 $tmpDX3

  return
}

