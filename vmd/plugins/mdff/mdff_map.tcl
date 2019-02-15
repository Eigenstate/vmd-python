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
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $       $Date: 2019/01/10 16:06:22 $
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

  set MAPMOL [mol new $inMap]
  voltool clamp -min $threshold -mol $MAPMOL
  voltool smult -amt -1 -mol $MAPMOL
  voltool range -minmax {0 1} -mol $MAPMOL -o $outDX
  mol delete $MAPMOL
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
  set MAPMOL [mol new $tmpDX]
  set INMAPMOL [mol new $inMap]
  voltool smult -amt -1 -mol $MAPMOL
  voltool range -minmax {0 1} -mol $MAPMOL
  voltool mult -union -mol1 $INMAPMOL -mol2 $MAPMOL -o $outMap 
  mol delete $MAPMOL
  mol delete $INMAPMOL
  file delete $tmpDX $tmpDX2

}

proc ::MDFF::Map::mdff_histogram_usage { } {

  variable defaultNBins

  puts "Usage: mdff hist -i <input map> ?options?"
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

  global MAPMOL
  set MAPMOL [mol new $inMap] 
  
  set histreturn [voltool hist -nbins $nbins -mol $MAPMOL]
  set minmax [voltool info minmax -mol $MAPMOL]
  set min [lindex $minmax 0]
  set max [lindex $minmax 1]
  set xlist [list]
  set delta [expr {($max - $min) / $nbins}]
 
  foreach {midpt hist} $histreturn {
    lappend xlist $midpt
    lappend histogram $hist
  }

  set highhistval 0.5
  set highhist 0
  for {set l 0} {$l < $nbins} {incr l} {
    if {[lindex $histogram $l] > $highhist} {
     set highhist [lindex $histogram $l]
     set highhistval [lindex $xlist $l] 
    }
  }
      
  set sorted [lsort -integer $histogram]
  set ymin [lindex $sorted 0]
  global ymax
  set ymax [lindex $sorted end]

   
 #normalize?
 # for {set z 0} {$z < $nbins} {incr z} {
 #   set oldval [lindex $histogram $z]
 #   lappend nhistogram [expr ($oldval - $ymin)/($ymax-$ymin)]
 # }
  
  mol modstyle 0 $MAPMOL Isosurface $highhistval 0 0 0 1 1
  
  global plot
  set plot [multiplot -x $xlist -y $histogram -title "Density histogram" -xlabel "Density" -ylabel "Number of voxels" -nolines -marker square -fill black -xmin [expr [lindex $xlist 0] - (0.5*$delta)] -xmax [expr [lindex $xlist end] + (0.5*$delta)] ]
  
  for {set j 0} {$j < $nbins} {incr j} {
    set left [expr [lindex $xlist $j] - (0.5 * $delta)]
    set right [expr [lindex $xlist $j] + (0.5 * $delta)]
    $plot draw rectangle $left 0 $right [lindex $histogram $j] -fill "#0000ff" -tags rect$j
   
    #$plot add [lindex $xlist $j] [lindex $histogram $j] -marker square -fillcolor black -radius [expr 0.5*$delta] -callback histclick
  }
 # puts [[$plot getpath].f.cf find withtag "rect0"]
  $plot replot
  global bpress
  set bpress 0
    
  global xmaxg
  global xplotming
  global xplotmaxg
  global scalexg
  global xming
  variable [$plot namespace]::xplotmin
  variable [$plot namespace]::xplotmax
  variable [$plot namespace]::scalex
  variable [$plot namespace]::xmin
  variable [$plot namespace]::xmax
  set xplotming $xplotmin
  set xplotmaxg $xplotmax
  set scalexg $scalex
  set xming $xmin
  set xmaxg $xmax
  
  bind [$plot getpath].f.cf <ButtonPress> {
    set bpress 1    
    variable [$plot namespace]::xplotmin
    set x [expr (%x - $xplotming)/$scalexg + $xming]
    if {$x >= $xming && $x <= $xmaxg} { 
      $plot undraw "line"
      $plot draw line $x 0 $x $ymax -tag "line"
      mol modstyle 0 $MAPMOL Isosurface $x 0 0 0 1 1
    }
  }
  
  bind [$plot getpath].f.cf <ButtonRelease> {
    set bpress 0 
  }
  
  bind [$plot getpath].f.cf <Motion> {
    if {$bpress && $x >= $xming && $x <= $xmaxg} {
      variable [$plot namespace]::xplotmin
      set x [expr (%x - $xplotming)/$scalexg + $xming]
      $plot undraw "line"
      $plot draw line $x 0 $x $ymax -tag "line"
      mol modstyle 0 $MAPMOL Isosurface $x 0 0 0 1 1
    }
  }
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

  set MAPMOL [mol new $inMap]
  
  voltool binmask -mol $MAPMOL
  voltool smult -amt -1 -mol $MAPMOL
  voltool sadd -amt 1 -mol $MAPMOL
  
  set MAPMOL2 [mol new $inMap]
  voltool -smooth -sigma $defaultSmoothKernel -mol $MAPMOL2
  voltool mult -mol1 $MAPMOL -mol2 $MAPMOL2 -o $tmpDX3

  set MAPMOL3 [mol new $inMap]
  set MAPMOL4 [mol new $tmpDX3]
  voltool add -mol1 $MAPMOL3 -mol2 $MAPMOL4 -o $outDX
     
  mol delete $MAPMOL
  mol delete $MAPMOL2
  mol delete $MAPMOL3
  mol delete $MAPMOL4
  file delete $tmpDX $tmpDX2 $tmpDX3

  return
}

