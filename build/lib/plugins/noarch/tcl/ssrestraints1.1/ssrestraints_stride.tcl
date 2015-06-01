#
# $Id: ssrestraints_stride.tcl,v 1.3 2013/04/15 17:43:07 johns Exp $
#
# Original documentation of the script:
#
# stride_workaround - works around a limitation in STRIDE (used by
#                     VMD to determine secondary structure of proteins) 
#                     when dealing with very large structures.
#
# Sometimes STRIDE fails when trying to determine the secondary structure
# of proteins in very large structures. This script works around this
# problem by running STRIDE separately on each protein and creating
# atomselection macros that can be later used to assign the secondary 
# structure information.
#
# Follow these steps:
#
# 1. source stride_workaround.tcl
#
# 2. create_structure_macros
#
#    Once your molecule is loaded, run the command
#    'create_structure_macros'.  This will create a file with
#    atomselection macros. You only need to run this command once; you
#    can re-use the generated file whenever you analyse your structure
#    in VMD again.
#
# 3. set_structure
#
#    Run the command 'set_structure' to assign the secondary structure
#    based on a previously generated macros file.
#
# 
# Version: 1.0
#
# Leonardo Trabuco <ltrabuco@ks.uiuc.edu> - Thu Jul 12 13:54:26 CDT 2007
#
#
# The original script has been adapted and included in the 
# ssrestraints package on Thu May 14 20:48:49 CDT 2009

package provide ssrestraints_stride 1.1

namespace eval ::SSRestraints::STRIDE:: {

  variable defaultMolid {top}
  variable defaultSeltext {protein}
  variable defaultSplit {chain}
  variable prefix {tmp_stride_}

}

# proc create_structure_macros { args } { return [eval ::SSRestraints::STRIDE::create_structure_macros $args] }

# proc set_structure { args } { return [eval ::SSRestraints::STRIDE::set_structure $args] }


# For reference, here are the structure codes used by AtomSel.C:
#
# case SS_HELIX_ALPHA: data[i] = "H"; break;
# case SS_HELIX_3_10 : data[i] = "G"; break;
# case SS_HELIX_PI   : data[i] = "I"; break;
# case SS_BETA       : data[i] = "E"; break;
# case SS_BRIDGE     : data[i] = "B"; break;
# case SS_TURN       : data[i] = "T"; break;
# default:
# case SS_COIL       : data[i] = "C"; break;

proc ::SSRestraints::STRIDE::create_structure_macros_usage {} {

  puts "Usage: create_structure_macros -o <macros file> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  puts "  -seltext <selection text> (default: protein)"
  puts "  -split <chain|segname> (default: chain)"

  return

}

proc ::SSRestraints::STRIDE::set_structure_usage {} {

  puts "Usage: set_structure <macros file> \[-mol <molid> (default: top)\]"

  return

}

proc ::SSRestraints::STRIDE::create_structure_macros { args } {

  variable defaultMolid
  variable defaultSeltext
  variable defaultSplit
  variable prefix

  set nargs [llength $args]
  if {$nargs == 0} {
    create_structure_macros_usage
    return
  } elseif {$nargs % 2} {
    create_structure_macros_usage
    error "Odd number of arguments $args"
  }

  foreach {name val} $args {
    switch -- $name {
      -mol { set arg(molid) $val }
      -seltext { set arg(seltext) $val }
      -split { set arg(split) $val }
      -o { set arg(o) $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if [info exists arg(molid)] {
    set molid $arg(molid)
  } else {
    set molid $defaultMolid
  }

  if { $molid == {top} } {
    set molid [molinfo top]
  }
  if { $molid == -1 } {
    error "Please load a molecule first."
  }

  if [info exists arg(seltext)] {
    set seltext $arg(seltext)
  } else {
    set seltext $defaultSeltext
  }

  if [info exists arg(split)] {
    if { $arg(split) != {chain} && $arg(split) != {segname} } {
      create_structure_macros_usage
      error "Unrecognized value for option -split"
    } else {
      set split $arg(split)
    }
  } else {
    set split $defaultSplit
  }

  if [info exists arg(o)] {
    set out $arg(o)
  } else {
    create_structure_macros_usage
    error "Missing output file."
  }

  set out [open $out w]

  set sel [atomselect $molid "protein and ($seltext)"]
  if { [$sel num] == 0 } {
    $sel delete
    error "Atom selection does not contain protein."
  }
  
  if { $split == {chain} } {
    set chainList [lsort -unique [$sel get chain]]
  } else {
    set segnameList [lsort -unique [$sel get segname]]
  }

  $sel delete

  set seltextH [list none]
  set seltextG [list none] 
  set seltextI [list none] 
  set seltextE [list none] 
  set seltextB [list none] 
  set seltextT [list none] 
  set seltextC [list none] 

  if { $split == {chain} } {
    
    foreach chain $chainList {

      set sel [atomselect $molid "chain $chain"]
      $sel writepsf "$prefix$chain.psf"
      $sel writepdb "$prefix$chain.pdb"
      $sel delete
      mol new "$prefix$chain.psf" waitfor all
      mol addfile "$prefix$chain.pdb" waitfor all
    
      # SS_HELIX_ALPHA
      set selH [atomselect top {structure H}]
      if [$selH num] {
        set residH [lsort -unique [$selH get resid]]
        lappend seltextH "or (chain $chain and resid $residH)"
      }
      $selH delete
    
      # SS_HELIX_3_10
      set selG [atomselect top {structure G}]
      if [$selG num] {
        set residG [lsort -unique [$selG get resid]]
        lappend seltextG "or (chain $chain and resid $residG)"
      }
      $selG delete
    
      # SS_HELIX_PI
      set selI [atomselect top {structure I}]
      if [$selI num] {
        set residI [lsort -unique [$selI get resid]]
        lappend seltextI "or (chain $chain and resid $residI)"
      }
      $selI delete
    
      # SS_BETA
      set selE [atomselect top {structure E}]
      if [$selE num] {
        set residE [lsort -unique [$selE get resid]]
        lappend seltextE "or (chain $chain and resid $residE)"
      }
      $selE delete
    
      # SS_BRIDGE
      set selB [atomselect top {structure B}]
      if [$selB num] {
        set residB [lsort -unique [$selB get resid]]
        lappend seltextB "or (chain $chain and resid $residB)"
      }
      $selB delete
    
      # SS_TURN
      set selT [atomselect top {structure T}]
      if [$selT num] {
        set residT [lsort -unique [$selT get resid]]
        lappend seltextT "or (chain $chain and resid $residT)"
      }
      $selT delete
    
      # SS_COIL
      set selC [atomselect top {structure C}]
      if [$selC num] {
        set residC [lsort -unique [$selC get resid]]
        lappend seltextC "or (chain $chain and resid $residC)"
      }
      $selC delete
    
      mol delete top
      file delete "$prefix$chain.psf" "$prefix$chain.pdb"

    }

  } else {

    foreach seg $segnameList {

      set sel [atomselect $molid "segname $seg"]
      $sel writepsf "$prefix$seg.psf"
      $sel writepdb "$prefix$seg.pdb"
      $sel delete
      mol new "$prefix$seg.psf" waitfor all
      mol addfile "$prefix$seg.pdb" waitfor all
    
      # SS_HELIX_ALPHA
      set selH [atomselect top {structure H}]
      if [$selH num] {
        set residH [lsort -unique [$selH get resid]]
        lappend seltextH "or (segname $seg and resid $residH)"
      }
      $selH delete
    
      # SS_HELIX_3_10
      set selG [atomselect top {structure G}]
      if [$selG num] {
        set residG [lsort -unique [$selG get resid]]
        lappend seltextG "or (segname $seg and resid $residG)"
      }
      $selG delete
    
      # SS_HELIX_PI
      set selI [atomselect top {structure I}]
      if [$selI num] {
        set residI [lsort -unique [$selI get resid]]
        lappend seltextI "or (segname $seg and resid $residI)"
      }
      $selI delete
    
      # SS_BETA
      set selE [atomselect top {structure E}]
      if [$selE num] {
        set residE [lsort -unique [$selE get resid]]
        lappend seltextE "or (segname $seg and resid $residE)"
      }
      $selE delete
    
      # SS_BRIDGE
      set selB [atomselect top {structure B}]
      if [$selB num] {
        set residB [lsort -unique [$selB get resid]]
        lappend seltextB "or (segname $seg and resid $residB)"
      }
      $selB delete
    
      # SS_TURN
      set selT [atomselect top {structure T}]
      if [$selT num] {
        set residT [lsort -unique [$selT get resid]]
        lappend seltextT "or (segname $seg and resid $residT)"
      }
      $selT delete
    
      # SS_COIL
      set selC [atomselect top {structure C}]
      if [$selC num] {
        set residC [lsort -unique [$selC get resid]]
        lappend seltextC "or (segname $seg and resid $residC)"
      }
      $selC delete
    
      mol delete top
      file delete "$prefix$seg.psf" "$prefix$seg.pdb"

    }

  }

  puts $out "atomselect macro ssH {[join $seltextH]}"
  puts $out "atomselect macro ssG {[join $seltextG]}"
  puts $out "atomselect macro ssI {[join $seltextI]}"
  puts $out "atomselect macro ssE {[join $seltextE]}"
  puts $out "atomselect macro ssB {[join $seltextB]}"
  puts $out "atomselect macro ssT {[join $seltextT]}"
  puts $out "atomselect macro ssC {[join $seltextC]}"
  
  close $out

  unset seltextH
  unset seltextG 
  unset seltextI 
  unset seltextE 
  unset seltextB 
  unset seltextT 
  unset seltextC 

  return

}

proc ::SSRestraints::STRIDE::set_structure { args } {

  variable defaultMolid

  set nargs [llength $args]
  if {$nargs == 0} {
    set_structure_usage
    return
  } elseif {$nargs != 1 && $nargs != 3} {
    create_structure_macros_usage
    error "Wrong number of arguments"
  }

  set macrosFile [lindex $args 0]

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(molid) $val }
      default { error "unkown argument: $name $val" }
    }
  }

  if [info exists arg(molid)] {
    set molid $arg(molid)
  } else {
    set molid $defaultMolid
  }

  source $macrosFile

  set selH [atomselect $molid {ssH}]
  $selH set structure H
  $selH delete
  
  set selG [atomselect $molid {ssG}]
  $selG set structure G
  $selG delete
  
  set selI [atomselect $molid {ssI}]
  $selI set structure I
  $selI delete
  
  set selE [atomselect $molid {ssE}]
  $selE set structure E
  $selE delete
  
  set selB [atomselect $molid {ssB}]
  $selB set structure B
  $selB delete
  
  set selT [atomselect $molid {ssT}]
  $selT set structure T
  $selT delete
  
  set selC [atomselect $molid {ssC}]
  $selC set structure C
  $selC delete

  return

}
