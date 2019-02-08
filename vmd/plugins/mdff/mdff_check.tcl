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
#       $RCSfile: mdff_check.tcl,v $
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $       $Date: 2018/09/12 17:39:39 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>
#
# mdff check 
#

# XXX - 'mdff check -ccc' introduces code duplication; this functionality
#        should probably be provided by 'mdff ccc' directly
#       'mdff check -rmsd' duplicates code from other plugins; should
#        probably consolidade RMSD calculation text command outside mdff

package require multiplot
package require mdff_tmp
package provide mdff_check 0.3

namespace eval ::MDFF::Check:: {
  
  variable defaultRMSDsel {backbone}
  variable defaultCCsel {protein or nucleic}
  variable defaultMolid {top}
  variable defaultFrames {all}
  variable defaultCCDeprecate 0
  
  # XXX - Need to read in grid spacing instead
  variable defaultGridSpacing 1.0 

}


proc ::MDFF::Check::mdff_check_usage { } {
  
  variable defaultRMSDsel
  variable defaultCCsel
  variable defaultMolid
  variable defaultFrames
  variable defaultGridSpacing
  variable defaultCCDeprecate
  
  puts "Usage: mdff check ?options?"
  puts "Options:" 
  puts "  -mol <molid> (default: $defaultMolid)"
  puts "  -frames <begin:end> or <begin:step:end> or all or now (default: $defaultFrames)"
  puts ""
  puts "  -rmsd -- calculate the RMSD with respect to the refence structure"
  puts "  -rmsdseltext <selection text for calculating the RMSD> (default: $defaultRMSDsel)"
  puts "  -refpdb <pdb file> -- reference for RMSD calculation (default: frame 0)"
  puts "  -rmsdfile <file to write RMSD> (default: none)"

  puts ""
  puts "  -ccc -- calculate cross correlation coefficient"
  puts "  -map <input map> (required by -ccc)"
  puts "  -res <map resolution in Angstroms> (required by -ccc)"
  puts "  -spacing <grid spacing in Angstroms> (default based on res, otherwise if using -deprecate: $defaultGridSpacing)"
  puts "  -cccseltext <selection text for calculating the cross correlation (default: $defaultCCsel)"
  puts "  -cccfile  <file to write cross correlation> (default: none)"
  puts "  -threshold <x> (ignores voxels with values below x threshold. If using -deprecate, x is sigmas)"
  puts "  -deprecate <use the older, slower correlation algorithm (ccc only)> (on: 1 off: 0, default:$defaultCCDeprecate)"

  return
}


proc ::MDFF::Check::mdff_check { args } {

  variable defaultMolid
  variable defaultFrames
  variable defaultRMSDsel
  variable defaultCCsel
  variable defaultGridSpacing
  variable molid
  variable rmsd
  variable refpdb
  variable rmsdfile
  variable rmsdseltext
  variable cc
  variable ccfile
  variable ccseltext
  variable map
  variable deprecate
  variable resolution
  variable use_threshold
  variable threshold
  variable gridspacing
  variable use_gridspacingdefault
  variable defaultCCDeprecate

  set nargs [llength $args]
  if {$nargs == 0} {
    mdff_check_usage
    error ""
  }

  # should we calculate the RMSD?
  set pos [lsearch -exact $args {-rmsd}]
  if { $pos != -1 } {
    set rmsd 1
    set args [lreplace $args $pos $pos]
  } else {
    set rmsd 0
  }

  # should we calculate the CCC?
  set pos [lsearch -exact $args {-ccc}]
  if { $pos != -1 } {
    set cc 1
    set args [lreplace $args $pos $pos]
  } else {
    set cc 0
  }

  # parse switches
  foreach {name val} $args {
    switch -- $name {
      -mol          {set arg(mol)         $val }
      -refpdb       {set arg(refpdb)      $val }
      -rmsdfile     {set arg(rmsdfile)    $val }
      -rmsdseltext  {set arg(rmsdseltext) $val }
      -cccfile      {set arg(cccfile)     $val }
      -cccseltext   {set arg(cccseltext)  $val }
      -frames       {set arg(frames)      $val }
      -map          {set arg(map)         $val }
      -res          {set arg(res)         $val }
      -threshold    {set arg(threshold)   $val }
      -spacing      {set arg(spacing)     $val }
      -deprecate    {set arg(deprecate)   $val }
    }
  }

  if $rmsd {

    if { [info exists arg(refpdb)] } {
      set refpdb $arg(refpdb)
    } else {
      set refpdb 0
    }

    if { [info exists arg(rmsdfile)] } {
      set rmsdfile $arg(rmsdfile)
    } else {
      set rmsdfile 0
    }

    if { [info exists arg(rmsdseltext)] } {
      set rmsdseltext $arg(rmsdseltext)
    } else {
      set rmsdseltext $defaultRMSDsel
    }

  }
  
  if $cc {

    if { [info exists arg(map)] } {
      set map $arg(map)
    } else {
      error "option -ccc require input map (-map)."
    }

    if { [info exists arg(res)] } {
      set resolution $arg(res)
    } else {
      error "option -ccc requires map resolution (-res)."
    }

    if { [info exists arg(spacing)] } {
      set gridspacing $arg(spacing)
      set use_gridspacingdefault 0
    } else {
      set gridspacing $defaultGridSpacing
      set use_gridspacingdefault 1
    }

    if { [info exists arg(threshold)] } {
      set use_threshold 1
      set threshold $arg(threshold)
    } else {
      set use_threshold 0
    }

    if { [info exists arg(cccfile)] } {
      set ccfile $arg(cccfile)
    } else {
      set ccfile 0
    }

    if { [info exists arg(cccseltext)] } {
      set ccseltext $arg(cccseltext)
    } else {
      set ccseltext $defaultCCsel
    }
    
    if { [info exists arg(deprecate)] } {
      set deprecate $arg(deprecate)
    } else {
      set deprecate $defaultCCDeprecate
    }

  }

  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid $defaultMolid
  }
  if { $molid == "top" } {
    set molid [molinfo top]
  }

  # get frames
  if { ! [info exists arg(frames)] } { set arg(frames) $defaultFrames }
  set frames [::MDFF::Tmp::getFrames -molid $molid -frames $arg(frames)]
  set frames_begin [lindex $frames 0]
  set frames_step [lindex $frames 1]
  set frames_end [lindex $frames 2]
  
  if $rmsd {
    ::MDFF::Check::mdff_check_rmsd $frames_begin $frames_step $frames_end
  }
  if $cc {
    ::MDFF::Check::mdff_check_cc $frames_begin $frames_step $frames_end
  }

}


proc ::MDFF::Check::mdff_check_rmsd {frames_begin frames_step frames_end} {

  global tk_version
  variable molid
  variable refpdb
  variable rmsdfile
  variable rmsdseltext

  puts "Calculating RMSD..."
    
  if {$refpdb == 0} {
    puts "Using first frame as reference"
    set ref [atomselect $molid $rmsdseltext frame 0]
  } else {
    puts "Using coordinates in $refpdb as reference"
    set oldtop [molinfo top]
    set molref [mol new $refpdb type pdb waitfor all]
    mol top $oldtop
    set ref [atomselect $molref $rmsdseltext]
    if { [$ref num] == 0 } {
      $ref delete
      mol delete $molref
      error "Empty atom selection for reference RMSD calculation."
    }
  }

  set sel [atomselect $molid $rmsdseltext]
  if { [$sel num] == 0 } {
    $sel delete
    $ref delete
    if { $refpdb != 0 } {
      mol delete $molref
    }
    error "Empty atom selection for RMSD calculation."
  } elseif { [$sel num] != [$ref num] } {
    $sel delete
    $ref delete
    if { $refpdb != 0 } {
      mol delete $molref
    }
    error "Atom selection and reference atom selection for RMSD calculation do not contain the same number of atoms."
  }
  
  if { $rmsdfile != 0 } {
    puts "Opening $rmsdfile"
    set rmsdout [open $rmsdfile w]
  }

  set xlist {}
  set ylit {}  
  
  for {set f $frames_begin} {$f <= $frames_end} {incr f $frames_step} {
    $sel frame $f
    lappend xlist $f
    lappend ylist [measure rmsd $sel $ref]
    if { $rmsdfile != 0 } {
      puts $rmsdout "$f [measure rmsd $sel $ref]"
    }
  }

  if { $rmsdfile != 0 } {
    close $rmsdout
  }

  $ref delete
  $sel delete

  if { $refpdb != 0 } {
    mol delete $molref
  }

  if [info exists tk_version] {
    set plot [multiplot -x $xlist -y $ylist -title "RMSD" -xlabel "Frames" -ylabel "Angstroms" -lines -legend $rmsdseltext -plot]
  }
  
  return 

}


proc ::MDFF::Check::mdff_check_cc { frames_begin frames_step frames_end } {

  global tk_version
  variable molid
  variable map
  variable deprecate
  variable resolution
  variable gridspacing
  variable use_gridspacingdefault
  variable use_threshold
  variable threshold
  variable ccfile
  variable ccseltext

  puts "Calculating the cross correlation..."
    
  if { $ccfile != 0 } {
    set ccout [open $ccfile w]
  }
  
  set sel [atomselect $molid $ccseltext]
  set xlist {}
  set ylist {}  
  
  #added for mdffi compatability
  if {!$deprecate} {
    set mapid [mol new $map]
    mol top $molid
  }

  for {set f $frames_begin} {$f <= $frames_end} {incr f $frames_step} {
    $sel frame $f
    if $use_threshold {
      if {$deprecate} {
        set ccc [mdff ccc $sel -i $map -res $resolution -spacing $gridspacing -threshold $threshold]
      } elseif {$use_gridspacingdefault} {
        set ccc [mdffi cc $sel -mol $mapid -res $resolution -thresholddensity $threshold]
      } else {
        set ccc [mdffi cc $sel -mol $mapid -res $resolution -thresholddensity $threshold -spacing $gridspacing]
      }
    } else {
      if {$deprecate} {
        set ccc [mdff ccc $sel -i $map -res $resolution -spacing $gridspacing]
      } elseif {$use_gridspacingdefault} {
        set ccc [mdffi cc $sel -mol $mapid -res $resolution]
      } else {
        set ccc [mdffi cc $sel -mol $mapid -res $resolution -spacing $gridspacing]
      }
    }
    lappend xlist $f
    lappend ylist $ccc
    if { $ccfile != 0 } {
      puts $ccout "$f $ccc"
    }
  }

  if { $ccfile != 0 } {
    close $ccout
  }

  $sel delete

  if [info exists tk_version] {
    set plot [multiplot -x $xlist -y $ylist -title "Cross-correlation coefficient"  -xlabel "Frames" -ylabel "CCC" -lines -legend $ccseltext -plot]
  }

  return 
}


