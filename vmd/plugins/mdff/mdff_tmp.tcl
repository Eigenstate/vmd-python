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
#       $RCSfile: mdff_tmp.tcl,v $
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.6 $       $Date: 2017/01/13 18:26:46 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>

# These are procs that could be used by other plugins and should
# eventually be moved out of the MDFF package.

package require molefacture
package provide mdff_tmp 0.2

namespace eval ::MDFF::Tmp {

}

# Code from vmdmovie.tcl to determine directory for tmp files
proc ::MDFF::Tmp::tmpdir {} {
  global env
  if { [info exists env(TMPDIR)] && [vmdinfo arch] != "WIN32" } {
    set tmpdir $env(TMPDIR)
  } else {
    switch [vmdinfo arch] { 
      WIN64 -
      WIN32 {
        if [info exists env(TMP)] {
          set tmpdir $env(TMP)
        } else {
          set tmpdir "c:/"
        }     
      }
      MACOSXX86_64 -
      MACOSXX86 -
      MACOSX {
        set tmpdir "/"
      }
      default {
        set tmpdir "/tmp"
      } 
    }
  }

  if [file writable $tmpdir] {
    return $tmpdir
  } else {
    set curdir [pwd]
    if [file writable $curdir] {
      puts "Warning) Temporary directory $tmpdir is not writable. Defaulting to current directory $curdir"
      return $curdir
    } else {
      error "ERROR) Could not find a writable directory for temporary files. You can specify a temporary directory by setting the environment variable TMPDIR, e.g.:\n  set env(TMPDIR) /tmp"
    }
  }

}

# Return a random file name given an optional prefix and/or suffix.
# If a directory is also given, make sure the file doesn't already exist.
proc ::MDFF::Tmp::tmpfilename { args } {

  set prefix {}
  set suffix {}

  foreach {name val} $args {
    switch -- $name {
      -prefix { set prefix $val }
      -suffix { set suffix $val }
      -tmpdir { set tmpdir $val }
    }
  }

  set filename "$prefix[lindex [split [expr rand()] .] 1]$suffix"

  if { [info exists tmpdir] } {
    while { [file exists [file join $tmpdir $filename]] } {
      puts "File [file join $tmpdir $filename] already exists. Generating a new filename..."
      set filename "$prefix[lindex [split [expr rand()] .] 1]$suffix"
    }
  }

  return $filename

}

# This is a more forgiving version of 'atomselect get atomicnumber'. In
# case atomic number information is present, defaults to the above
# command. In case it is not present, it uses a molefacture function
# to guess the atomic number.
proc ::MDFF::Tmp::getAtomicNumber { sel } {

  if { [$sel num] == 0 } {
    error "getAtomicNumber: empty atomselection."
  }

  set atomicNumber [$sel get atomicnumber]

  # If atomic number information is available for the entire selection
  # simply return the result of 'atomselect get atomicnumber'
  if { [lsearch -regexp $atomicNumber {-1|0}] == -1 } {
    return $atomicNumber
  }

  # Otherwise, guess missing elements
  puts "Warning: guessing atomic number for atoms with unknown element..."
  set indexList [$sel get index]
  set nameList [$sel get name]
  set resnameList [$sel get resname]
  set massList [$sel get mass]
  for {set i 0} {$i < [llength $atomicNumber]} {incr i} {
    if { [lindex $atomicNumber $i] <= 0 } {
      set element [::Molefacture::get_element [lindex $nameList $i] [lindex $resnameList $i] [lindex $massList $i]]
      lset atomicNumber $i [lsearch -exact $::Molefacture::periodic $element]
    }
  }
  unset nameList
  unset resnameList
  unset massList
  $sel set atomicnumber $atomicNumber
  if { [lsearch -exact $atomicNumber -1] != -1 } {
    error "Error: failed to guess atomic number."
  }

  return $atomicNumber

}

# Frame selection utility. This code is duplicated several times in 
# the plugin tree.
#
# Options: -frames <begin:end> or <begin:step:end> or all or now (default: all)"
#          -molid <molid> (default: top)"
#
proc ::MDFF::Tmp::getFrames { args } {

  set molid {top}
  foreach {name val} $args {
    switch -- $name {
      -molid { set molid $val }
      -frames {set frames $val }
    }
  }

  set nowframe [molinfo $molid get frame]
  set lastframe [expr [molinfo $molid get numframes] - 1]

  if [info exists frames] {
    set fl [split $frames :]
    puts $fl
    puts [llength $fl]
    switch -- [llength $fl] {
      1 {
        switch -- $fl {
          all {
            set frames_begin 0
            set frames_end $lastframe
          }
          now {
            set frames_begin $nowframe
          }
          last {
            set frames_begin $lastframe
          }
          default {
            set frames_begin $fl
          }
        }
      }
      2 {
        set frames_begin [lindex $fl 0]
        set frames_end [lindex $fl 1]
      }
      3 {
        puts "okay"
        set frames_begin [lindex $fl 0]
        set frames_step [lindex $fl 1]
        set frames_end [lindex $fl 2]
      }
      default { error "bad -frames arg: $frames" }
    }
  } else {
    set frames_begin 0
  }
  if { ! [info exists frames_step] } { set frames_step 1 }
  if { ! [info exists frames_end] } { set frames_end $lastframe }
    switch -- $frames_end {
      end - last { set frames_end $lastframe }
  }
  if { [ catch {
    if { $frames_begin < 0 } {
      set frames_begin [expr $lastframe + 1 + $frames_begin]
    }
    if { $frames_end < 0 } {
      set frames_end [expr $lastframe + 1 + $frames_end]
    }
    if { ! ( [string is integer $frames_begin] && \
      ( $frames_begin >= 0 ) && ( $frames_begin <= $lastframe ) && \
	  [string is integer $frames_end] && \
  	  ( $frames_end >= 0 ) && ( $frames_end <= $lastframe ) && \
  	  ( $frames_begin <= $frames_end ) && \
  	  [string is integer $frames_step] && ( $frames_step > 0 ) ) } {
        error
      }
  } ok ] } { error "bad -frames arg: $frames" }

  return [list $frames_begin $frames_step $frames_end]

}
