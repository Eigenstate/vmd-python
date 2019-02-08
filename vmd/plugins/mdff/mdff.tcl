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
#       $RCSfile: mdff.tcl,v $
#       $Author: ryanmcgreevy $        $Locker:  $             $State: Exp $
#       $Revision: 1.7 $       $Date: 2019/01/10 16:06:22 $
#
############################################################################


# MDFF package
# Authors: Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
#          Elizabeth Villa <villa@ks.uiuc.edu>

package require mdff_check
package require mdff_correlation
package require mdff_map
package require mdff_setup
package require mdff_sim
#package require mdff_gui

package provide mdff 0.5

namespace eval ::MDFF {

}

proc mdff { args } { return [eval ::MDFF::mdff $args] }

proc ::MDFF::mdff_usage { } {
  puts "Usage: mdff <command> \[args...\]"
  puts "Commands:"
  puts "  ccc        -- calculates the cross-correlation coefficient"
  puts "  check      -- monitors the fitting via RMSD and CCC"
  puts "  constrain  -- creates a pdb file for restraining atoms"
  puts "  delete     -- deletes volume corresponding to atomic structure"
  puts "  edges      -- creates a map with smooth edges"
  puts "  fix        -- creates a pdb file for fixing atoms"
  puts "  griddx     -- creates a map for docking"
  puts "  gridpdb    -- creates a pdb file with atomic masses in the beta field"
  puts "  hist       -- calculates a density histogram"
  puts "  setup      -- writes a NAMD configuration file for MDFF"
  puts "  sim        -- creates a simulated map from an atomic structure"
  return

}

proc ::MDFF::mdff { args } {

  set nargs [llength $args]
  if { $nargs == 0 } {
    mdff_usage
    error ""
  }

  # parse command
  set command [lindex $args 0]
  set args [lreplace $args 0 0]

  if { $command == "check" } {
    return [eval ::MDFF::Check::mdff_check $args]
  } elseif { $command == "constrain" } {
    return [eval ::MDFF::Setup::mdff_constrain $args]
  } elseif { $command == "delete" } {
    return [eval ::MDFF::Map::mdff_delete $args]
  } elseif { $command == "fix" } {
    return [eval ::MDFF::Setup::mdff_fix $args]
  } elseif { $command == "griddx" } {
    return [eval ::MDFF::Map::mdff_griddx $args]
  } elseif { $command == "gridpdb" } {
    return [eval ::MDFF::Setup::mdff_gridpdb $args]
  } elseif { $command == "hist" } {
    return [eval ::MDFF::Map::mdff_histogram $args]
  } elseif { $command == "ccc" } {
    return [eval ::MDFF::Correlation::mdff_ccc $args]
  } elseif { $command == "setup" } {
    return [eval ::MDFF::Setup::mdff_setup $args]
  } elseif { $command == "sim" } {
    return [eval ::MDFF::Sim::mdff_sim $args]
  } elseif { $command == "edges" } {
    return [eval ::MDFF::Map::mdff_edge $args]
  } elseif { $command == "hist" } {
    return [eval ::MDFF::Map::mdff_histogram $args]
  } else {
    mdff_usage
    error "Unrecognized command."
  }

  return

}

