#!/usr/local/bin/vmd
# replace some water molecules with Na and Cl ions
# Ilya Balabin, July 15, 2003
# 
# Partially rewritten by Leonardo Trabuco, July 2010
# $Id: autoionize.tcl,v 1.25 2015/04/20 20:41:13 ryanmcgreevy Exp $

# TODO:
# - allow user to specify topology file
# - add switch to use Amber force field instead of CHARMM
# - remove dead code related to -is option, since that is
#   no longer supported.

package require psfgen
package require readcharmmtop
package provide autoionize 1.4

namespace eval ::autoionize:: {
  namespace export autoionize

  # Maximum number of ion placement attempts
  variable maxTries 10

  # Minimum distance from molecule
  variable defaultFromDistance 5

  # Minimum distance between ions
  variable defaultBetweenDistance 5

  variable defaultPrefix {ionized}
  variable defaultSegname {ION}

  # XXX - Note that the modes neutralize, ionic strength, and salt
  #       concentration) currently support only ions with charge 
  #       -2, -1, +1, and +2. If you add an ion with a different charge 
  #       from the above, make sure you update the code for these modes
  #       accordingly.
  variable supportedIons {
    {SOD SOD 1  sodium    Na+ }
    {MG  MG  2  magnesium Mg2+}
    {POT POT 1  potassium K+  }
    {CES CES 1  cesium    Cs+ }
    {CAL CAL 2  calcium   Ca2+}
    {ZN2 ZN  2  zinc      Zn2+}
    {CLA CLA -1 chloride  Cl- }
  }

}

proc autoionize { args } { return [eval ::autoionize::autoionize $args] }

proc ::autoionize::autoionize_usage { } {
  variable defaultFromDistance
  variable defaultBetweenDistance
  variable defaultSegname
  variable defaultPrefix
  variable supportedIons

  puts "Usage: autoionize -psf file.psf -pdb file.pdb <mode> \[options\]"
  puts "Ion placement mode (choose one):"
  puts "  -neutralize              -- only neutralize system"
  puts "  -sc <salt concentration> -- neutralize and set salt concentration (mol/L)"
  # puts "  -is <ionic strength>     -- neutralize and set ionic strength (mol/L)"
  puts "  -nions {{ion1 num1} {ion2 num2} ...} -- user defined number of ions"
  puts "Other options:"
  puts "  -cation <ion resname>    -- default: SOD"
  puts "  -anion <ion resname>     -- default: CLA"
  puts "  -o <prefix>              -- output file prefix (default: ionized)"
  puts "  -from <distance>         -- min. distance from solute (default: 5A)"
  puts "  -between <distance>      -- min. distance between ions (default: 5A)"
  puts "  -seg <segname>           -- specify new segment name (default: ION)"
  puts "Supported ions (CHARMM force field resnames):"
  foreach ion $supportedIons {
    puts [format "   %-3s -- %-9s (%s)" [lindex $ion 0] [lindex $ion 3] [lindex $ion 4]]
  }

  error ""
}

proc ::autoionize::autoionize {args} {
  global errorInfo errorCode
  set oldcontext [psfcontext new]  ;# new context
  set errflag [catch { eval ::autoionize::autoionize_core $args } errMsg]
  set savedInfo $errorInfo
  set savedCode $errorCode
  psfcontext $oldcontext delete  ;# revert to old context
  if $errflag { error $errMsg $savedInfo $savedCode }
}

proc ::autoionize::autoionize_core {args} {
  variable maxTries
  variable defaultFromDistance
  variable defaultBetweenDistance
  variable defaultPrefix
  variable defaultSegname
  global env

  set n [llength $args]
  if {$n == 0} {autoionize_usage}

  #####################################################################
  # Begin of command-line parsing code

  # Neutralize system?
  set pos [lsearch -exact $args {-neutralize}]
  if { $pos != -1 } {
    set mode {neutralize}
    set modeflag 1
    set args [lreplace $args $pos $pos]
  } else {
    set modeflag 0
  }

  # get all options
  for { set i 0 } { $i < $n } { incr i 2 } {
    set key [lindex $args $i]
    set val [lindex $args [expr $i + 1]]
    set cmdline($key) $val
  }

  # Each element of nIonsList contains the following information about
  # the ions to be placed: residue name, charge, number of ions.
  set nIonsList {}

  ### Parse options for each ion placement mode

  # Option -nna/-ncl (deprecated)
  #
  # Note that the -nna and -ncl options are deprecated. The current version
  # still supports them, but they are now longer documented.
  if { ( [info exists cmdline(-nna)] && ![info exists cmdline(-ncl)] ) ||
       ( [info exists cmdline(-ncl)] && ![info exists cmdline(-nna)] ) } {
    error "Autoionize) ERROR: Both -nna and -ncl need to be specified."
  }
  if { [info exists cmdline(-nna)] && [info exists cmdline(-ncl)] } {
    set mode {nions}
    incr modeflag
    set nNa $cmdline(-nna)
    set nCl $cmdline(-ncl)

    foreach ion {SOD CLA} charge {1 -1} num [list $nNa $nCl] {
      if {![string is integer $num] || $num < 0} {
        error "Autoionize) ERROR: Expected positive integer number of ions but got '$num'."
      } elseif {$num == 0} {
        puts "Autoionize) WARNING: Requested placement of 0 $ion ions. Ignoring..."
        continue
      }
      lappend nIonsList [list $ion $charge $num]
    }
  }

  # Option -is
  if { [info exists cmdline(-is)] } {
    set mode {is}
    set ionicStrength $cmdline(-is)
    if {$ionicStrength < 0} {
      error "Autoionize) ERROR: Cannot set the ionic strength to a negative value."
    }
    incr modeflag
  }

  # Option -sc
  if { [info exists cmdline(-sc)] } {
    set mode {sc}
    set saltConcentration $cmdline(-sc)
    if {$saltConcentration < 0} {
      error "Autoionize) ERROR: Cannot set the salt concentration to a negative value."
    }
    incr modeflag
  }

  # Option -nions
  if { [info exists cmdline(-nions)] } {
    set mode {nions}
    incr modeflag

    foreach ion_num $cmdline(-nions) {
      set ion [lindex $ion_num 0]
      set num [lindex $ion_num 1]

      if {![string is integer $num] || $num < 0} {
        error "Autoionize) ERROR: Expected positive integer number of ions but got '$num'."
      } elseif {$num == 0} {
        puts "Autoionize) WARNING: Requested placement of 0 $ion ions. Ignoring..."
        continue
      }

      set charge [ionGetCharge $ion]
      lappend nIonsList [list $ion $charge $num]
    }
  }

  if { $modeflag == 0 } {
    error "Autoionize) ERROR: Ion placement mode was not specified."
  }

  if { $modeflag > 1 } {
    error "Autoionize) ERROR: Multiple ion placement modes requested."
  }

  # Check if no ions were requested via options -nions or -nna/-ncl
  if {$mode == {nions}} {
    if {[llength $nIonsList] == 0} {
      error "ERROR: Requested 0 ions to be placed."
    }
  }

  if { [info exists cmdline(-psf)] } {
    set psffile $cmdline(-psf)
  } else {
    error "Autoionize) ERROR: Missing psf file."
  }

  if { [info exists cmdline(-pdb)] } {
    set pdbfile $cmdline(-pdb)
  } else {
    error "Autoionize) ERROR: Missing pdb file."
  }

  # set optional parameters
  if { [info exists cmdline(-o)] } {
    set prefix $cmdline(-o)
  } else {
    set prefix $defaultPrefix
  }
  if { [info exists cmdline(-from)] } {
    set from $cmdline(-from)
  } else {
    set from $defaultFromDistance
  }
  if { [info exists cmdline(-between)] } {
    set between $cmdline(-between)
  } else {
    set between $defaultBetweenDistance
  }
  if { [info exists cmdline(-seg)] } {
    set segname $cmdline(-seg)
  } else {
    set segname $defaultSegname
  }

  if { [info exists cmdline(-cation)] } {
    set cation $cmdline(-cation)
    set cationCharge [ionGetCharge $cation]
    if {$cationCharge <= 0} {
      error "Autoionize) ERROR: Requested cation $cation has non-positive charge $cationCharge."
    }
    if {$mode == {nions}} {
      error "Autoionize) ERROR: Cannot use option -cation togeth with -nions."
    }
  } else {
    set cation {SOD}
    set cationCharge 1
  }

  if { [info exists cmdline(-anion)] } {
    set anion $cmdline(-anion)
    set anionCharge [ionGetCharge $anion]
    if {$anionCharge >= 0} {
      error "Autoionize) ERROR: Requested anion $anion has non-negative charge $anionCharge."
    }
    if {$mode == {nions}} {
      error "Autoionize) ERROR: Cannot use option -anion togeth with -nions."
    }
  } else {
    set anion {CLA}
    set anionCharge -1
  }

  # The previous version of autoionize used a wrong definition of ionic 
  # strength. The code has been updated to use the correct definition of
  # ionic strength; however, this breaks backward compatibility with
  # any script that used the option -is. Thus, we'll phase out the use
  # of -is instead by removing it from the documentation, exposing only
  # the new option -sc (salt concentration). We'll allow the used to 
  # still use -is with the old (wrong) behavior, provided the salt has not 
  # been changed to anything other than NaCl. 
  if {$mode == {is}} {
    if {$cation != {SOD} || $anion != {CLA}} {
      error "Autoionize) The option -is is no longer supported."
    } else {
      puts "Autoionize) WARNING: The option -is is deprecated; please use -sc instead."
      set mode {sc}
      set saltConcentration [expr 0.5 * $ionicStrength]
    }
  }

  # End of command-line parsing code
  #####################################################################

  # Read in system
  puts "Autoionize) Reading ${psffile}/${pdbfile}..."
  resetpsf
  readpsf $psffile
  coordpdb $pdbfile
  mol new $psffile type psf waitfor all
  mol addfile $pdbfile type pdb waitfor all

  # Get pbc info for later
  set xdim [molinfo top get a]
  set ydim [molinfo top get b]
  set zdim [molinfo top get c]

  # Compute net charge of the system
  set sel [atomselect top all]
  set netCharge [eval "vecadd [$sel get charge]"]
  set roundNetCharge [expr round($netCharge)]
  $sel delete
  puts "Autoionize) System net charge before adding ions: ${netCharge}e."

  set done 0 ;# flag to tell if there are ions to be placed
  set nCation 0
  set nAnion 0

  # For each ion placement mode, calculate the number of each ion
  # and set the nIonsList accordingly.

  ###
  ### Ion placement mode 'neutralize'. Also called in combination 
  ### with other modes.
  ###
  if {$mode != "nions"} {

    # XXX - The following implementation will work only for ions with 
    #       charge -2, -1, +1, and +2. 

    set errflag 0
    if {$roundNetCharge > 0} {
      if {$anionCharge == -1} {
        set nAnion $roundNetCharge
        set nCation 0
      } elseif {$anionCharge == -2} {
        if {[expr {$roundNetCharge % 2}] == 0} { ;# total charge is even
          set nAnion [expr {$roundNetCharge / 2}]
          set nCation 0
        } else { ;# total charge is odd
          if {$cationCharge == 1} {
            set nAnion [expr {1+int($roundNetCharge/2)}]
            set nCation 1
          } else {
            set errflag 1
          }
        }
      }
    } elseif {$roundNetCharge < 0} {
      set roundNetChargeAbs [expr {-1*$roundNetCharge}]
      if {$cationCharge == 1} {
        set nAnion 0
        set nCation $roundNetChargeAbs
      } elseif {$cationCharge == 2} {
        if {[expr {$roundNetChargeAbs % 2}] == 0} { ;# total charge is even
          set nAnion 0
          set nCation [expr {$roundNetChargeAbs / 2}]
        } else { ;# total charge is odd
          if {$anionCharge == -1} {
            set nAnion 1
            set nCation [expr {1+int($roundNetChargeAbs/2)}]
          } else {
            set errflag 1
          }
        }
      }
    } elseif {$mode == "neutralize"} { ;# we were only requested to neutralize
      puts "Autoionize) The system is already neutral."
      set done 1
    }

    # FIXME -- Maybe do the best possible job instead of bailing out...
    if {$errflag == 1} {
      error "Autoionize) ERROR: Could not neutralize system."
    }

    if {$done == 0} {
      puts "Autoionize) Number of ions required for neutralizing the system: $nCation $cation and $nAnion $anion."
      set newTotalCharge [expr $nAnion * $anionCharge + $nCation * $cationCharge + $roundNetCharge]
      if {$newTotalCharge != 0} {
        error "Autoionize) ERROR: Could not neutralize system."
      }
    }
    #puts "DEBUG:   Ionized system would have charge $newTotalCharge"

  }

  ###
  ### Ion placement modes 'sc' = salt concentration
  ###                     'is' = ionic strength
  ###
  if {$mode == "sc"} {
    puts "Autoionize) Desired salt concentration: ${saltConcentration} mol/L."
  } elseif {$mode == "is"} {
    puts "Autoionize) Desired ionic strength: ${ionicStrength} mol/L."
  }
  
  if {$mode == "sc" || $mode == "is"} {

    set sel [atomselect top "water and noh"]
    set nWater [$sel num]
    $sel delete

    if {$nWater == 0} {
      error "Autoionize) ERROR: Cannot add ions to unsolvated system."
    }

    # Guess chemical formula. 
    # XXX - We currently only support ions with charge -2, -1, +1, and +2.
    if {$cationCharge == 1 && $anionCharge == -1} { ;# e.g., NaCl, KCl, ...
      set cationStoich 1
      set anionStoich 1
    } elseif {$cationCharge == 2 && $anionCharge == -1} { ;# e.g., MgCl2
      set cationStoich 1
      set anionStoich 2
    } elseif {$cationCharge == 1 && $anionCharge == -2} {
      set cationStoich 2
      set anionStoich 1
    } elseif {$cationCharge == 2 && $anionCharge == -2} {
      set cationStoich 1
      set anionStoich 1
    } else {
      error "Autoionize) ERROR: Unsupported ion charge; cannot guess chemical formula."
    }

    if {$mode == "is"} { ;# convert ionic strength to salt concentration
      set cationConcentration [expr {2 * $ionicStrength / ( sqrt($cationCharge * $cationCharge * $anionCharge * $anionCharge) + $cationCharge * $cationCharge)}]
      set saltConcentration [expr {$cationConcentration * $cationStoich}]
    }

    # As long as saltConcentration and ionicStrength are non-negative,
    # no error checking is needed here...
    set num [expr {int(0.5 + 0.0187 * $saltConcentration * $nWater)}]
    set nCation [expr {$nCation + $cationStoich * $num}]
    set nAnion [expr {$nAnion + $anionStoich * $num}]

  }

  if {$mode != "nions"} {
    if {$nCation > 0} {
      lappend nIonsList [list $cation $cationCharge $nCation]
    } 
    if {$nAnion > 0} {
      lappend nIonsList [list $anion $anionCharge $nAnion]
    }
    if {$nCation == 0 && $nAnion == 0} {
      # Just in case the system is neutral and the requested ionic
      # strength or salt concentration is zero, in which case the
      # behavior is the same as -neutralize, i.e., copy files and exit
      # normally.
      set done 1
    }
    if {$nCation < 0 || $nAnion < 0} {
      error "Autoionize) ERROR: Internal error; negative number of ions."
    }
  }

  if {$done == 1} {
    puts "Autoionize) Nothing to be done; copying system to ${prefix}.psf/${prefix}.pdb..."
    file copy -force $psffile ${prefix}.psf
    file copy -force $pdbfile ${prefix}.pdb
  } else {

    puts "Autoionize) Ions to be placed:"
    set nIons 0
    foreach ion $nIonsList {
      puts "Autoionize)   [lindex $ion 2] [lindex $ion 0]"
      set nIons [expr {$nIons + [lindex $ion 2]}]
    }

    puts "Autoionize) Required min distance from molecule: ${from}A."
    puts "Autoionize) Required min distance between ions: ${between}A."
    puts "Autoionize) Output file prefix \'${prefix}\'."

    # Make sure requested segname does not already exist
    set seltest [atomselect top "segname $segname"]
    if { [$seltest num] != 0 } {
      set segnameOK 0
      set segnameWarn 1
    } else {
      set segnameOK 1
      set segnameWarn 0
    }
    $seltest delete
  
    # If segname is duplicate, try using instead ION1, ION2, ...
    if {$segnameOK == 0} {
      for {set i 1} {$i <= 9} {incr i} { 
        set tryseg "ION${i}"
        set seltest [atomselect top "segname $tryseg"]
        if { [$seltest num] == 0 } {
          set segname $tryseg
          set segnameOK 1
          $seltest delete
          break
        }
        $seltest delete
      }
    }
  
    # If segname is still duplicate, try using instead IN1, IN2, ...
    if {$segnameOK == 0} {
      for {set i 1} {$i <= 99} {incr i} {
        set tryseg "IN${i}"
        set seltest [atomselect top "segname $tryseg"]
        if { [$seltest num] == 0 } {
          set segname $tryseg
          set segnameOK 1
          $seltest delete
          break
        }
        $seltest delete
      }
    }
  
    if {$segnameWarn} {
      if {$segnameOK} {
        puts "Autoionize) WARNING: Ions will be added to segname $segname to avoid duplication."
      } else {
        error "Autoionize) ERROR: Could not determine a segname to avoid duplication."
      }
    } else {
      puts "Autoionize) Ions will be added to segname $segname."
    }

    # Find water oxygens to replace with ions
  
    set nTries 0
    while {1} {
      set ionList {}
      set sel [atomselect top "noh and water and not (within $from of not water)"]
      set watIndex [$sel get index]
      set watSize [llength $watIndex]
  
      set count 0
  
      while {[llength $ionList] < $nIons} {
        if {!$watSize} {break}
        set thisIon [lindex $watIndex [expr int($watSize * rand())]]
        if {[llength $ionList]} {
          set tempsel [atomselect top "index [concat $ionList] and within $between of (index $thisIon)"]
        } else {
          set tempsel [atomselect top "water and not water"]
        }
        if {![$tempsel num]} {
          lappend ionList $thisIon
        }
        $tempsel delete
      }
      $sel delete
      if {[llength $ionList] == $nIons} {break}
      incr nTries
      if {$nTries == $maxTries} {
        puts "Autoionize) ERROR: Failed to add ions after $maxTries tries"
        puts "Autoionize) Try decreasing -from and/or -between parameters,"
        puts "Autoionize) decreasing ion concentration, or adding more water molecules..."
        exit
      }	
    }
    puts "Autoionize) Obtained positions for $nIons ions."

    # Select and delete the waters but store the coordinates!
    set sel [atomselect top "index $ionList"]
    set waterPos [$sel get {x y z}]
    set num1 [llength $waterPos]
    puts "Autoionize) Tagged ${num1} water molecules for deletion."
  
    set num1 0
    foreach segid [$sel get segid] resid [$sel get resid] {
      delatom $segid $resid
      incr num1
    }
    puts "Autoionize) Deleted ${num1} water molecules."

    # Read in topology file
    puts "Autoionize) Reading CHARMM topology file..."
    set topfile [file join $env(CHARMMTOPDIR) toppar_water_ions_namd.str]
    topology $topfile

    # Make topology entries
    set resid 1
    segment $segname {
      first NONE
      last NONE
      foreach ion $nIonsList {
        set resname [lindex $ion 0]
        set num [lindex $ion 2]
        for {set i 0} {$i < $num} {incr i} {
          residue $resid $resname
          incr resid
        }
      }
    }

    # Randomize ion positions (otherwise Cl ions tend to stick together)
    puts "Autoionize) Randomizing ion positions..."
    set newPos {}
    while {[llength $waterPos] > 0} {
      set thisNum [expr [llength $waterPos] * rand()]
      set thisNum [expr int($thisNum)]
      lappend newPos [lindex $waterPos $thisNum]
      set waterPos [lreplace $waterPos $thisNum $thisNum]
    }
    set waterPos $newPos

    # Assign ion coordinates
    puts "Autoionize) Assigning ion coordinates..."
    set resid 1
    foreach ion $nIonsList {
      set name [ionGetName [lindex $ion 0]]
      set startpos [expr {$resid - 1}]
      set endpos [expr {[lindex $ion 2] - 1 + $startpos}]
      foreach pos [lrange $waterPos $startpos $endpos] {
        # puts "DEBUG: coord $segname $resid $name $pos"
        coord $segname $resid $name $pos
        incr resid
      }
    }
    
    writepsf $prefix.psf
    writepdb $prefix.pdb

  }

  # Update displayed molecule
  puts "Autoionize) Loading new system with added ions..."
  mol delete top
  mol new $prefix.psf type psf waitfor all
  mol addfile $prefix.pdb type pdb waitfor all

  # Re-write the pdb including periodic cell info
  molinfo top set a $xdim
  molinfo top set b $ydim
  molinfo top set c $zdim
  set sel [atomselect top all]
  $sel writepdb $prefix.pdb 

  # Re-compute net charge of the system
  set netCharge [eval "vecadd [$sel get charge]"]
  $sel delete
  puts "Autoionize) System net charge after adding ions: ${netCharge}e."
  if {[expr abs($netCharge - round($netCharge))] > 0.001} {
    if {[winfo exists .autoigui]} {
      tk_messageBox -icon warning -message "System has a non-integer total charge. There was likely a problem in the process of building it."
    } else {
      puts "Autoionize) WARNING: System has a non-integer total charge. There was likely a problem in the process of building it."
    }
  }
    
  puts "Autoionize) All done."
}

proc ::autoionize::ionGetCharge {ion} {
  variable supportedIons

  foreach sion $supportedIons {
    if {$ion == [lindex $sion 0]} {
      set charge [lindex $sion 2]
      return $charge
    }
  }
  error "Autoionize) ERROR: Unsupported ion $ion."
}

proc ::autoionize::ionGetName {ion} {
  variable supportedIons

  foreach sion $supportedIons {
    if {$ion == [lindex $sion 0]} {
      set name [lindex $sion 1]
      return $name
    }
  }
  error "Autoionize) ERROR: Unsupported ion $ion."
}
