#
# cispeptide - identifies and fixes cis peptide bonds in proteins
#
# Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
# Eduard Schreiner <eschrein@ks.uiuc.edu>
#
# $Id: cispeptide.tcl,v 1.20 2015/04/01 21:11:54 ryanmcgreevy Exp $
#

# TODO: 
# - add regression tests
# - add a 'move undo'
# - add option to move all
# - implement 'I'm feeling lucky' feature
# - pull cispeptide_check_files out of the chirality plugin since it is
#   useful for other plugins such as the chirality one
# - force GUI list update when running from the command line
# - check for CISPEP in the PDB file?
# - make molecule visible when showing selected cis peptide bond

package provide cispeptide 1.2
if [info exists tk_version] {
  package require autoimd
}

namespace eval ::cispeptide:: {

  variable defaultSelText {same fragment as protein}

  # Cut-off to identify a cis peptide bond based on a dihedral angle
  variable dihedCutoff 85

  # Cut-offs to check for clashes when moving atoms
  variable clashWithinCutoff 4
  variable clashCutoff 0.5

  # Cut-off for selecting water and ions for AutoIMD
  variable minWithinCutoff 10

  # Force constant for dihedral restraints
  variable cispeptideExtraBondsK 200

  variable cispeptideSelRestrainAtomsSet 0

  variable cispeptideReps {}

  proc cispeptide_reset_reps { } {
    variable cispeptideReps

    foreach {molid repname} $cispeptideReps {
      if { [lsearch [molinfo list] $molid] != -1 } {
        set repid [mol repindex $molid $repname]
        mol delrep $repid $molid
      }
    }
    set cispeptideReps {}

    return
  }

  proc cispeptide_reset { } {

    # List of pairs of residue numbers involved in a cis peptide bond
    variable cispeptideResiduesList {}

    # List of per cis peptide bond action, flagging which atom should be
    # moved, if any
    variable cispeptideActionsList {}

    # List of per cis peptide bond index of moved atom
    variable cispeptideMovedList {}

    variable cispeptideCurrent 0

    variable idHListAutoIMD {} 
    variable idNListAutoIMD {}
    variable idCListAutoIMD {}
    variable idOListAutoIMD {}

    variable cispeptideSelRestrainAtomsSet
    if $cispeptideSelRestrainAtomsSet {
      set cispeptideSelRestrainAtomsSet 0
      uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsH delete
      uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsN delete
      uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsC delete
      uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsO delete
    }

    return
  }
  cispeptide_reset

}

# Regression tests for identifying cis peptide bonds from PDB structures.
#
# Make sure you run ::cispeptide::run_tests after each change. If a tes
# fails, check manually for correctness. If any structure revealed a bug,
# make sure to add it to regressionTests below.
#
proc ::cispeptide::run_tests {} {

  # Each list element contains a PDB id and expected number of cis 
  # peptide bonds. Only the first frame of each molecule is considered.
  set regressionTests {
    {2afw 8} 
    {1an5 0}
    {2b5m 1}
    {2b5r 5}
    {2b83 0}
    {316d 4}
    {209d 4}
    {487d 0}
    {103m 0}
    {205l 0}
    {208l 0}
    {308d 0}
    {408d 0}
    {209d 4}
    {209l 2}
    {20gs 2}
    {110l 0}
    {212l 0}
    {312d 0}
    {3a0n 2}
    {3a0r 2}
    {3a0o 8}
    {3b82 6}
    {3e7p 0}
    {2flq 2}
    {3l4g 8}
    {2piy 9}
    {3a0u 0}
    {2a1v 1}
    {3a2o 0}
    {2a8s 1}
    {3ak5 1}
  }

  set numTests [llength $regressionTests]
  set failCount 0
  set passCount 0
  cispeptide reset
  for {set i 0} {$i < $numTests} {incr i} {
    set pdbId [lindex $regressionTests $i 0]
    set expectedCount [lindex $regressionTests $i 1]
    mol new $pdbId waitfor all
    animate goto 0
    set count [cispeptide check -mol top]
    if {$count != $expectedCount} {
      puts "cispeptide) TEST [expr {$i + 1}]/$numTests: FAILED"
      puts "cispeptide) pdb: $pdbId, expected: $expectedCount, actual: $count"
      incr failCount
    } else {
      puts "cispeptide) TEST [expr {$i + 1}]/$numTests: PASSED"
      incr passCount
    }
    cispeptide reset
    mol delete top
  }

  puts "\ncispeptide) SUMMARY OF REGRESSION TESTS"
  puts "cispeptide) ---------------------------"
  puts "cispeptide) $passCount test(s) passed."
  puts "cispeptide) $failCount test(s) failed."

  return
}

proc cispeptide { args } { return [eval ::cispeptide::cispeptide $args] }

proc ::cispeptide::cispeptide_usage { } {
  puts "Usage: cispeptide <command> \[args...\]"
  puts "Commands:"
  puts "  check    -- identify cis peptide bonds"
  puts "  list     -- list identified cis peptide bonds"
  puts "  minimize -- fix cis peptide bonds using energy minimization"
  puts "  move     -- move specified atom to convert to trans"
  puts "  reset    -- reinitialize plugin state"
  puts "  restrain -- generate NAMD extrabonds file to restrain peptide bonds"
  puts "  set      -- define how to modify a given cis peptide bond"
  puts "  show     -- visualize identified cis peptide bonds"
  return
}

proc ::cispeptide::cispeptide { args } {

  set nargs [llength $args]
  if { $nargs == 0 } {
    cispeptide_usage
    error ""
  }

  # parse command
  set command [lindex $args 0]
  set args [lreplace $args 0 0]

  if { $command == "check" } {
    return [eval cispeptide_check $args]
  } elseif { $command == "list" } {
    return [eval cispeptide_list $args]
  } elseif { $command == "minimize" } {
    return [eval cispeptide_minimize $args]
  } elseif { $command == "move" } {
    return [eval cispeptide_move $args]
  } elseif { $command == "reset" } {
    cispeptide_reset_reps
    cispeptide_reset
  } elseif { $command == "restrain" } {
    return [eval cispeptide_restrain $args]
  } elseif { $command == "set" } {
    return [eval cispeptide_set $args]
  } elseif { $command == "show" } {
    return [eval cispeptide_show $args]
  } elseif { $command == "extrab" } {
    return [eval cispeptide_extrab $args]
  } else {
    cispeptide_usage
    error "Unrecognized command."
  }
  
  return
  
}

proc ::cispeptide::cispeptide_check_usage { } {

  variable defaultSelText
  puts "Usage: cispeptide check -mol <molid> ?options?"
  puts "Options:"
  puts "  -seltext <atom selection text> (default: $defaultSelText)"
  puts "  -labelcis <TRUE|FALSE>   label all cis bonds with a 1 in the user column"
  return

}

proc ::cispeptide::cispeptide_check { args } {

  variable defaultSelText
  variable labelcis FALSE

  set nargs [llength $args]
  if {$nargs == 0} {
    cispeptide_check_usage
    error ""
  }

  foreach {name val} $args {
    switch -- $name {
      -mol { set arg(mol) $val }
      -seltext { set arg(seltext) $val }
      # -split option is no longer used but kept here for 
      # backward compatibility
      -labelcis {set labelcis $val}
      -split { set arg(split) $val } 
      -gui { set arg(gui) $val }
      default { 
        cispeptide_check_usage
        error "Unrecognized argument." 
      }
    }
  }

  if { [info exists arg(mol)] } {
    set molid $arg(mol)
    if {$molid == {top}} {
      set molid [molinfo top]
    }
  } else {
    cispeptide_check_usage
    error "Missing required argument -mol."
  }

  if { [info exists arg(seltext)] } {
    set seltext $arg(seltext)
  } else {
    set seltext $defaultSelText
  }

  # Was I called from the GUI?
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  if {[cispeptide_check_files $molid 0 $gui] == 1} {
    return -1
  }

  set peptideBonds [::cispeptide::find_peptide_bonds $seltext $molid]
  set indO [lindex $peptideBonds 0]
  set indC [lindex $peptideBonds 1]
  set indN [lindex $peptideBonds 2]
  set indCA [lindex $peptideBonds 3]
  set count [::cispeptide::cispeptide_check_core $indO $indC $indN $indCA $molid $labelcis]

  # puts "STEREO> CISPEPTIDE COUNT: $count"
  return $count

}


# Wrapper to choose appropriate algorithm to find peptide bonds.
proc ::cispeptide::find_peptide_bonds {seltext molid} {

  # If a psf file was loaded, use fast algorithm.
  if {[lsearch -exact [join [molinfo $molid get filetype]] psf] != -1} {
    set peptBonds [::cispeptide::find_peptide_bonds_psf $seltext $molid]
    # If there was an error, fall back to robust algorithm.
    if {$peptBonds == -1} {
      puts "WARNING: Fast algorithm for detecting peptide bonds failed. Falling back to robust (slow) algorithm..."
      set peptBonds [::cispeptide::find_peptide_bonds_robust $seltext $molid]
    }
  } else {
    # Otherwise, use robust (slow) algorithm.
    set peptBonds [::cispeptide::find_peptide_bonds_robust $seltext $molid]
  }
    
  return $peptBonds
}


# Fast algorithm to identify peptide bonds that assumes a psf file was loaded.
proc ::cispeptide::find_peptide_bonds_psf {seltext molid} {

  puts "cispeptide) Using fast algorithm for detecting peptide bonds..."

  # XXX - Duplicated code; need to pull this out...
  # If there are alternative conformations, pick the first one in
  # the atom selections below
  set selTest [atomselect $molid "name CA N C O and protein and ($seltext)"]
  set altlocList [lsort -unique [$selTest get altloc]]
  $selTest delete
  set altlocSelector ""
  if {[llength $altlocList] > 1} {
    if {[lindex $altlocList 0] == ""} {
      set altlocSelector "and altloc \"\" [lindex $altlocList 1]"
    } else {
      set altlocSelector "and altloc [lindex $altlocList 0]"
    }
  }

  set indO {}
  set indC {}
  set indN {}
  set indCA {}

  # Split protein chains according to segments.
  set sel [atomselect $molid "protein and ($seltext)"]
  foreach seg [lsort -unique [$sel get segname]] {

    set selO [atomselect $molid "segname $seg and name O and protein and ($seltext)"]
    set selC [atomselect $molid "segname $seg and name C and protein and ($seltext)"]
    set selN [atomselect $molid "segname $seg and name N and protein and ($seltext)"]
    set selCA [atomselect $molid "segname $seg and name CA and protein and ($seltext)"]

    set thisIndO  [lrange [$selO  get index] 0 end]
    set thisIndC  [lrange [$selC  get index] 0 end-1]
    set thisIndN  [lrange [$selN  get index] 1 end]
    set thisIndCA [lrange [$selCA get index] 1 end]

    set numCA [$selCA num]

    $selCA delete
    $selN delete
    $selC delete
    $selO delete

    # If there is a segment with a single amino acid, i.e., no peptide
    # bonds, skip it.
    if {$numCA < 2} {
      continue
    }

    lappend indO $thisIndO
    lappend indC $thisIndC
    lappend indN $thisIndN
    lappend indCA $thisIndCA

  }
  $sel delete

  set indO [join $indO]
  set indC [join $indC]
  set indN [join $indN]
  set indCA [join $indCA]

  # Check for consistency in the size of index lists.
  set len [llength $indO]
  if {[llength $indC] != $len || [llength $indN] != $len || 
      [llength $indCA] != $len } {
    return -1
  }

  return [list $indO $indC $indN $indCA]

}


# Robust algorithm that explicitly identifies each peptide bond given the bond
# list derived by VMD.
proc ::cispeptide::find_peptide_bonds_robust {seltext molid} {

  # If there are alternative conformations, pick the first one in
  # the atom selections below
  set selTest [atomselect $molid "name CA N C O and protein and ($seltext)"]
  set altlocList [lsort -unique [$selTest get altloc]]
  $selTest delete
  set altlocSelector ""
  if {[llength $altlocList] > 1} {
    if {[lindex $altlocList 0] == ""} {
      set altlocSelector "and altloc \"\" [lindex $altlocList 1]"
    } else {
      set altlocSelector "and altloc [lindex $altlocList 0]"
    }
  }

  # Find all residue pairs (i,j) with residue i containing atoms O and C,
  # residue j containing atoms N, CA, and atom C(i) bound to atom N(j).
  # Generate a list of indices for atoms O, C, N, and CA that accurately
  # reflect all peptide bonds in the system.

  set indO {}
  set indC {}
  set indN {}
  set indCA {}

  set allO [atomselect $molid "name O OT1 and ($seltext) $altlocSelector"]
  set iAllO [$allO get index]
  $allO delete 
  foreach iO $iAllO {

    set selO [atomselect $molid "index $iO"]
    set bondsO [$selO getbonds]
    $selO delete
    if {[llength [lindex $bondsO 0]] == 0} {
      continue
    }

    set selC [atomselect $molid "(same residue as index $iO) and name C and index [join $bondsO] and ($seltext) $altlocSelector"]
    if {[$selC num] == 0} {
      $selC delete
      continue
    }
    if {[$selC num] > 1} {
      puts "WARNING: Atom O (index $iO) is bound to multiple C atoms. Ignoring..."
      $selC delete
      continue
    }
    set iC [$selC get index]
    set bondsC [$selC getbonds]
    $selC delete
    if {[llength [lindex $bondsC 0]] == 0} {
      continue
    }

    set selN [atomselect $molid "name N and index [join $bondsC] and ($seltext) $altlocSelector"]
    if {[$selN num] == 0} {
      $selN delete
      continue
    }
    if {[$selN num] > 1} {
      puts "WARNING: Atom C (index $iC) is bound to multiple N atoms. Ignoring..."
      $selN delete
      continue
    }
    set iN [$selN get index]
    set bondsN [$selN getbonds]
    $selN delete
    if {[llength [lindex $bondsN 0]] == 0} {
      continue
    }

    set selCA [atomselect $molid "(same residue as index $iN) and name CA and index [join $bondsN] and ($seltext) $altlocSelector"]
    if {[$selCA num] == 0} {
      $selCA delete
      continue
    }
    if {[$selCA num] > 1} {
      puts "WARNING: Atom N (index $iN) is bound to multiple CA atoms. Ignoring..."
      $selCA delete
      continue
    }
    set iCA [$selCA get index]
    $selCA delete

    lappend indO $iO
    lappend indC $iC
    lappend indN $iN
    lappend indCA $iCA

  }

  return [list $indO $indC $indN $indCA]

}

# Detects cis peptide bonds given lists of indices of O, C, N, and CA atoms
# correspoding to peptide bonds. It is the job of the caller to make sure the
# lists are correct, i.e., that element i from each list correspond to a real
# peptide bond.
proc ::cispeptide::cispeptide_check_core {indO indC indN indCA molid labelcis} {

  variable dihedCutoff
  variable cnBondCutoff
  variable cispeptideResiduesList
  variable cispeptideActionsList
  variable cispeptideMovedList
  
  set count [llength $cispeptideActionsList]
  foreach o $indO c $indC n $indN ca $indCA {

    set d [measure dihed [list $o $c $n $ca] molid $molid]
    if { [expr {abs($d) > $dihedCutoff}] } {

      set sel1 [atomselect $molid "index $o"]
      set sel2 [atomselect $molid "index $ca"]
      set residue1 [$sel1 get residue]
      set residue2 [$sel2 get residue]
      if {$labelcis} {
	set labelres [atomselect $molid "residue $residue2"]
	$labelres set user 1
	$labelres delete
      }
      $sel1 delete
      $sel2 delete
      set residues [list $residue1 $residue2]

      puts "cispeptide) Found cis peptide bond between residues $residue1 and $residue2:"
      puts "  [cispeptide_residue_info $residue1 $molid string]"
      puts "  [cispeptide_residue_info $residue2 $molid string]"

      # Don't duplicate entries of the cis peptide bond list
      if { [lsearch $cispeptideResiduesList $residues] != -1 } {
        puts "  WARNING: Duplicated cis peptide bond. Ignoring..."
      } else {
        puts "  Current cis peptide bond number is $count."
        incr count
        lappend cispeptideResiduesList $residues
        lappend cispeptideActionsList X
        lappend cispeptideMovedList -1
      }
      puts ""
    }
  }

  return $count

}

proc ::cispeptide::cispeptide_show_usage { } {

  puts "Usage: cispeptide show <current|next|cis peptide number|none> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return

}

proc ::cispeptide::cispeptide_show { args } {

  global tk_version
  variable cispeptideResiduesList
  variable cispeptideCurrent
  variable cispeptideReps

  if {![info exists tk_version]} {
    error "The 'cispeptide show' command requires VMD running in graphics mode."
  }

  set nargs [llength $args]
  if {$nargs < 1} {
    cispeptide_show_usage
    error ""
  }
  set cmd [lindex $args 0]

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      default {
        cispeptide_show_usage
        error "Unrecognized argument."
      }
    }
  }
  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }

  set numBonds [llength $cispeptideResiduesList]

  # Validate argument
  if { $cmd == {none} } {
  } elseif { $cmd == {next} } {
    incr cispeptideCurrent
  } elseif { $cmd != {current} } {

    # Make sure argument is an integer and in range
    if { [string is integer $cmd] != 1 } {
      cispeptide_show_usage
      error "Recognized non-integer argument."
    } elseif { $cmd < 0 || $cmd >= $numBonds } {
      cispeptide_show_usage
      error "Cis peptide bond requested is out of range."
    } 

    set cispeptideCurrent $cmd
  }

  # Clean up previous reps
  cispeptide_reset_reps

  if { $cmd == {none} } {
    return
  }

  # Wrap around if needed
  if { $cispeptideCurrent == $numBonds } {
    puts "cispeptide) Reached end of list. Wrapping around to first cis peptide bond."
    set cispeptideCurrent 0
  }

  set residue1 [lindex [lindex $cispeptideResiduesList $cispeptideCurrent] 0]
  set residue2 [lindex [lindex $cispeptideResiduesList $cispeptideCurrent] 1]
  puts "cispeptide) Showing cis peptide bond ${cispeptideCurrent}:"
  puts "  [cispeptide_residue_info $residue1 $molid string]"
  puts "  [cispeptide_residue_info $residue2 $molid string]"

  mol color Name
  mol representation CPK 1.0 0.3 8.0 6.0
  mol material Opaque
  mol selection "(residue $residue1 and name O OT1 C) or (residue $residue2 and name HN H N)"
  mol addrep $molid
  set repid [expr [molinfo $molid get numreps] - 1]
  # repname is guaranteed to be unique within a molecule
  set repname [mol repname $molid $repid]
  lappend cispeptideReps $molid
  lappend cispeptideReps $repname

  mol color Segname
  mol representation NewRibbons 0.300000 6.000000 3.000000 0
  mol material Opaque
  mol selection "same residue as (within 10 of (residue $residue1 $residue2))"
  mol addrep $molid
  set repid [expr [molinfo $molid get numreps] - 1]
  # repname is guaranteed to be unique within a molecule
  set repname [mol repname $molid $repid]
  lappend cispeptideReps $molid
  lappend cispeptideReps $repname

  mol color Name
  mol representation Licorice 0.1 10.0 10.0
  mol material Opaque
  mol selection "same residue as (within 10 of (residue $residue1 $residue2))"
  mol addrep $molid
  set repid [expr [molinfo $molid get numreps] - 1]
  # repname is guaranteed to be unique within a molecule
  set repname [mol repname $molid $repid]
  lappend cispeptideReps $molid
  lappend cispeptideReps $repname


  # Center view on the current residues
  set sel [atomselect $molid "(residue $residue1 and name C) or (residue $residue2 and name N)"]
  set center [measure center $sel]
  $sel delete
  foreach mol [molinfo list] {
    molinfo $mol set center [list $center]
  }
  scale to 0.5
  translate to 0 0 0
  display update

  return

}

proc ::cispeptide::cispeptide_list_usage { } {
  puts "Usage: cispeptide list <list of cis peptide bonds|all> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return
}

proc ::cispeptide::cispeptide_list { args } {

  variable cispeptideResiduesList
  variable cispeptideActionsList
  variable cispeptideMovedList 

  set nargs [llength $args]
  if {$nargs < 1} {
    cispeptide_list_usage
    error ""
  }

  if { [lindex $args 0] == {all} } {
    set selectedBonds {}
    for {set i 0} {$i < [llength $cispeptideActionsList]} {incr i} {
      lappend selectedBonds $i
    }
  } else {
    set selectedBonds [lindex $args 0]
  }

  # Check list of selected bonds for errors
  set numBonds [llength $cispeptideActionsList]
  foreach thisBond $selectedBonds {
    if { [string is integer $thisBond] != 1 } {
      cispeptide_list_usage
      error "Recognized non-integer argument."
    } elseif { $thisBond < 0 || $thisBond >= $numBonds } {
      cispeptide_list_usage
      error "Cis peptide bond requested is out of range."
    } 
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -gui { set arg(gui) $val }
      default {
        cispeptide_list_usage
        error "Unrecognized argument."
      }
    }
  }
  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }

  # Was I called from the GUI?
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  foreach i $selectedBonds {
    set pair [lindex $cispeptideResiduesList $i]
    set action [lindex $cispeptideActionsList $i]
    set moved [lindex $cispeptideMovedList $i]
    set residue1 [lindex $pair 0]
    set residue2 [lindex $pair 1]
    if $gui {
      lappend returnList [cispeptide_residue_info $residue1 $molid list]
      lappend returnList [cispeptide_residue_info $residue2 $molid list]
      lappend returnList $action
      lappend returnList $moved
    } else {
      puts "cispeptide) Residue pair $i (action = $action; moved = $moved):"
      puts "  [cispeptide_residue_info $residue1 $molid string]"
      puts "  [cispeptide_residue_info $residue2 $molid string]"
    }
    incr i
  }

  if {$gui && [info exists returnList]} {
    return $returnList
  }
  return

}

# Internal function: return information about a given residue
# outformat can be list or string
proc ::cispeptide::cispeptide_residue_info { residue mol outformat } {
  set sel [atomselect $mol "residue $residue"]
  set returnList {}
  set resid [lsort -unique [$sel get resid]]
  set resname [lsort -unique [$sel get resname]]
  set chain [lsort -unique [$sel get chain]]
  set segname [lsort -unique [$sel get segname]]
  $sel delete

  if { $outformat == {list} } {
    set returnList {}
    lappend returnList $residue
    lappend returnList $resid
    lappend returnList $resname
    lappend returnList $chain
    lappend returnList $segname
  } elseif { $outformat == {string} } {
    set returnList "residue $residue, resid $resid, resname $resname, chain $chain, segname $segname"
  } else {
    error "Internal error in cispeptide_residue_info."
  }

  return $returnList
}

proc ::cispeptide::cispeptide_set_usage { } {
  puts "Usage: cispeptide set <H|O|X> ?options?"
  puts "Defines which action to take when modifying (at a later step)"
  puts "the given cis peptide bond."
  puts "  H: move hydrogen atom (not allowed for X-PRO)"
  puts "  O: move oxygen atom"
  puts "  X: don't modify this cis peptide bond"
  puts "Options:"
  puts "  -cpn <cis peptide number> (default: current)"
  puts "  -mol <molid> (default: top)"
  return
}

# XXX - Need to allow for "set all"
proc ::cispeptide::cispeptide_set { args } {
  variable cispeptideCurrent
  variable cispeptideActionsList
  variable cispeptideResiduesList
  variable cispeptideMovedList

  set nargs [llength $args]
  if {$nargs < 1} {
    cispeptide_set_usage
    error ""
  }
  set action [lindex $args 0]

  if { $action != {H} && $action != {O} && $action != {X} } {
    cispeptide_set_usage
    error "Unrecognized argument."
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -cpn { set arg(cpn) $val }
      -gui { set arg(gui) $val }
      default {
        cispeptide_set_usage
        error "Unrecognized argument."
      }
    }
  }

  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }

  if { [info exists arg(cpn)] } {
    set thisBond $arg(cpn)
  } else {
    set thisBond $cispeptideCurrent
  }

  # Make sure argument is an integer and in range
  set numBonds [llength $cispeptideActionsList]
  if { [string is integer $thisBond] != 1 } {
    cispeptide_set_usage
    error "Recognized non-integer argument."
  } elseif { $thisBond < 0 || $thisBond >= $numBonds } {
    cispeptide_set_usage
    error "Cis peptide bond requested is out of range."
  }
  
  # Was I called from the GUI?
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  if {[cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  # Action H is not allowed for X-PRO
  if { $action == {H} } {
    set residue [lindex [lindex $cispeptideResiduesList $thisBond] 1]
    set sel [atomselect $molid "residue $residue and name CA"]
    set resname [$sel get resname]
    $sel delete
    if { $resname == {PRO} } {
      if $gui {
        tk_messageBox -type ok -message "There is no hydrogen to be moved when the second residue in a peptide bond is a proline."
        return
      } else {
        cispeptide_set_usage
        error "Action \"H\" is not allowed for X-PRO."
      }
    }
  }

  # Assign action to the given cis peptide bond if it was not moved
  if { [lindex $cispeptideMovedList $thisBond] == -1 } {
    lset cispeptideActionsList $thisBond $action
  } else {
    if $gui {
        tk_messageBox -type ok -message "You cannot tag an atom for moving that was already moved."
      return
    } else {
      error "Cannot change action for atom that was already moved."
    }
  }

  return
}

# XXX - add an option 'move all'
proc ::cispeptide::cispeptide_move_usage { } {
  puts "Usage: cispeptide move <cis peptide number|current> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return
}

proc ::cispeptide::cispeptide_move { args } {
  variable cispeptideCurrent
  variable cispeptideResiduesList
  variable cispeptideActionsList
  variable cispeptideMovedList
  variable clashWithinCutoff 
  variable clashCutoff 

  set nargs [llength $args]
  if {$nargs < 1} {
    cispeptide_move_usage
    error ""
  }

  if { [lindex $args 0] == {current} } {
    set thisBond $cispeptideCurrent
  } else {
    set thisBond [lindex $args 0]
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -gui { set arg(gui) $val }
      default {
        cispeptide_move_usage
        error "Unrecognized argument."
      }
    }
  }
  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }

  # Make sure argument is an integer and in range
  set numBonds [llength $cispeptideActionsList]
  if { [string is integer $thisBond] != 1 } {
    cispeptide_move_usage
    error "Recognized non-integer argument."
  } elseif { $thisBond < 0 || $thisBond >= $numBonds } {
    cispeptide_move_usage
    error "Cis peptide bond requested is out of range."
  } 

  # Was I called from the GUI?
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  if {[cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  if { [lindex $cispeptideMovedList $thisBond] != -1 } {
    if $gui {
      tk_messageBox -type ok -message "You cannot move an atom that was already moved."
      return
    } else {
      error "Cannot move an atom that was already moved."
    }
  }

  set moveAtom [lindex $cispeptideActionsList $thisBond]
  set residue1 [lindex [lindex $cispeptideResiduesList $thisBond] 0]
  set residue2 [lindex [lindex $cispeptideResiduesList $thisBond] 1]
  switch $moveAtom {
    X { 
      if $gui {
        tk_messageBox -type ok -message "You need to tag an atom for moving first."
        return
      } else {
        error "Action flag for cis peptide bond $thisBond is set to X. Did you run 'cispeptide set'? Aborting..."
      }
    }
    H { 
      set selAtom1 [atomselect $molid "residue $residue2 and name HN H"]
      set selAtom2 [atomselect $molid "residue $residue2 and name N"]
    } 
    O {
      set selAtom1 [atomselect $molid "residue $residue1 and name O OT1"]
      set selAtom2 [atomselect $molid "residue $residue1 and name C"]

    }
    default { error "Internal error in cispeptide_move: Unrecognized action." }
  }

  if { [$selAtom1 num] != 1 || [$selAtom2 num] != 1 } {
    $selAtom1 delete
    $selAtom2 delete
    error "Selected wrong number of atoms."
  }

  set coord1 [join [$selAtom1 get {x y z}]]
  set coord2 [join [$selAtom2 get {x y z}]]
  set bondVector [vecsub $coord2 $coord1]
  $selAtom1 moveby [vecscale 2 $bondVector]

  # Move atom a little bit less if there are clashes
  set id [$selAtom1 get index]
  set sel [atomselect $molid "within $clashWithinCutoff of index $id"]
  if { [llength [lindex [measure contacts $clashCutoff $selAtom1 $sel] 0]] } {
    puts "cispeptide) Warning: moving atom (index $id) introduced clash(es). Moving back by 0.25 times the bond length."
    $selAtom1 moveby [vecscale -0.25 $bondVector]
  }
  $sel delete

  # Store index of the atoms that was moved
  lset cispeptideMovedList $thisBond $id

  $selAtom1 delete
  $selAtom2 delete

  return
}

proc ::cispeptide::cispeptide_minimize_usage { } {
  puts "Usage: cispeptide minimize <list of cis peptide bonds|all> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return
}

proc ::cispeptide::cispeptide_minimize { args } {
  global tk_version
  variable cispeptideResiduesList
  variable cispeptideMovedList
  variable cispeptideActionsList
  variable minWithinCutoff
  variable cispeptideSelRestrainAtomsH
  variable cispeptideSelRestrainAtomsN
  variable cispeptideSelRestrainAtomsC
  variable cispeptideSelRestrainAtomsO
  variable cispeptideSelRestrainAtomsSet
  variable molidBeforeAutoIMD

  if {![info exists tk_version]} {
    error "The 'cispeptide minimize' command requires VMD running in graphics mode."
  }

  set nargs [llength $args]
  if {$nargs < 1} {
    cispeptide_minimize_usage
    error ""
  }

  if { [lindex $args 0] == {all} } {
    set selectedBonds {}
    for {set i 0} {$i < [llength $cispeptideActionsList]} {incr i} {
      lappend selectedBonds $i
    }
  } else {
    set selectedBonds [lindex $args 0]
  }

  # Check list of selected bonds for errors
  set numBonds [llength $cispeptideActionsList]
  foreach thisBond $selectedBonds {
    if { [string is integer $thisBond] != 1 } {
      cispeptide_minimize_usage
      error "Recognized non-integer argument."
    } elseif { $thisBond < 0 || $thisBond >= $numBonds } {
      cispeptide_minimize_usage
      error "Cis peptide bond requested is out of range."
    } 
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -gui { set arg(gui) $val }
      -mdff { set arg(mdff) $val}
      default {
        cispeptide_minimize_usage
        error "Unrecognized argument."
      }
    }
  }

  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }
  set molidBeforeAutoIMD $molid

  # Was I called from the GUI?
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  if {[cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  # Minimize everything that has an action set, was moved and selected
  set includeResidues {}
  set includeResidues1 {}
  set includeResidues2 {}
  foreach i $selectedBonds {
    set residues [lindex $cispeptideResiduesList $i]
    set movedAtom [lindex $cispeptideMovedList $i]
    set action [lindex $cispeptideActionsList $i]
    if { $action != {X} } {
      if { $movedAtom == -1 } {
        error "Flagged atom for cis peptide bond $i has not been moved yet. Did you run 'cispeptide move'? Aborting..."
      }
      lappend includeResidues $residues
      lappend includeResidues1 [lindex $residues 0]
      lappend includeResidues2 [lindex $residues 1]
    }
  }

  # Create a selection containing all H-N-C-O atoms involved in cis 
  # peptide bonds being considered. This selection will later be used
  # to create an extrabonds file containing dihedral restraints once 
  # autoimd is invoked. In the case of X-PRO, use CD-N-C-O instead.
  if $cispeptideSelRestrainAtomsSet {
    uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsH delete
    uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsN delete
    uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsC delete
    uplevel #0 $::cispeptide::cispeptideSelRestrainAtomsO delete
  }
  set cispeptideSelRestrainAtomsH [atomselect $molid "(name H HN and residue [join $includeResidues2]) or (name CD and resname PRO and residue [join $includeResidues2])"]
  set cispeptideSelRestrainAtomsN [atomselect $molid "name N and residue [join $includeResidues2]"]
  set cispeptideSelRestrainAtomsC [atomselect $molid "name C and residue [join $includeResidues1]"]
  set cispeptideSelRestrainAtomsO [atomselect $molid "name O OT1 and residue [join $includeResidues1]"]

  $cispeptideSelRestrainAtomsH global
  $cispeptideSelRestrainAtomsN global
  $cispeptideSelRestrainAtomsC global
  $cispeptideSelRestrainAtomsO global

  set cispeptideSelRestrainAtomsSet 1
  if { [$cispeptideSelRestrainAtomsH num] != [llength $includeResidues1] ||
       [$cispeptideSelRestrainAtomsN num] != [llength $includeResidues1] ||
       [$cispeptideSelRestrainAtomsC num] != [llength $includeResidues1] ||
       [$cispeptideSelRestrainAtomsO num] != [llength $includeResidues1] } {
    error "Internal error in ::cispeptide::cispeptide_minimize: Incorrect number of atoms in atom selection for dihedral restraints."
  }

  set residuesSeltext {none}
  foreach residue [join $includeResidues] {
    set sel [atomselect $molid "residue $residue"]
    set segname [lsort -unique [$sel get segname]]
    set resid [lsort -unique [$sel get resid]]
    $sel delete
    lappend residuesSeltext " or (segname $segname and resid $resid)"
  }

  set residuesSeltext [join $residuesSeltext]
  set mymoltenseltext "same residue as within $minWithinCutoff of ($residuesSeltext)"


  if { [info exists arg(mdff)] && $arg(mdff) == 1 } {
  
    ::MDFFGUI::gui::mdffgui
    set ::MDFFGUI::settings::CispeptideRestraints 1
    #unused currently
    #set ::MDFFGUI::settings::IMDSelText $mymoltenseltext
    set ::MDFFGUI::settings::GridPDBSelText $mymoltenseltext
    set ::MDFFGUI::settings::FixedPDBSelText "not ($mymoltenseltext)"
    set ::MDFFGUI::settings::MolID $molid
    set ::MDFFGUI::settings::IMD 1
    set ::MDFFGUI::settings::IMDWait 1

  } else {

    autoimd set moltenseltext $mymoltenseltext
    autoimd set sim_mode minimize
    autoimd set usecispeptide 1

    # Open autoimd window
    autoimd showwindow
  }
  return
}

# Internal function to be called only by autoimd (not exposed to the user).
# This proc creates an extrabonds file containing dihedral restraints to 
# fix cis peptide bonds.
# Usage: cispeptide extrab <extrabonds filename> <autoimd pdb filename>
#
# Note that, for PRO, atom CD is used instead of H below! That is achieved
# by abusing the definition of cispeptideSelRestrainAtomsH to include CD
# instead of H for PRO.
proc ::cispeptide::cispeptide_extrab { args } {

  variable cispeptideSelRestrainAtomsH
  variable cispeptideSelRestrainAtomsN
  variable cispeptideSelRestrainAtomsC
  variable cispeptideSelRestrainAtomsO
  variable cispeptideSelRestrainAtomsSet
  variable cispeptideExtraBondsK
  variable idHListAutoIMD 
  variable idNListAutoIMD 
  variable idCListAutoIMD 
  variable idOListAutoIMD 

  if { $cispeptideSelRestrainAtomsSet == 0 } {
    error "Internal error in ::cispeptide::cispeptide_extrab: cispeptideSelRestrainAtomsSet is 0."
  }

  set nargs [llength $args]
  if {$nargs != 2} {
    error "Internal error in ::cispeptide::cispeptide_extrab: Wrong number of arguments."
  }
  set extrabFileName [lindex $args 0]
  set autoimdPDB [lindex $args 1]

  set tmpmol [mol new $autoimdPDB type pdb waitfor all]
  set selAutoIMD [atomselect $tmpmol "protein and ((name H HN N C O) or (resname PRO and name CD))"]
  set cH [measure contacts 0.1 $selAutoIMD $cispeptideSelRestrainAtomsH]
  set cN [measure contacts 0.1 $selAutoIMD $cispeptideSelRestrainAtomsN]
  set cC [measure contacts 0.1 $selAutoIMD $cispeptideSelRestrainAtomsC]
  set cO [measure contacts 0.1 $selAutoIMD $cispeptideSelRestrainAtomsO]
  $selAutoIMD delete

  set selH [atomselect $tmpmol "index [lindex $cH 0]"]
  set selN [atomselect $tmpmol "index [lindex $cN 0]"]
  set selC [atomselect $tmpmol "index [lindex $cC 0]"]
  set selO [atomselect $tmpmol "index [lindex $cO 0]"]

  if { [$selH num] != [$cispeptideSelRestrainAtomsH num] ||
       [$selN num] != [$cispeptideSelRestrainAtomsN num] ||
       [$selC num] != [$cispeptideSelRestrainAtomsC num] ||
       [$selO num] != [$cispeptideSelRestrainAtomsO num] } {
    $selH delete
    $selN delete
    $selC delete
    $selO delete
    mol delete $tmpmol
    error "Internal error in ::cispeptide::cispeptide_extrab: atom selections of inconsistent lenghts."
  }

  set idHListAutoIMD [$selH get index]
  set idNListAutoIMD [$selN get index]
  set idCListAutoIMD [$selC get index]
  set idOListAutoIMD [$selO get index]

  set out [open $extrabFileName w]
  foreach idH [$selH get index] idN [$selN get index] idC [$selC get index] idO [$selO get index] {
    puts $out "dihedral $idH $idN $idC $idO $cispeptideExtraBondsK 180.0"
  }
  close $out

  $selH delete
  $selN delete
  $selC delete
  $selO delete
  mol delete $tmpmol

  return
}

# Internal function: set up appropriate reps. To be called by autoimd.
proc ::cispeptide::cispeptide_autoimd_reps { } {
  variable molidBeforeAutoIMD
  variable idHListAutoIMD 
  variable idNListAutoIMD 
  variable idCListAutoIMD 
  variable idOListAutoIMD 

  mol off $molidBeforeAutoIMD

  set molAutoIMD [molinfo top]
  mol showrep $molAutoIMD 0 0
  mol showrep $molAutoIMD 1 0
  mol showrep $molAutoIMD 2 0

  mol color Name
  mol representation CPK 1.0 0.3 8.0 6.0
  mol material Opaque
  mol selection "index $idHListAutoIMD $idNListAutoIMD $idCListAutoIMD $idOListAutoIMD"
  mol addrep $molAutoIMD

  mol color Segname
  mol representation NewRibbons 0.300000 6.000000 3.000000 0
  mol material Opaque
  mol selection "same residue as (within 10 of (same residue as index $idHListAutoIMD))"
  mol addrep $molAutoIMD

  mol color Name
  mol representation Licorice 0.1 10.0 10.0
  mol material Opaque
  mol selection "same residue as (within 10 of (same residue as index $idHListAutoIMD))"
  mol addrep $molAutoIMD

  display update
  # Center view on the current residues
  set sel [atomselect $molAutoIMD "index $idNListAutoIMD $idCListAutoIMD"]
  set center [measure center $sel]
  $sel delete
  molinfo $molAutoIMD set center [list $center]
  scale to 0.5
  translate to 0 0 0
  display update

  return
}

# Internal function: Turn original molecule back on. To be called by autoimd.
proc ::cispeptide::cispeptide_autoimd_finish { } {
  variable molidBeforeAutoIMD
  mol on $molidBeforeAutoIMD
  autoimd set usecispeptide 0
  return
}

proc ::cispeptide::cispeptide_restrain_usage { } {

  variable defaultSelText
  puts "Usage: cispeptide restrain -o <extrabonds file> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  puts "  -seltext <atom selection text> (default: $defaultSelText)"
  # puts "  -split <chain|segname> (default: segname)"
  return

}

# Generate an extrabonds file for NAMD defining dihedral restraints to
# prevent flipping of peptide bonds from trans to cis or cis to trans, i.e.,
# the equilibrium value is taken to be the current value.
proc ::cispeptide::cispeptide_restrain { args } {

  variable defaultSelText
  variable cispeptideExtraBondsK

  set nargs [llength $args]
  if {$nargs == 0} {
    cispeptide_restrain_usage
    error ""
  }

  foreach {name val} $args {
    switch -- $name {
      -mol { set arg(mol) $val }
      -o { set arg(o) $val }
      -seltext { set arg(seltext) $val }
      # -split option is no longer used but kept here for 
      # backward compatibility
      -split { set arg(split) $val } 
      -gui { set arg(gui) $val }
      default { 
        cispeptide_restrain_usage
        error "Unrecognized argument." 
      }
    }
  }

  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }

  if { [info exists arg(seltext)] } {
    set seltext $arg(seltext)
  } else {
    set seltext $defaultSelText
  }

  # Was I called from the GUI?
  # Currently not applicable
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  if { [info exists arg(o)] } {
    set outFile $arg(o)
  } else {
    if $gui {
      tk_messageBox -type ok -message "Missing output extrabonds file name."
      return
    } else {
      cispeptide_restrain_usage
      error "Missing output file."
    }
  }

  if {[cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  set peptideBonds [::cispeptide::find_peptide_bonds $seltext $molid]
  set indO [lindex $peptideBonds 0]
  set indC [lindex $peptideBonds 1]
  set indN [lindex $peptideBonds 2]
  set indCA [lindex $peptideBonds 3]

  set out [open $outFile w]
  foreach o $indO c $indC n $indN ca $indCA {
    set d [measure dihed [list $o $c $n $ca] molid $molid]
    puts $out "dihedral $o $c $n $ca $cispeptideExtraBondsK $d"
  }
  close $out
  
  return
}

# XXX - pull this proc out of the cispeptide plugin; it is useful for
#       other plugins such as the chirality one.
proc ::cispeptide::cispeptide_check_files { molid require_psf {gui 0} } {

  if $require_psf {
    # Check that a PSF was loaded (code adapted from autoimd)
    set psffile ""
    foreach filename [lindex [molinfo $molid get filename] 0] filetype [lindex [molinfo $molid get filetype] 0] {
    # make sure to get the *last* psf file in the list
      if {![string compare "$filetype" "psf"]} {
        set psffile "$filename"
      }
    }
    # make sure that we have a PSF - we need this for psfgen
    if { "$psffile" == "" } {
      if $gui {
        tk_messageBox -type ok -message "You must have a PSF file loaded."
        return 1
      } else {
        error "You must have a PSF file loaded."
      }
    }
  }

  # Check that the given molecule constains at least one frame
  if {![molinfo $molid get numframes]} {
    if $gui {
      tk_messageBox -type ok -message "Molecule contains no frames. Did you load a PDB file?"
      return 1
    } {
      error "Molecule contains no frames. Did you load a PDB file?"
    }
  }

  return

}

