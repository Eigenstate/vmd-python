#
# chirality - identifies and fixes chirality errors in proteins/nucleic acids
#
# Leonardo Trabuco <ltrabuco@ks.uiuc.edu>
# Eduard Schreiner <eschrein@ks.uiuc.edu>
#

# TODO
# - Maybe warn the user if we don't recognize a certain resname
# - Force GUI list update when running from the command line
# - Add support for DNA residue names in Amber
# - Make molecule visible when showing selected chiral center
# - add a 'move undo'
# - implement "I'm feeling lucky" feature
# - restrain chiral centers that we're not messing with during AutoIMD

package provide chirality 1.2
if [info exists tk_version] {
  package require autoimd
}

# Temporarily require cispeptide package until we pull out
# cispeptide_check_files out of the package to a more centralized location
package require cispeptide

namespace eval ::chirality:: {

  variable defaultSelText {all}

  # Chiral centers with heavy atoms: used to identify chirality problems.
  # The atom order was chosen so that wrong chiralities can be identified
  # by the sign (negative) of the corresponding improper.
  # Valid for standard PDBs, as well as Amber and CHARMM force fields.

  # Each chiral center also contains a list of possible associated
  # hydrogen names. If a structure has hydrogens, we find which atom
  # naming convention it uses and stick with that.
  #
  # N.B. The order H2' H2'' is important for the chiral center
  # associated with O2' (RNA). In a structure containing both RNA and
  # DNA (e.g., 1hhx), having the opposite order will match H2'' on 
  # DNA and will miss all H2' on RNA.

  # IMPORTANT! If you edit the chiralCenters list below, make sure you:
  #
  # 1. Run the regression tests (see ::chirality::run_tests)
  # 
  # 2. Update the docs by running from the plugin source directory:
  #    ::chirality::generate_docs doc/supported_chiral_centers.html

  variable chiralCenters {
    {{A G ADE ADN GUA RA RG DA DG RA3 RG3 RA5 RG5 ATP ADP AMP GTP GDP GMP 1MG 2MA 2MG 6MA 7MG DMA MA6 MRG OMG SPA MIA} {{H1' C1' C2' N9 O4'}}}
    {{C U CYT URA RC DC RU T DT RC3 RU3 RC5 RU5 3AU 3MU 4SU 5MC 5MU DHU M4C MRC OMC MRU OMU RCP} {{H1' C1' C2' N1 O4'}}}
    {{PSU 3MP} {{H1' C1' C2' C5 O4'}}}
    {{ADE ADN A RA DA RA3 RA5 CYT C RC DC RC3 RC5 RCP GUA G RG DG RG3 RG5 URA U RU T DT RU3 RU5 GTP GDP GMP ATP ADP AMP}  {{H3' C3' C4' C2' O3'} {H4' C4' C5' C3' O4'} {{H2'1 H2' H2''} C2' C3' C1' O2'}}}
    {{CH 1MG 2MA 2MG 3AU 3MP 3MU 4SU 5MC 5MU 6MA 7MG DHU H2U DMA MA6 M4C MRC OMC MRG OMG MRU OMU PSU SPA MIA}  {{H2' C2' C3' C1' O2'}}}
    {{3AU} {{H24 C12 N40 C13 C11}}}
    {{ALA ARG ASN ASP CYM CYS GLN GLU HSP HSE HSD HIP HIE HID HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL} {{HA CA N C CB}}}
    {{THR} {{HB CB CA OG1 CG2}}}
    {{ILE} {{HB CB CA CG1 CG2}}}
  {{BGLN NAG} {{H1 C1 {O1 O2 O3 O4 O6 ND2} O5 C2} {H2 C2 C1 N C3} {H3 C3 O3 C2 C4} {H4 C4 C5 C3 O4} {H5 C5 C4 O5 C6}}}
  {{BMAN BMA} {{H1 C1 {O1 O2 O3 O4 O6} O5 C2} {H2 C2 O2 C1 C3} {H3 C3 O3 C2 C4} {H4 C4 C5 C3 O4} {H5 C5 C4 O5 C6}}}
  {{AMAN MAN} {{H1 C1 {O1 O2 O3 O4 O6} C2 O5} {H2 C2 O2 C1 C3} {H3 C3 O3 C2 C4} {H4 C4 C5 C3 O4} {H5 C5 C4 O5 C6}}}
  {{AFUC FUC} {{H1 C1 {O1 O2 O3 O4 O6} O5 C2} {H2 C2 O2 C1 C3} {H3 C3 C2 O3 C4} {H4 C4 C5 C3 O4} {H5 C5 O5 C4 C6}}}
  }
  



  # Cut-offs to check for clashes when moving atoms
  variable clashWithinCutoff 4
  variable clashCutoff 0.5

  # Cut-off for selecting water and ions for AutoIMD
  variable minWithinCutoff 10

  variable chiralSelRestrainAtomsSet 0
  variable chiralExtraBondsK 50
  variable targetImprpH 67
  variable targetImprpNoH 35

  variable chiralReps {}

  proc chirality_reset_reps { } {
    variable chiralReps

    foreach {molid repname} $chiralReps {
      if { [lsearch [molinfo list] $molid] != -1 } {
        set repid [mol repindex $molid $repname]
        mol delrep $repid $molid
      }
    }
    set chiralReps {}

    return
  }

  proc chirality_reset { } {

    # List of residue numbers with chirality errors
    variable chiralResiduesList {}

    # List of impropers to be restrained to fix/prevent chirality errors
    variable chiralImpropersList {}

    # List of heavy atoms identifying each chiral center
    variable chiralNamesList {}

    # List of per chirality error action, flagging which atom should be
    # moved, if any
    variable chiralActionsList {}

    # List of per chiral center index of moved atom
    variable chiralMovedList {}

    variable chiralCurrent 0

    variable id0ListAutoIMD {}  
    variable id1ListAutoIMD {}
    variable id2ListAutoIMD {}
    variable id3ListAutoIMD {}
    variable id4ListAutoIMD {}

    variable chiralSelRestrainAtomsSet
    if $chiralSelRestrainAtomsSet {
      set chiralSelRestrainAtomsSet 0
      uplevel #0 $::chirality::chiralSelRestrainAtoms0 delete
      uplevel #0 $::chirality::chiralSelRestrainAtoms1 delete 
      uplevel #0 $::chirality::chiralSelRestrainAtoms2 delete
      uplevel #0 $::chirality::chiralSelRestrainAtoms3 delete
      uplevel #0 $::chirality::chiralSelRestrainAtoms4 delete
    }
    return
  }
  chirality_reset


}

# Generate HTML list with supported chiral centers for documentation.
# Usage: 
# ::chirality::generate_docs doc/supported_chiral_centers.html
proc ::chirality::generate_docs { htmlFile } {
  variable chiralCenters

  set out [open $htmlFile w]
  puts $out "<!-- DO NOT EDIT THIS FILE!\nRegenerate it by running ::chirality::generate_docs within VMD. -->"
  puts $out "<table border=1 width=50%><tr><th>Residue names</th><th>Atom names</th></tr>"
  foreach center $chiralCenters {
    set resnames [lindex $center 0]
    set atoms [lindex $center 1]
    puts $out "<tr><td>$resnames</td><td>$atoms</td></tr>"
  }

  puts $out "</table>"
  flush $out
  close $out

  return
}

# Regression tests for identifying chirality errors from PDB structures.
#
# Make sure you run ::chirality::run_tests after each change. If a test
# fails, check manually for correctness. If support for a new chiral 
# center was implemented, add a representative entry to regressionTests below.
#
proc ::chirality::run_tests {} {

  # Each list element contains a PDB id and expected number of chiral errors.
  # Only the first frame of each molecule is considered.
  set regressionTests {
    {3bc5 0} 
    {1pxq 0} 
    {2oii 0}
    {3bc5 0}
    {3hhs 0}
    {1hhx 0}
    {200d 0}
    {183l 0}
    {2a2n 0}
    {3a4y 0}
    {3a6d 0}
    {1aew 0}
    {2aou 0}
    {1bcf 0}
    {3co2 0}
    {1e4c 0}
    {173l 1}
    {1a2s 1}
    {2a2e 1}
    {1a34 0}
    {2a3d 2}
    {3a35 2}
    {2a5b 1}
    {1a7s 1}
    {2a9x 1}
    {2aa3 3}
    {1abi 1}
    {1acp 5}
    {2adw 8}
    {3ces 1}
    {2cfz 0}
    {3cf5 2}
    {1chm 11}
    {1cir 0}
  }

  set numTests [llength $regressionTests]
  set failCount 0
  set passCount 0
  chirality reset
  for {set i 0} {$i < $numTests} {incr i} {
    set pdbId [lindex $regressionTests $i 0]
    set expectedCount [lindex $regressionTests $i 1]
    mol new $pdbId waitfor all
    animate goto 0
    set count [chirality check -mol top]
    if {$count != $expectedCount} {
      puts "chirality) TEST [expr {$i + 1}]/$numTests: FAILED"
      puts "chirality) pdb: $pdbId, expected: $expectedCount, actual: $count"
      incr failCount
    } else {
      puts "chirality) TEST [expr {$i + 1}]/$numTests: PASSED"
      incr passCount
    } 
    chirality reset
    mol delete top
  }

  puts "\nchirality) SUMMARY OF REGRESSION TESTS"
  puts "chirality) ---------------------------"
  puts "chirality) $passCount test(s) passed."
  puts "chirality) $failCount test(s) failed."

  return
}


proc chirality { args } { return [eval ::chirality::chirality $args] }

proc ::chirality::chirality_usage { } {
  puts "Usage: chirality <command> \[args...\]"
  puts "Commands:"
  puts "  check    -- identify chirality errors"
  puts "  list     -- list identified chirality errors"
  puts "  minimize -- fix chirality errors using energy minimization"
  puts "  move     -- move hydrogen atom to fix chirality"
  puts "  reset    -- reinitialize plugin state"
  puts "  restrain -- generate NAMD extrabonds file to prevent chirality changes"
  puts "  show     -- visualize identified chirality errors"
  return
}

proc ::chirality::chirality { args } {

  set nargs [llength $args]
  if { $nargs == 0 } {
    chirality_usage
    error ""
  }

  # parse command
  set command [lindex $args 0]
  set args [lreplace $args 0 0]

  if { $command == "check" } {
    return [eval chirality_check $args]
  } elseif { $command == "list" } {
    return [eval chirality_list $args]
  } elseif { $command == "minimize" } {
    return [eval chirality_minimize $args]
  } elseif { $command == "move" } {
    return [eval chirality_move $args]
  } elseif { $command == "reset" } {
    chirality_reset_reps
    chirality_reset
  } elseif { $command == "restrain" } {
    return [eval chirality_restrain $args]
  } elseif { $command == "set" } {
    return [eval chirality_set $args]
  } elseif { $command == "show" } {
    return [eval chirality_show $args]
  } elseif { $command == "extrab" } {
    return [eval chirality_extrab $args]
  } else {
    chirality_usage
    error "Unrecognized command."
  }
  
  return
  
}

proc ::chirality::chirality_check_usage { } {

  variable defaultSelText
  puts "Usage: chirality check -mol <molid>"
  puts "Options:"
  puts "  -seltext <atom selection text> (default: $defaultSelText)"
  return

}

proc ::chirality::chirality_check { args } {

  variable defaultSelText
  variable chiralCenters
  variable chiralImpropersList
  variable chiralResiduesList
  variable chiralActionsList
  variable chiralMovedList
  variable chiralNamesList

  set nargs [llength $args]
  if {$nargs == 0} {
    chirality_check_usage
    error ""
  }

  foreach {name val} $args {
    switch -- $name {
      -mol { set arg(mol) $val }
      -seltext { set arg(seltext) $val }
      -gui { set arg(gui) $val }
      default { 
        chirality_check_usage
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
    chirality_check_usage
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

  # FIXME: This proc should be taken out of the cispeptide plugin
  if {[::cispeptide::cispeptide_check_files $molid 0 $gui] == 1} {
    return -1
  }

  # Get list of atom ids for all supported chiral centers
  set chiralAtoms [::chirality::find_chiral_centers $seltext $molid]
  set count [::chirality::chirality_check_core $chiralAtoms $molid]

  # puts "STEREO> CHIRALITY COUNT: $count"
  return $count

}

# Find all atoms corresponding to complete, supported chiral centers
proc ::chirality::find_chiral_centers {seltext molid} {
  variable chiralCenters 

  # List returns by this proc. Each element corresponds to an 
  # entry in chiralCenters, which contains five lists of atoms,
  # plus a flag if the chiral centers contain hydrogens or not,
  # plus the selected atom names for each chiral center.
  set returnList {}

  # Check each chiral center for correctness
  for {set i 0} {$i < [llength $chiralCenters]} {incr i} {
    
    set resnamesList  [lindex [lindex $chiralCenters $i] 0]
    set atomnamesList [lindex [lindex $chiralCenters $i] 1]
    set sel [atomselect $molid "($seltext) and resname $resnamesList"]

    if {[$sel num] > 0} {

      # If there are alternative conformations, pick the first one in
      # the atom selections below
      set altlocSelector ""
      set altlocList [lsort -unique [$sel get altloc]]
      if {[llength $altlocList] > 1} {
        if {[lindex $altlocList 0] == ""} {
          set altlocSelector "and altloc \"\" [lindex $altlocList 1]"
        } else {
          set altlocSelector "and altloc [lindex $altlocList 0]"
        }
      }

      foreach atomnames $atomnamesList {
        set selatom0 [atomselect $molid "($seltext) and resname $resnamesList and name [lindex $atomnames 1] $altlocSelector"]
        set selatomH [atomselect $molid "($seltext) and resname $resnamesList and name [lindex $atomnames 0] $altlocSelector"]
		
		if {[$selatomH num] > 0} {
			set hasH 1
		} else {
			set hasH 0
		}
		
		set atomidH {}
		set atomid0 [$selatom0 get index]
		set atomid1 {}
		set atomid2 {}
		set atomid3 {}
		
		for {set sel0index 0} {$sel0index < [llength $atomid0]} {incr sel0index} {
			set currentCenterIndex [lindex $atomid0 $sel0index]
			set currentCenterSel [atomselect top "index $currentCenterIndex"]
			set bondedList [lindex [$currentCenterSel getbonds] 0]
			set bondedsel [atomselect $molid "index $bondedList"]
			set bondedIndexList [$bondedsel get index]
			set bondedNameList [$bondedsel get name]
			$bondedsel delete
			$currentCenterSel delete
    
        # Work through the $atomnames set to match to $bondedNameList and thereby assign an index.  This is a general
        # approach which allows multiple names for all but the chiral carbon, and doesn't require all atoms to be on the
        # same residue
			if {$hasH} {
				foreach name [lindex $atomnames 0] {
					set bondedNameListPos [lsearch -exact $bondedNameList $name]
					if {$bondedNameListPos != -1} {
						lappend atomidH [lindex $bondedIndexList $bondedNameListPos]
					}
				}
			}
			
			foreach name [lindex $atomnames 2] {
				set bondedNameListPos [lsearch -exact $bondedNameList $name]
				if {$bondedNameListPos != -1} {
					lappend atomid1 [lindex $bondedIndexList $bondedNameListPos]
				}
			}
			

			foreach name [lindex $atomnames 3] {
				set bondedNameListPos [lsearch -exact $bondedNameList $name]
				if {$bondedNameListPos != -1} {
					lappend atomid2 [lindex $bondedIndexList $bondedNameListPos]
				}
			}
			
			foreach name [lindex $atomnames 4] {
				set bondedNameListPos [lsearch -exact $bondedNameList $name]
				if {$bondedNameListPos != -1} {
					lappend atomid3 [lindex $bondedIndexList $bondedNameListPos]
				}
			}
		}
		
			


        # If any of the lists is empty, i.e., all chiral centers contain
        # missing atoms, move on to the next chiral center.
        if {([llength $atomid0] == 0 || [llength $atomid1] == 0 ||
             [llength $atomid2] == 0 || [llength $atomid3] == 0) ||
             ($hasH && [llength $atomidH] == 0)} {
          puts "WARNING: Ignoring residue(s) with missing atoms..."
          continue
        }
		
	set selatom1 [atomselect $molid "index $atomid1"]
        set selatom2 [atomselect $molid "index $atomid2"]
        set selatom3 [atomselect $molid "index $atomid3"]


        # Check if index lists have the same length. Otherwise, prune
        # residues with missing atoms from the lists.
        set listLengthOK 1
        set listLength [llength $atomid0]
        if {[llength $atomid1] != $listLength ||
            [llength $atomid2] != $listLength ||
            [llength $atomid3] != $listLength} {
	    set listLengthOK 0
        }
        if {$hasH && [llength $atomidH] != $listLength} {
          set listLengthOK 0
        }
        if {!$listLengthOK} {
          puts "WARNING: Ignoring residue(s) with missing atoms..."
        }

        # Force per-residue check for missing atoms for robustness
        # XXX - need to eliminate the checks above for index 
        #       lengths and implement a similar check after residue
        #       pruning to report an accurate warning. Clean up!
        if {0} {
          set residuesList {}
          lappend residuesList [$selatom0 get residue]
          lappend residuesList [$selatom1 get residue]
          lappend residuesList [$selatom2 get residue]
          lappend residuesList [$selatom3 get residue]

          if {$hasH} {
            lappend residuesList [$selatomH get residue]
          }
          set residuesIntersection [::chirality::list_intersect $residuesList]
          $selatom0 delete
          $selatom1 delete
          $selatom2 delete
          $selatom3 delete

          # If after pruning indices due to missing atoms we end up with an empty
          # index list, silently move on to the next kind of chiral center.
          if {([llength $atomid0] == 0 || [llength $atomid1] == 0 ||
               [llength $atomid3] == 0) || ($hasH && [llength $atomidH] == 0)} {

            continue
          }

          # An empty intersection of residues is VERY likely an error
          if {[llength $residuesIntersection] == 0} {
            error "Empty residuesList, which indicates an error with the dictionary describing chiral centers. Please report this error to the plugin authors."
          }

          set selatom0 [atomselect $molid "(index $atomid0) and (residue $residuesIntersection)"]
          set selatom1 [atomselect $molid "(index $atomid1) and (residue $residuesIntersection)"]
          set selatom2 [atomselect $molid "(index $atomid2) and (residue $residuesIntersection)"]
          set selatom3 [atomselect $molid "(index $atomid3) and (residue $residuesIntersection)"]



          set atomid0 [$selatom0 get index]
          set atomid1 [$selatom1 get index]
          set atomid2 [$selatom2 get index]
          set atomid3 [$selatom3 get index]
          if {$hasH} {
            $selatomH delete
            set selatomH [atomselect $molid "(index $atomidH) and (residue $residuesIntersection)"]
            set atomidH [$selatomH get index]
          }
        }

        $selatom0 delete
        $selatom1 delete
        $selatom2 delete
        $selatom3 delete
        $selatomH delete

        lappend returnList [list $atomidH $atomid0 $atomid1 $atomid2 $atomid3 $hasH $atomnames]

      }
    }

    $sel delete
  }

  return $returnList
}

# Detects chiral errors given a list generated by 
# ::chirality::find_chiral_centers
proc ::chirality::chirality_check_core {atomList molid} {
  variable chiralResiduesList
  variable chiralImpropersList
  variable chiralActionsList
  variable chiralMovedList
  variable chiralNamesList

  set count [llength $chiralResiduesList]
  foreach chiral $atomList {
    set atomidH [lindex $chiral 0]
    set atomid0 [lindex $chiral 1]
    set atomid1 [lindex $chiral 2]
    set atomid2 [lindex $chiral 3]
    set atomid3 [lindex $chiral 4]
    set hasH [lindex $chiral 5]
    set atomnames [lindex $chiral 6]

    foreach idH $atomidH id0 $atomid0 id1 $atomid1 id2 $atomid2 id3 $atomid3 {
      set impr [measure imprp [list $id0 $id1 $id2 $id3] molid $molid]
      set imprH 0
      if $hasH {
        set imprH [measure imprp [list $idH $id1 $id2 $id3] molid $molid]
      }
      # puts "[[atomselect $molid "index $idH $id0 $id1 $id2 $id3"] get name]"
      if {$impr < 0 || $imprH < 0} {
        set seltmp [atomselect $molid "index $id0"]
        set thisResidue [$seltmp get residue]
        $seltmp delete
        puts "chirality) Chirality error in residue $thisResidue (atoms $atomnames):"
        puts "  [chirality_residue_info $thisResidue $molid string]"

        # Don't duplicate entries of chirality errors
        set isDuplicated 0
        set searchResults [lsearch -all $chiralResiduesList $thisResidue]
        if { $searchResults != -1 } {
          foreach lid $searchResults {
            if { [lindex $chiralNamesList $lid] == $atomnames } {
              set isDuplicated 1
            }
          }
        } 
        
        if $isDuplicated {
          puts "  WARNING: Duplicated chirality error. Ignoring..."
        } else {
          puts "  Current chirality error number is $count."
          incr count
          set indexList  [list $id0 $id1 $id2 $id3]
          set indexListH [list $idH $id1 $id2 $id3]
          lappend chiralImpropersList [list $indexList $indexListH]
          lappend chiralResiduesList $thisResidue
          lappend chiralActionsList X
          lappend chiralMovedList -1
          lappend chiralNamesList $atomnames
        }
        puts ""

      }
    }

  }

  return $count
}

# Internal function: return information about a given residue
# outformat can be list or string
proc ::chirality::chirality_residue_info { residue mol outformat } {
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
    error "chirality) Internal error in chirality_residue_info."
  }

  return $returnList
}

proc ::chirality::chirality_show_usage { } {

  puts "Usage: chirality show <current|next|chirality error number|none> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return

}

proc ::chirality::chirality_show { args } {

  global tk_version
  variable chiralResiduesList
  variable chiralNamesList
  variable chiralCurrent
  variable chiralReps

  if {![info exists tk_version]} {
    error "The 'chirality show' command requires VMD running in graphics mode."
  }

  set nargs [llength $args]
  if {$nargs < 1} {
    chirality_show_usage
    error ""
  }
  set cmd [lindex $args 0]

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      default {
        chirality_show_usage
        error "Unrecognized argument."
      }
    }
  }
  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }

  set numErrors [llength $chiralResiduesList]

  # Validate argument
  if { $cmd == {none} } {
  } elseif { $cmd == {next} } {
    incr chiralCurrent
  } elseif { $cmd != {current} } {

    # Make sure argument is an integer and in range
    if { [string is integer $cmd] != 1 } {
      chirality_show_usage
      error "Recognized non-integer argument."
    } elseif { $cmd < 0 || $cmd >= $numErrors } {
      chirality_show_usage
      error "Chiral error requested is out of range."
    } 

    set chiralCurrent $cmd
  }

  # Clean up previous reps
  chirality_reset_reps

  if { $cmd == {none} } {
    return
  }

  # Wrap around if needed
  if { $chiralCurrent == $numErrors } {
    puts "chirality) Reached end of list. Wrapping around to first chirality error."
    set chiralCurrent 0
  }

  set thisResidue [lindex $chiralResiduesList $chiralCurrent]
  set names [lindex $chiralNamesList $chiralCurrent]
  puts "chirality) Showing chirality error ${chiralCurrent}:"
  puts "  [chirality_residue_info $thisResidue $molid string]"

  mol color Name
  mol representation CPK 1.0 0.3 8.0 6.0
  mol material Opaque
  mol selection "residue $thisResidue and name $names"
  mol addrep $molid
  set repid [expr [molinfo $molid get numreps] - 1]
  # repname is guaranteed to be unique within a molecule
  set repname [mol repname $molid $repid]
  lappend chiralReps $molid
  lappend chiralReps $repname

  mol color ColorID 11
  mol representation VDW 0.5 8.0
  mol material Transparent
  mol selection "residue $thisResidue and name [lindex $names 1]"
  mol addrep $molid
  set repid [expr [molinfo $molid get numreps] - 1]
  # repname is guaranteed to be unique within a molecule
  set repname [mol repname $molid $repid]
  lappend chiralReps $molid
  lappend chiralReps $repname

  mol color Segname
  mol representation NewRibbons 0.300000 6.000000 3.000000 0
  mol material Opaque
  mol selection "same residue as (within 30 of (residue $thisResidue))"
  mol addrep $molid
  set repid [expr [molinfo $molid get numreps] - 1]
  # repname is guaranteed to be unique within a molecule
  set repname [mol repname $molid $repid]
  lappend chiralReps $molid
  lappend chiralReps $repname

  mol color Name
  mol representation Licorice 0.1 10.0 10.0
  mol material Opaque
  mol selection "same residue as (within 5 of (residue $thisResidue))"
  mol addrep $molid
  set repid [expr [molinfo $molid get numreps] - 1]
  # repname is guaranteed to be unique within a molecule
  set repname [mol repname $molid $repid]
  lappend chiralReps $molid
  lappend chiralReps $repname


  # Center view on the current residues
  set sel [atomselect $molid "residue $thisResidue and name [lindex $names 1]"]
  set center [measure center $sel]
  $sel delete
  foreach mol [molinfo list] {
    molinfo $mol set center [list $center]
  }
  scale to 0.4
  translate to 0 0 0
  display update

  return

}

proc ::chirality::chirality_list_usage { } {
  puts "Usage: chirality list <list of chirality errors|all> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return
}

proc ::chirality::chirality_list { args } {
  variable chiralResiduesList
  variable chiralActionsList
  variable chiralMovedList
  variable chiralNamesList

  set nargs [llength $args]
  if {$nargs < 1} {
    chirality_list_usage
    error ""
  }

  if { [lindex $args 0] == {all} } {
    set selectedErrors {}
    for {set i 0} {$i < [llength $chiralResiduesList]} {incr i} {
      lappend selectedErrors $i
    }
  } else {
    set selectedErrors [lindex $args 0]
  }

  # Check list of selected chirality errors for errors
  set numErrors [llength $chiralResiduesList]
  foreach thisError $selectedErrors {
    if { [string is integer $thisError] != 1 } {
      chirality_list_usage
      error "Recognized non-integer argument."
    } elseif { $thisError < 0 || $thisError >= $numErrors } {
      chirality_list_usage
      error "Chirality error requested is out of range."
    } 
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -gui { set arg(gui) $val }
      default {
        chirality_list_usage
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

  foreach i $selectedErrors {
    set thisResidue [lindex $chiralResiduesList $i]
    set names [lindex $chiralNamesList $i]
    set action [lindex $chiralActionsList $i]
    set moved [lindex $chiralMovedList $i]
    if $gui {
      lappend returnList [chirality_residue_info $thisResidue $molid list]
      lappend returnList $names
      lappend returnList $action
      lappend returnList $moved
    } else {
      puts "chirality) Chirality error $i (action = $action; moved = $moved):"
      puts "  [chirality_residue_info $thisResidue $molid string]"
      puts "  Atom names: $names"
    }
    incr i
  }

  if {$gui && [info exists returnList]} {
    return $returnList
  }
  return

}

proc ::chirality::chirality_set_usage { } {
  puts "Usage: chirality set <H|X> ?options?"
  puts "Defines which action to take when modifying (at a later step)"
  puts "the given chirality error."
  puts "  H: move hydrogen atom"
  puts "  X: don't modify this chirality error"
  puts "Options:"
  puts "  -cen <chirality error number> (default: current)"
  puts "  -mol <molid> (default: top)"
  return
}

# XXX - Need to allow for "set all"
proc ::chirality::chirality_set { args } {
  variable chiralCurrent
  variable chiralActionsList
  variable chiralResiduesList

  set nargs [llength $args]
  if {$nargs < 1} {
    chirality_set_usage
    error ""
  }
  set action [lindex $args 0]

  if { $action != {H} && $action != {X} } {
    chirality_set_usage
    error "Unrecognized argument."
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -cen { set arg(cen) $val }
      -gui { set arg(gui) $val }
      default {
        chirality_set_usage
        error "Unrecognized argument."
      }
    }
  }

  if { [info exists arg(mol)] } {
    set molid $arg(mol)
  } else {
    set molid [molinfo top]
  }

  if { [info exists arg(cen)] } {
    set thisError $arg(cen)
  } else {
    set thisError $chiralCurrent
  }

  # Make sure argument is an integer and in range
  set numErrors [llength $chiralActionsList]
  if { [string is integer $thisError] != 1 } {
    chirality_set_usage
    error "Recognized non-integer argument."
  } elseif { $thisError < 0 || $thisError >= $numErrors } {
    chirality_set_usage
    error "Chirality error requested is out of range."
  } 

  # Was I called from the GUI?
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  # FIXME: This proc should be taken out of the cispeptide plugin
  if {[::cispeptide::cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  # Assign action to the given chirality error
  lset chiralActionsList $thisError $action

  return
}

# XXX - add an option 'move all'
proc ::chirality::chirality_move_usage { } {
  puts "Usage: chirality move <chiral error number|current> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return
}

proc ::chirality::chirality_move { args } {
  variable chiralCurrent
  variable chiralResiduesList
  variable chiralActionsList
  variable chiralMovedList
  variable chiralImpropersList
  variable clashWithinCutoff 
  variable clashCutoff 

  set nargs [llength $args]
  if {$nargs < 1} {
    chirality_move_usage
    error ""
  }

  if { [lindex $args 0] == {current} } {
    set thisError $chiralCurrent
  } else {
    set thisError [lindex $args 0]
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -gui { set arg(gui) $val }
      default {
        chirality_move_usage
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
  set numErrors [llength $chiralActionsList]
  if { [string is integer $thisError] != 1 } {
    chirality_move_usage
    error "Recognized non-integer argument."
  } elseif { $thisError < 0 || $thisError >= $numErrors } {
    chirality_move_usage
    error "Chirality error requested is out of range."
  } 

  # Was I called from the GUI?
  if { [info exists arg(gui)] } {
    set gui 1
  } else {
    set gui 0
  }

  # FIXME: This proc should be taken out of the cispeptide plugin
  if {[::cispeptide::cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  if { [lindex $chiralMovedList $thisError] != -1 } {
    if $gui {
      tk_messageBox -type ok -message "You cannot move an atom that was already moved."
      return
    } else {
      error "Cannot move an atom that was already moved."
    }
  }

  set moveAtom [lindex $chiralActionsList $thisError]
  set residue [lindex [lindex $chiralResiduesList $thisError] 0]
  switch $moveAtom {
    X { 
      if $gui {
        tk_messageBox -type ok -message "You need to tag an atom for moving first."
        return
      } else {
        error "Action flag for chirality error $thisError is set to X. Did you run 'chirality set'? Aborting..."
      }
    }
    H { 
      set id  [lindex [lindex [lindex $chiralImpropersList $thisError] 1] 0]
      set id2 [lindex [lindex [lindex $chiralImpropersList $thisError] 0] 0]
      set selAtom1 [atomselect $molid "index $id"]
      set selAtom2 [atomselect $molid "index $id2"]
    } 
    default { error "Internal error in chirality_move: Unrecognized action." }
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
  set sel [atomselect $molid "within $clashWithinCutoff of index $id"]
  if { [llength [lindex [measure contacts $clashCutoff $selAtom1 $sel] 0]] } {
    puts "chirality) Warning: moving atom (index $id) introduced clash(es). Moving back by 0.25 times the bond length."
    $selAtom1 moveby [vecscale -0.25 $bondVector]
  }
  $sel delete

  # Store index of the atoms that was moved
  lset chiralMovedList $thisError $id

  $selAtom1 delete
  $selAtom2 delete

  return
}

proc ::chirality::chirality_minimize_usage { } {
  puts "Usage: chirality minimize <list of chirality errors|all> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  return
}

proc ::chirality::chirality_minimize { args } {
  
  global tk_version
  variable chiralResiduesList
  variable chiralMovedList
  variable chiralActionsList
  variable chiralNamesList
  variable minWithinCutoff
  variable chiralSelRestrainAtoms0
  variable chiralSelRestrainAtoms1
  variable chiralSelRestrainAtoms2
  variable chiralSelRestrainAtoms3
  variable chiralSelRestrainAtoms4
  variable chiralSelRestrainAtomsSet
  variable molidBeforeAutoIMD

  if {![info exists tk_version]} {
    error "The 'chirality minimize' command requires VMD running in graphics mode."
  }

  set nargs [llength $args]
  if {$nargs < 1} {
    chirality_minimize_usage
    error ""
  }

  if { [lindex $args 0] == {all} } {
    set selectedErrors {}
    for {set i 0} {$i < [llength $chiralActionsList]} {incr i} {
      lappend selectedErrors $i
    }
  } else {
    set selectedErrors [lindex $args 0]
  }

  # Check list of selected chiral centers for errors
  set numErrors [llength $chiralActionsList]
  foreach thisError $selectedErrors {
    if { [string is integer $thisError] != 1 } {
      chirality_minimize_usage
      error "Recognized non-integer argument."
    } elseif { $thisError < 0 || $thisError >= $numErrors } {
      chirality_minimize_usage
      error "Chiral center requested is out of range."
    } 
  }

  foreach {name val} [lrange $args 1 end] {
    switch -- $name {
      -mol { set arg(mol) $val }
      -gui { set arg(gui) $val }
      -mdff { set arg(mdff) $val}
      default {
        chirality_minimize_usage
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

  # FIXME: This proc should be taken out of the cispeptide plugin
  if {[::cispeptide::cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  # Minimize everything that has an action set, was moved and selected
  set includeResidues {}
  foreach i $selectedErrors {
    set residues [lindex $chiralResiduesList $i]
    set movedAtom [lindex $chiralMovedList $i]
    set action [lindex $chiralActionsList $i]
    if { $action != {X} } {
      if { $movedAtom == -1 } {
        error "Flagged atom for chirality error $i has not been moved yet. Did you run 'chirality move'? Aborting..."
      }
      lappend includeResidues $residues
    }
  }

  # Create selections containing all atoms involved in impropers we
  # want to restrain in order to fix chirality errors. This selections 
  # will later be used to create an extrabonds file containing improper
  # restraints once autoimd is invoked.
  if $chiralSelRestrainAtomsSet {
    uplevel #0 $::chirality::chiralSelRestrainAtoms0 delete
    uplevel #0 $::chirality::chiralSelRestrainAtoms1 delete
    uplevel #0 $::chirality::chiralSelRestrainAtoms2 delete
    uplevel #0 $::chirality::chiralSelRestrainAtoms3 delete
    uplevel #0 $::chirality::chiralSelRestrainAtoms4 delete
  }

  set id0List {}
  set id1List {}
  set id2List {}
  set id3List {}
  set id4List {}
  foreach i $selectedErrors {
    set thisResidue [lindex $chiralResiduesList $i]
    set atomnames [lindex $chiralNamesList $i]
    set sel0 [atomselect $molid "residue $thisResidue and name [lindex $atomnames 0]"]
    set sel1 [atomselect $molid "residue $thisResidue and name [lindex $atomnames 1]"]
    set sel2 [atomselect $molid "residue $thisResidue and name [lindex $atomnames 2]"]
    set sel3 [atomselect $molid "residue $thisResidue and name [lindex $atomnames 3]"]
    set sel4 [atomselect $molid "residue $thisResidue and name [lindex $atomnames 4]"]
    lappend id0List [$sel0 get index]
    lappend id1List [$sel1 get index]
    lappend id2List [$sel2 get index]
    lappend id3List [$sel3 get index]
    lappend id4List [$sel4 get index]
    $sel0 delete
    $sel1 delete
    $sel2 delete
    $sel3 delete
    $sel4 delete
  }

  set chiralSelRestrainAtoms0 [atomselect $molid "index $id0List"]
  set chiralSelRestrainAtoms1 [atomselect $molid "index $id1List"]
  set chiralSelRestrainAtoms2 [atomselect $molid "index $id2List"]
  set chiralSelRestrainAtoms3 [atomselect $molid "index $id3List"]
  set chiralSelRestrainAtoms4 [atomselect $molid "index $id4List"]

  $chiralSelRestrainAtoms0 global
  $chiralSelRestrainAtoms1 global
  $chiralSelRestrainAtoms2 global
  $chiralSelRestrainAtoms3 global
  $chiralSelRestrainAtoms4 global

  set chiralSelRestrainAtomsSet 1
  if { [$chiralSelRestrainAtoms0 num] != [llength $includeResidues] ||
       [$chiralSelRestrainAtoms1 num] != [llength $includeResidues] ||
       [$chiralSelRestrainAtoms2 num] != [llength $includeResidues] ||
       [$chiralSelRestrainAtoms3 num] != [llength $includeResidues] ||
       [$chiralSelRestrainAtoms4 num] != [llength $includeResidues] } {
    error "Internal error in ::chirality::chirality_minimize: Incorrect number of atoms in atom selection for improper restraints."
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
  set mymoltenseltext "($residuesSeltext) or ((water or ion) and same residue as within $minWithinCutoff of ($residuesSeltext))"

  if { [info exists arg(mdff)] && $arg(mdff) == 1 } {
  
    ::MDFFGUI::gui::mdffgui
    set ::MDFFGUI::settings::ChiralityRestraints 1
    set ::MDFFGUI::settings::GridPDBSelText $mymoltenseltext
    #unused currently
    #set ::MDFFGUI::settings::IMDSelText $mymoltenseltext
    set ::MDFFGUI::settings::FixedPDBSelText "not ($mymoltenseltext)"
    set ::MDFFGUI::settings::MolID $molid
    set ::MDFFGUI::settings::IMD 1
    set ::MDFFGUI::settings::IMDWait 1

  } else {
  
    autoimd set moltenseltext $mymoltenseltext
    autoimd set sim_mode minimize
    autoimd set usechirality 1

    # Open autoimd window
    autoimd showwindow
  }
  return
}

# Internal function to be called only by autoimd (not exposed to the user).
# This proc creates an extrabonds file containing improper restraints to 
# fix chirality errors.
# Usage: chirality extrab <extrabonds filename> <autoimd pdb filename>
proc ::chirality::chirality_extrab { args } {

  variable chiralSelRestrainAtoms0
  variable chiralSelRestrainAtoms1
  variable chiralSelRestrainAtoms2
  variable chiralSelRestrainAtoms3
  variable chiralSelRestrainAtoms4
  variable chiralSelRestrainAtomsSet
  variable chiralExtraBondsK
  variable targetImprpH
  variable targetImprpNoH
  variable id0ListAutoIMD
  variable id1ListAutoIMD
  variable id2ListAutoIMD
  variable id3ListAutoIMD
  variable id4ListAutoIMD

  if { $chiralSelRestrainAtomsSet == 0 } {
    error "Internal error in ::chirality::chirality_extrab: chiralSelRestrainAtomsSet is 0."
  }

  set nargs [llength $args]
  if {$nargs != 2} {
    error "Internal error in ::chirality::chirality_extrab: Wrong number of arguments."
  }
  set extrabFileName [lindex $args 0]
  set autoimdPDB [lindex $args 1]

  set tmpmol [mol new $autoimdPDB type pdb waitfor all]
  set selAutoIMD [atomselect $tmpmol "all"]
  set c0 [measure contacts 0.1 $selAutoIMD $chiralSelRestrainAtoms0]
  set c1 [measure contacts 0.1 $selAutoIMD $chiralSelRestrainAtoms1]
  set c2 [measure contacts 0.1 $selAutoIMD $chiralSelRestrainAtoms2]
  set c3 [measure contacts 0.1 $selAutoIMD $chiralSelRestrainAtoms3]
  set c4 [measure contacts 0.1 $selAutoIMD $chiralSelRestrainAtoms4]
  $selAutoIMD delete

  set sel0 [atomselect $tmpmol "index [lindex $c0 0]"]
  set sel1 [atomselect $tmpmol "index [lindex $c1 0]"]
  set sel2 [atomselect $tmpmol "index [lindex $c2 0]"]
  set sel3 [atomselect $tmpmol "index [lindex $c3 0]"]
  set sel4 [atomselect $tmpmol "index [lindex $c4 0]"]

  if { [$sel0 num] != [$chiralSelRestrainAtoms0 num] ||
       [$sel1 num] != [$chiralSelRestrainAtoms1 num] ||
       [$sel2 num] != [$chiralSelRestrainAtoms2 num] ||
       [$sel3 num] != [$chiralSelRestrainAtoms3 num] ||
       [$sel4 num] != [$chiralSelRestrainAtoms4 num] } {
    $sel0 delete
    $sel1 delete
    $sel2 delete
    $sel3 delete
    $sel4 delete
    mol delete $tmpmol
    error "Internal error in ::chirality::chirality_extrab: atom selections of inconsistent lenghts."
  }

  # Save indices of atoms in active chiral centers so that we can adjust
  # the AutoIMD representations later
  set id0ListAutoIMD [$sel0 get index] 
  set id1ListAutoIMD [$sel1 get index] 
  set id2ListAutoIMD [$sel2 get index] 
  set id3ListAutoIMD [$sel3 get index] 
  set id4ListAutoIMD [$sel4 get index] 

  set out [open $extrabFileName w]
  foreach id0 $id0ListAutoIMD id1 $id1ListAutoIMD id2 $id2ListAutoIMD id3 $id3ListAutoIMD id4 $id4ListAutoIMD {
    # Improper involving hydrogen
    puts $out "improper $id0 $id2 $id3 $id4 $chiralExtraBondsK $targetImprpH"
    # Improper not involving hydrogen
    puts $out "improper $id1 $id2 $id3 $id4 $chiralExtraBondsK $targetImprpNoH"
  }
  close $out

  $sel0 delete
  $sel1 delete
  $sel2 delete
  $sel3 delete
  $sel4 delete
  mol delete $tmpmol

  return
}

# Internal function: set up appropriate reps. To be called by autoimd.
proc ::chirality::chirality_autoimd_reps { } {
  variable molidBeforeAutoIMD
  variable id0ListAutoIMD
  variable id1ListAutoIMD
  variable id2ListAutoIMD
  variable id3ListAutoIMD
  variable id4ListAutoIMD

  mol off $molidBeforeAutoIMD

  set molAutoIMD [molinfo top]
  mol showrep $molAutoIMD 0 0
  mol showrep $molAutoIMD 1 0
  mol showrep $molAutoIMD 2 0

  mol color Name
  mol representation CPK 1.0 0.3 8.0 6.0
  mol material Opaque
  mol selection "index $id0ListAutoIMD $id1ListAutoIMD $id2ListAutoIMD $id3ListAutoIMD $id4ListAutoIMD"
  mol addrep $molAutoIMD

  mol color ColorID 11
  mol representation VDW 0.5 8.0
  mol material Transparent
  mol selection "index $id1ListAutoIMD"
  mol addrep $molAutoIMD

  mol color Segname
  mol representation NewRibbons 0.300000 6.000000 3.000000 0
  mol material Opaque
  mol selection "same residue as (within 30 of (same residue as index $id1ListAutoIMD))"
  mol addrep $molAutoIMD

  mol color Name
  mol representation Licorice 0.1 10.0 10.0
  mol material Opaque
  mol selection "same residue as (within 5 of (same residue as index $id1ListAutoIMD))"
  mol addrep $molAutoIMD

  display update
  # Center view on the current residues
  set sel [atomselect $molAutoIMD "index $id1ListAutoIMD"]
  set center [measure center $sel]
  $sel delete
  molinfo $molAutoIMD set center [list $center]
  scale to 0.4
  translate to 0 0 0
  display update

  return
}

# Internal function: Turn original molecule back on. To be called by autoimd.
proc ::chirality::chirality_autoimd_finish { } {
  variable molidBeforeAutoIMD
  mol on $molidBeforeAutoIMD
  autoimd set usechirality 0
  return
}

proc ::chirality::chirality_restrain_usage { } {

  variable defaultSelText
  puts "Usage: chirality restrain -o <extrabonds file> ?options?"
  puts "Options:"
  puts "  -mol <molid> (default: top)"
  puts "  -seltext <atom selection text> (default: $defaultSelText)"
  return

}

# Generate an extrabonds file for NAMD defining improper restraints (two per
# chiral center) to prevent chirality changes. Here we assume the structure
# has hydrogen atoms.
proc ::chirality::chirality_restrain { args } {

  variable defaultSelText
  variable chiralCenters
  variable chiralExtraBondsK

  set nargs [llength $args]
  if {$nargs == 0} {
    chirality_restrain_usage
    error ""
  }

  foreach {name val} $args {
    switch -- $name {
      -mol { set arg(mol) $val }
      -o { set arg(o) $val }
      -seltext { set arg(seltext) $val }
      -gui { set arg(gui) $val }
      default { 
        chirality_restrain_usage
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
    set molid [molinfo top]
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

  if { [info exists arg(o)] } {
    set outFile $arg(o)
  } else {
    if $gui {
      tk_messageBox -type ok -message "Missing output extrabonds file name."
      return
    } else {
      chirality_restrain_usage
      error "Missing output file."
    }
  }

  # FIXME: This proc should be taken out of the cispeptide plugin
  if {[::cispeptide::cispeptide_check_files $molid 1 $gui] == 1} {
    return 1
  }

  set chiralAtoms [::chirality::find_chiral_centers $seltext $molid]
  set out [open $outFile w]

  foreach chiral $chiralAtoms {
    set atomidH [lindex $chiral 0]
    set atomid0 [lindex $chiral 1]
    set atomid1 [lindex $chiral 2]
    set atomid2 [lindex $chiral 3]
    set atomid3 [lindex $chiral 4]

    foreach idH $atomidH id0 $atomid0 id1 $atomid1 id2 $atomid2 id3 $atomid3 {
      set impr [measure imprp [list $id0 $id1 $id2 $id3] molid $molid]
      set imprH [measure imprp [list $idH $id1 $id2 $id3] molid $molid]

      puts $out "improper $id0 $id1 $id2 $id3 $chiralExtraBondsK $impr"
      puts $out "improper $idH $id1 $id2 $id3 $chiralExtraBondsK $imprH"
    }

  }

  close $out
  return
}

# Return a list with the intersection of the input lists. Note that this
# functionality is provided in TclX, but we code it here to avoid
# dependencies.
proc ::chirality::list_intersect { listOfLists } {

  set numLists [llength $listOfLists]

  # Start with the first list. For each of the remaining
  # lists, remove missing elements.
  set returnList [lindex $listOfLists 0]
  for {set i 1} {$i < $numLists} {incr i} {
    set copyList $returnList
    set testList [lindex $listOfLists $i]
    for {set j 0} {$j < [llength $copyList]} {incr j} {
      set pos [lsearch -exact $testList [lindex $copyList $j]]
      if {$pos != -1} { 
        set testList [lreplace $testList $pos $pos]
      } else {
        set pos2 [lsearch -exact $returnList [lindex $copyList $j]]
        set returnList [lreplace $returnList $pos2 $pos2]
      }
    }
  }

  return $returnList

}

